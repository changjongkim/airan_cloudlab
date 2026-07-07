// PROOF 6 — defer cudaFree but actually free at exit.
//
// Goal: Show that if cudaFree's host blocking is removed, L1 latency recovers.
// Method: Intercept cudaFree, push ptr to FIFO, return success immediately.
//         At process exit (atexit), drain the FIFO by calling real cudaFree on all.
//         No memory leak across process lifetime; per-call host blocking eliminated.
//
// Build: gcc -shared -fPIC -O2 -o e6_defer.so proof_56_shim.c -ldl -lpthread
// Use:   LD_PRELOAD=/path/e6_defer.so python3 real_l1.py ...

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <time.h>
#include <pthread.h>

typedef int (*cudaFree_t)(void *);
static cudaFree_t real_cudaFree = NULL;

#define CAP (1 << 20)   // 1M ptr capacity
static void   *queue[CAP];
static long    head = 0;
static long    tail = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static long long n_intercepted = 0;
static long long n_drained     = 0;
static double    t_intercept_us = 0.0;
static double    t_drain_us     = 0.0;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void load_real(void) {
    if (!real_cudaFree) {
        real_cudaFree = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
        fprintf(stderr, "[e6_defer] real_cudaFree=%p\n", real_cudaFree);
    }
}

int cudaFree(void *devPtr) {
    if (!devPtr) return 0;  // cudaSuccess on NULL
    load_real();
    double t0 = now_us();
    pthread_mutex_lock(&mu);
    if (tail - head < CAP) {
        queue[tail % CAP] = devPtr;
        tail++;
        n_intercepted++;
    } else {
        // queue full — fall back to real cudaFree to avoid OOM
        pthread_mutex_unlock(&mu);
        return real_cudaFree(devPtr);
    }
    pthread_mutex_unlock(&mu);
    t_intercept_us += (now_us() - t0);
    if ((n_intercepted % 1000) == 0)
        fprintf(stderr, "[e6_defer] intercepted=%lld queued (avg intercept=%.2fus)\n",
                n_intercepted, t_intercept_us / n_intercepted);
    return 0;  // cudaSuccess
}

// Same hook for versioned symbol (versioned linker resolution catches this).
int cudaFree_v3020(void *devPtr) { return cudaFree(devPtr); }

__attribute__((destructor))
static void drain_at_exit(void) {
    load_real();
    if (!real_cudaFree) return;
    fprintf(stderr, "[e6_defer] DRAIN START: queued=%ld\n", tail - head);
    double t0 = now_us();
    pthread_mutex_lock(&mu);
    while (head < tail) {
        void *p = queue[head % CAP];
        head++;
        pthread_mutex_unlock(&mu);
        real_cudaFree(p);
        n_drained++;
        pthread_mutex_lock(&mu);
    }
    pthread_mutex_unlock(&mu);
    t_drain_us = now_us() - t0;
    fprintf(stderr, "[e6_defer] FINAL intercepted=%lld drained=%lld total_intercept=%.1fms total_drain=%.1fms\n",
            n_intercepted, n_drained, t_intercept_us/1000.0, t_drain_us/1000.0);
}
