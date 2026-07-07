// E9 LD_PRELOAD shim — calls cudaDeviceSynchronize before every cudaFree.
// Goal: discriminate whether cudaFree's blocking time is from
//   H1: waiting for pending kernel work to finish (kernel completion sync), or
//   H2/H3: memory-subsystem state / driver lock contention.
//
// If H1 dominates: after explicit sync, real cudaFree returns quickly
// If H2/H3:        real cudaFree still slow even after sync drained the work queue.
//
// Build:  gcc -shared -fPIC -o e9_sync.so e9_sync_shim.c -ldl
// Use:    LD_PRELOAD=/path/e9_sync.so python3 real_l1.py ...

#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <time.h>

typedef int (*cudaFree_t)(void *);
typedef int (*cudaDeviceSync_t)(void);

static cudaFree_t          real_free = NULL;
static cudaDeviceSync_t    real_sync = NULL;
static long long           free_count = 0;
static double              sync_total_us = 0.0;
static double              free_total_us = 0.0;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int cudaFree(void *devPtr) {
    if (!real_free) {
        real_free = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
        real_sync = (cudaDeviceSync_t)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
        fprintf(stderr, "[e9_shim] hooked cudaFree (real_free=%p sync=%p)\n", real_free, real_sync);
    }
    if (!devPtr) return real_free(devPtr);

    double t0 = now_us();
    if (real_sync) real_sync();
    double t1 = now_us();
    int rc = real_free(devPtr);
    double t2 = now_us();

    sync_total_us += (t1 - t0);
    free_total_us += (t2 - t1);
    free_count++;

    if ((free_count % 1000) == 0) {
        fprintf(stderr, "[e9_shim] n=%lld avg_sync=%.2fus avg_free=%.2fus\n",
                free_count,
                sync_total_us / free_count,
                free_total_us / free_count);
    }
    return rc;
}

__attribute__((destructor))
static void e9_summary(void) {
    if (free_count > 0) {
        fprintf(stderr, "[e9_shim] FINAL n=%lld avg_sync=%.2fus avg_free=%.2fus total_sync=%.2fms total_free=%.2fms\n",
                free_count,
                sync_total_us / free_count,
                free_total_us / free_count,
                sync_total_us / 1000.0,
                free_total_us / 1000.0);
    }
}
