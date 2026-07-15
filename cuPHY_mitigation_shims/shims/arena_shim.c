/*
 * arena_shim.c — Hypothesis A: Persistent Buffer Pool simulation
 *
 * Simulates cuPHY refactored to reuse buffers across frames.
 *
 * Behavior:
 *   cudaMalloc(size):
 *     - Look up size in cache
 *     - If HIT: return cached pointer (no real cudaMalloc)  ← key
 *     - If MISS: real cudaMalloc + cache
 *   cudaFree(ptr):
 *     - Just mark pointer as "available for reuse"
 *     - No real cudaFree (so no host-blocking, no implicit sync)
 *     - Buffer stays allocated across frames
 *
 * This differs from:
 *   - proof_56_shim: defers cudaFree, but still cudaMalloc's every frame
 *   - cudaMemPool_shim: uses cudaMallocFromPoolAsync, but still allocates every frame
 *
 * arena_shim: eliminates BOTH per-frame malloc AND per-frame free.
 * Should reveal whether the true fix is "no cudaFree in hot path" OR
 * "no cudaMalloc/cudaFree at all in hot path".
 *
 * Env:
 *   ARENA_LOG=1  → print hit/miss stats
 *   ARENA_STRICT=1 → return error if buffer already in use (default: reuse anyway)
 *
 * Build: gcc -shared -fPIC -O2 arena_shim.c -o arena.so -ldl -lpthread
 * Use:   LD_PRELOAD=/path/arena.so ./app
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>

typedef int cudaError_t;
typedef cudaError_t (*cudaMalloc_t)(void **, size_t);
typedef cudaError_t (*cudaFree_t)(void *);

static cudaMalloc_t real_cudaMalloc = NULL;
static cudaFree_t   real_cudaFree   = NULL;

#define MAX_ENTRIES 4096
typedef struct {
    size_t size;
    void  *ptr;
    int    in_use;
} arena_entry_t;
static arena_entry_t entries[MAX_ENTRIES];
static int n_entries = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static long n_malloc_hits    = 0;
static long n_malloc_misses  = 0;
static long n_free_intercepts = 0;
static long n_free_notfound  = 0;
static long bytes_allocated  = 0;
static int  log_enabled      = 0;
static int  strict_mode      = 0;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;

static void init_shim(void) {
    real_cudaMalloc = (cudaMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
    if (!real_cudaMalloc) {
        void *lib = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
        if (!lib) lib = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
        if (lib) real_cudaMalloc = (cudaMalloc_t)dlsym(lib, "cudaMalloc");
    }
    real_cudaFree = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
    if (!real_cudaFree) {
        void *lib = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
        if (!lib) lib = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
        if (lib) real_cudaFree = (cudaFree_t)dlsym(lib, "cudaFree");
    }
    log_enabled = getenv("ARENA_LOG") && strcmp(getenv("ARENA_LOG"), "1") == 0;
    strict_mode = getenv("ARENA_STRICT") && strcmp(getenv("ARENA_STRICT"), "1") == 0;
    if (log_enabled) {
        fprintf(stderr, "[arena] init: real_cudaMalloc=%p real_cudaFree=%p strict=%d\n",
                real_cudaMalloc, real_cudaFree, strict_mode);
    }
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    pthread_once(&init_once, init_shim);
    if (!devPtr) return 1;

    pthread_mutex_lock(&mu);
    // Look for cached free buffer of matching size
    for (int i = 0; i < n_entries; i++) {
        if (entries[i].size == size && !entries[i].in_use) {
            entries[i].in_use = 1;
            *devPtr = entries[i].ptr;
            n_malloc_hits++;
            pthread_mutex_unlock(&mu);
            if (log_enabled && (n_malloc_hits % 1000) == 0)
                fprintf(stderr, "[arena] hits=%ld misses=%ld entries=%d bytes=%ld\n",
                        n_malloc_hits, n_malloc_misses, n_entries, bytes_allocated);
            return 0;
        }
    }
    // No hit — real allocation
    pthread_mutex_unlock(&mu);
    cudaError_t rc = real_cudaMalloc(devPtr, size);
    if (rc != 0) return rc;
    pthread_mutex_lock(&mu);
    if (n_entries < MAX_ENTRIES) {
        entries[n_entries].size = size;
        entries[n_entries].ptr = *devPtr;
        entries[n_entries].in_use = 1;
        n_entries++;
        bytes_allocated += size;
    }
    n_malloc_misses++;
    pthread_mutex_unlock(&mu);
    return 0;
}

cudaError_t cudaFree(void *devPtr) {
    pthread_once(&init_once, init_shim);
    if (!devPtr) return 0;

    pthread_mutex_lock(&mu);
    for (int i = 0; i < n_entries; i++) {
        if (entries[i].ptr == devPtr) {
            entries[i].in_use = 0;
            n_free_intercepts++;
            pthread_mutex_unlock(&mu);
            return 0;  // cudaSuccess, but no real free
        }
    }
    n_free_notfound++;
    pthread_mutex_unlock(&mu);
    // Not tracked → real free (safety)
    if (real_cudaFree) return real_cudaFree(devPtr);
    return 0;
}

// Versioned symbol aliases
cudaError_t cudaMalloc_v3020(void **devPtr, size_t size) {
    return cudaMalloc(devPtr, size);
}
cudaError_t cudaFree_v3020(void *devPtr) {
    return cudaFree(devPtr);
}

__attribute__((destructor)) static void report(void) {
    fprintf(stderr, "[arena] FINAL malloc_hits=%ld misses=%ld free_intercepts=%ld not_found=%ld entries=%d bytes=%ld\n",
            n_malloc_hits, n_malloc_misses, n_free_intercepts, n_free_notfound, n_entries, bytes_allocated);

    // Best-effort release at shutdown
    if (real_cudaFree) {
        for (int i = 0; i < n_entries; i++) {
            real_cudaFree(entries[i].ptr);
        }
    }
}
