/*
 * cudaMemPool_shim.c — Option B mitigation shim (FULL)
 *
 * Replaces both cudaMalloc and cudaFree with stream-ordered memory pool
 * variants. This is the COMPLETE variant — every allocation goes through
 * an async pool, so every free is truly async.
 *
 * Behavior:
 *   cudaMalloc(ptr, size)      → cudaMallocFromPoolAsync(ptr, size, pool, stream)
 *   cudaFree(ptr)              → cudaFreeAsync(ptr, stream)
 *   (fallback to sync on error)
 *
 * Advantages over Option A:
 *   - Guaranteed async free (pointer definitely from async pool)
 *   - Pool internally recycles memory → less fragmentation
 *
 * Risks:
 *   - Some code may assume cudaMalloc allocates immediately (before kernel)
 *   - Pool may hold onto memory longer than expected
 *   - cuPHY may not tolerate this
 *
 * Build: gcc -shared -fPIC -O2 cudaMemPool_shim.c -o cudaMemPool.so -ldl -lpthread
 * Use:   LD_PRELOAD=/path/to/cudaMemPool.so ./app
 *
 * Runtime env:
 *   CUPOOL_LOG=1  → debug info
 *   CUPOOL_RELEASE_THRESHOLD_MB=<n> → mempool release threshold (default 1024)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>

typedef int    cudaError_t;
typedef void*  cudaStream_t;
typedef struct cudaMemPool_st* cudaMemPool_t;

typedef cudaError_t (*cudaMalloc_t)(void **, size_t);
typedef cudaError_t (*cudaFree_t)(void *);
typedef cudaError_t (*cudaMallocFromPoolAsync_t)(void **, size_t, cudaMemPool_t, cudaStream_t);
typedef cudaError_t (*cudaFreeAsync_t)(void *, cudaStream_t);
typedef cudaError_t (*cudaMallocAsync_t)(void **, size_t, cudaStream_t);
typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t *);
typedef cudaError_t (*cudaDeviceGetDefaultMemPool_t)(cudaMemPool_t *, int);
typedef cudaError_t (*cudaMemPoolSetAttribute_t)(cudaMemPool_t, int, void *);

static cudaMalloc_t                real_cudaMalloc               = NULL;
static cudaFree_t                  real_cudaFree                 = NULL;
static cudaMallocAsync_t           real_cudaMallocAsync          = NULL;
static cudaMallocFromPoolAsync_t   real_cudaMallocFromPoolAsync  = NULL;
static cudaFreeAsync_t             real_cudaFreeAsync            = NULL;

static cudaStream_t   pool_stream = NULL;
static cudaMemPool_t  pool        = NULL;
static pthread_once_t init_once   = PTHREAD_ONCE_INIT;
static int            log_enabled = 0;
static long           n_malloc    = 0;
static long           n_free      = 0;
static long           n_fallback  = 0;

static void* resolve_sym(const char *name) {
    void *sym = dlsym(RTLD_NEXT, name);
    if (sym) return sym;
    void *libcudart = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
    if (!libcudart) libcudart = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
    return libcudart ? dlsym(libcudart, name) : NULL;
}

static void init_shim(void) {
    real_cudaMalloc              = (cudaMalloc_t) resolve_sym("cudaMalloc");
    real_cudaFree                = (cudaFree_t) resolve_sym("cudaFree");
    real_cudaMallocAsync         = (cudaMallocAsync_t) resolve_sym("cudaMallocAsync");
    real_cudaMallocFromPoolAsync = (cudaMallocFromPoolAsync_t) resolve_sym("cudaMallocFromPoolAsync");
    real_cudaFreeAsync           = (cudaFreeAsync_t) resolve_sym("cudaFreeAsync");

    cudaStreamCreate_t create = (cudaStreamCreate_t) resolve_sym("cudaStreamCreate");
    if (create) create(&pool_stream);

    cudaDeviceGetDefaultMemPool_t get_pool =
        (cudaDeviceGetDefaultMemPool_t) resolve_sym("cudaDeviceGetDefaultMemPool");
    if (get_pool) get_pool(&pool, 0);

    // Set release threshold — keep memory in pool between iterations
    cudaMemPoolSetAttribute_t set_attr =
        (cudaMemPoolSetAttribute_t) resolve_sym("cudaMemPoolSetAttribute");
    const char *env_thresh = getenv("CUPOOL_RELEASE_THRESHOLD_MB");
    long thresh_mb = env_thresh ? atol(env_thresh) : 1024;
    long thresh_bytes = thresh_mb * 1024L * 1024L;
    if (set_attr && pool) {
        set_attr(pool, /* cudaMemPoolAttrReleaseThreshold = */ 4, &thresh_bytes);
    }

    log_enabled = (getenv("CUPOOL_LOG") != NULL);
    if (log_enabled) {
        fprintf(stderr, "[cudaMemPool_shim] init: malloc=%p free=%p mallocAsync=%p freeAsync=%p pool=%p stream=%p thresh=%ldMB\n",
                real_cudaMalloc, real_cudaFree, real_cudaMallocAsync, real_cudaFreeAsync,
                pool, pool_stream, thresh_mb);
    }
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    pthread_once(&init_once, init_shim);

    if (real_cudaMallocFromPoolAsync && pool) {
        cudaError_t rc = real_cudaMallocFromPoolAsync(devPtr, size, pool, pool_stream);
        if (rc == 0) {
            __sync_fetch_and_add(&n_malloc, 1);
            return rc;
        }
    } else if (real_cudaMallocAsync) {
        cudaError_t rc = real_cudaMallocAsync(devPtr, size, pool_stream);
        if (rc == 0) {
            __sync_fetch_and_add(&n_malloc, 1);
            return rc;
        }
    }

    __sync_fetch_and_add(&n_fallback, 1);
    if (real_cudaMalloc) return real_cudaMalloc(devPtr, size);
    return 1;
}

cudaError_t cudaMalloc_v3020(void **devPtr, size_t size) {
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaFree(void *devPtr) {
    pthread_once(&init_once, init_shim);
    if (!devPtr) return 0;

    if (real_cudaFreeAsync) {
        cudaError_t rc = real_cudaFreeAsync(devPtr, pool_stream);
        if (rc == 0) {
            __sync_fetch_and_add(&n_free, 1);
            return rc;
        }
    }

    __sync_fetch_and_add(&n_fallback, 1);
    if (real_cudaFree) return real_cudaFree(devPtr);
    return 1;
}

cudaError_t cudaFree_v3020(void *devPtr) {
    return cudaFree(devPtr);
}

__attribute__((destructor)) static void report(void) {
    if (log_enabled || getenv("CUPOOL_LOG")) {
        fprintf(stderr, "[cudaMemPool_shim] malloc=%ld free=%ld fallback=%ld\n",
                n_malloc, n_free, n_fallback);
    }
}
