/*
 * cudaFreeAsync_shim.c — Option A mitigation shim
 *
 * Replaces synchronous cudaFree() calls with cudaFreeAsync() on a
 * shim-owned stream. This is the SIMPLE variant.
 *
 * Behavior:
 *   - Original code calls cudaFree(ptr)
 *   - Shim intercepts and calls cudaFreeAsync(ptr, shim_stream)
 *   - Host does NOT block (in theory)
 *
 * Caveat:
 *   cudaFreeAsync accepts pointers regardless of allocation source.
 *   HOWEVER, if the pointer was allocated by regular cudaMalloc (not
 *   cudaMallocAsync/cudaMallocFromPoolAsync), cudaFreeAsync may still
 *   fall back to sync internally. Test empirically.
 *
 * Build: gcc -shared -fPIC -O2 cudaFreeAsync_shim.c -o cudaFreeAsync.so -ldl -lpthread
 * Use:   LD_PRELOAD=/path/to/cudaFreeAsync.so ./app
 *
 * Runtime env:
 *   CUFREE_ASYNC_LOG=1  → print debug info
 *   CUFREE_ASYNC_STREAM=0 → use NULL/default stream (else create shim stream)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>

typedef int cudaError_t;
typedef void* cudaStream_t;

typedef cudaError_t (*cudaFree_t)(void *);
typedef cudaError_t (*cudaFreeAsync_t)(void *, cudaStream_t);
typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t *);

static cudaFree_t       real_cudaFree       = NULL;
static cudaFreeAsync_t  real_cudaFreeAsync  = NULL;
static cudaStream_t     shim_stream         = NULL;
static pthread_once_t   init_once           = PTHREAD_ONCE_INIT;
static int              use_default_stream  = 0;
static int              log_enabled         = 0;
static long             n_intercepted       = 0;
static long             n_fallback          = 0;

static void init_shim(void) {
    // Resolve real cudaFree
    real_cudaFree = dlsym(RTLD_NEXT, "cudaFree");
    if (!real_cudaFree) {
        void *libcudart = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
        if (!libcudart) libcudart = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
        if (libcudart) real_cudaFree = dlsym(libcudart, "cudaFree");
    }
    real_cudaFreeAsync = dlsym(RTLD_NEXT, "cudaFreeAsync");
    if (!real_cudaFreeAsync) {
        void *libcudart = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
        if (!libcudart) libcudart = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
        if (libcudart) real_cudaFreeAsync = dlsym(libcudart, "cudaFreeAsync");
    }

    const char *env_stream = getenv("CUFREE_ASYNC_STREAM");
    use_default_stream = (env_stream && strcmp(env_stream, "0") == 0);
    const char *env_log = getenv("CUFREE_ASYNC_LOG");
    log_enabled = (env_log && strcmp(env_log, "1") == 0);

    if (!use_default_stream) {
        cudaStreamCreate_t create = dlsym(RTLD_NEXT, "cudaStreamCreate");
        if (create) create(&shim_stream);
    }
    if (log_enabled) {
        fprintf(stderr, "[cudaFreeAsync_shim] init: real_cudaFree=%p real_cudaFreeAsync=%p stream=%p\n",
                real_cudaFree, real_cudaFreeAsync, shim_stream);
    }
}

cudaError_t cudaFree(void *devPtr) {
    pthread_once(&init_once, init_shim);
    if (!devPtr) return 0;

    if (real_cudaFreeAsync) {
        cudaError_t rc = real_cudaFreeAsync(devPtr, shim_stream);
        __sync_fetch_and_add(&n_intercepted, 1);
        return rc;
    }

    __sync_fetch_and_add(&n_fallback, 1);
    if (real_cudaFree) return real_cudaFree(devPtr);
    return 1;
}

cudaError_t cudaFree_v3020(void *devPtr) {
    return cudaFree(devPtr);
}

__attribute__((destructor)) static void report(void) {
    if (log_enabled || getenv("CUFREE_ASYNC_LOG")) {
        fprintf(stderr, "[cudaFreeAsync_shim] intercepted=%ld fallback=%ld\n",
                n_intercepted, n_fallback);
    }
}
