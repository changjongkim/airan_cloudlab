"""CUDA green contexts through the driver API (ctypes): a context that may use only some SMs of the GPU."""
import ctypes

cuda = ctypes.CDLL("libcuda.so.1")


class SmRes(ctypes.Structure):
    _fields_ = [("smCount", ctypes.c_uint)]


class _U(ctypes.Union):
    _fields_ = [("sm", SmRes), ("_oversize", ctypes.c_ubyte * 48)]


class DevRes(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("_pad", ctypes.c_ubyte * 92), ("u", _U)]


def check(code, what):
    if code != 0:
        raise RuntimeError(f"{what}: CUDA error {code}")


def init(device=0):
    check(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
    primary = ctypes.c_void_p()
    check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(primary), dev), "cuDevicePrimaryCtxRetain")
    check(cuda.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
    res = DevRes()
    check(cuda.cuDeviceGetDevResource(dev, ctypes.byref(res), 1), "cuDeviceGetDevResource")
    return dev, primary, res


def green(dev, res, sms, rest_side=False):
    """A green context on the first `sms` SMs of the split, or with rest_side on the SMs that the split leaves
    -> (green ctx, CUcontext, SMs of the context, SMs of the other side)."""
    groups = (DevRes * 1)()
    n = ctypes.c_uint(1)
    rest = DevRes()
    check(cuda.cuDevSmResourceSplitByCount(groups, ctypes.byref(n), ctypes.byref(res), ctypes.byref(rest), 0, sms), "split")
    desc = ctypes.c_void_p()
    if rest_side:
        groups, rest = (DevRes * 1)(rest), groups[0]
    check(cuda.cuDevResourceGenerateDesc(ctypes.byref(desc), groups, 1), "desc")
    g = ctypes.c_void_p()
    check(cuda.cuGreenCtxCreate(ctypes.byref(g), desc, dev, 1), "cuGreenCtxCreate")
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxFromGreenCtx(ctypes.byref(ctx), g), "cuCtxFromGreenCtx")
    return g, ctx, int(groups[0].u.sm.smCount), int(rest.u.sm.smCount)


def current():
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    return ctx.value


def use(ctx):
    check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
