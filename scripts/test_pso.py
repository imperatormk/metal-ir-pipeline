#!/usr/bin/env python3
"""Test if an IR file produces a working metallib through our serializer."""
import sys, ctypes, os, tempfile

IR_FILE = sys.argv[1]
ir = open(IR_FILE).read()

lib = ctypes.CDLL(os.environ.get('METALIR_DYLIB_PATH',
    os.path.expanduser('~/projects/oss/metal-ir-pipeline/build/lib/Bridge/libMetalIRBridge.dylib')))
lib.metalir_compile.restype = ctypes.c_void_p
lib.metalir_compile.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p, ctypes.c_int]
lib.metalir_free.argtypes = [ctypes.c_void_p]

out_len = ctypes.c_uint64(0)
errbuf = ctypes.create_string_buffer(512)
ptr = lib.metalir_compile(ir.encode(), ctypes.byref(out_len), errbuf, 512)
if not ptr:
    print(f"COMPILE_FAIL: {errbuf.value.decode()}")
    sys.exit(1)

data = bytes((ctypes.c_ubyte * out_len.value).from_address(ptr))
lib.metalir_free(ptr)

# Write to temp file and test PSO
tmp = tempfile.NamedTemporaryFile(suffix='.metallib', delete=False)
tmp.write(data)
tmp.close()

try:
    sys.path.insert(0, os.path.expanduser('~/projects/oss/triton-apple/src'))
    from triton_apple_backend import metal_utils
    mlib = metal_utils.load_metallib(data)
    # Find first function
    names = mlib.function_names
    fn = mlib.get_function(names[0])
    print(f"OK threads={fn.max_total_threads_per_threadgroup}")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
finally:
    os.unlink(tmp.name)
