#!/usr/bin/env python3
"""Binary-search which IR body lines crash the Metal GPU JIT compiler.

Uses metal-ir-pipeline (metal-ir-opt) to compile, then loads via metal_utils.

Usage:
    python3 bisect_ir.py <path-to-failing.ll>
    python3 bisect_ir.py <path-to-failing.ll> --verbose
"""

import subprocess, sys, tempfile, os, re, ctypes

IR_FILE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/failing.ll"
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
METAL_IR_OPT = os.path.expanduser("~/projects/oss/metal-ir-pipeline/build/tools/metal-ir-opt")
TIMEOUT = 30

with open(IR_FILE) as f:
    lines = f.readlines()

# ── Split into header / body / footer ──────────────────────────────
body_start = None
body_end = None
for i, line in enumerate(lines):
    if body_start is None and line.strip().startswith("define "):
        body_start = i + 1
    if body_start and body_end is None and line.strip() == "}":
        body_end = i

if body_start is None or body_end is None:
    print("ERROR: Could not find `define ... { ... }` block")
    sys.exit(1)

header = lines[:body_start]
footer = lines[body_end:]
body = lines[body_start:body_end]
body = [l for l in body if l.strip() != "ret void"]

# Find kernel name for PSO creation
kernel_name = None
for line in header:
    m = re.search(r'define\s+void\s+@(\w+)\(', line)
    if m:
        kernel_name = m.group(1)
        break

print(f"Kernel: {kernel_name}")
print(f"Header: lines 1-{body_start}")
print(f"Body:   lines {body_start+1}-{body_end} ({len(body)} instructions)")
print(f"Footer: lines {body_end+1}-{len(lines)}")
print()

# ── Helpers ────────────────────────────────────────────────────────
_DEF_RE = re.compile(r'^\s+(%[\w.]+)\s*=')
_REF_RE = re.compile(r'%([\w.]+)')
_PARAM_RE = re.compile(r'define\s+\w+\s+@\w+\(([^)]*)\)')
param_count = 0
for line in header:
    m = _PARAM_RE.search(line)
    if m:
        param_count = m.group(1).count(',') + 1
        break
PARAM_SSAS = {f'%{i}' for i in range(param_count)}
# Also add named params
for line in header:
    for m in re.finditer(r'%(\w+)', line):
        if not m.group(1).isdigit():
            PARAM_SSAS.add(f'%{m.group(1)}')


def make_ir(body_lines):
    defined = set(PARAM_SSAS)
    kept = []
    for line in body_lines:
        dm = _DEF_RE.match(line)
        def_name = dm.group(1) if dm else None
        if dm:
            rhs = line.split('=', 1)[1]
        else:
            rhs = line
        refs = {f'%{m}' for m in _REF_RE.findall(rhs)}
        if refs <= defined:
            kept.append(line)
            if def_name:
                defined.add(def_name)
        elif VERBOSE:
            missing = refs - defined
            print(f"    [skip] {line.rstrip()[:80]}  (missing: {missing})")
    return ''.join(header + kept + ["  ret void\n"] + footer)


def test_ir(ir_text):
    """Returns True if metallib loads + PSO creates OK."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
        f.write(ir_text)
        ll_path = f.name
    mlib_path = ll_path.replace('.ll', '.metallib')
    try:
        # Compile IR → metallib
        result = subprocess.run(
            [METAL_IR_OPT, ll_path, "-o", mlib_path],
            capture_output=True, text=True, timeout=TIMEOUT
        )
        if result.returncode != 0:
            if VERBOSE:
                print(f"    [compile fail] {result.stderr[:100]}")
            return False
        if not os.path.exists(mlib_path):
            return False

        # Try loading metallib + creating PSO
        dylib_path = os.environ.get('METALIR_DYLIB_PATH',
            os.path.expanduser('~/projects/oss/metal-ir-pipeline/build/lib/Bridge/libMetalIRBridge.dylib'))
        # Use a subprocess to test PSO creation (isolates crashes)
        test_code = f"""
import sys
sys.path.insert(0, '{os.path.expanduser("~/projects/oss/triton-apple/src")}')
from triton_apple_backend import metal_utils
data = open('{mlib_path}', 'rb').read()
lib = metal_utils.load_metallib(data)
fn = lib.get_function('{kernel_name}')
print('OK')
"""
        env = {**os.environ,
               'METALIR_DYLIB_PATH': dylib_path,
               'TRITON_PASS_PLUGIN_PATH': os.environ.get('TRITON_PASS_PLUGIN_PATH', '')}
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True, text=True, timeout=TIMEOUT,
            env=env
        )
        return "OK" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        if VERBOSE:
            print(f"    [exception] {e}")
        return False
    finally:
        for p in [ll_path, mlib_path]:
            try: os.unlink(p)
            except: pass


# ── Sanity checks ─────────────────────────────────────────────────
print("Testing empty body (just ret void)...")
ok = test_ir(''.join(header + ["  ret void\n"] + footer))
print(f"  Empty body: {'PASS' if ok else 'FAIL'}")
if not ok:
    print("ERROR: Even empty body crashes! Problem is in header/metadata/declares.")
    sys.exit(1)

print("Testing full body...")
full_ir = make_ir(body)
ok = test_ir(full_ir)
print(f"  Full body:  {'PASS' if ok else 'FAIL'}")
if ok:
    print("Full body passes! Nothing to bisect.")
    sys.exit(0)

# ── Binary search ─────────────────────────────────────────────────
lo, hi = 0, len(body)
print(f"\nBisecting {len(body)} body lines...")

while hi - lo > 1:
    mid = (lo + hi) // 2
    ir = make_ir(body[:mid])
    ok = test_ir(ir)
    status = "PASS" if ok else "FAIL"
    print(f"  body[:{mid:3d}] → {status}   (file line {body_start + mid})")
    if ok:
        lo = mid
    else:
        hi = mid

print(f"\n{'='*60}")
print(f"CRASH introduced at body line {hi} (file line {body_start + hi}):")
print(f"  {body[hi-1].rstrip()}")
print(f"\nContext:")
for i in range(max(0, hi - 5), min(len(body), hi + 3)):
    marker = ">>>" if i == hi - 1 else "   "
    print(f"  {marker} {body_start + i + 1:3d}: {body[i].rstrip()}")

# ── Save the minimal failing IR ───────────────────────────────────
minimal = make_ir(body[:hi])
out_path = IR_FILE.replace('.ll', '_minimal.ll')
with open(out_path, 'w') as f:
    f.write(minimal)
print(f"\nMinimal failing IR saved to: {out_path}")
