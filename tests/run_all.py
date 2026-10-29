"""Run the GUI tests: python tests/run_all.py [test name filter]

Each test drives the real windows with simulated mouse / key events on the sample studies in `samples/` and must
end with "errors: 0". Screenshots go to tests/out. Exit code 0 only when every test passed.
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = [
    "test_3d_measure", "test_scalpel_loading", "test_units_archive", "test_main_ui", "test_vr_ui", "test_mpr_ui",
    "test_dti_fusion_ui", "test_drag_series", "test_open_window_tf", "test_captions_snapshots",
    "test_reformats_rim_cursor", "test_dsa",
]


def main():
    wanted = [t for t in TESTS if not sys.argv[1:] or any(a in t for a in sys.argv[1:])]
    out = os.path.join(ROOT, "tests", "out")
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, SHIM_MCCOMPAT="0x800000001", PYTHONIOENCODING="utf-8")
    results = []
    for name in wanted:
        t = time.perf_counter()
        p = subprocess.run([sys.executable, os.path.join(ROOT, "tests", name + ".py"), ROOT,
                            os.path.join(ROOT, "samples", "head_ct"), out],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        lines = p.stdout.splitlines()
        fails = [ln for ln in lines if ln.startswith("FAIL")]
        ok = p.returncode == 0 and "errors: 0" in lines and not fails
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({time.perf_counter() - t:.0f} s)", flush=True)
        if not ok:
            for ln in fails or lines[-15:] or p.stderr.splitlines()[-15:]:
                print("      " + ln)
    print(f"\n{sum(results)}/{len(results)} tests passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
