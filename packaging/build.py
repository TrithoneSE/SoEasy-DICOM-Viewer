"""Build SoEasy DICOM Viewer for Windows.

    python packaging/build.py              # dist/SoEasy DICOM Viewer/  (PyInstaller one-folder build)
    python packaging/build.py --installer  # + dist/SoEasy-DICOM-Viewer-<version>-Setup.exe (NSIS) and a portable .zip

Every build is checked by running the built program's own --selftest twice, on the graphics driver and on software
OpenGL, with the sample studies (2D, MPR, 3D, DSA, DTI, snapshot movie). A build with a failed check stops here.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "packaging")
WORK = os.path.join(PKG, "build")
DIST = os.path.join(ROOT, "dist")
APP = "SoEasy DICOM Viewer"
sys.path.insert(0, ROOT)
from dcv import __version__  # noqa: E402

VERSION_INFO = """VSVersionInfo(
  ffi=FixedFileInfo(filevers=({v4}), prodvers=({v4}), mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
                    date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'SoEasy DICOM Viewer contributors'),
      StringStruct('FileDescription', '{app}'),
      StringStruct('FileVersion', '{v}'),
      StringStruct('InternalName', '{app}'),
      StringStruct('LegalCopyright', 'GPL-3.0-or-later'),
      StringStruct('OriginalFilename', '{app}.exe'),
      StringStruct('ProductName', '{app}'),
      StringStruct('ProductVersion', '{v}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def run(cmd, **kw):
    print("+", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def folder_size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def pyinstaller():
    os.makedirs(WORK, exist_ok=True)
    nums = [int(x) for x in __version__.split(".")[:3]] + [0]
    with open(os.path.join(WORK, "version_info.txt"), "w", encoding="utf-8") as f:
        f.write(VERSION_INFO.format(v=__version__, v4=", ".join(map(str, nums)), app=APP))
    run([sys.executable, os.path.join(ROOT, "tools", "make_icon.py")])
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--workpath", os.path.join(WORK, "work"),
         "--distpath", DIST, os.path.join(PKG, "soeasy.spec")])
    return os.path.join(DIST, APP)


def selftest(app_dir, modes):
    exe = os.path.join(app_dir, APP + ".exe")
    ok = True
    for mode in modes:
        report = os.path.join(WORK, f"selftest_{mode}.txt")
        cmd = [exe, "--selftest", report, os.path.join(ROOT, "samples")] + (["--software-gl"] if mode == "software"
                                                                           else [])
        code = subprocess.run(cmd).returncode
        text = open(report, encoding="utf-8").read() if os.path.isfile(report) else "(no report)"
        print(f"\n--- self-test, {mode} OpenGL (exit {code}) ---\n{text}")
        ok &= code == 0 and "FAILED" not in text
    return ok


def makensis():
    exe = shutil.which("makensis") or r"C:\Program Files (x86)\NSIS\makensis.exe"
    if not os.path.isfile(exe):
        sys.exit("NSIS (makensis) not found: install it from https://nsis.sourceforge.io or `winget install NSIS.NSIS`")
    return exe


def installer(app_dir):
    out = os.path.join(DIST, f"SoEasy-DICOM-Viewer-{__version__}-Setup.exe")
    run([sys.executable, os.path.join(PKG, "make_installer_art.py")])
    run([makensis(),f"/DVERSION={__version__}", f"/DSRC={app_dir}", f"/DOUT={out}", f"/DROOT={ROOT}",
         os.path.join(PKG, "installer.nsi")])
    portable = os.path.join(DIST, f"SoEasy-DICOM-Viewer-{__version__}-portable.zip")
    with zipfile.ZipFile(portable, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d, _, fs in os.walk(app_dir):
            for f in fs:
                full = os.path.join(d, f)
                z.write(full, os.path.join(APP, os.path.relpath(full, app_dir)))
    return out, portable


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--installer", action="store_true", help="also build the NSIS installer and a portable zip")
    ap.add_argument("--skip-selftest", action="store_true", help="do not run the built program's self-test")
    ap.add_argument("--selftest-modes", default="hardware,software",
                    help="OpenGL modes to self-test (CI machines without a GPU: software)")
    args = ap.parse_args()
    app_dir = pyinstaller()
    print(f"\nbuild: {app_dir} ({folder_size(app_dir) / 2**20:.0f} MB)")
    if not args.skip_selftest and not selftest(app_dir, args.selftest_modes.split(",")):
        sys.exit("self-test FAILED - see the report above")
    if args.installer:
        lines = []
        for path in installer(app_dir):
            with open(path, "rb") as f:
                lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {os.path.basename(path)}")
            print(f"{os.path.basename(path)}: {os.path.getsize(path) / 2**20:.0f} MB")
        with open(os.path.join(DIST, "SHA256SUMS.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
