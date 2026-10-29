"""Open DICOM studies packed in archives (.zip, .rar, .7z, .tar...).

Archives are extracted to a private temp folder, because series are (re)loaded from file paths (memory trimming
reloads unused series from disk). ZIP uses Python's zipfile; RAR / 7z / tar use Windows' own tar.exe (bsdtar /
libarchive, built into Windows 10 1803+ and 11), with 7-Zip or WinRAR as a fallback when they are installed.
The folders are deleted when the app exits (and leftovers from a crash after 12 hours).
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
import time
import zipfile

ARCHIVE_EXT = (".zip", ".rar", ".7z", ".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz")
BASE = os.path.join(tempfile.gettempdir(), "DicomColorViewer", "archives")
_ours = []


class ArchiveError(RuntimeError):
    pass


def is_archive(path) -> bool:
    return str(path).lower().endswith(ARCHIVE_EXT) and os.path.isfile(path)


def _cleanup():
    for d in _ours:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup)


def _remove_stale():
    try:
        for n in os.listdir(BASE):
            d = os.path.join(BASE, n)
            if d not in _ours and time.time() - os.path.getmtime(d) > 12 * 3600:
                shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


def _tools():
    """Command lines that extract `a` into `d`, tried in order."""
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    pf = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")]
    out = []
    tar = os.path.join(sysroot, "System32", "tar.exe")
    if os.path.isfile(tar):
        out.append(lambda a, d: [tar, "-xf", a, "-C", d])
    for base in filter(None, pf):
        z = os.path.join(base, "7-Zip", "7z.exe")
        if os.path.isfile(z):
            out.append(lambda a, d, z=z: [z, "x", "-y", "-bd", f"-o{d}", a])
        r = os.path.join(base, "WinRAR", "UnRAR.exe")
        if os.path.isfile(r):
            out.append(lambda a, d, r=r: [r, "x", "-o+", "-inul", a, d + os.sep])
    return out


def _extract_zip(path, dest, progress, cancelled):
    try:
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            for i, m in enumerate(members):
                if cancelled():
                    raise ArchiveError("Cancelled")
                if m.flag_bits & 0x1:
                    raise ArchiveError(f"{os.path.basename(path)} is password protected.")
                if progress and i % 20 == 0:
                    progress(i, len(members), f"Extracting {os.path.basename(path)}...")
                z.extract(m, dest)       # zipfile strips absolute paths and '..'
    except (zipfile.BadZipFile, NotImplementedError) as ex:
        raise ArchiveError(f"{os.path.basename(path)}: {ex}") from ex


def _extract_cmd(path, dest, progress):
    if progress:
        progress(0, 0, f"Extracting {os.path.basename(path)}...")
    tools = _tools()
    if not tools:
        raise ArchiveError("No program to extract this archive was found (needs Windows tar.exe, 7-Zip or WinRAR).")
    last = ""
    for make in tools:
        cmd = make(path, dest)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as ex:
            last = str(ex)
            continue
        if r.returncode == 0 and any(files for _, _, files in os.walk(dest)):
            return
        last = (r.stderr or r.stdout).strip().splitlines()[-1:] or [f"exit code {r.returncode}"]
        last = last[0]
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest, exist_ok=True)
    raise ArchiveError(f"Could not extract {os.path.basename(path)} (damaged or password protected?)\n{last}")


def extract_archive(path, progress=None, cancelled=lambda: False) -> str:
    """Extract `path` to a new temp folder and return that folder."""
    os.makedirs(BASE, exist_ok=True)
    _remove_stale()
    stem = os.path.basename(path)
    for ext in ARCHIVE_EXT:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    dest = tempfile.mkdtemp(prefix=f"{stem[:40]}_", dir=BASE)
    _ours.append(dest)
    try:
        if path.lower().endswith(".zip"):
            _extract_zip(path, dest, progress, cancelled)
        else:
            _extract_cmd(path, dest, progress)
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return dest
