"""OpenGL for the 3D views: the graphics driver, or software rendering on PCs without a usable one.

VTK needs OpenGL 3.2. Without a GPU driver (only "Microsoft Basic Display Adapter"), in many virtual machines and
over some remote-desktop setups Windows offers just OpenGL 1.1. Qt ships Mesa's llvmpipe renderer as
`opengl32sw.dll` (OpenGL 3.3 on the CPU); loading it under the name `opengl32.dll` before VTK starts makes VTK use
it, because Windows reuses a module that is already loaded under the same name. Rendering is then slower but
complete (same images). `--software-gl` or SOEASY_SOFTWARE_GL=1 forces it.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

SOFTWARE_FLAG = "--software-gl"
mode = "hardware"          # "hardware" / "software", shown in Help → About


def display_adapters():
    """Names of the display adapters Windows knows (EnumDisplayDevices)."""
    class DisplayDevice(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("DeviceName", ctypes.c_wchar * 32),
                    ("DeviceString", ctypes.c_wchar * 128), ("StateFlags", ctypes.c_ulong),
                    ("DeviceID", ctypes.c_wchar * 128), ("DeviceKey", ctypes.c_wchar * 128)]
    names, i = [], 0
    while True:
        d = DisplayDevice()
        d.cb = ctypes.sizeof(d)
        if not ctypes.windll.user32.EnumDisplayDevicesW(None, i, ctypes.byref(d), 0):
            return names
        if d.DeviceString and d.DeviceString not in names:
            names.append(d.DeviceString)
        i += 1


def software_requested():
    if SOFTWARE_FLAG in sys.argv or os.environ.get("SOEASY_SOFTWARE_GL") == "1":
        return True
    try:
        adapters = display_adapters()
    except (AttributeError, OSError):
        return False
    return bool(adapters) and all("microsoft basic" in a.lower() for a in adapters)


def use_software_opengl():
    """Load Mesa as opengl32.dll; must run before anything imports VTK's rendering modules."""
    global mode
    import PySide6
    src = os.path.join(os.path.dirname(PySide6.__file__), "opengl32sw.dll")
    if not os.path.isfile(src):
        return False
    folder = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "SoEasyDICOMViewer", "softgl")
    dst = os.path.join(folder, "opengl32.dll")
    try:
        if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
            os.makedirs(folder, exist_ok=True)
            shutil.copyfile(src, dst)
        ctypes.WinDLL(dst)
    except OSError:
        return False
    mode = "software"
    return True


def hardware_opengl_ok():
    """Whether the driver's OpenGL is enough for VTK (an off-screen window, a few ms with a GPU)."""
    from . import vtk_lite as vtk
    vtk.vtkOutputWindow.GetInstance().SetDisplayMode(0)     # no error pop-up from the probe itself
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    try:
        return bool(rw.SupportsOpenGL())
    finally:
        rw.Finalize()


def relaunch_with_software_opengl():
    """Start this program again with --software-gl (same arguments) and tell the caller to quit."""
    args = [a for a in sys.argv[1:] if a != SOFTWARE_FLAG] + [SOFTWARE_FLAG]
    cmd = [sys.executable] + args if getattr(sys, "frozen", False) else [sys.executable, sys.argv[0]] + args
    subprocess.Popen(cmd, close_fds=True)


def setup():
    """Choose the OpenGL before VTK loads. Returns False when the program was restarted in software mode."""
    if software_requested():
        use_software_opengl()
        return True
    if not hardware_opengl_ok():
        relaunch_with_software_opengl()
        return False
    return True
