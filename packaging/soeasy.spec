# PyInstaller spec for SoEasy DICOM Viewer (one-folder build: fast start, nothing unpacked at run time).
# Build with:  python packaging/build.py      (runs this spec, then checks the result with --selftest)
#
# Size: only what the application imports is packed. VTK comes in module by module (dcv/vtk_lite.py), and the
# Qt parts the viewer never loads are dropped below. Nothing is compressed or stripped inside the program files,
# so the installed application is byte-for-byte what PyInstaller collected.
import os
import sys

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, ROOT)
from dcv import __version__  # noqa: E402

APP = "SoEasy DICOM Viewer"

# Qt / Python parts never used by the viewer (checked against the modules loaded at run time)
DROP = (
    "Qt6Quick", "Qt6Qml", "Qt6Pdf", "Qt6VirtualKeyboard", "Qt6Network", "Qt6OpenGL.dll", "Qt6OpenGLWidgets",
    "QtOpenGL.pyd", "QtOpenGLWidgets.pyd", "QtNetwork.pyd",
    os.path.join("PySide6", "translations"),
    os.path.join("plugins", "platforminputcontexts"), os.path.join("plugins", "networkinformation"),
    os.path.join("plugins", "tls"), os.path.join("plugins", "generic"),
    "libcrypto-", "libssl-", "_ssl.pyd", "_hashlib.pyd",          # no network; hashlib falls back to built-ins
    "PIL\\_avif", "PIL\\_webp", "PIL\\_imagingtk",                  # Pillow writes the GIF / PNG movies only
)

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    datas=[(os.path.join(ROOT, "assets"), "assets")]
    + copy_metadata("pydicom") + copy_metadata("pylibjpeg") + copy_metadata("pylibjpeg-libjpeg")
    + copy_metadata("pylibjpeg-openjpeg"),
    # pydicom finds its pixel decoders at run time; pylibjpeg finds its plugins through package metadata
    hiddenimports=collect_submodules("pydicom.pixels") + ["libjpeg", "openjpeg", "pylibjpeg"],
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6", "PySide2", "vtk",
              "PySide6.QtNetwork", "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtQml",
              "PySide6.QtQuick", "PySide6.QtPdf", "ssl", "_ssl", "_hashlib"],
    noarchive=False,
)
a.binaries = [b for b in a.binaries if not any(d in b[0] for d in DROP)]
a.datas = [d for d in a.datas if not any(x in d[0] for x in DROP)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP,
    icon=os.path.join(ROOT, "assets", "icon.ico"),
    version=os.path.join(SPECPATH, "build", "version_info.txt"),
    console=False,
    upx=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name=APP, upx=False)
