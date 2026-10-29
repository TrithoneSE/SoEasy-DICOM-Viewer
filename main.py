"""SoEasy DICOM Viewer - run: python main.py [--software-gl] [DICOM folders / files / archives]"""

import os
import sys

# prefer the discrete NVIDIA GPU on laptops with two GPUs
os.environ.setdefault("SHIM_MCCOMPAT", "0x800000001")

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dcv import graphics, theme  # noqa: E402   (no VTK yet: graphics.setup() picks the OpenGL first)


def dark_palette(app):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor(37, 37, 41))
    p.setColor(QPalette.WindowText, Qt.white)
    p.setColor(QPalette.Base, QColor(25, 25, 28))
    p.setColor(QPalette.AlternateBase, QColor(45, 45, 50))
    p.setColor(QPalette.ToolTipBase, QColor(theme.BG))
    p.setColor(QPalette.ToolTipText, QColor(theme.TEXT))
    p.setColor(QPalette.Text, Qt.white)
    p.setColor(QPalette.Button, QColor(50, 50, 56))
    p.setColor(QPalette.ButtonText, Qt.white)
    p.setColor(QPalette.Highlight, QColor(theme.ACCENT))
    p.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(p)
    app.setStyleSheet(theme.TOOLTIP)


def install_logging():
    """Python exceptions and VTK/OpenGL errors go to %APPDATA%/DicomColorViewer/error.log
    (pythonw has no console, so without this errors would be invisible)."""
    import datetime
    import traceback
    from dcv import vtk_lite as vtk
    d = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "DicomColorViewer")
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, "error.log")
    ow = vtk.vtkFileOutputWindow()
    ow.SetFileName(os.path.join(d, "vtk_errors.log"))
    ow.AppendOn()
    vtk.vtkOutputWindow.SetInstance(ow)

    def hook(tp, val, tb):
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            f.write("".join(traceback.format_exception(tp, val, tb)))
        sys.__excepthook__(tp, val, tb)

    sys.excepthook = hook


def resource(*parts):
    """Path to a bundled file, both when running from source and from the PyInstaller .exe."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        if graphics.SOFTWARE_FLAG in sys.argv:
            graphics.use_software_opengl()
        from dcv import selftest
        data = next((a for a in sys.argv[3:] if not a.startswith("--")), None)
        sys.exit(0 if selftest.run(sys.argv[2], data) else 1)
    if not graphics.setup():
        return                          # restarted with software OpenGL
    install_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("SoEasy DICOM Viewer")
    from PySide6.QtGui import QIcon
    app.setWindowIcon(QIcon(resource("assets", "icon.png")))
    dark_palette(app)
    from dcv.main_window import MainWindow
    w = MainWindow()
    w.show()
    if graphics.mode == "software":
        w.statusBar().showMessage("No usable graphics driver: 3D uses software rendering (complete, but slower).",
                                  15000)
    paths = [a for a in sys.argv[1:] if not a.startswith("--") and os.path.exists(a)]
    if paths:
        QTimer.singleShot(200, lambda: w.open_paths(paths))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
