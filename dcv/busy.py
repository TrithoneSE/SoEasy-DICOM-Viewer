"""'Please wait' panel for work the user starts with one click (opening 3D, segmenting a tumor).

Without it the window just freezes for a few seconds and looks like a hang. Heavy NumPy / VTK work runs in a plain
Python thread (NumPy releases the GIL) while a local event loop keeps the panel and its moving bar painted; work
that must stay on the GUI thread (creating VTK / OpenGL windows) runs between stages with the panel already shown.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QProgressBar, QVBoxLayout

from . import theme


class BusyDialog(QDialog):
    def __init__(self, parent, title):
        super().__init__(parent, Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("busy")
        self.setStyleSheet(
            f"QDialog#busy {{ background: {theme.SURFACE}; border: 1px solid {theme.ACCENT}; border-radius: 8px; }}"
            f" QLabel {{ color: {theme.TEXT}; background: transparent; }}"
            f" QProgressBar {{ background: {theme.BG}; border: 1px solid {theme.BORDER}; border-radius: 4px; }}"
            f" QProgressBar::chunk {{ background: {theme.ACCENT}; border-radius: 3px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 18, 24, 18)
        self.title = QLabel(title)
        self.title.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.title.setAlignment(Qt.AlignCenter)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)               # moving "busy" bar
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(14)
        self.detail = QLabel("Starting...")
        self.detail.setAlignment(Qt.AlignCenter)
        self.detail.setStyleSheet(f"color: {theme.MUTED};")
        for w in (self.title, self.bar, self.detail):
            lay.addWidget(w)
        self.setFixedSize(440, 112)

    def stage(self, text):
        """Show the next step and repaint now (before blocking GUI-thread work)."""
        self.detail.setText(text)
        if not self.isVisible():
            p = self.parentWidget()
            if p is not None:
                g = p.frameGeometry()
                self.move(g.center().x() - self.width() // 2, g.center().y() - self.height() // 2)
            self.show()
        for _ in range(3):
            QApplication.processEvents()

    def run(self, text, fn, *args):
        """Run fn(*args) in a worker thread while the panel stays live; return its result (re-raise its error)."""
        self.stage(text)
        box = {}

        def work():
            try:
                box["result"] = fn(*args)
            except BaseException as ex:  # noqa: BLE001 - re-raised on the GUI thread
                box["error"] = ex

        t = threading.Thread(target=work, daemon=True)
        t.start()
        loop = QEventLoop()
        timer = QTimer()
        timer.timeout.connect(lambda: None if t.is_alive() else loop.quit())
        timer.start(30)
        if t.is_alive():
            loop.exec()
        timer.stop()
        t.join()
        if "error" in box:
            raise box["error"]
        return box.get("result")
