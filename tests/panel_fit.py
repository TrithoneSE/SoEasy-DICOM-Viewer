"""Shared check for the GUI tests: every control of a dock panel must lie inside the visible width.
(The 3D Segments tab once was wider than the dock: spin-box arrows, combo arrows and buttons were cut off.)"""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractSlider, QAbstractSpinBox, QApplication, QComboBox, QDockWidget, QScrollArea,
    QToolButton,
)

CONTROLS = (QAbstractSpinBox, QComboBox, QAbstractButton, QAbstractSlider)


def expand_collapsibles(win):
    """Open every '▸ ...' section: tallest content, so the vertical scrollbar takes its width too."""
    for b in win.findChildren(QToolButton):
        if b.isCheckable() and b.text()[:1] in ("▸", "▾"):
            b.setChecked(True)


def clipped_controls(win, pages):
    """Shrink the window's dock to its minimum width; return [(page, control, overflow px)] for controls that
    stick out of the visible part of their scroll area / page."""
    dock = win.findChild(QDockWidget)
    win.resizeDocks([dock], [dock.minimumWidth()], Qt.Horizontal)
    expand_collapsibles(win)
    app = QApplication.instance()
    out = []
    for name, page, show in pages:
        show()
        app.processEvents()
        view = page.viewport() if isinstance(page, QScrollArea) else page
        inner = page.widget() if isinstance(page, QScrollArea) else page
        right = view.width()
        controls = {id(c): c for t in CONTROLS for c in inner.findChildren(t)}.values()
        for c in controls:
            if not c.isVisible():
                continue
            x = c.mapTo(view, QPoint(c.width(), 0)).x()
            if x > right + 1:
                out.append((name, type(c).__name__, (c.text() if hasattr(c, "text") else "")[:30], x - right))
    return out
