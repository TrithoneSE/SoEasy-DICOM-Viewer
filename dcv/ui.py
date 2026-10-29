"""Small widgets shared by the windows: muted hint text, '▸ title' collapsible sections, color swatches."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QToolButton, QVBoxLayout, QWidget

from . import theme


def hint(html, muted=True):
    lb = QLabel(html)
    lb.setWordWrap(True)
    lb.setStyleSheet(f"color: {theme.MUTED if muted else theme.TEXT}; font-size: {11 if muted else 12}px;")
    return lb


def collapsible(title, content, expanded=False):
    """A '▸ title' toggle that shows / hides `content`: long help and rarely used settings stay out of the way."""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)
    b = QToolButton()
    b.setCheckable(True)
    b.setAutoRaise(True)
    b.setToolButtonStyle(Qt.ToolButtonTextOnly)
    b.setStyleSheet(f"QToolButton, QToolButton:checked, QToolButton:hover {{ color: {theme.MUTED}; background: "
                    f"transparent; border: none; padding: 2px 0; font-size: 12px; }} "
                    f"QToolButton:hover {{ color: {theme.TEXT}; }}")

    def upd(on):
        b.setText(("▾  " if on else "▸  ") + title)
        content.setVisible(on)

    b.toggled.connect(upd)
    b.setChecked(expanded)
    upd(expanded)
    v.addWidget(b)
    v.addWidget(content)
    return w


def cmap_icon(lut, w=44, h=12):
    """Gradient swatch of a (256, 3) uint8 color map, for color map menus."""
    import numpy as np
    xs = np.linspace(0, 255, w).astype(int)
    row = np.ascontiguousarray(np.repeat(lut[xs][None], h, axis=0))
    img = QImage(row.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QIcon(QPixmap.fromImage(img))


def color_dot(rgb, size=14):
    """Round swatch for an (r, g, b) color in 0..1."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    p.drawEllipse(QRectF(1, 1, size - 2, size - 2))
    p.end()
    return QIcon(pm)
