"""Draw the installer bitmaps from the app logo: packaging/build/wizard.bmp (164 x 314, welcome / finish pages) and
header.bmp (150 x 57, top right of the other pages). Run by build.py before makensis."""

import os
import sys

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "packaging", "build")


def main():
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])    # text needs a real platform (fonts)
    os.makedirs(OUT, exist_ok=True)
    app.setApplicationName("installer art")
    logo = QImage(os.path.join(ROOT, "assets", "icon.png"))

    w = QImage(164, 314, QImage.Format_RGB32)
    p = QPainter(w)
    g = QLinearGradient(0, 0, 0, 314)
    g.setColorAt(0, QColor(13, 45, 92))
    g.setColorAt(1, QColor(21, 101, 192))
    p.fillRect(w.rect(), g)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    p.drawImage(QRectF(32, 40, 100, 100), logo)
    p.setPen(QColor(235, 243, 254))
    p.setFont(QFont("Segoe UI", 15, QFont.Bold))
    p.drawText(QRectF(0, 158, 164, 60), Qt.AlignHCenter | Qt.AlignTop, "SoEasy\nDICOM Viewer")
    p.setPen(QColor(170, 200, 235))
    p.setFont(QFont("Segoe UI", 8))
    p.drawText(QRectF(8, 250, 148, 50), Qt.AlignHCenter | Qt.TextWordWrap, "2D · MPR · 3D · DSA · DTI")
    p.end()
    w.save(os.path.join(OUT, "wizard.bmp"))

    h = QImage(150, 57, QImage.Format_RGB32)
    h.fill(QColor(255, 255, 255))
    p = QPainter(h)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    p.drawImage(QRectF(96, 5, 47, 47), logo)
    p.end()
    h.save(os.path.join(OUT, "header.bmp"))


if __name__ == "__main__":
    main()
