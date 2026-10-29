"""Draw the app logo (S E = SoEasy DICOM Viewer) and write icon.png + icon.ico.

The logo is a brain seen from above on a background split left / right (two blues). The left hemisphere is dark on
the light half, the right one light on the dark half. The letters S (left) and E (right) are drawn as sulci: grooves
in the background colour whose ends taper out like real sulci, between rounded gyri along the edge. Below 40 px the
small folds are left out and the letters get thicker so "SE" stays readable in the taskbar.
All geometry is in unit coordinates (0..1) and scaled to the icon size.

Run: .venv\\Scripts\\python.exe tools\\make_icon.py
"""

import math
import os
import struct
import sys

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPainterPathStroker, QPen, QTransform

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "assets")

LEFT_BG = QColor(214, 232, 252)     # light blue  (also the colour of the left sulci and the S)
RIGHT_BG = QColor(21, 101, 192)     # blue        (also the colour of the right sulci and the E)
LEFT_BRAIN = QColor(15, 63, 134)    # navy hemisphere
RIGHT_BRAIN = QColor(234, 243, 254)
BORDER = QColor(10, 50, 110)

GAP = 0.010                         # half width of the fissure between the hemispheres
N_LOBES = 8                         # gyri along the edge of one hemisphere
FLIP = QTransform(-1, 0, 0, 1, 1, 0)   # left hemisphere -> right hemisphere


def _base_curve():
    """Smooth outline of the left hemisphere, from the frontal pole (top) to the occipital pole (bottom)."""
    b = QPainterPath(QPointF(0.5 - GAP, 0.115))
    b.cubicTo(0.31, 0.085, 0.145, 0.20, 0.140, 0.43)
    b.cubicTo(0.135, 0.68, 0.275, 0.895, 0.5 - GAP, 0.890)
    return b


def _knots():
    b = _base_curve()
    ts = [i / N_LOBES for i in range(N_LOBES + 1)]
    return b, ts, [b.pointAtPercent(t) for t in ts]


def _normal(b, t):
    """Outward unit normal of the base curve at t."""
    a, c = b.pointAtPercent(max(0, t - .005)), b.pointAtPercent(min(1, t + .005))
    dx, dy = c.x() - a.x(), c.y() - a.y()
    n = math.hypot(dx, dy) or 1
    return -dy / n, dx / n


def hemisphere():
    """Left hemisphere: each stretch between two knots bulges outward like a gyrus; closed along the fissure."""
    b, ts, P = _knots()
    path = QPainterPath(P[0])
    for i in range(N_LOBES):
        tm = (ts[i] + ts[i + 1]) / 2
        mid, (nx, ny) = b.pointAtPercent(tm), _normal(b, tm)
        h = math.hypot(P[i + 1].x() - P[i].x(), P[i + 1].y() - P[i].y()) * 0.34
        path.quadTo(QPointF(mid.x() + nx * h, mid.y() + ny * h), P[i + 1])
    path.closeSubpath()
    return path


def edge_sulci():
    """Short curved grooves going inward from the dents between the gyri."""
    b, ts, P = _knots()
    path = QPainterPath()
    for i in range(1, N_LOBES):
        q, (nx, ny) = P[i], _normal(b, ts[i])
        tx, ty = -ny, nx
        d = 0.075 if i % 2 else 0.05
        path.moveTo(q.x() + nx * 0.004, q.y() + ny * 0.004)
        path.quadTo(QPointF(q.x() - nx * d * 0.6 + tx * 0.012, q.y() - ny * d * 0.6 + ty * 0.012),
                    QPointF(q.x() - nx * d - tx * 0.006, q.y() - ny * d - ty * 0.006))
    return path


def letter_s():
    p = QPainterPath(QPointF(0.405, 0.335))
    p.cubicTo(0.380, 0.272, 0.248, 0.272, 0.253, 0.372)
    p.cubicTo(0.258, 0.462, 0.405, 0.457, 0.405, 0.582)
    p.cubicTo(0.405, 0.697, 0.268, 0.717, 0.243, 0.642)
    return p


def letter_e():
    p = QPainterPath(QPointF(0.600, 0.318))
    p.lineTo(0.600, 0.682)
    for y, x1, amp in ((0.318, 0.755, 0.012), (0.500, 0.730, -0.012), (0.682, 0.755, 0.012)):
        p.moveTo(0.600, y)
        p.cubicTo(0.650, y - amp, 0.700, y + amp, x1, y)     # slightly wavy arms, like sulci
    return p


# letter ends that continue as tapering sulci (cubic control points)
TAILS_S = [((0.243, 0.642), (0.212, 0.600), (0.185, 0.585), (0.160, 0.588))]
TAILS_E = [((0.755, 0.318), (0.800, 0.318), (0.830, 0.305), (0.880, 0.285)),
           ((0.755, 0.682), (0.800, 0.682), (0.830, 0.695), (0.880, 0.715))]


def _cubic(c, s):
    path = QPainterPath(QPointF(*c[0]) * s)
    path.cubicTo(QPointF(*c[1]) * s, QPointF(*c[2]) * s, QPointF(*c[3]) * s)
    return path


def _taper(p, tails, s, w0, w1, color, n=48):
    """Stroke each tail with a width easing from w0 to w1."""
    for c in tails:
        path = _cubic(c, s)
        pts = [path.pointAtPercent(i / n) for i in range(n + 1)]
        for i in range(n):
            p.setPen(QPen(color, w0 + (w1 - w0) * (i / (n - 1)) ** 0.7, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(pts[i], pts[i + 1])


def draw(size):
    s = size
    img = QImage(s, s, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    sc = QTransform().scale(s, s)
    m = max(1.0, s * 0.02)
    bg = QPainterPath()
    bg.addRoundedRect(QRectF(m, m, s - 2 * m, s - 2 * m), s * 0.12, s * 0.12)
    p.setClipPath(bg)
    p.fillRect(QRectF(0, 0, s / 2, s), LEFT_BG)
    p.fillRect(QRectF(s / 2, 0, s / 2, s), RIGHT_BG)
    p.setClipping(False)
    small = s < 40
    lw = (0.088 if small else 0.066) * s
    for T, letter, tails, fill, groove in ((sc, letter_s(), TAILS_S, LEFT_BRAIN, LEFT_BG),
                                           (FLIP * sc, letter_e(), TAILS_E, RIGHT_BRAIN, RIGHT_BG)):
        shape = T.map(hemisphere())
        g = sc.map(letter)
        p.fillPath(shape, fill)
        p.setClipPath(shape)
        if not small:
            st = QPainterPathStroker()
            st.setWidth(lw + 0.055 * s)
            st.setCapStyle(Qt.RoundCap)
            halo = st.createStroke(g)
            for c in tails:
                halo = halo.united(st.createStroke(_cubic(c, s)))
            pen = QPen(groove, s * 0.021, Qt.SolidLine, Qt.RoundCap)
            for poly in T.map(edge_sulci()).toSubpathPolygons():
                sp = QPainterPath()
                sp.addPolygon(poly)
                if not halo.intersects(QPainterPathStroker(pen).createStroke(sp)):   # keep folds clear of letters
                    p.strokePath(sp, pen)
        _taper(p, tails, s, lw, s * 0.014, groove)
        p.setClipping(False)
        p.strokePath(g, QPen(groove, lw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setPen(QPen(BORDER, max(1.0, s * 0.025)))
    p.setBrush(Qt.NoBrush)
    p.drawPath(bg)
    p.end()
    return img


def png_bytes(img):
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


def write_ico(path, sizes):
    """Multi-resolution .ico with PNG-compressed entries (supported since Windows Vista)."""
    pngs = [png_bytes(draw(s)) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries = b""
    for s, data in zip(sizes, pngs):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    with open(path, "wb") as f:
        f.write(header + entries + b"".join(pngs))


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    os.makedirs(OUT, exist_ok=True)
    draw(512).save(os.path.join(OUT, "icon.png"))
    write_ico(os.path.join(OUT, "icon.ico"), [16, 20, 24, 32, 40, 48, 64, 128, 256])
    print("written to", OUT)
