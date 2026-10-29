"""Line icons for the toolbars (own drawings, 24×24 grid, one stroke style) rendered from inline SVG.

icon(name) returns a QIcon tinted for the dark theme: normal, hover (active), disabled and checked ("on") states.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from . import theme

_DOT = '<circle cx="{}" cy="{}" r="1.6" fill="currentColor" stroke="none"/>'

SVG = {
    # files / layout
    "open": '<path d="M3 7.5A1.5 1.5 0 0 1 4.5 6H9l2 2h8.5A1.5 1.5 0 0 1 21 9.5v8a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 '
            '17.5z"/>',
    "archive": '<rect x="3" y="4" width="18" height="5" rx="1"/><path d="M5 9v10h14V9M10 13h4"/>',
    "layout": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18M3 12h18"/>',
    "link": '<path d="M10 14a4 4 0 0 0 5.66 0l3-3a4 4 0 0 0-5.66-5.66l-1 1"/>'
            '<path d="M14 10a4 4 0 0 0-5.66 0l-3 3a4 4 0 0 0 5.66 5.66l1-1"/>',
    "annot": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/><path d="M8 10h8M8 14h5"/>',
    "fullscreen": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    # mouse tools
    "wl": '<circle cx="12" cy="12" r="8.5"/><path d="M12 3.5a8.5 8.5 0 0 1 0 17z" fill="currentColor"/>',
    "pan": '<path d="M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
    "zoom": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-4.8-4.8M7.5 10.5h6M10.5 7.5v6"/>',
    "browse": '<path d="M4 8.5l8-4 8 4-8 4z"/><path d="M4 12.5l8 4 8-4"/><path d="M4 16.5l8 4 8-4"/>',
    "edit": '<path d="M5 3.5l6.5 16 2.3-6.7 6.7-2.3z"/>',
    "cursor3d": '<circle cx="12" cy="12" r="5.5"/><path d="M12 2.5v5M12 16.5v5M2.5 12h5M16.5 12h5"/>' + _DOT.format(12, 12),
    "eyedropper": '<path d="M15.5 3.5l5 5-3 3-5-5z"/><path d="M13.5 8.5L5 17v2.5h2.5L16 11"/>',
    "brush": '<path d="M20.5 3.5l-9 9"/><path d="M11.5 12.5c-2-1.2-5.5-.4-5.5 3 0 1.8-1 2.8-2.5 3 3.6 1.4 9 .3 8-6z"/>',
    "seed": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/>' + _DOT.format(12, 12),
    "wlroi": '<rect x="3" y="3" width="18" height="18" rx="1.5" stroke-dasharray="3 2.6"/><circle cx="12" cy="12" r="4.5"/>'
             '<path d="M12 7.5a4.5 4.5 0 0 1 0 9z" fill="currentColor"/>',
    # measurements
    "ruler": '<path d="M3.5 16.5l13-13 4 4-13 13z"/><path d="M7.5 12.5l2 2M10.5 9.5l2 2M13.5 6.5l2 2"/>',
    "length": '<path d="M5 19L19 5"/>' + _DOT.format(5, 19) + _DOT.format(19, 5),
    "ellipse": '<ellipse cx="12" cy="12" rx="9" ry="6.5"/>',
    "rect": '<rect x="3.5" y="6" width="17" height="12" rx="1"/>',
    "polygon": '<path d="M5 9l7-5.5 7.5 6-2.5 10H7.5z"/>',
    "openpoly": '<path d="M3 18l5-9 6 6 7-10"/>',
    "angle": '<path d="M20 19H4L15 5"/><path d="M9.5 19a5.5 5.5 0 0 0-2.1-4.3"/>',
    "cobb": '<path d="M3.5 7l17 3M3.5 18l17-4"/><path d="M11 8.5l1 7.5" stroke-dasharray="2 2"/>',
    "deviation": '<path d="M7 3v18"/><path d="M7 12h10" stroke-dasharray="2 2"/>' + _DOT.format(17.5, 12),
    "arrow": '<path d="M5 19L19 5M10 5h9v9"/>',
    "pencil": '<path d="M4 20l1-4.5L16 4.5l3.5 3.5L8.5 19z"/><path d="M14 7l3.5 3.5"/>',
    "text": '<path d="M5 7V4.5h14V7M12 4.5v15M9 19.5h6"/>',
    # image
    "image": '<path d="M4 6.5h9M17 6.5h3M4 12h3M11 12h9M4 17.5h11M19 17.5h1"/><circle cx="15" cy="6.5" r="2"/>'
             '<circle cx="9" cy="12" r="2"/><circle cx="17" cy="17.5" r="2"/>',
    "rotate_r": '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.5 4v4.5H15"/>',
    "rotate_l": '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4v4.5H9"/>',
    "flip_h": '<path d="M12 3v18" stroke-dasharray="2 2"/><path d="M9 7L4 17h5zM15 7l5 10h-5z"/>',
    "flip_v": '<path d="M3 12h18" stroke-dasharray="2 2"/><path d="M7 9L17 4v5zM7 15l10 5v-5z"/>',
    "invert": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M20 4v16H4z" fill="currentColor"/>',
    "reset": '<path d="M3.5 11L12 4l8.5 7"/><path d="M5.5 9.5V20h13V9.5"/>',
    # views / windows
    "cine": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9l5 3-5 3z" fill="currentColor"/>',
    # cine bar: previous series, previous frame, play / pause, next frame, next series
    "c_prev_series": '<path d="M6 5v14"/><path d="M18 5.5L9.5 12l8.5 6.5z"/>',
    "c_prev": '<path d="M16 5.5L7.5 12l8.5 6.5z"/>',
    "c_play": '<path d="M8 5l11 7-11 7z" fill="currentColor"/>',
    "c_pause": '<path d="M8.5 5v14M15.5 5v14" stroke-width="3"/>',
    "c_next": '<path d="M8 5.5l8.5 6.5L8 18.5z"/>',
    "c_next_series": '<path d="M18 5v14"/><path d="M6 5.5l8.5 6.5L6 18.5z"/>',
    # DSA: a vessel tree, the mask moved by hand, the auto-shift square
    "dsa": '<path d="M12 21v-7M12 14L6 5M12 14l6-8M9 9.5L4.5 11M15.2 10l4.3 1.5"/>',
    "dsa_shift": '<rect x="7.5" y="7.5" width="9" height="9" rx="1"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3"/>',
    "dsa_auto": '<rect x="4" y="4" width="16" height="16" rx="1.5" stroke-dasharray="3 2.2"/>'
                '<path d="M12 8.5v7M8.5 12h7"/>',
    "mpr": '<rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/>'
           '<rect x="3" y="13" width="8" height="8" rx="1"/><path d="M13 17h8M17 13v8"/>',
    "3d": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
    "fusion": '<circle cx="9" cy="12" r="6"/><circle cx="15" cy="12" r="6"/>',
    "dti": '<path d="M3 8c5-4.5 13-4.5 18 0"/><path d="M3 12.5c5-3.5 13-3.5 18 0"/><path d="M3 17c5-2.5 13-2.5 18 0"/>',
    "help": '<circle cx="12" cy="12" r="8.5"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 1-1 1.7"/>'
            + _DOT.format(12, 16.8),
    "tags": '<path d="M4 4h16v16H4z"/><path d="M8 9h8M8 13h8M8 17h5"/>',
    "trash": '<path d="M4 7h16M9 7V4.5h6V7M6 7l1 13h10l1-13"/>',
    # 3D window
    "eye": '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
    "zoom_out": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-4.8-4.8M7.5 10.5h6"/>',
    "crop": '<path d="M6 2.5V16a2 2 0 0 0 2 2h13.5"/><path d="M18 21.5V8a2 2 0 0 0-2-2H2.5"/>',
    "scalpel": '<path d="M3 21l8.5-8.5"/><path d="M11.5 12.5L19.5 4.5c1.4 1.4 1.2 3.3-.6 5.1l-3.9 3.9z"/>',
    "inspect": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-4.8-4.8M10.5 9.8v3.7"/>' + _DOT.format(10.5, 7.4),
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    "camera": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "quality": '<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-6"/>' + _DOT.format(12, 18),
    "palette": '<path d="M12 3a9 9 0 1 0 0 18c1.2 0 1.8-.8 1.8-1.7 0-.5-.2-.9-.5-1.2-.3-.3-.5-.7-.5-1.2 0-1 .8-1.7 '
               '1.7-1.7H16a5 5 0 0 0 5-5c0-4-4-7.2-9-7.2z"/>' + _DOT.format(7.5, 11) + _DOT.format(10, 7.5)
               + _DOT.format(14.5, 7.5),
    "segment": '<path d="M7 6.5c3-3 9-2.2 11 1.6s.3 8.3-3 10.2-8.8 1.8-10.8-1.8S4 9.5 7 6.5z"/>'
               '<path d="M9 11c1.5-1.5 4-1.2 5 .3" stroke-dasharray="2 2"/>',
    "save": '<path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
    "more": _DOT.format(6, 12) + _DOT.format(12, 12) + _DOT.format(18, 12),
    # tractography / fusion
    "run": '<path d="M7 4.5v15l12-7.5z"/>',
    "roi_include": '<circle cx="12" cy="12" r="8.5"/><path d="M12 8v8M8 12h8"/>',
    "roi_exclude": '<circle cx="12" cy="12" r="8.5"/><path d="M9 9l6 6M15 9l-6 6"/>',
    "export": '<path d="M12 15V3.5M7.5 8L12 3.5 16.5 8"/><path d="M4.5 13v6.5h15V13"/>',
    # 3D rotate modes
    "orbit": '<circle cx="12" cy="12" r="3"/><path d="M3.5 10a8.5 4.5 0 0 1 17 0"/><path d="M20.5 14a8.5 4.5 0 0 1-17 0"/>'
             '<path d="M18 6.8l2.6 3.1-3.9.6M6 17.2l-2.6-3.1 3.9-.6"/>',
    "roll": '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.5 4v4.5H15"/>'
            '<rect x="9" y="9" width="6" height="6" rx="1" transform="rotate(25 12 12)"/>',
    "turn": '<path d="M12 2.5v19" stroke-dasharray="2 2"/><path d="M4 11a8 3.5 0 0 0 16 0"/>'
            '<path d="M16.6 15.6l3.4-4.6.8 4.6"/>',
    # MPR
    "axes": '<path d="M12 3v18M3 12h18"/><circle cx="12" cy="17.5" r="1.8"/><circle cx="17.5" cy="12" r="1.8"/>',
}
SVG["transform"] = SVG["reload"] = SVG["rotate_r"]      # same arrow, different meaning

_cache = {}


def _render(name, color, px):
    body = SVG[name].replace("currentColor", color)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="1.7" '
           'stroke-linecap="round" stroke-linejoin="round">{}</svg>').format(color, body)
    r = QSvgRenderer(QByteArray(svg.encode()))
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p, QRectF(0, 0, px, px))
    p.end()
    return QPixmap.fromImage(img)


def icon(name) -> QIcon:
    """Theme-tinted QIcon; unknown names give an empty icon (callers fall back to text)."""
    if name not in SVG:
        return QIcon()
    if name in _cache:
        return _cache[name]
    ic = QIcon()
    for px in (16, 20, 24, 32, 40, 48):
        ic.addPixmap(_render(name, theme.ICON, px), QIcon.Normal, QIcon.Off)
        ic.addPixmap(_render(name, theme.ICON_HOVER, px), QIcon.Active, QIcon.Off)
        ic.addPixmap(_render(name, theme.ICON_DISABLED, px), QIcon.Disabled, QIcon.Off)
        ic.addPixmap(_render(name, theme.ACCENT_TEXT, px), QIcon.Normal, QIcon.On)
        ic.addPixmap(_render(name, theme.ACCENT_TEXT, px), QIcon.Active, QIcon.On)
    _cache[name] = ic
    return ic
