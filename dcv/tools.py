"""Tool buttons shared by the main window and the MPR window.

Left-click a tool button  -> assign it to the LEFT mouse button  (blue underline)
Right-click a tool button -> assign it to the RIGHT mouse button (pink underline)
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu, QToolButton

from . import theme
from .icons import icon
from .measure import TOOL_NAMES

# (key, short label under the icon, tooltip, shortcut); the icon has the same name as the key (icons.py)
BASIC_TOOLS = [("wl", "W/L", "Window / Level", "W"), ("pan", "Pan", "Pan", "M"), ("zoom", "Zoom", "Zoom", "Z"),
               ("browse", "Browse", "Browse slices by dragging", "S"),
               ("edit", "Select", "Select / move measurements", "")]
EXTRA_TOOLS = {
    "cursor3d": ("3D cursor", "3D cursor: synchronise position in all views", "Q"),
    "eyedropper": ("Picker", "Color picker: click a tissue → create a 3D color curve from its values", "K"),
    "brush": ("Brush", "Segment brush (Shift = erase, Shift+wheel = size)", "B"),
    "seed": ("Seed", "Segmentation seed (same as Ctrl+click)", "J"),
    "wlroi": ("W/L ROI", "W/L ROI: drag a box → window from its contents", "R"),
    "roll": ("Roll", "Roll: drag around the centre to turn this view in its own plane (MPR)", ""),
}
# mouse tools of the DSA mode (DSA ▾ menu in the main window, not toolbar buttons of their own)
DSA_TOOLS = {
    "dsa_shift": ("Pixel-shift", "DSA manual pixel-shift: drag to move the mask (Shift+arrows 1 px, "
                                 "Ctrl+Shift+arrows 0.1 px)"),
    "dsa_auto": ("Auto shift", "DSA auto pixel-shift: click = align the mask inside the yellow square"),
}
MEASURE_ORDER = ["length", "ellipse", "rect", "polygon", "openpoly", "angle", "cobb", "deviation", "arrow", "pencil",
                 "text"]
# marks the tool on the left (blue) / right (pink) mouse button
MOUSE_STYLE = {
    "left": f"background: {theme.ACCENT_WEAK}; border-bottom: 3px solid {theme.ACCENT};",
    "right": f"background: {theme.RIGHT_WEAK}; border-bottom: 3px solid {theme.RIGHT};",
    "both": f"background: {theme.ACCENT_WEAK}; border-top: 3px solid {theme.RIGHT}; "
            f"border-bottom: 3px solid {theme.ACCENT};",
}


def short_name(kind):
    return TOOL_NAMES[kind][0].split(" (")[0]


def abc2_dialog(parent, lengths_cm):
    """ABC/2 volume calculator, prefilled with the longest length measurements (cm)."""
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel
    lengths = sorted(lengths_cm, reverse=True)
    dlg = QDialog(parent)
    dlg.setWindowTitle("ABC/2 volume (e.g. intracerebral haemorrhage)")
    f = QFormLayout(dlg)
    spins = []
    for i, name in enumerate(("A (cm) — largest diameter", "B (cm) — perpendicular to A",
                              "C (cm) — number of slices × thickness")):
        sp = QDoubleSpinBox()
        sp.setRange(0, 100)
        sp.setDecimals(2)
        if i < len(lengths):
            sp.setValue(lengths[i])
        f.addRow(name, sp)
        spins.append(sp)
    res = QLabel()
    f.addRow("Volume:", res)

    def upd():
        a, b, c = (sp.value() for sp in spins)
        res.setText(f"<b>{a * b * c / 2:.1f} ml</b>")

    for sp in spins:
        sp.valueChanged.connect(upd)
    upd()
    bb = QDialogButtonBox(QDialogButtonBox.Close)
    bb.rejected.connect(dlg.reject)
    f.addRow(bb)
    dlg.exec()


class ToolBank(QObject):
    changed = Signal(str)       # status text

    def __init__(self, window, tools: dict, extra=("cursor3d", "eyedropper", "brush", "seed", "wlroi")):
        super().__init__(window)
        self.window = window
        self.tools = tools
        self.extra = extra
        self.buttons = {}
        self.measure_tool = "length"
        self.measure_btn = None
        self.measure_menu = None

    def _new_button(self, icon_name, text):
        b = QToolButton()
        b.setIcon(icon(icon_name))
        b.setText(text)
        b.setIconSize(QSize(22, 22))
        b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        b.setAutoRaise(True)
        return b

    def _button(self, key, text, tip, shortcut):
        b = self._new_button(key, text)
        b.setToolTip(f"{tip}{'  [' + shortcut + ']' if shortcut else ''}\n"
                     "Left-click = assign to LEFT mouse button, right-click = assign to RIGHT mouse button")
        b.clicked.connect(lambda: self.assign("left", key))
        b.setContextMenuPolicy(Qt.CustomContextMenu)
        b.customContextMenuRequested.connect(lambda _: self.assign("right", key))
        self.buttons[key] = b
        if shortcut:
            a = QAction(self.window)
            a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(lambda: self.assign("left", key))
            self.window.addAction(a)
        return b

    def populate(self, tb):
        for key, text, tip, sc in BASIC_TOOLS:
            tb.addWidget(self._button(key, text, tip, sc))
        mb = self._new_button(self.measure_tool, short_name(self.measure_tool))
        mb.setPopupMode(QToolButton.MenuButtonPopup)
        menu = QMenu(mb)
        for k in MEASURE_ORDER:
            name, sc = TOOL_NAMES[k]
            a = menu.addAction(icon(k), f"{name}\t{sc}")
            a.triggered.connect(lambda _=False, kk=k: self.select_measure(kk))
            sa = QAction(self.window)
            sa.setShortcut(QKeySequence(sc))
            sa.triggered.connect(lambda _=False, kk=k: self.select_measure(kk))
            self.window.addAction(sa)
        menu.addSeparator()
        mb.setMenu(menu)
        mb.clicked.connect(lambda: self.assign("left", self.measure_tool))
        mb.setContextMenuPolicy(Qt.CustomContextMenu)
        mb.customContextMenuRequested.connect(lambda _: self.assign("right", self.measure_tool))
        mb.setToolTip("Measurement tools — ▾ chooses the tool\n"
                      "Left-click = assign to LEFT mouse button, right-click = assign to RIGHT mouse button")
        self.measure_btn, self.measure_menu = mb, menu
        tb.addWidget(mb)
        for key in self.extra:
            text, tip, sc = EXTRA_TOOLS[key]
            tb.addWidget(self._button(key, text, tip, sc))
        self.refresh()

    def select_measure(self, kind):
        self.measure_tool = kind
        self.measure_btn.setIcon(icon(kind))
        self.measure_btn.setText(short_name(kind))
        self.assign("left", kind)

    @staticmethod
    def name_of(key):
        names = {k: label for k, label, _, _ in BASIC_TOOLS}
        names.update({k: v[0] for k, v in EXTRA_TOOLS.items()})
        names.update({k: v[0] for k, v in DSA_TOOLS.items()})
        names.update({k: short_name(k) for k in TOOL_NAMES})
        return names.get(key, key)

    def mouse_html(self):
        """Status bar text: what the left / right mouse buttons do now."""
        n = self.name_of
        return (f'<span style="color:{theme.ACCENT}">■</span> Left: {n(self.tools["left"])}&nbsp;&nbsp;&nbsp;'
                f'<span style="color:{theme.RIGHT}">■</span> Right: {n(self.tools["right"])}&nbsp;&nbsp;&nbsp;'
                f'Middle: pan&nbsp;&nbsp;&nbsp;Wheel: slice')

    def assign(self, button, key):
        self.tools[button] = key
        self.refresh()
        names = {k: tip for k, _, tip, _ in BASIC_TOOLS}
        names.update({k: v[1] for k, v in EXTRA_TOOLS.items()})
        names.update({k: v[1] for k, v in DSA_TOOLS.items()})
        names.update({k: v[0] for k, v in TOOL_NAMES.items()})
        self.changed.emit(f"Left mouse: {names.get(self.tools['left'])}   |   Right mouse: "
                          f"{names.get(self.tools['right'])}   |   Middle: pan   |   Wheel: slice")

    def refresh(self):
        items = list(self.buttons.items()) + ([("__m", self.measure_btn)] if self.measure_btn else [])
        for k, b in items:
            kk = self.measure_tool if k == "__m" else k
            left, right = self.tools["left"] == kk, self.tools["right"] == kk
            mode = "both" if left and right else "left" if left else "right" if right else None
            b.setStyleSheet(f"QToolButton {{ {MOUSE_STYLE[mode]} }}" if mode else "")
