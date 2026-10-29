"""Colour tokens and the Qt stylesheet of the dark theme.

Neutral, slightly cool greys around the images (a coloured surround would bias how grey values are read), one blue
accent taken from the logo, pink only for "assigned to the right mouse button".
"""

BG = "#17191D"              # window background, menu bar
SURFACE = "#202328"         # toolbars, docks, status bar, menus
SURFACE_2 = "#2A2E35"       # cards, list items, inputs
HOVER = "#323741"
PRESSED = "#3A404B"
BORDER = "#363B45"
TEXT = "#E4E7EC"
MUTED = "#9AA2AF"
ACCENT = "#3B8EEA"          # logo blue, lightened for dark backgrounds
ACCENT_WEAK = "#1F3A5C"
ACCENT_TEXT = "#7DB6F5"
RIGHT = "#E64A8E"           # right mouse button
RIGHT_WEAK = "#47223A"
ACTIVE_VIEWPORT = "#E8B400"  # yellow frame of the active viewport (radiology convention)

ICON = "#D3D8E0"
ICON_HOVER = "#FFFFFF"
ICON_DISABLED = "#5B6270"


def _arrow(direction):
    """Small triangle PNG for spin box buttons (Qt style sheets need an image there). Needs a running QApplication."""
    import os
    import tempfile

    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPolygonF

    d = os.path.join(tempfile.gettempdir(), "DicomColorViewer", "ui")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"arrow_{direction}_{MUTED[1:]}.png")
    if not os.path.exists(path):
        img = QImage(16, 16, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(MUTED))
        tri = [(2, 12), (14, 12), (8, 4)] if direction == "up" else [(2, 4), (14, 4), (8, 12)]
        p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in tri]))
        p.end()
        img.save(path)
    return path.replace("\\", "/")


def main_stylesheet():
    return f"""
QMainWindow, QMainWindow > QWidget {{ background: {BG}; }}
QMenuBar {{ background: {BG}; color: {TEXT}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 9px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {HOVER}; }}
QMenu {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 26px 5px 10px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_WEAK}; }}
QMenu::item:disabled {{ color: {ICON_DISABLED}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; margin-left: 4px; }}

QToolBar {{ background: {SURFACE}; border: none; border-bottom: 1px solid {BORDER}; padding: 3px 6px; spacing: 1px; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 8px 6px; }}
QToolButton {{ color: {TEXT}; background: transparent; border: 1px solid transparent; border-radius: 6px;
               padding: 3px 3px 2px 3px; font-size: 11px; }}
QToolButton:hover {{ background: {HOVER}; border-color: {BORDER}; }}
QToolButton:pressed {{ background: {PRESSED}; }}
QToolButton:checked {{ background: {ACCENT_WEAK}; border-color: {ACCENT}; color: {ACCENT_TEXT}; }}
QToolButton[popupMode="1"] {{ padding-right: 13px; }}
QToolButton[popupMode="2"] {{ padding-right: 11px; }}
QToolButton::menu-button {{ border: none; width: 12px; }}
QToolButton::menu-button:hover {{ background: {PRESSED}; border-top-right-radius: 6px;
                                  border-bottom-right-radius: 6px; }}

QDockWidget {{ color: {MUTED}; font-size: 11px; }}
QDockWidget::title {{ background: {SURFACE}; padding: 6px 8px; border-bottom: 1px solid {BORDER}; }}
QStatusBar {{ background: {SURFACE}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 6px; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 4px; padding: 3px 6px; selection-background-color: {ACCENT}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QAbstractSpinBox {{ padding-right: 18px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 16px; border: none;
    border-left: 1px solid {BORDER}; background: {SURFACE_2}; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 4px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 4px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {HOVER}; }}
QAbstractSpinBox::up-arrow {{ image: url({_arrow("up")}); width: 8px; height: 8px; }}
QAbstractSpinBox::down-arrow {{ image: url({_arrow("down")}); width: 8px; height: 8px; }}
QPushButton {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 5px;
               padding: 5px 10px; }}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {PRESSED}; }}
QPushButton:disabled {{ color: {ICON_DISABLED}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; padding: 7px 12px; }}
QPushButton#primary:hover {{ background: #529BEE; }}
QGroupBox {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; margin-top: 16px;
             padding: 12px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {TEXT}; font-weight: 600; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 7px 12px; border: none;
                border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}
QListWidget {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 6px; outline: none; }}
QListWidget::item {{ padding: 3px 4px; }}
QListWidget::item:selected {{ background: {ACCENT_WEAK}; color: {TEXT}; }}
QScrollArea {{ background: transparent; border: none; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: white; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {PRESSED}; border-radius: 4px; min-height: 30px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: #4A515E; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {PRESSED}; border-radius: 4px; min-width: 30px; margin: 2px; }}
"""


TOOLTIP = f"QToolTip {{ background: #0E1013; color: {TEXT}; border: 1px solid {BORDER}; padding: 5px 7px; }}"
