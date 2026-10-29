"""One 2D viewport: a frameless image with an active-viewport border; accepts series dragged from the list."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout

from . import theme
from .slice_view import SliceView

SERIES_MIME = "application/x-dcv-series"


class ViewportFrame(QFrame):
    activated = Signal(object)
    dropped = Signal(object, str)
    maximizeRequested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.state = None
        v = QVBoxLayout(self)
        v.setContentsMargins(1, 1, 1, 1)
        v.setSpacing(0)
        self.slice = SliceView("axial")
        v.addWidget(self.slice, 1)
        self.slice.activated.connect(lambda _: self.activated.emit(self))
        self.slice.maximizeRequested.connect(lambda _: self.maximizeRequested.emit(self))
        self.set_active(False)

    def set_state(self, state, orientation="axial"):
        self.state = state
        keep = self.slice.state is state
        self.slice.set_state(state, keep_window=keep)
        if self.slice.orientation != orientation:
            self.slice.set_orientation(orientation)

    def set_active(self, on):
        self._active = on
        self.setStyleSheet("ViewportFrame { border: 1px solid %s; }" % (theme.ACTIVE_VIEWPORT if on else "#2a2a2a"))

    def mousePressEvent(self, e):
        self.activated.emit(self)
        super().mousePressEvent(e)

    # a series dragged from the list: highlight this viewport while it is over it, open it here on drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(SERIES_MIME):
            e.acceptProposedAction()
            self.setStyleSheet("ViewportFrame { border: 3px solid %s; }" % theme.ACCENT)

    def dragLeaveEvent(self, e):
        self.set_active(self._active)

    def dropEvent(self, e):
        self.set_active(self._active)
        uid = bytes(e.mimeData().data(SERIES_MIME)).decode()
        self.dropped.emit(self, uid)
        e.acceptProposedAction()
