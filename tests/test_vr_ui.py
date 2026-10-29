"""GUI test: the redesigned 3D window. One toolbar row with icons, VR/MIP/MinIP and resolution menus, shortcuts of
actions that only sit in ▾ menus, collapsed help, ABC/2 from 3D lengths (label text is mm or cm, the calculator must
use the points). Run via tests\\run_tests.bat (args: app_dir data_dir out_dir)."""
import sys
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QToolBar, QToolButton  # noqa: E402
import dcv.vr_window as vrw  # noqa: E402
import numpy as np  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402
from dcv.volume_view import Measure3D  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, str(a[1:2])), QMessageBox.Yes)[1]))
abc2_calls = []
vrw.abc2_dialog = lambda parent, lengths: abc2_calls.append(lengths)


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
w.open_paths([data_dir])


def run():
    if w.viewports[0].state is None:
        QTimer.singleShot(300, run)
        return
    try:
        st = w.viewports[0].state
        vw = w.vr_window(st)
        vw.resize(1300, 850)
        app.processEvents()
        bars = [t for t in vw.findChildren(QToolBar) if t.isVisible()]
        check(len(bars) == 1, f"one toolbar row ({len(bars)})")
        buttons = [b for b in bars[0].findChildren(QToolButton) if b.isVisible() and b.text()]
        check(all(not b.icon().isNull() for b in buttons), f"every 3D toolbar button has an icon ({len(buttons)})")
        emoji = [b.text() for b in buttons if any(ord(ch) > 0x2000 for ch in b.text())]
        check(not emoji, f"no emoji in 3D toolbar labels {emoji}")

        # render mode menu
        vw.blend_acts["mip"][0].trigger()
        app.processEvents()
        check(vw._blend == "mip" and vw.blend_btn.text() == "MIP", "render mode -> MIP")
        vw.blend_acts["composite"][0].trigger()
        check(vw._blend == "composite" and vw.blend_btn.text() == "VR", "render mode -> VR")

        # resolution menu
        vw.res_acts[2].trigger()
        check(st.render_factor == 2 and st._render_f == 2, f"resolution half (factor {st._render_f})")
        vw.res_acts[1].trigger()
        check(st.render_factor == 1 and st._render_f == 1, "resolution full")

        # shortcuts of actions that only live in dropdown menus are registered on the window
        keys = {a.shortcut().toString() for a in vw.actions()}
        missing = [k for k in ("1", "6", "+", "0", "H", "Shift+C", "Ctrl+Z", "F5", "F1", "X", "I") if k not in keys]
        check(not missing, f"window shortcuts present (missing: {missing})")

        # long help is collapsed by default
        tips = [lb for lb in vw.findChildren(QLabel) if "Double-click empty space" in lb.text()]
        check(tips and not tips[0].isVisible(), "curve editor tips collapsed")
        tog = next(b for b in vw.findChildren(QToolButton) if "Curve editor tips" in b.text())
        tog.setChecked(True)
        app.processEvents()
        check(tips[0].isVisible(), "tips expand on click")

        # measure button: icon + name, select via menu
        vw.select_measure("angle")
        check(vw.measure_btn.text() == "Angle" and vw.measure_btn.isChecked(), "measure menu -> Angle (tool on)")
        vw._release_tool()

        # ---- rotate modes (left-drag): roll = screen plane only, turn = head axis only, free = trackball
        vr = vw.vr
        iren = vr.iren
        W, H = vr.rw.GetSize()
        cx, cy = W // 2, H // 2

        def ev(name, x, y):
            iren.SetEventInformation(int(x), int(y), 0, 0, "\0", 0)
            iren.InvokeEvent(name)

        def drag(pts):
            ev("LeftButtonPressEvent", *pts[0])
            for p in pts[1:]:
                ev("MouseMoveEvent", *p)
            ev("LeftButtonReleaseEvent", *pts[-1])

        cam = vr.renderer.GetActiveCamera()
        vr.view("anterior")

        def angle(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))

        vw.rot_acts["roll"][0].trigger()          # the Roll button next to Rotate
        check(vw.vr.rotate_mode == "roll" and vw.rot_acts["roll"][0].isChecked()
              and not vw.rot_acts["free"][0].isChecked(), "Roll button selects roll (and only roll)")
        dop, up = cam.GetDirectionOfProjection(), cam.GetViewUp()
        arc = [(cx + 150 * np.cos(t), cy + 150 * np.sin(t)) for t in np.linspace(0, np.pi / 2, 10)]   # 90 deg CCW
        drag(arc)
        check(angle(dop, cam.GetDirectionOfProjection()) < 0.5, "Roll keeps the viewing direction")
        check(abs(angle(up, cam.GetViewUp()) - 90) < 3, f"Roll turned the picture {angle(up, cam.GetViewUp()):.0f} deg")
        vr.view("anterior")
        vw.set_rotate_mode("turn")
        axis = np.asarray(vr.vol.direction, float).T @ np.array([0, 0, 1.0])
        dop = np.array(cam.GetDirectionOfProjection())
        drag([(cx + dx, cy + 7 * np.sin(dx)) for dx in np.linspace(0, 100, 8)])            # 100 px -> 50 deg
        new = np.array(cam.GetDirectionOfProjection())
        flat = lambda v: v - (v @ axis) * axis                                               # noqa: E731
        check(abs(dop @ axis - new @ axis) < 0.01, "Turn keeps the head axis (vertical drag wobble ignored)")
        check(abs(angle(flat(dop), flat(new)) - 50) < 3, f"Turn rotated {angle(flat(dop), flat(new)):.0f} deg")
        vr.view("anterior")
        vw.set_rotate_mode("free")
        dop = cam.GetDirectionOfProjection()
        drag([(cx, cy), (cx + 60, cy + 40), (cx + 120, cy + 80)])
        check(angle(dop, cam.GetDirectionOfProjection()) > 5, "free mode: trackball rotation as before")
        # a measuring tool still gets the drag while Roll is selected
        vw.set_rotate_mode("roll")
        vr.view("anterior")
        n = len(vr.measurements3d)
        vw.select_measure("length")
        drag([(cx - 80, cy), (cx, cy), (cx + 80, cy)])
        check(len(vr.measurements3d) == n + 1, "Length tool works with Roll selected")
        vr.measurements3d.clear()
        vw.set_rotate_mode("free")

        # ---- bone lock toggle: no frame may show re-encoded data with the old table (the brief glitch)
        from dcv.segment import BONE_OFFSET
        bad = []

        def on_render(*_):
            table_has_bone = vr.ctf.GetRange()[0] < -BONE_OFFSET / 2
            if vr.state.render_data is not None and table_has_bone != bool(vr.state.bone_encoded):
                bad.append((table_has_bone, vr.state.bone_encoded))
        obs = vr.rw.AddObserver("StartEvent", on_render)
        for on in (not vw.bone_chk.isChecked(), vw.bone_chk.isChecked()):
            vw.bone_chk.setChecked(on)
            app.processEvents()
        vr.rw.RemoveObserver(obs)
        check(not bad, f"bone lock toggle: every frame uses a matching table ({len(bad)} mismatched)")

        # ---- every control of the 3D panel fits the dock at its narrowest (Segments tab was cut off)
        from panel_fit import clipped_controls
        pages = [(vw.tabs.tabText(i), vw.tabs.widget(i), lambda i=i: vw.tabs.setCurrentIndex(i))
                 for i in range(vw.tabs.count())]
        cut = clipped_controls(vw, pages)
        check(not cut, f"3D panel: no control cut off at minimum width {cut[:4]}")
        vw.tabs.setCurrentIndex(0)

        # ABC/2 uses the real length, whatever unit the label shows
        vw.vr.measurements3d.append(Measure3D("length", [(0, 0, 0), (0, 0, 45.0)]))   # label "4.50 cm"
        vw.vr.measurements3d.append(Measure3D("length", [(0, 0, 0), (0, 8.0, 0)]))    # label "8.0 mm"
        vw._abc2()
        got = sorted(round(x, 3) for x in abc2_calls[-1])
        check(got == [0.8, 4.5], f"ABC/2 lengths in cm {got}")
        vw.vr.measurements3d.clear()
        vw.close()
    except Exception:
        traceback.print_exc()
        errors.append("exception")
    print("errors:", len(errors))
    app.quit()


QTimer.singleShot(500, run)
app.exec()
