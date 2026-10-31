import "./style.css";
import { normal } from "./dicom/header";
import { expand, firstHeader, groupByStudy, scan, seriesCount, type Progress, type Series } from "./dicom/scan";
import { makeThumbnail } from "./dicom/thumb";
import { openSeries, Volume } from "./dicom/volume";
import { ICONS } from "./icons";
import { fmtMl } from "./seg/segment";
import { SegmentPanel } from "./seg/panel";
import { Layer, Segments, type Segment } from "./seg/segments";
import { openCollage } from "./view/collage";
import { KINDS, MPRModel, mprHooks, PlaneSource, SLABS, TITLES } from "./view/mpr";
import { Linker } from "./view/linked";
import { CurveEditor } from "./vr/editor";
import { builtinPresets, copyTF, fromDict, usesVeins, type TransferFunction } from "./vr/tf";
import { boneAndVessels, forceGridStep } from "./vr/bone";
import { MAX_SEGMENTS_3D, type SegmentMask } from "./vr/renderer";
import { QUALITY, VIEWS, VRView, type Resolution } from "./vr/view3d";
import { Viewport, type Tool } from "./view/viewport";

/** W/L presets of the desktop app (dcv/main_window.py DEFAULT_WL): name, WW, WL. */
const PRESETS: [string, number, number][] = [
  ["Brain", 80, 40], ["Subdural", 250, 80], ["Stroke", 40, 40], ["Abdomen / Mediastinum", 400, 40], ["Liver", 150, 60],
  ["Lung", 1500, -600], ["Bone", 1800, 400], ["Angio", 600, 170], ["Temporal bone", 4000, 700],
];

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const IOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
// iOS kills a tab that holds much more than ~1–1.5 GB; keep decoded series well below that (3D adds a GPU copy)
const BUDGET = IOS ? 450e6 : 2000e6;

/** The 2D viewports of the layout (1×1 uses the first); `cellSeries` = the series each one shows. */
const views2d = [...document.querySelectorAll<HTMLCanvasElement>("#grid canvas")].map((c) => new Viewport(c));
const cellSeries: (Series | null)[] = views2d.map(() => null);
const openTicket = views2d.map(() => 0);
const LAYOUTS: [string, number, string][] = [["1x1", 1, "1 × 1"], ["1x2", 2, "1 × 2"], ["1x3", 3, "1 × 3"], ["2x2", 4, "2 × 2"]];
let cells = 1;
const linker = new Linker(views2d, () => cells);
/** The active 2D viewport: the series list loads into it. */
let vp2d = views2d[0];
/** The viewport the toolbar acts on: the active 2D view, or the MPR view touched last. */
let vp = vp2d;
const mprViews = [...document.querySelectorAll<HTMLCanvasElement>("#mpr canvas")].map((c) => new Viewport(c));
let mode: "2d" | "mpr" | "3d" = "2d";
let mpr: { m: MPRModel; off: () => void } | null = null;
const state = {
  series: [] as Series[],
  current: null as Series | null,
  volumes: new Map<Series, Volume>(),
  used: [] as Series[],            // least recently viewed first
};

// ================================================================== toolbar
interface Btn {
  el: HTMLButtonElement;
  update?: () => void;
}
const buttons: Btn[] = [];

function button(icon: string, label: string, title: string, onClick: (b: HTMLButtonElement) => void,
  update?: (b: HTMLButtonElement) => void): HTMLButtonElement {
  const b = document.createElement("button");
  b.className = "tool";
  b.innerHTML = `${ICONS[icon]}<span>${label}</span>`;
  b.title = title;
  b.addEventListener("click", () => onClick(b));
  $("tools").append(b);
  buttons.push({ el: b, update: update && (() => update(b)) });
  return b;
}

function sep() {
  const s = document.createElement("div");
  s.className = "sep";
  $("tools").append(s);
}

const TOOLS: [Tool, string, string, string][] = [
  ["wl", "wl", "W/L", "Window / level: drag (W)"],
  ["pan", "pan", "Pan", "Pan (M)"],
  ["zoom", "zoom", "Zoom", "Zoom: drag up / down (Z) — or pinch with two fingers"],
  ["scroll", "scroll", "Scroll", "Scroll through the images: drag up / down (S)"],
  ["locate", "locate", "Locate", "3D cursor (C): tap a point — the other views of the layout go to it. Shift + press works with any tool"],
];

button("open", "Open", "Open DICOM files, a folder or a .zip archive", (b) => showMenu(b, [
  ["Files or .zip archive…", () => $("pick-files").click()],
  ["Folder…", () => $("pick-folder").click()],
]));
button("series", "Series", "Show / hide the series list", () => toggleSide(), (b) => b.classList.toggle("on", $("side").classList.contains("open")));
sep();
for (const [tool, icon, label, title] of TOOLS) {
  button(icon, label, title, () => setTool(tool), (b) => {
    b.disabled = mode === "3d" || !vp.vol || (tool === "locate" && mode !== "2d");   // 3D has its own tools
    b.classList.toggle("on", !b.disabled && vp.tool === tool);
  });
}
const MEASURES: [Tool, string, string][] = [
  ["length", "Length", "L"], ["ellipse", "Ellipse (ROI): area, mean, SD, min, max", "E"], ["angle", "Angle", "Shift+A"], ["cobb", "Cobb angle", "Shift+C"],
];
button("length", "Measure", "Length, ellipse ROI, angle, Cobb angle. Drag a point to move it, the line to move all of it", (b) => {
  showMenu(b, MEASURES.map(([tool, label, key]): MenuItem => [`${vp.tool === tool ? "✓ " : ""}${label}`, () => setTool(tool), key]));
}, (b) => {
  b.disabled = mode === "3d" || !vp.vol;
  b.classList.toggle("on", !b.disabled && MEASURES.some(([t]) => t === vp.tool));
});
button("seed", "Seed", "Seed segmentation: tap inside a lesion (or Ctrl + click with any tool). The segment shows in 2D, MPR and 3D with its volume", (b) => {
  const v = currentVolume();
  const list = segs.list(v);
  if (!list.length && mode !== "3d") {
    armSeed();
    return;
  }
  showMenu(b, [
    ...(mode === "3d" ? [] : [["New seed: tap inside the lesion", () => armSeed(), "Ctrl+click"] as MenuItem, null as MenuItem]),
    ...list.map((s): MenuItem => [`Edit ${s.name} · ${s.grown ? fmtMl(s.ml) : "empty"}`, () => segPanel.open(v!, s)]),
  ]);
}, (b) => {
  const v = currentVolume();
  b.disabled = !v || (mode === "3d" ? !segs.list(v).length : !mprAvailable());
  b.classList.toggle("on", !b.disabled && mode !== "3d" && vp.tool === "seed");
});
button("presets", "Window", "W/L presets, auto window, invert", (b) => {
  const items: MenuItem[] = [];
  if (vp.vol?.ww) items.push([`Default (${fmtNum(vp.vol.ww)} / ${fmtNum(vp.vol.wl ?? 0)})`, () => vp.setWindow(vp.vol!.ww!, vp.vol!.wl!)]);
  items.push(["Auto (1–99 %)", () => vp.autoWindow()]);
  items.push(null);
  PRESETS.forEach(([name, ww, wl], i) => items.push([`${name}  (${ww} / ${wl})`, () => vp.setWindow(ww, wl), `${i + 1}`]));
  items.push(null);
  items.push([`${vp.invert ? "✓ " : ""}Invert`, () => { vp.invert = !vp.invert; vp.requestRender(); }, "I"]);
  showMenu(b, items);
}, (b) => (b.disabled = mode === "3d" || !vp.vol));
button("layout", "Layout", "Several series side by side, linked by reference lines (MRI: axial / sagittal / coronal)", (b) => {
  const k = $("grid").dataset.layout;
  showMenu(b, [
    ...LAYOUTS.map(([id, , label]): MenuItem => [`${k === id ? "✓ " : ""}${label}`, () => setLayout(id)]),
    null,
    [`${linker.showLines ? "✓ " : ""}Reference lines`, () => {
      linker.showLines = !linker.showLines;
      for (const v of views2d) v.requestRender();
    }],
    ["Hide the 3D cursor", () => linker.clearCursor(), "Esc"],
  ]);
}, (b) => {
  b.disabled = mode !== "2d" || !state.series.length;
  b.classList.toggle("on", cells > 1 && mode === "2d");
});
sep();
button("mpr", "MPR", "Multiplanar reconstruction: sagittal / coronal / axial, tilt the axes, slab MIP", (b) => {
  if (mode !== "mpr") {
    openMpr();
    return;
  }
  const m = mpr!.m;
  const layout = $("mpr").dataset.layout ?? "all";
  showMenu(b, [
    ["Back to the series (2D)", () => setMode("2d")],
    null,
    ...SLABS.map(([label, mm, md]): MenuItem => [`${m.slabMm === mm && (mm === 0 || m.slabMode === md) ? "✓ " : ""}${label}`, () => {
      m.slabMm = mm;
      m.slabMode = md;
      m.changed();
    }]),
    null,
    ...(["all", ...KINDS] as const).map((k): MenuItem => [`${layout === k ? "✓ " : ""}${k === "all" ? "All three views" : TITLES[k] + " only"}`,
      () => setMprLayout(k)]),
    null,
    [`${m.showLines ? "✓ " : ""}Show axis lines`, () => {
      m.showLines = !m.showLines;
      m.changed();
    }],
    ["Reset axes (no tilt, centre)", () => m.reset()],
  ]);
}, (b) => {
  b.disabled = !mprAvailable();
  b.classList.toggle("on", mode === "mpr");
});
sep();
button("vr", "3D", "3D volume rendering with colour curves, bone lock, MIP", (b) => {
  if (mode !== "3d") {
    open3d();
    return;
  }
  const v = vr();
  const panel = !$("vr").classList.contains("nopanel");
  showMenu(b, [
    ["Back to the series (2D)", () => setMode("2d")],
    null,
    ...([["VR (colour curves)", 0], ["MIP", 1], ["MinIP", 2]] as const).map(([label, k]): MenuItem => [`${v.blend === k ? "✓ " : ""}${label}`, () => {
      v.blend = k;
      v.requestRender();
      vrLabel();
    }]),
    null,
    ...Object.entries(VIEWS).map(([k, [label]]): MenuItem => [`View: ${label}`, () => v.view(k)]),
    null,
    ...Object.keys(QUALITY).map((q): MenuItem => [`${v.quality === q ? "✓ " : ""}Ray detail: ${q}`, () => {
      v.quality = q;
      v.requestRender();
    }]),
    [`${panel ? "✓ " : ""}Curve panel`, () => $("vr").classList.toggle("nopanel")],
    null,
    [`${v.crop.on ? "✓ " : ""}Crop box`, () => v.setCrop(!v.crop.on), "Shift+C"],
    [`${v.crop.on && !v.cropEdit ? "✓ " : ""}Hide the box (keep the crop)`, () => v.setCropBox(!v.cropEdit), "B"],
    ["Reset crop box", () => v.resetCrop()],
    [`${v.cutOutside ? "✓ " : ""}Scalpel keeps the inside (cuts outside the lasso)`, () => {
      v.cutOutside = !v.cutOutside;
    }],
  ]);
}, (b) => {
  b.disabled = !mprAvailable();
  b.classList.toggle("on", mode === "3d");
});
button("collage", "Collage", "Collage: this series on one 5 × 5 / 4 × 4 / 4 × 3 image, one slice every 5 mm (Ctrl+G)", () => openCollage(vp, toast, IOS, segs),
  (b) => (b.disabled = mode !== "2d" || vp.count() < 2));
button("dsa", "DSA", "Digital subtraction (XA runs): on / off, mask, pixel-shift", (b) => {
  const on = !!vp.dsa;
  showMenu(b, [
    [`${on ? "✓ " : ""}Subtraction on / off`, () => vp.setDsa(!on), "/"],
    ["Use displayed image as mask", () => vp.dsaMaskCurrent(), "?"],
    ["Reset mask (default mask, no shift)", () => vp.dsaReset()],
    null,
    [`${vp.tool === "dsaShift" ? "✓ " : ""}Manual pixel-shift (drag the mask)`, () => dsaTool("dsaShift"), "⇧ arrows"],
    [`${vp.tool === "dsaAuto" ? "✓ " : ""}Auto pixel-shift (tap in the yellow square)`, () => dsaTool("dsaAuto"), "Ctrl+M"],
  ]);
}, (b) => {
  b.disabled = mode !== "2d" || !vp.dsaAvailable();
  b.classList.toggle("on", !!vp.dsa);
});
button("cine", "Cine", "Play / pause (Space)", () => vp.toggleCine(), (b) => {
  b.disabled = mode !== "2d" || vp.count() < 2;
  b.classList.toggle("on", vp.cinePlaying);
});
button("fit", "Fit", "Fit the image to the window (double tap)", () => vp.resetView(), (b) => (b.disabled = mode === "3d" || !vp.vol));
button("trash", "Delete", "Delete the selected measurement (all on this image when none is selected)", () => {
  if (!vp.deleteSelected()) vp.clearMeasurements();
}, (b) => (b.disabled = mode === "3d" || !vp.vol));
sep();
button("info", "About", "About, privacy, memory, shortcuts", () => showAbout());

let toolBeforeSeed: Tool = "wl";
function armSeed() {
  if (vp.tool !== "seed") toolBeforeSeed = vp.tool;
  setTool("seed");
  toast("Tap inside the lesion");
}

/** Grow a segment from a voxel of the active series (all of its slices are needed first). */
async function placeSeed(view: Viewport, voxel: [number, number, number]) {
  const v = currentVolume();
  if (!v || !mprAvailable()) {
    toast("Seed works on a stack of images, not on a run or a single image");
    return;
  }
  const [z, y, x] = voxel;
  if (z < 0 || y < 0 || x < 0 || z >= v.n || y >= v.rows || x >= v.cols) return;
  if (view.tool === "seed") setTool(toolBeforeSeed);    // one seed, then the tool from before (the desktop's habit)
  if (!v.complete) {
    progress(v.readyCount, v.n, "Loading all images…");
    const poll = window.setInterval(() => progress(v.readyCount, v.n, "Loading all images…"), 200);
    await v.want(Array.from({ length: v.n }, (_, k) => k));
    clearInterval(poll);
    hideProgress();
    if (currentVolume() !== v) return;
  }
  const t0 = performance.now();
  const seg = segs.seed(v, voxel);
  if (!seg) {
    toast("Nothing grows from this point");
    return;
  }
  toast(`${seg.name}: ${fmtMl(seg.ml)} (${Math.round(performance.now() - t0)} ms). Drag the handles to change what enters it`);
  segPanel.open(v, seg);
}

function setTool(t: Tool) {
  if (t === "locate" && mode !== "2d") return;
  if (t !== "locate") linker.clearCursor();
  for (const v of [...views2d, ...mprViews]) {
    v.cancelDraft();
    v.tool = t;
    v.requestRender();
  }
  refresh();
}

async function dsaTool(t: Tool) {
  if (await vp.setDsa(true)) setTool(t);
}

function fmtNum(v: number) {
  return Math.round(v).toString();
}

// ================================================================== menus
type MenuItem = [string, () => void, string?] | null;
function showMenu(anchor: HTMLElement, items: MenuItem[]) {
  const m = $("menu");
  m.innerHTML = "";
  for (const it of items) {
    if (!it) {
      m.append(Object.assign(document.createElement("hr")));
      continue;
    }
    const b = document.createElement("button");
    b.innerHTML = `<span>${it[0]}</span>${it[2] ? `<kbd>${it[2]}</kbd>` : ""}`;
    b.addEventListener("click", () => {
      hideMenu();
      it[1]();
      refresh();
    });
    m.append(b);
  }
  m.hidden = false;
  const r = anchor.getBoundingClientRect();
  const w = m.offsetWidth;
  m.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, r.left))}px`;
  m.style.top = `${r.bottom + 4}px`;
}
function hideMenu() {
  $("menu").hidden = true;
}
document.addEventListener("pointerdown", (e) => {
  if (!$("menu").hidden && !$("menu").contains(e.target as Node) && !(e.target as HTMLElement).closest?.(".tool")) hideMenu();
});

// ================================================================== side bar / series list
/** Phone in either direction: the short side of the screen is small. */
function isPhone(): boolean {
  return Math.min(window.innerWidth, window.innerHeight) < 600;
}

function toggleSide(open?: boolean) {
  $("side").classList.toggle("open", open);
  refresh();
}

function renderSeriesList() {
  const box = $("series");
  box.innerHTML = "";
  for (const g of groupByStudy(state.series)) {
    const ph = document.createElement("div");
    ph.className = "patient";
    ph.innerHTML = `<b></b><span></span>`;
    ph.querySelector("b")!.textContent = g.patient || "(no name)";
    ph.querySelector("span")!.textContent = [g.patientId, fmtDate(g.birthDate)].filter(Boolean).join(" · ");
    box.append(ph);
    const sh = document.createElement("div");
    sh.className = "study";
    sh.textContent = [fmtDate(g.studyDate), g.studyDesc].filter(Boolean).join(" — ") || "Study";
    box.append(sh);
    for (const s of g.series) {
      const it = document.createElement("button");
      it.className = "item";
      it.classList.toggle("active", s === state.current);
      it.classList.toggle("shown", s !== state.current && cellSeries.slice(0, cells).includes(s));
      const h = firstHeader(s);
      const th = document.createElement("div");
      th.className = "thumb";
      if (s.thumb) th.append(cloneCanvas(s.thumb));
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.innerHTML = "<b></b><span></span>";
      meta.querySelector("b")!.textContent = s.description || `Series ${h.seriesNumber}`;
      const n = seriesCount(s);
      meta.querySelector("span")!.textContent = `${h.modality} · ${n} ${s.multiframe ? "frames" : n === 1 ? "image" : "images"}`;
      it.append(th, meta);
      it.addEventListener("click", () => {
        show(s);
        if (isPhone()) toggleSide(false);
      });
      (it as HTMLButtonElement & { series?: Series }).series = s;
      box.append(it);
    }
  }
}

function cloneCanvas(c: HTMLCanvasElement) {
  const o = document.createElement("canvas");
  o.width = c.width;
  o.height = c.height;
  o.getContext("2d")!.drawImage(c, 0, 0);
  return o;
}

function fmtDate(d: string) {
  if (!/^\d{8}$/.test(d)) return d;
  const m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][+d.slice(4, 6) - 1];
  return `${+d.slice(6, 8)} ${m} ${d.slice(0, 4)}`;
}

async function makeThumbnails(list: Series[]) {
  for (const s of list) {
    if (state.series !== list) return;
    try {
      s.thumb = await makeThumbnail(s);
    } catch {
      s.thumb = null;
    }
    const el = [...$("series").querySelectorAll<HTMLButtonElement & { series?: Series }>(".item")].find((e) => e.series === s);
    if (el && s.thumb) el.querySelector(".thumb")!.replaceChildren(cloneCanvas(s.thumb));
  }
}

// ================================================================== opening
let progressTimer = 0;
const progress: Progress = (done, total, text) => {
  const p = $("progress");
  p.hidden = false;
  (p.querySelector(".bar > div") as HTMLElement).style.width = total ? `${(100 * done) / total}%` : "100%";
  p.querySelector(".bar")!.classList.toggle("busy", !total);
  p.querySelector("span")!.textContent = total ? `${text} ${done}/${total}` : text;
  clearTimeout(progressTimer);
};
function hideProgress() {
  progressTimer = window.setTimeout(() => ($("progress").hidden = true), 150);
}

async function openFiles(files: File[]) {
  if (!files.length) return;
  hideMenu();
  try {
    progress(0, 0, "Reading…");
    const { sources, errors } = await expand(files, progress);
    const list = await scan(sources, progress);
    hideProgress();
    if (!list.length) {
      toast(errors[0] ?? "No DICOM images found in what was opened.");
      return;
    }
    for (const v of state.volumes.values()) v.dispose();
    state.volumes.clear();
    state.used = [];
    state.series = list;
    state.current = null;
    views2d.forEach((v, i) => {
      v.setVolume(null);
      cellSeries[i] = null;
    });
    segPanel.close();
    linker.cursor = null;
    $("welcome").hidden = true;
    setMode("2d");
    setActive(views2d[0]);
    renderSeriesList();
    toggleSide(!isPhone());                          // on a phone the list would cover the image: Series opens it
    makeThumbnails(list);
    await show(list[0]);
    await fillLayout();
  } catch (e) {
    hideProgress();
    toast(`Could not open: ${(e as Error).message ?? e}`);
  }
}

/** Show a series in a viewport (the active one by default); decoded series stay in memory while they fit the
 *  budget (least recently viewed dropped first, never one on screen). */
async function show(s: Series, view = vp2d) {
  const i = views2d.indexOf(view);
  if (cellSeries[i] === s && view.vol) return;
  const ticket = ++openTicket[i];
  cellSeries[i] = s;
  if (view === vp2d) {
    if (state.current !== s) segPanel.close();
    state.current = s;
  }
  renderSeriesList();
  const r = await load(s);
  if (ticket !== openTicket[i]) return;
  if (!r) {
    cellSeries[i] = view.vol ? cellSeries[i] : null;
    return;
  }
  if (view === vp2d && mode !== "2d") setMode("2d");
  view.setVolume(r.vol, r.first);
  if (r.vol.multiframe && r.vol.frameTimeMs) view.playCine();      // a run plays on opening
  linker.changed(view);
  refresh();
}

async function load(s: Series): Promise<{ vol: Volume; first: number } | null> {
  let vol = state.volumes.get(s);
  let first = 0;
  if (!vol) {
    const h = firstHeader(s);
    const need = seriesCount(s) * h.rows * h.cols * (h.samples === 3 ? 3 : 4);
    const onScreen = new Set(cellSeries.slice(0, cells));
    for (const old of [...state.used]) {
      if (totalBytes() + need <= BUDGET) break;
      if (old === s || onScreen.has(old)) continue;
      state.volumes.get(old)?.dispose();
      state.volumes.delete(old);
      state.used = state.used.filter((x) => x !== old);
    }
    try {
      progress(0, 0, `Opening ${s.description || "series"}…`);
      const r = await openSeries(s);
      hideProgress();
      vol = r.vol;
      first = r.first;
      state.volumes.set(s, vol);
    } catch (e) {
      hideProgress();
      toast(`Could not open this series: ${(e as Error).message ?? e}`);
      return null;
    }
  } else first = vol.multiframe ? 0 : Math.floor(vol.n / 2);
  state.used = [...state.used.filter((x) => x !== s), s];
  return { vol, first };
}

// ================================================================== layout (several series side by side)
function setLayout(k: string) {
  const entry = LAYOUTS.find(([id]) => id === k);
  if (!entry) return;
  cells = entry[1];
  $("grid").dataset.layout = k;
  views2d.forEach((v, i) => {
    if (i < cells) return;
    openTicket[i]++;
    v.setVolume(null);
    cellSeries[i] = null;
  });
  setActive(views2d.indexOf(vp2d) < cells ? vp2d : views2d[0]);
  renderSeriesList();
  for (const v of views2d) v.requestRender();
  if (cells > 1) toast("Tap a view, then a series to show it there. Locate (C): tap a point, the other views go to it");
  fillLayout();
}

/** Plane of a series from its first header: 0 sagittal, 1 coronal, 2 axial, -1 unknown. */
function planeOfSeries(s: Series): number {
  const iop = firstHeader(s).iop;
  if (!iop) return -1;
  const n = normal(iop).map(Math.abs);
  return n.indexOf(Math.max(...n));
}

/** Empty views get series of the same study: planes not yet shown first (MRI: axial + sagittal + coronal), then the
 *  series nearest in the list; localizers (fewer than 3 images) last. */
async function fillLayout() {
  const ref = state.current ?? cellSeries.find(Boolean) ?? null;
  if (!ref) return;
  const study = firstHeader(ref).studyUid;
  const at = (s: Series) => state.series.indexOf(s);
  for (let i = 0; i < cells; i++) {
    if (cellSeries[i]) continue;
    const used = cellSeries.slice(0, cells).filter(Boolean) as Series[];
    const planes = new Set(used.map(planeOfSeries));
    const cand = state.series.filter((s) => firstHeader(s).studyUid === study && !used.includes(s));
    if (!cand.length) return;
    const score = (s: Series) => {
      const p = planeOfSeries(s);
      return (p >= 0 && !planes.has(p) ? 0 : 1e6) + (seriesCount(s) < 3 ? 1e5 : 0) + Math.abs(at(s) - at(ref)) + (at(s) < at(ref) ? 0.5 : 0);
    };
    cand.sort((a, b) => score(a) - score(b));
    await show(cand[0], views2d[i]);
  }
}

function totalBytes() {
  let t = 0;
  for (const v of state.volumes.values()) t += v.bytes;
  return t;
}

$<HTMLInputElement>("pick-files").addEventListener("change", (e) => {
  const inp = e.target as HTMLInputElement;
  openFiles([...(inp.files ?? [])]);
  inp.value = "";
});
$<HTMLInputElement>("pick-folder").addEventListener("change", (e) => {
  const inp = e.target as HTMLInputElement;
  openFiles([...(inp.files ?? [])]);
  inp.value = "";
});
$("w-files").addEventListener("click", () => $("pick-files").click());
$("w-folder").addEventListener("click", () => $("pick-folder").click());

// drag & drop (desktop browsers), folders included
const view = $("view");
view.addEventListener("dragover", (e) => {
  e.preventDefault();
  view.classList.add("drop");
});
view.addEventListener("dragleave", () => view.classList.remove("drop"));
view.addEventListener("drop", async (e) => {
  e.preventDefault();
  view.classList.remove("drop");
  const items = [...(e.dataTransfer?.items ?? [])];
  const entries = items.map((i) => i.webkitGetAsEntry?.()).filter(Boolean) as FileSystemEntry[];
  const files: File[] = [];
  const walk = async (en: FileSystemEntry, path: string): Promise<void> => {
    if (en.isFile) {
      const f = await new Promise<File>((res, rej) => (en as FileSystemFileEntry).file(res, rej));
      files.push(new File([f], f.name, { type: f.type, lastModified: f.lastModified }));
      Object.defineProperty(files[files.length - 1], "webkitRelativePath", { value: path + f.name });
    } else if (en.isDirectory) {
      const reader = (en as FileSystemDirectoryEntry).createReader();
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const c of batch) await walk(c, `${path}${en.name}/`);
      }
    }
  };
  if (entries.length) for (const en of entries) await walk(en, "");
  else files.push(...(e.dataTransfer?.files ?? []));
  openFiles(files);
});

// ================================================================== cine bar
const slider = $<HTMLInputElement>("c-slider");
slider.addEventListener("input", () => {
  vp.stopCine();
  vp.setIndex(+slider.value - 1);
});
$("c-play").addEventListener("click", () => vp.toggleCine());
$("c-slower").addEventListener("click", () => vp.setFps(vp.fps - (vp.fps > 10 ? 5 : 1)));
$("c-faster").addEventListener("click", () => vp.setFps(vp.fps + (vp.fps >= 10 ? 5 : 1)));

// DSA nudge pad (touch): 1 px or 0.1 px steps, like Shift / Ctrl+Shift + arrows
let padStep = 1;
$("pad-step").addEventListener("click", () => {
  padStep = padStep === 1 ? 0.1 : 1;
  $("pad-step").textContent = `${padStep} px`;
});
for (const b of $("dsapad").querySelectorAll<HTMLButtonElement>("button[data-d]")) {
  b.addEventListener("click", () => {
    const [dy, dx] = b.dataset.d!.split(",").map(Number);
    vp.dsaMove(dy * padStep, dx * padStep);
  });
}

// ================================================================== status
function refresh() {
  for (const b of buttons) b.update?.();
  const n = vp.count();
  $("cine").hidden = n < 2 || mode !== "2d";
  slider.max = String(Math.max(1, n));
  slider.value = String(vp.index + 1);
  $("c-pos").textContent = `${vp.index + 1}/${n}`;
  $("c-play").innerHTML = vp.cinePlaying ? ICONS.pause : ICONS.play;
  $("c-fps").textContent = `${vp.fps} fps`;
  $("dsapad").hidden = !(vp.dsa && vp.tool === "dsaShift");
}
const segs = new Segments();
const segPanel = new SegmentPanel(segs);

function masksOf(list: Segment[]): SegmentMask[] {
  return list.filter((s) => s.visible && s.grown).map((s) => ({ id: s.id, rev: s.rev, color: s.color, sub: s.grown!.sub, off: s.grown!.off, dims: s.grown!.dims }));
}
function sync3dSegments(fast: boolean) {
  const v = view3d;
  if (!v?.vol || mode !== "3d") return;
  const masks = masksOf(segs.list(v.vol));
  v.setSegments(masks, fast);
  if (masks.length > MAX_SEGMENTS_3D) toast(`3D shows the first ${MAX_SEGMENTS_3D} segments; hide one to see another`);
}
segs.onChanged = () => {
  for (const v of [...views2d, ...mprViews]) v.requestRender();
  sync3dSegments(segPanel.dragging);
  refresh();
};
segPanel.onSettled = () => sync3dSegments(false);

views2d.forEach((view) => {
  const layer = new Layer();
  view.layer = (k) => {
    const v = view.vol;
    return v instanceof Volume && segs.list(v).length ? layer.get(`${segs.version}|${k}`, v.cols, v.rows, (img) => segs.paintSlice(v, k, img)) : null;
  };
  view.onSeed = (r, c) => {
    const v = view.vol;
    if (v) void placeSeed(view, [view.index, Math.floor(r), Math.floor(c)]);
  };
  view.canvas.addEventListener("pointerdown", () => {
    if (view !== vp) setActive(view);
  }, { capture: true });
  view.onStatus = (t) => toast(t);
  view.hooks = linker.hooks(view);
  let shown: [unknown, number] = [null, -1];
  view.onChange = () => {
    if (view.vol !== shown[0] || view.index !== shown[1]) {      // another slice: the reference lines move
      shown = [view.vol, view.index];
      linker.changed(view);
    }
    if (view === vp) refresh();
  };
});

// ================================================================== MPR
function currentVolume(): Volume | null {
  return state.current ? state.volumes.get(state.current) ?? null : null;
}

function mprAvailable(): boolean {
  const v = currentVolume();
  return !!v && !v.rgb && v.n >= 3 && !(v.multiframe && v.frameTimeMs);   // a stack, not a run / loop
}

function setActive(v: Viewport) {
  vp = v;
  const i = views2d.indexOf(v);
  if (i >= 0) {
    vp2d = v;
    if (cellSeries[i] !== state.current) {
      segPanel.close();                      // the editor belongs to the series it was opened on
      state.current = cellSeries[i];
      renderSeriesList();
    }
  }
  for (const o of mprViews) {
    o.active = mode === "mpr" && o === v;
    o.requestRender();
  }
  for (const o of views2d) {
    const on = mode === "2d" && cells > 1 && o === v;
    if (o.active !== on) {
      o.active = on;
      o.requestRender();
    }
  }
  refresh();
}

function setMode(next: typeof mode) {
  if (next !== "3d" && view3d?.vol) {
    view3d.dispose();                        // frees the graphics memory of the volume
    store(OPEN_KEY, null);
  }
  $("vr").hidden = next !== "3d";
  if (next !== "mpr" && mpr) {
    mpr.off();
    mpr = null;
    for (const v of mprViews) v.setVolume(null);
  }
  if (next !== "2d") for (const v of views2d) v.stopCine();
  mode = next;
  $("seg").classList.toggle("over3d", mode === "3d");
  (mode === "3d" ? document.querySelector(".vr-view")! : $("view")).append($("seg"));   // beside the curve panel, not over it
  $("grid").hidden = mode !== "2d";
  $("mpr").hidden = mode !== "mpr";
  if (mode === "2d") {
    setActive(vp2d);
    views2d.forEach((v, i) => {                 // views emptied for MPR / 3D memory come back
      const s = cellSeries[i];
      if (i < cells && s && !v.vol) show(s, v);
    });
  } else setActive(vp);
  refresh();
}

function setMprLayout(k: string) {
  $("mpr").dataset.layout = k;
  if (k !== "all") setActive(mprViews[KINDS.indexOf(k as never)]);
}

let settleTimer = 0;
function settle() {
  clearTimeout(settleTimer);
  settleTimer = window.setTimeout(() => {
    if (!mpr) return;
    mpr.m.fast = false;
    for (const v of mprViews) v.requestRender();
  }, 180);
}

/** MPR / 3D hold more per series: other decoded series are dropped (they reload when chosen again). Series in the
 *  other views of the layout stay while they are small together (an MRI series is a few MB). */
function keepOnlyCurrent() {
  const others = new Set(cellSeries.slice(0, cells).filter((s) => s && s !== state.current));
  let othersBytes = 0;
  for (const s of others) othersBytes += state.volumes.get(s!)?.bytes ?? 0;
  for (const [s, v] of [...state.volumes]) {
    if (s === state.current || (others.has(s) && othersBytes <= BUDGET / 4)) continue;
    views2d.forEach((view) => {
      if (view.vol === v) view.setVolume(null);         // cellSeries stays: it reloads on the way back to 2D
    });
    v.dispose();
    state.volumes.delete(s);
  }
  state.used = state.used.filter((s) => state.volumes.has(s));
}

/** All slices are needed: wait for the rest of the series, then three planes through its centre. */
async function openMpr() {
  const v = currentVolume();
  if (!v || !mprAvailable()) return;
  keepOnlyCurrent();
  vp2d.stopCine();
  if (!v.complete) {
    progress(v.readyCount, v.n, "Loading all images…");
    const poll = window.setInterval(() => progress(v.readyCount, v.n, "Loading all images…"), 200);
    await v.want(Array.from({ length: v.n }, (_, k) => k));
    clearInterval(poll);
    hideProgress();
    if (currentVolume() !== v) return;
  }
  const m = new MPRModel(v);
  setMode("mpr");
  KINDS.forEach((k, i) => {
    const view = mprViews[i];
    view.setVolume(new PlaneSource(m, k), m.index(k));
    view.hooks = mprHooks(m, k, view, settle);
    const layer = new Layer();
    view.layer = () => {
      const src = view.vol as PlaneSource | null;
      if (!src || !segs.list(v).length) return null;
      return layer.get(`${segs.version}|${m.planeKey(k)}`, src.cols, src.rows, (img) => {
        const [sz, sy, sx] = v.spacing;
        const c0 = m.toWorld(k, 0, 0), c1 = m.toWorld(k, 0, 1), r1 = m.toWorld(k, 1, 0);
        const vox = (w: number[]): [number, number, number] => [w[0] / sx, w[1] / sy, w[2] / sz];
        const o = vox(c0), a = vox(c1), b = vox(r1);
        return segs.paintPlane(v, o, [a[0] - o[0], a[1] - o[1], a[2] - o[2]], [b[0] - o[0], b[1] - o[1], b[2] - o[2]], img);
      });
    };
    view.onSeed = (r, c) => {
      const [sz, sy, sx] = v.spacing;
      const w = m.toWorld(k, r, c);
      void placeSeed(view, [Math.round(w[2] / sz), Math.round(w[1] / sy), Math.round(w[0] / sx)]);
    };
    view.tool = vp2d.tool === "dsaShift" || vp2d.tool === "dsaAuto" || vp2d.tool === "locate" ? "wl" : vp2d.tool;
    view.invert = vp2d.invert;
    view.setWindow(vp2d.ww, vp2d.wl);
  });
  const off = m.onChange(() => {
    mprViews.forEach((view, i) => {
      view.index = m.index(KINDS[i]);
      view.requestRender();
    });
    refresh();
  });
  mpr = { m, off };
  setActive(mprViews[2]);
}

// ================================================================== 3D
let view3d: VRView | null = null;
let editor: CurveEditor | null = null;
const USER_PRESETS = "soeasy.vr.presets";        // the owner's own presets (not patient data), this browser only

const RES_KEY = "soeasy.vr.resolution";          // the owner's choice (not patient data), this browser only
const OPEN_KEY = "soeasy.vr.open";                // set while 3D is open: still there on loading = iOS closed the page
function stored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function store(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch { /* private window: forgotten */ }
}
let ranOut = stored(OPEN_KEY) === "1";            // Auto stays at half resolution until the page is reloaded again
const pageWasClosed = ranOut;
store(OPEN_KEY, null);
window.addEventListener("pagehide", () => store(OPEN_KEY, null));   // a normal close is not a memory problem

function vr(): VRView {
  if (!view3d) {
    const v = new VRView($<HTMLCanvasElement>("vr-canvas"));
    view3d = v;
    v.resChoice = (stored(RES_KEY) as Resolution | null) ?? "auto";
    v.ranOut = ranOut;
    v.onChange = () => syncPanel();
    v.onStatus = (t) => toast(t);
    v.onLost = () => {
      ranOut = true;
      const canvas = $("vr-canvas");
      setMode("2d");
      v.destroy();
      view3d = null;
      canvas.replaceWith(canvas.cloneNode());     // a new context next time: this one is gone
      toast("3D ran out of graphics memory and was closed. Open it again: Auto now uses half resolution.");
    };
  }
  return view3d;
}

let snapIndex = -1;
function renderSnaps() {
  const v = vr();
  const box = $("vr-snaps");
  box.hidden = !v.snapshots.length;
  box.replaceChildren(...v.snapshots.map((s, i) => {
    const b = document.createElement("div");
    b.className = "snap" + (i === snapIndex ? " cur" : "");
    b.title = `Snapshot ${i + 1}: tap to go back to this view`;
    const img = document.createElement("img");
    img.src = s.thumb;
    img.alt = `Snapshot ${i + 1}`;
    const x = document.createElement("button");
    x.className = "x";
    x.textContent = "×";
    x.setAttribute("aria-label", "Delete snapshot");
    x.addEventListener("click", (e) => {
      e.stopPropagation();
      v.snapshots.splice(i, 1);
      snapIndex = -1;
      renderSnaps();
    });
    b.append(img, x);
    b.addEventListener("click", () => goSnap(i));
    return b;
  }));
}

function goSnap(i: number) {
  const v = vr();
  if (!v.snapshots.length || !editor) return;
  snapIndex = (i + v.snapshots.length) % v.snapshots.length;
  v.applySnapshot(v.snapshots[snapIndex]);
  editor.setTF(v.tf);
  syncPanel();
  vrLabel();
  renderSnaps();
}

function userPresets(): TransferFunction[] {
  try {
    return (JSON.parse(localStorage.getItem(USER_PRESETS) ?? "[]") as Record<string, unknown>[]).map(fromDict);
  } catch {
    return [];
  }
}

function storeUserPresets(list: TransferFunction[]) {
  try {
    localStorage.setItem(USER_PRESETS, JSON.stringify(list));
  } catch {
    toast("Could not save presets in this browser (private mode or storage full).");
  }
}

function fillPresets() {
  const sel = $<HTMLSelectElement>("vr-preset");
  const v = vr();
  sel.innerHTML = "";
  const add = (label: string, list: TransferFunction[], prefix: string) => {
    if (!list.length) return;
    const g = document.createElement("optgroup");
    g.label = label;
    list.forEach((tf, i) => g.append(new Option(tf.name, `${prefix}${i}`)));
    sel.append(g);
  };
  add("Built-in", builtinPresets(), "b");
  add("My presets", userPresets(), "u");
  const cur = [...sel.options].find((o) => o.text === v.tf.name);
  if (cur) sel.value = cur.value;
  else {
    sel.prepend(new Option(`${v.tf.name} (edited)`, "cur"));
    sel.value = "cur";
  }
}

/** Busy text over the view while something heavy runs on the main thread (bone area, upload). */
function busy(text: string, fn: () => void) {
  progress(0, 0, text);
  setTimeout(() => {
    try {
      fn();
    } catch (e) {
      toast(`3D: ${(e as Error).message ?? e}`);
    }
    hideProgress();
  }, 30);
}

function vrLabel() {
  const v = vr();
  if (!v.vol) return;
  const mode3 = ["VR", "MIP", "MinIP"][v.blend];
  const res = v.resolution === 2 ? " · half resolution" : "";
  $("vr-label").textContent = `${v.vol.patient}\n${v.vol.description}   [${mode3}${res}]`;
}

const rimToSlider = (t: number) => Math.log(t / 5) / Math.log(40);           // 5..200 HU, logarithmic
const sliderToRim = (s: number) => 5 * 40 ** s;

function syncPanel() {
  const v = vr();
  if (!v.vol || !editor) return;
  const rims = v.tf.curves.some((c) => c.hide_edges);
  $<HTMLInputElement>("vr-gain").value = String(Math.log2(v.gain));
  $<HTMLInputElement>("vr-rim").value = String(rimToSlider(v.tf.edge_threshold));
  $<HTMLInputElement>("vr-rim").disabled = !rims;
  $("vr-rim-v").textContent = `${Math.round(v.tf.edge_threshold)} HU`;
  $("vr-bone-l").hidden = !v.ct;
  $<HTMLInputElement>("vr-bone").checked = v.boneLock;
  $("vr-bone-v").textContent = v.boneLock ? `≥ ${v.boneHu} HU` : "";
  $("vr-art-l").hidden = v.arteryCore === null;
  const art = $<HTMLInputElement>("vr-art");
  if (document.activeElement !== art) {
    art.value = v.arteryHu === null ? "" : String(v.arteryHu);
    art.placeholder = v.arteryCore === null ? "" : `${Math.round(v.arteryCore)}`;
  }
  $<HTMLInputElement>("vr-shade").checked = v.tf.shading;
  editor.boneHu = v.ct && v.boneLock ? v.boneHu : null;
  editor.gain = v.gain;
  const c = editor.selectedCurve();
  for (const id of ["ct-color", "ct-bone", "ct-rims", "ct-del"]) $<HTMLButtonElement>(id).disabled = !c;
  $("ct-bone").classList.toggle("on", c?.layer === "bone");
  $("ct-rims").classList.toggle("on", !!c?.hide_edges);
  $("t3-pan").classList.toggle("on", v.tool === "pan");
  $("t3-measure").classList.toggle("on", v.tool === "measure");
  $("t3-mdel").hidden = !v.measures.length;
  $<HTMLSelectElement>("vr-res").value = v.resChoice;
  $("t3-crop").classList.toggle("on", v.crop.on);
  $<HTMLButtonElement>("t3-box").disabled = !v.crop.on;
  $("t3-box").textContent = v.crop.on && !v.cropEdit ? "Show box" : "Hide box";
  $("t3-cut").classList.toggle("on", v.tool === "scalpel");
  $<HTMLButtonElement>("t3-undo").disabled = !v.canUndo;
  $<HTMLButtonElement>("t3-reset").disabled = !v.hasCuts;
  renderSnaps();
  $("ct-del").textContent = editor.sel?.p !== null && editor.sel?.p !== undefined && c && c.points.length > 2 ? "Delete point" : "Delete curve";
  $("ct-del").title = "Delete the selected point or curve";
  fillPresets();
  editor.draw();
}

/** Every slice decoded, the bone area, then the volume on the GPU (heavy: once per opening). */
async function open3d() {
  const v = currentVolume();
  if (!v || !mprAvailable()) return;
  keepOnlyCurrent();
  vp2d.stopCine();
  if (!v.complete) {
    const poll = window.setInterval(() => progress(v.readyCount, v.n, "Loading all images…"), 200);
    progress(v.readyCount, v.n, "Loading all images…");
    await v.want(Array.from({ length: v.n }, (_, k) => k));
    clearInterval(poll);
    hideProgress();
    if (currentVolume() !== v) return;
  }
  let view: VRView;
  try {
    view = vr();
  } catch (e) {
    toast(`3D is not available: ${(e as Error).message ?? e}`);
    return;
  }
  setMode("3d");
  if (pageWasClosed && view.resChoice === "auto") toast("3D closed the page last time: Auto uses half resolution until the next reload");
  store(OPEN_KEY, "1");
  busy(view.ct ? "Bone area and upload to the graphics card…" : "Upload to the graphics card…", () => {
    view.open(v);
    sync3dSegments(false);
    if (!editor) {
      editor = new CurveEditor($("vr-editor"));
      editor.onChange = (fast) => {
        vr().applyTF(fast);
        if (!fast) syncPanel();
      };
      editor.onSelect = () => syncPanel();
      editor.onBone = (hu) => busy("Recomputing the bone area…", () => {
        vr().boneHu = hu;
        vr().upload();
        syncPanel();
      });
    }
    editor.setData(view.tf, view.lo, view.hi, view.hist);
    vrLabel();
    syncPanel();
  });
}

$("t3-pan").addEventListener("click", () => {
  const v = vr();
  v.setTool(v.tool === "pan" ? "rotate" : "pan");
});
$("t3-measure").addEventListener("click", () => {
  const v = vr();
  v.setTool(v.tool === "measure" ? "rotate" : "measure");
  if (v.tool === "measure") toast("Drag from one point of the surface to the other");
});
$("t3-mdel").addEventListener("click", () => vr().deleteMeasure());
$("vr-res").addEventListener("change", () => {
  const choice = $<HTMLSelectElement>("vr-res").value as Resolution;
  store(RES_KEY, choice);
  busy("Uploading the volume…", () => {
    vr().setResChoice(choice);
    vrLabel();
    syncPanel();
  });
});
$("t3-crop").addEventListener("click", () => vr().setCrop(!vr().crop.on));
$("t3-box").addEventListener("click", () => vr().setCropBox(!vr().cropEdit));
$("t3-cut").addEventListener("click", () => {
  const v = vr();
  v.setTool(v.tool === "scalpel" ? "rotate" : "scalpel");
  if (v.tool === "scalpel") toast(`Draw around what to ${v.cutOutside ? "keep" : "remove"} — through the whole depth`);
});
$("t3-undo").addEventListener("click", () => busy("Undo…", () => vr().undoCut()));
$("t3-reset").addEventListener("click", () => busy("Restoring…", () => vr().resetCuts()));
$("t3-snap").addEventListener("click", () => {
  const v = vr();
  v.addSnapshot();
  snapIndex = v.snapshots.length - 1;
  renderSnaps();
  toast(`Snapshot ${v.snapshots.length} kept (until this series is closed)`);
});
$("t3-png").addEventListener("click", async () => {
  const blob = await vr().png();
  if (!blob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "soeasy-3d.png";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

$("vr-preset").addEventListener("change", () => {
  const val = $<HTMLSelectElement>("vr-preset").value;
  const list = val.startsWith("u") ? userPresets() : builtinPresets();
  const tf = list[+val.slice(1)];
  if (!tf || !editor) return;
  const apply = () => {
    vr().setPreset(tf);
    editor!.setTF(vr().tf);
    syncPanel();
  };
  if (vr().ct && usesVeins(tf)) {
    busy("Separating arteries and veins…", apply);
    toast("Arteries and veins separate on an arterial-phase CTA. On a late phase both are equally bright, so they cannot be told apart.");
  } else busy("Applying…", apply);
});
$("vr-art").addEventListener("change", () => {
  const raw = $<HTMLInputElement>("vr-art").value.trim();
  const hu = raw === "" ? null : Math.max(100, Math.min(1000, +raw || 0));
  busy("Separating arteries and veins…", () => vr().setArteryHu(hu));
});
$("vr-save").addEventListener("click", () => {
  const v = vr();
  const name = prompt("Preset name", v.tf.name.replace(/ \(edited\)$/, ""));
  if (!name) return;
  v.tf.name = name;
  const list = userPresets().filter((p) => p.name !== name);
  list.push(copyTF(v.tf));
  storeUserPresets(list);
  toast(`Preset "${name}" saved in this browser`);
  syncPanel();
});
$("vr-more").addEventListener("click", (e) => {
  const v = vr();
  const mine = userPresets();
  showMenu(e.currentTarget as HTMLElement, [
    ["Export these curves (.json, desktop format)", () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([JSON.stringify(v.tf, null, 2)], { type: "application/json" }));
      a.download = `${v.tf.name.replace(/[^\w\- ]+/g, "_")}.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }],
    ["Import a preset (.json from the desktop app)…", () => $("pick-preset").click()],
    null,
    ...mine.map((p): MenuItem => [`Delete my preset "${p.name}"`, () => {
      storeUserPresets(mine.filter((q) => q !== p));
      syncPanel();
    }]),
  ]);
});
$<HTMLInputElement>("pick-preset").addEventListener("change", async (e) => {
  const inp = e.target as HTMLInputElement;
  const f = inp.files?.[0];
  inp.value = "";
  if (!f || !editor) return;
  try {
    const tf = fromDict(JSON.parse(await f.text()));
    storeUserPresets([...userPresets().filter((p) => p.name !== tf.name), tf]);
    vr().setPreset(tf);
    editor.setTF(vr().tf);
    syncPanel();
  } catch {
    toast("This file is not a colour preset.");
  }
});
$("vr-gain").addEventListener("input", () => {
  const v = vr();
  v.gain = 2 ** +$<HTMLInputElement>("vr-gain").value;
  v.applyTF(true);
  if (editor) editor.gain = v.gain;
  editor?.draw();
});
$("vr-gain").addEventListener("change", () => vr().applyTF(false));
$("vr-rim").addEventListener("input", () => {
  const v = vr();
  v.tf.edge_threshold = Math.round(sliderToRim(+$<HTMLInputElement>("vr-rim").value));
  $("vr-rim-v").textContent = `${v.tf.edge_threshold} HU`;
  v.applyTF(true);
});
$("vr-rim").addEventListener("change", () => vr().applyTF(false));
$("vr-bone").addEventListener("change", () => busy("Bone area…", () => {
  const v = vr();
  v.boneLock = $<HTMLInputElement>("vr-bone").checked;
  v.upload();
  syncPanel();
}));
$("vr-shade").addEventListener("change", () => {
  vr().tf.shading = $<HTMLInputElement>("vr-shade").checked;
  vr().requestRender();
});
$("ct-add").addEventListener("click", () => editor?.addCurve((vr().lo + Math.min(vr().hi, 1500)) / 2));
$("ct-color").addEventListener("click", () => editor?.pickColor());
$("ct-del").addEventListener("click", () => editor?.deleteSelected());
for (const [id, key] of [["ct-bone", "layer"], ["ct-rims", "hide_edges"]] as const) {
  $(id).addEventListener("click", () => {
    const c = editor?.selectedCurve();
    if (!c) return;
    if (key === "layer") c.layer = c.layer === "bone" ? "normal" : "bone";
    else c.hide_edges = !c.hide_edges;
    vr().applyTF(false);
    syncPanel();
  });
}

mprViews.forEach((view) => {
  view.canvas.addEventListener("pointerdown", () => setActive(view), { capture: true });
  view.onStatus = (t) => toast(t);
  view.onChange = () => {
    if (view === vp) {
      for (const o of mprViews) {           // one window for the three planes, like the desktop MPR window
        if (o !== view && (o.ww !== view.ww || o.wl !== view.wl || o.invert !== view.invert)) {
          o.ww = view.ww;
          o.wl = view.wl;
          o.invert = view.invert;
          o.requestRender();
        }
      }
    }
    refresh();
  };
});

let toastTimer = 0;
function toast(text: string) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => (t.hidden = true), 3500);
}

// ================================================================== keyboard
window.addEventListener("keydown", (e) => {
  if ((e.target as HTMLElement).closest?.("input, textarea, dialog[open]")) return;
  const k = e.key;
  if (mode === "3d") {                                 // keys of the desktop 3D window
    const v = view3d;
    if (!v?.vol) return;
    const low = k.toLowerCase();
    if (low === "c" && e.shiftKey) v.setCrop(!v.crop.on);
    else if (low === "b" && !e.ctrlKey && !e.metaKey && v.crop.on) v.setCropBox(!v.cropEdit);
    else if (low === "x" && !e.ctrlKey && !e.metaKey) v.setTool(v.tool === "scalpel" ? "rotate" : "scalpel");
    else if (low === "p" && !e.ctrlKey && !e.metaKey) v.setTool(v.tool === "pan" ? "rotate" : "pan");
    else if (low === "l" && !e.ctrlKey && !e.metaKey) v.setTool(v.tool === "measure" ? "rotate" : "measure");
    else if ((k === "Delete" || k === "Backspace") && v.measures.length) v.deleteMeasure();
    else if (low === "z" && (e.ctrlKey || e.metaKey)) busy("Undo…", () => v.undoCut());
    else if (k === "Insert") $("t3-snap").click();
    else if (k === "PageUp") goSnap(snapIndex - 1);
    else if (k === "PageDown") goSnap(snapIndex + 1);
    else if (k === "Escape") v.setTool("rotate");
    else return;
    e.preventDefault();
    return;
  }
  const dsaKey = k === "/" || k === "?" || ((e.ctrlKey || e.metaKey || e.shiftKey) && k.toLowerCase() === "m");
  if (mode !== "2d" && dsaKey) return;                 // DSA works on the acquired frames only
  let used = true;
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "g") {
    if (mode === "2d" && vp.count() >= 2) openCollage(vp, toast, IOS, segs);
  } else if (k === "/" && !e.shiftKey) vp.setDsa(!vp.dsa);
  else if (k === "?" && (e.ctrlKey || e.metaKey)) vp.dsaReset();
  else if (k === "?") vp.dsaMaskCurrent();
  else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "m") { if (vp.dsa) setTool("dsaAuto"); }
  else if (e.shiftKey && k.toLowerCase() === "m") dsaTool("dsaShift");
  else if (!e.ctrlKey && !e.metaKey && !e.altKey && /^[1-9]$/.test(k)) {
    const p = PRESETS[+k - 1];
    if (p) vp.setWindow(p[1], p[2]);
  } else if (!e.ctrlKey && !e.metaKey && !e.altKey && k.length === 1 && "wmzslc".includes(k.toLowerCase()) && !e.shiftKey) {
    setTool({ w: "wl", m: "pan", z: "zoom", s: "scroll", l: "length", c: "locate" }[k.toLowerCase()] as Tool);
  } else if (!e.ctrlKey && !e.metaKey && !e.altKey && k.toLowerCase() === "e" && !e.shiftKey) setTool("ellipse");
  else if (!e.ctrlKey && !e.metaKey && !e.altKey && e.shiftKey && k.toLowerCase() === "a") setTool("angle");
  else if (!e.ctrlKey && !e.metaKey && !e.altKey && e.shiftKey && k.toLowerCase() === "c") setTool("cobb");
  else if (k.toLowerCase() === "i" && !e.ctrlKey && !e.metaKey) {
    vp.invert = !vp.invert;
    vp.requestRender();
  } else if (k.toLowerCase() === "a" && !e.ctrlKey && !e.metaKey) vp.autoWindow();
  else {
    if (k === "Escape") linker.clearCursor();
    used = vp.key(e);
  }
  if (used) {
    e.preventDefault();
    refresh();
  }
});

// ================================================================== about
function showAbout() {
  const d = $<HTMLDialogElement>("about");
  const mem = performance as Performance & { memory?: { usedJSHeapSize: number } };
  const mb = (b: number) => `${(b / 1e6).toFixed(0)} MB`;
  d.innerHTML = `
    <h2>SoEasy DICOM Viewer — web</h2>
    <p class="disclaimer">Personal visualisation aid — <b>not a certified diagnostic device</b>.</p>
    <p>All DICOM processing runs in this browser. Images are never uploaded; the app sends no analytics and works
    offline once installed (Share → Add to Home Screen on iPad / iPhone).</p>
    <p><b>Memory:</b> decoded images ${mb(totalBytes())} in ${state.volumes.size} series
    ${mem.memory ? ` · JS heap ${mb(mem.memory.usedJSHeapSize)}` : ""}</p>
    <h3>Touch</h3>
    <p>One finger: the selected tool. Two fingers: pinch to zoom, move to pan. Double tap: fit.</p>
    <h3>Layout</h3>
    <p>Layout → 1 × 2, 1 × 3, 2 × 2: tap a view, then a series to show it there. Lines show where the slice of each
    other view cuts this one. Locate (C, or Shift + press): tap a point, the other views go to the slice through it.</p>
    <h3>Seed</h3>
    <p>Seed (or Ctrl + click): tap inside a lesion. The segment shows in 2D, MPR and 3D with its volume; the two handles of
    the histogram change what enters it. Press Seed again to edit it. 3D shows up to ${MAX_SEGMENTS_3D} segments.</p>
    <h3>Collage</h3>
    <p>Collage (Ctrl+G): the active series on one image, one slice every 5 mm, centred; the slider moves the run. A stack can be shown as axial, coronal or sagittal (reformatted).</p>
    <h3>Keys</h3>
    <p>↑ ↓ / wheel: images · Space: cine · W M Z S L E C: W/L, pan, zoom, scroll, length, ellipse, locate · Shift+A: angle · Shift+C: Cobb · 1–9: presets · A: auto
    window · I: invert · /: DSA on/off · ?: displayed image as mask · Ctrl+?: reset mask · Shift (+Ctrl) + arrows:
    move the mask 1 (0.1) px · Shift+M: pixel-shift tool · Ctrl+M: auto pixel-shift · Delete: measurement</p>
    <form method="dialog"><button class="big">Close</button></form>`;
  d.showModal();
}

refresh();

// ================================================================== offline (service worker, production only)
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js").catch(() => undefined);
}

// ================================================================== dev: test data and a handle for automated checks
if (import.meta.env.DEV) {
  (window as unknown as { __dcv: unknown }).__dcv = {
    boneAndVessels, forceGridStep, segs, segPanel, get vp() { return vp; }, get vp2d() { return vp2d; }, views2d, cellSeries, linker, setLayout, get cells() { return cells; }, mprViews, get mpr() { return mpr; }, state, openFiles, totalBytes, openMpr, setMode,
    open3d, get view3d() { return view3d; }, get editor() { return editor; },
  };
  const q = new URLSearchParams(location.search).get("testdata");
  if (q) {
    (async () => {
      const list: string[] = await (await fetch(`/__testdata/list?path=${encodeURIComponent(q)}`)).json();
      const files: File[] = [];
      for (const p of list) {
        const blob = await (await fetch(`/__testdata/file?path=${encodeURIComponent(p)}`)).blob();
        const f = new File([blob], p.split("/").pop()!);
        Object.defineProperty(f, "webkitRelativePath", { value: p });
        files.push(f);
      }
      await openFiles(files);
    })();
  }
}
