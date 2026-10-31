/**
 * Colour-curve transfer functions — a port of dcv/tf_model.py, same presets and the same
 * JSON (presets saved by the desktop app load here and the other way round).
 *
 * A transfer function has several curves; each curve is a polyline of (HU, opacity 0..1) points with a colour per
 * point, 0 outside its first / last point. Curves merge: opacity = the maximum, colour = opacity-weighted mean.
 * `hide_edges` curves colour only homogeneous tissue, not the partial-volume rim of brighter structures (vessel /
 * bone walls pass through the HU range of an enhancing tumour): rendered with a 2D (HU × gradient) table.
 * `layer: "bone"` curves colour only the bone area when the bone lock is on, `layer: "vein"` curves only the vessels that
 * are not arteries (see bone.ts).
 */

export type RGB = [number, number, number];
export interface TFPoint {
  x: number;
  y: number;
  color: RGB;
}
export interface TFCurve {
  name: string;
  points: TFPoint[];
  layer: "normal" | "bone" | "vein";
  hide_edges: boolean;
}
export interface TransferFunction {
  name: string;
  curves: TFCurve[];
  shading: boolean;
  ambient: number;            // lighting defaults of the desktop
  diffuse: number;
  specular: number;
  specular_power: number;
  background: RGB;
  edge_threshold: number;     // gradient (HU) above which hide_edges curves stop colouring
}

/** Gradient axis of the 2D table: 0..RIM_GRAD_MAX HU (dcv/volume_view.py). */
export const RIM_GRAD_MAX = 320;

const guessLayer = (name: string): TFCurve["layer"] => (/^(tulang|bone)/i.test(name) ? "bone" : /^(vein|vena)/i.test(name) ? "vein" : "normal");

export function makeTF(name: string, curves: TFCurve[], extra: Partial<TransferFunction> = {}): TransferFunction {
  return {
    name, curves, shading: true, ambient: 0.15, diffuse: 0.9, specular: 0.3, specular_power: 15,
    background: [0, 0, 0], edge_threshold: 60, ...extra,
  };
}

export function copyTF(tf: TransferFunction): TransferFunction {
  return JSON.parse(JSON.stringify(tf));
}

/** Desktop JSON (TransferFunction.to_dict) → TransferFunction; missing fields get the desktop defaults. */
export function fromDict(d: Record<string, unknown>): TransferFunction {
  const num = (v: unknown, dflt: number) => (typeof v === "number" && Number.isFinite(v) ? v : dflt);
  const curves = ((d.curves as Record<string, unknown>[]) ?? []).map((c) => {
    const name = String(c.name ?? "Curve");
    const pts = ((c.points as Record<string, unknown>[]) ?? []).map((p) => ({
      x: num(p.x, 0), y: Math.min(1, Math.max(0, num(p.y, 0))),
      color: (Array.isArray(p.color) ? (p.color as number[]).slice(0, 3).map((v) => num(v, 1)) : [1, 1, 1]) as RGB,
    })).sort((a, b) => a.x - b.x);
    return { name, points: pts, layer: (c.layer === "bone" || c.layer === "normal" || c.layer === "vein" ? c.layer : guessLayer(name)) as TFCurve["layer"],
      hide_edges: !!c.hide_edges };
  });
  return makeTF(String(d.name ?? "Preset"), curves, {
    shading: d.shading === undefined ? true : !!d.shading,
    ambient: num(d.ambient, 0.15), diffuse: num(d.diffuse, 0.9), specular: num(d.specular, 0.3),
    specular_power: num(d.specular_power, 15), edge_threshold: num(d.edge_threshold, 60),
    background: (Array.isArray(d.background) ? (d.background as number[]).slice(0, 3) : [0, 0, 0]) as RGB,
  });
}

function curve(name: string, pts: [number, number, RGB][], hideEdges = false): TFCurve {
  return { name, points: pts.map(([x, y, c]) => ({ x, y, color: c.map((v) => v / 255) as RGB })), layer: guessLayer(name), hide_edges: hideEdges };
}

const IVORY: RGB = [240, 225, 190], BONE_HI: RGB = [255, 250, 235], RED: RGB = [200, 20, 20], RED_HI: RGB = [255, 70, 60];
const VEIN: RGB = [40, 70, 200], VEIN_HI: RGB = [80, 130, 255], SKIN: RGB = [225, 160, 120], MUSCLE: RGB = [190, 70, 60];
const FAT: RGB = [240, 210, 120];

/** Built-in presets of the desktop app (starting points; change them and save as your own). */
export function builtinPresets(): TransferFunction[] {
  return [
    makeTF("CT - Bone", [curve("Bone", [[180, 0, IVORY], [300, 0.12, IVORY], [650, 0.55, BONE_HI], [3000, 0.85, BONE_HI]])]),
    makeTF("CT Angio - Vessels + Bone", [
      curve("Vessels", [[130, 0, RED], [190, 0.18, RED], [280, 0.45, RED_HI], [800, 0.45, RED_HI], [1000, 0.0, RED_HI]]),
      curve("Bone", [[380, 0, IVORY], [500, 0.35, IVORY], [900, 0.7, BONE_HI], [3000, 0.9, BONE_HI]]),
    ]),
    makeTF("CT Angio - Arteries / Veins + Bone", [
      curve("Veins", [[100, 0, VEIN], [140, 0.2, VEIN], [200, 0.45, VEIN_HI], [500, 0.45, VEIN_HI], [600, 0.0, VEIN_HI]]),
      curve("Arteries", [[110, 0, RED], [170, 0.2, RED], [280, 0.45, RED_HI], [450, 0.45, RED_HI], [500, 0.0, RED_HI]]),
      curve("Bone", [[450, 0, IVORY], [600, 0.4, IVORY], [900, 0.7, BONE_HI], [3000, 0.9, BONE_HI]]),
    ]),
    makeTF("CT Angio - Vessels only", [curve("Vessels", [[140, 0, RED], [200, 0.25, RED], [320, 0.6, RED_HI], [420, 0.0, RED_HI]])]),
    makeTF("CT - Skin + Bone", [
      curve("Skin", [[-500, 0, SKIN], [-250, 0.06, SKIN], [-80, 0.12, SKIN], [0, 0.0, SKIN]]),
      curve("Bone", [[200, 0, IVORY], [400, 0.3, IVORY], [900, 0.7, BONE_HI], [3000, 0.85, BONE_HI]]),
    ]),
    makeTF("CT - Muscle + Vessels + Bone", [
      curve("Fat", [[-150, 0, FAT], [-90, 0.015, FAT], [-40, 0.0, FAT]]),
      curve("Muscle/organ", [[10, 0, MUSCLE], [40, 0.04, MUSCLE], [80, 0.06, MUSCLE], [120, 0.0, MUSCLE]]),
      curve("Vessels", [[140, 0, RED], [200, 0.3, RED_HI], [330, 0.45, RED_HI], [400, 0.0, RED_HI]]),
      curve("Bone", [[400, 0, IVORY], [600, 0.5, BONE_HI], [3000, 0.9, BONE_HI]]),
    ]),
    makeTF("CT - Enhancing lesion/tumor (start here)", [
      curve("Soft tissue", [[-20, 0, [150, 150, 170]], [30, 0.02, [150, 150, 170]], [60, 0.0, [150, 150, 170]]], true),
      curve("Enhancing lesion", [[70, 0, [40, 220, 60]], [95, 0.2, [40, 220, 60]], [140, 0.25, [120, 255, 120]], [170, 0.0, [120, 255, 120]]], true),
      curve("Vessels", [[180, 0, RED], [230, 0.35, RED_HI], [380, 0.4, RED_HI], [430, 0.0, RED_HI]]),
      curve("Bone", [[430, 0, IVORY], [650, 0.5, BONE_HI], [3000, 0.9, BONE_HI]]),
    ]),
    makeTF("CT - Lung", [
      curve("Lung", [[-950, 0, [120, 180, 255]], [-850, 0.04, [150, 200, 255]], [-600, 0.08, [200, 220, 255]], [-450, 0.0, [230, 230, 255]]]),
      curve("Bone", [[250, 0, IVORY], [500, 0.4, IVORY], [3000, 0.85, BONE_HI]]),
    ]),
  ];
}

/** MR / modalities without HU: one curve over the upper part of the data range. */
export function genericPreset(lo: number, hi: number): TransferFunction {
  const r = hi - lo;
  return makeTF("Generic (MR / other)", [curve("Curve 1", [[lo + 0.15 * r, 0, [200, 120, 90]], [lo + 0.35 * r, 0.25, [240, 190, 140]],
    [lo + 0.7 * r, 0.7, [255, 250, 230]], [hi, 0.9, [255, 255, 255]]])]);
}

/** Opacity and colour of a curve at xs (np.interp: colour clamped at the ends, opacity 0 outside). */
export function sampleCurve(c: TFCurve, xs: Float64Array): { a: Float64Array; rgb: Float64Array } {
  const n = xs.length;
  const a = new Float64Array(n), rgb = new Float64Array(n * 3);
  const p = c.points;
  let j = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    if (x <= p[0].x) {
      a[i] = x === p[0].x ? p[0].y : 0;
      rgb.set(p[0].color, i * 3);
      continue;
    }
    if (x >= p[p.length - 1].x) {
      a[i] = x === p[p.length - 1].x ? p[p.length - 1].y : 0;
      rgb.set(p[p.length - 1].color, i * 3);
      continue;
    }
    while (j > 0 && p[j].x > x) j--;
    while (j < p.length - 2 && p[j + 1].x < x) j++;
    const p0 = p[j], p1 = p[j + 1];
    const t = p1.x > p0.x ? (x - p0.x) / (p1.x - p0.x) : 0;
    a[i] = p0.y + (p1.y - p0.y) * t;
    for (let k = 0; k < 3; k++) rgb[i * 3 + k] = p0.color[k] + (p1.color[k] - p0.color[k]) * t;
  }
  return { a, rgb };
}

interface Sample {
  a: Float64Array;
  rgb: Float64Array;
  hide: boolean;
}

export function curveSamples(tf: TransferFunction, xs: Float64Array, layer: TFCurve["layer"][] | null): Sample[] {
  return tf.curves.filter((c) => c.points.length >= 2 && (layer === null || layer.includes(c.layer)))
    .map((c) => ({ ...sampleCurve(c, xs), hide: c.hide_edges }));
}

/** Merge curve samples: opacity = max, colour = opacity-weighted mean. fillBlack false = interpolate the colour
 *  across gaps (3D). keep: factor on hide_edges curves (a row of the 2D table). */
export function combine(samples: Sample[], n: number, fillBlack: boolean, keep: number | null = null): { a: Float64Array; rgb: Float64Array } {
  const alpha = new Float64Array(n), wsum = new Float64Array(n), rgb = new Float64Array(n * 3);
  for (const s of samples) {
    const f = s.hide && keep !== null ? keep : 1;
    for (let i = 0; i < n; i++) {
      const a = s.a[i] * f;
      if (a > alpha[i]) alpha[i] = a;
      wsum[i] += a;
      rgb[i * 3] += s.rgb[i * 3] * a;
      rgb[i * 3 + 1] += s.rgb[i * 3 + 1] * a;
      rgb[i * 3 + 2] += s.rgb[i * 3 + 2] * a;
    }
  }
  const idx: number[] = [];
  for (let i = 0; i < n; i++) {
    if (wsum[i] > 1e-9) {
      for (let k = 0; k < 3; k++) rgb[i * 3 + k] /= wsum[i];
      idx.push(i);
    }
  }
  if (!fillBlack && idx.length) {                     // np.interp over the gaps: nearest-left / right ends clamped
    let q = 0;
    for (let i = 0; i < n; i++) {
      if (wsum[i] > 1e-9) continue;
      while (q < idx.length - 1 && idx[q + 1] < i) q++;
      const i0 = idx[q], i1 = idx[Math.min(q + 1, idx.length - 1)];
      const t = i <= i0 ? 0 : i >= i1 ? 1 : (i - i0) / (i1 - i0);
      for (let k = 0; k < 3; k++) rgb[i * 3 + k] = rgb[i0 * 3 + k] + (rgb[i1 * 3 + k] - rgb[i0 * 3 + k]) * t;
    }
  } else if (fillBlack) {
    for (let i = 0; i < n; i++) if (wsum[i] <= 1e-9) rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = 0;
  }
  for (let i = 0; i < n * 3; i++) rgb[i] = Math.min(1, Math.max(0, rgb[i]));
  return { a: alpha, rgb };
}

export interface TableSpec {
  x0: number;               // value at column 0 (bone lock: -BONE_OFFSET - max)
  x1: number;               // value at the last column
  width: number;
  lo: number;               // lowest data value (domain start)
  bone: boolean;            // bone area stored mirrored as -offset - HU (bone curves only)
  boneOffset: number;
  /** Veins stored as veinOffset + HU (vein curves only). */
  vein?: boolean;
  veinOffset?: number;
  gain: number;
}

/** Whether these curves use the artery / vein separation. */
export function usesVeins(tf: TransferFunction): boolean {
  return tf.curves.some((c) => c.layer === "vein" && c.points.length >= 2);
}

/**
 * RGBA table for the renderer (Float32, row-major width × rows). 1 row when no curve hides edges, else `rows`
 * rows over the gradient axis 0..RIM_GRAD_MAX: hide_edges curves fade out from 0.6·T to 1.4·T (smoothstep), like
 * VolumeView._tf2d. Column value v → normal domain when v ≥ lo, bone domain (HU = -offset - v) when the bone lock is on.
 */
export function buildTable(tf: TransferFunction, s: TableSpec, rows = 64): { data: Float32Array; rows: number } {
  const twoD = tf.curves.some((c) => c.hide_edges && c.points.length >= 2);
  const H = twoD ? rows : 1;
  const W = s.width;
  const xs = new Float64Array(W);
  for (let i = 0; i < W; i++) xs[i] = s.x0 + ((s.x1 - s.x0) * i) / (W - 1);
  const out = new Float32Array(W * H * 4);
  // gradient rows: keep factor per row, merged once per distinct factor
  const keepOf = (r: number) => {
    if (!twoD) return 1;
    const g = ((r + 0.5) / H) * RIM_GRAD_MAX, T = tf.edge_threshold;
    const t = Math.min(1, Math.max(0, (g - 0.6 * T) / (0.8 * T)));
    return Math.round((1 - t * t * (3 - 2 * t)) * 1000) / 1000;
  };
  // a layer without its own domain falls back on the normal one (dcv/volume_view.py _domains)
  const domains: { cols: number[]; hu: Float64Array; layer: TFCurve["layer"][] | null }[] = [];
  const vein = !!s.vein, off = s.veinOffset ?? 0;
  const normalCols: number[] = [], boneCols: number[] = [], veinCols: number[] = [];
  for (let i = 0; i < W; i++) {
    if (vein && xs[i] >= off) veinCols.push(i);
    else if (xs[i] >= s.lo) {
      if (!vein || xs[i] <= off - 1000) normalCols.push(i);
    } else if (s.bone && xs[i] <= -s.boneOffset - s.lo) boneCols.push(i);
  }
  const normalLayers: TFCurve["layer"][] | null = vein && s.bone ? ["normal"] : vein ? ["normal", "bone"] : s.bone ? ["normal", "vein"] : null;
  domains.push({ cols: normalCols, hu: Float64Array.from(normalCols, (i) => xs[i]), layer: normalLayers });
  if (vein) domains.push({ cols: veinCols, hu: Float64Array.from(veinCols, (i) => xs[i] - off), layer: ["vein"] });
  if (s.bone) domains.push({ cols: boneCols, hu: Float64Array.from(boneCols, (i) => -s.boneOffset - xs[i]), layer: ["bone"] });
  for (const d of domains) {
    const m = d.cols.length;
    if (!m) continue;
    const samples = curveSamples(tf, d.hu, d.layer);
    if (!samples.length) continue;
    const base = combine(samples, m, false);         // colour where every curve has faded out
    const cache = new Map<number, { a: Float64Array; rgb: Float64Array }>();
    for (let r = 0; r < H; r++) {
      const k = keepOf(r);
      let c = cache.get(k);
      if (!c) {
        c = twoD ? combine(samples, m, true, k) : base;
        if (twoD) for (let i = 0; i < m; i++) if (c.a[i] <= 1e-9) for (let q = 0; q < 3; q++) c.rgb[i * 3 + q] = base.rgb[i * 3 + q];
        cache.set(k, c);
      }
      for (let i = 0; i < m; i++) {
        const o = (r * W + d.cols[i]) * 4;
        out[o] = c.rgb[i * 3];
        out[o + 1] = c.rgb[i * 3 + 1];
        out[o + 2] = c.rgb[i * 3 + 2];
        out[o + 3] = Math.min(1, Math.max(0, c.a[i] * s.gain));
      }
    }
  }
  return { data: out, rows: H };
}
