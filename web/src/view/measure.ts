/** Measurement labels, same rules as dcv/measure.py: mm below 1 cm, cm from 1 cm up. */

/** "8.4 mm" below 1 cm, "3.27 cm" from 1 cm up. */
export function fmtLen(mm: number): string {
  return Math.abs(round1(mm)) < 10 ? `${mm.toFixed(1)} mm` : `${(mm / 10).toFixed(2)} cm`;
}

/** Two sizes in one unit: "8.4 × 6.1 mm" or "4.52 × 3.10 cm". */
export function fmtDims(a: number, b: number): string {
  if (round1(Math.max(Math.abs(a), Math.abs(b))) < 10) return `${a.toFixed(1)} × ${b.toFixed(1)} mm`;
  return `${(a / 10).toFixed(2)} × ${(b / 10).toFixed(2)} cm`;
}

/** Area: mm² below 1 cm², cm² from 1 cm² up. */
export function fmtArea(mm2: number): string {
  return round1(mm2) < 100 ? `${mm2.toFixed(1)} mm²` : `${(mm2 / 100).toFixed(2)} cm²`;
}

// Python's round(x, 1) on the value the label shows (half-to-even is irrelevant at this precision)
function round1(v: number): number {
  return Math.round(v * 10) / 10;
}

export interface Length {
  a: [number, number];    // (row, column), continuous image coordinates
  b: [number, number];
}

/** Length in mm (spacing = [row, column] mm per pixel). */
export function lengthMm(m: Length, spacing: [number, number]): number {
  return Math.hypot((m.b[0] - m.a[0]) * spacing[0], (m.b[1] - m.a[1]) * spacing[1]);
}

export type MeasureKind = "length" | "ellipse" | "angle" | "cobb";

/** Points each tool needs: a line, the ellipse's two corners, angle = arm, vertex, arm; Cobb = two lines. */
export const REQUIRED: Record<MeasureKind, number> = { length: 2, ellipse: 2, angle: 3, cobb: 4 };

export interface Measurement {
  kind: MeasureKind;
  pts: [number, number][];       // (row, column), continuous image coordinates (pixel i spans i..i + 1)
  stats?: { key: string; value: RoiStats | null };
}

export interface RoiStats {
  n: number;
  mean: number;
  sd: number;
  min: number;
  max: number;
}

/** Angle between two (row, column) vectors in degrees, 0..180. */
export function angleDeg(v1: [number, number], v2: [number, number]): number {
  const a = ((Math.atan2(v1[0], v1[1]) - Math.atan2(v2[0], v2[1])) * 180) / Math.PI;
  return Math.abs(((((a + 180) % 360) + 360) % 360) - 180);
}

/** The angle at pts[1], in mm (sr, sc = mm per row, column). */
export function angleOf(P: [number, number][], sr: number, sc: number): number {
  return angleDeg([(P[0][0] - P[1][0]) * sr, (P[0][1] - P[1][1]) * sc], [(P[2][0] - P[1][0]) * sr, (P[2][1] - P[1][1]) * sc]);
}

/** Cobb angle: between line 0-1 and line 2-3, the acute one. */
export function cobbOf(P: [number, number][], sr: number, sc: number): number {
  const a = angleDeg([(P[1][0] - P[0][0]) * sr, (P[1][1] - P[0][1]) * sc], [(P[3][0] - P[2][0]) * sr, (P[3][1] - P[2][1]) * sc]);
  return Math.min(a, 180 - a);
}

/** Area of the ellipse in the box of the two corners, mm². */
export function ellipseAreaMm2(P: [number, number][], sr: number, sc: number): number {
  return (Math.PI * Math.abs(P[1][0] - P[0][0]) * sr * Math.abs(P[1][1] - P[0][1]) * sc) / 4;
}

/** Values of the pixels whose centre lies inside the ellipse. */
export function ellipseStats(plane: ArrayLike<number>, rows: number, cols: number, P: [number, number][]): RoiStats | null {
  const r0 = Math.min(P[0][0], P[1][0]), r1 = Math.max(P[0][0], P[1][0]);
  const c0 = Math.min(P[0][1], P[1][1]), c1 = Math.max(P[0][1], P[1][1]);
  const ra = (r1 - r0) / 2, ca = (c1 - c0) / 2, rm = (r0 + r1) / 2, cm = (c0 + c1) / 2;
  if (ra <= 0 || ca <= 0) return null;
  let n = 0, sum = 0, sq = 0, lo = Infinity, hi = -Infinity;
  for (let r = Math.max(0, Math.floor(r0)); r <= Math.min(rows - 1, Math.floor(r1)); r++) {
    const dr = (r + 0.5 - rm) / ra;
    const q = 1 - dr * dr;
    if (q < 0) continue;
    const half = ca * Math.sqrt(q);
    const a = Math.max(0, Math.ceil(cm - half - 0.5)), b = Math.min(cols - 1, Math.floor(cm + half - 0.5));
    for (let c = a, i = r * cols + a; c <= b; c++, i++) {
      const v = plane[i];
      n++;
      sum += v;
      sq += v * v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!n) return null;
  const mean = sum / n;
  return { n, mean, sd: Math.sqrt(Math.max(0, sq / n - mean * mean)), min: lo, max: hi };
}

/**
 * Caption lines, the same as dcv/measure.py. `sr`, `sc` = mm per row, column; `known` = false when the file has no
 * pixel spacing (lengths in pixels, no area). `stats` = the ellipse's values, null when they are not available.
 */
export function measureLines(m: Measurement, sr: number, sc: number, known: boolean, unit: string, stats: RoiStats | null): string[] {
  const P = m.pts;
  if (P.length < REQUIRED[m.kind]) return [];
  switch (m.kind) {
    case "length":
      return [known ? fmtLen(lengthMm({ a: P[0], b: P[1] }, [sr, sc])) : `${Math.hypot(P[1][0] - P[0][0], P[1][1] - P[0][1]).toFixed(1)} px`];
    case "angle":
      return [`${angleOf(P, sr, sc).toFixed(1)}°`];
    case "cobb":
      return [`Cobb: ${cobbOf(P, sr, sc).toFixed(1)}°`];
    case "ellipse": {
      const out: string[] = [];
      if (known) {
        out.push(`Area: ${fmtArea(ellipseAreaMm2(P, sr, sc))}`);
        out.push(`Ø ${fmtDims(Math.abs(P[1][1] - P[0][1]) * sc, Math.abs(P[1][0] - P[0][0]) * sr)}`);
      }
      if (stats) {
        out.push(`Mean: ${stats.mean.toFixed(1)}${unit}  SD: ${stats.sd.toFixed(1)}`);
        out.push(`Min: ${stats.min.toFixed(0)}  Max: ${stats.max.toFixed(0)}`);
      }
      return out;
    }
  }
}
