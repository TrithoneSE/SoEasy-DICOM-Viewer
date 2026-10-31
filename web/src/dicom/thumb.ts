/** Series thumbnails like dcv/loader.py: the spatially middle slice (one decoded frame of a run) in the
 *  window stored in the header, else the 1–99 % range. */
import { quantiles } from "../dsa";
import { openFrames } from "./frames";
import { decodeFrame, frameInfo, isNative, NO_CODECS, unsupportedReason } from "./pixels";
import { decodePool } from "./pool";
import type { Series } from "./scan";

export async function makeThumbnail(s: Series, size = 96): Promise<HTMLCanvasElement | null> {
  const inst = s.multiframe ? s.instances[0] : [...s.instances].sort((a, b) => a.key - b.key)[Math.floor(s.instances.length / 2)];
  if (unsupportedReason(inst.h)) return null;
  const reader = await openFrames(inst.src);         // one frame of a run, not the whole file
  const h = reader.h;
  const bytes = await reader.frame(s.multiframe ? Math.floor(h.frames / 2) : 0);
  const info = frameInfo(h);
  const d = isNative(h.transferSyntax)
    ? await decodeFrame(bytes, info, NO_CODECS)
    : await decodePool().decode(bytes, info);
  const { rows, cols } = h;
  const img = new ImageData(cols, rows);
  const px = new Uint32Array(img.data.buffer);
  if (d.rgb) {
    for (let i = 0; i < rows * cols; i++) px[i] = 0xff000000 | (d.rgb[i * 3 + 2] << 16) | (d.rgb[i * 3 + 1] << 8) | d.rgb[i * 3];
  } else if (d.gray) {
    const g = d.gray;
    const v = new Float32Array(g.length);
    for (let i = 0; i < g.length; i++) v[i] = g[i] * h.slope + h.intercept;     // the stored window is rescaled (HU)
    let lo: number, hi: number;
    if (h.ww && h.ww > 1 && h.wl !== null) [lo, hi] = [h.wl - h.ww / 2, h.wl + h.ww / 2];
    else {
      [lo, hi] = quantiles(v, 0.01, 0.99);
      if (hi <= lo) hi = lo + 1;
    }
    const inv = h.photometric === "MONOCHROME1";
    const sc = 255 / (hi - lo);
    for (let i = 0; i < v.length; i++) {
      let y = Math.max(0, Math.min(255, (v[i] - lo) * sc)) | 0;
      if (inv) y = 255 - y;
      px[i] = 0xff000000 | (y << 16) | (y << 8) | y;
    }
  } else return null;
  const full = document.createElement("canvas");
  full.width = cols;
  full.height = rows;
  full.getContext("2d")!.putImageData(img, 0, 0);
  const aspect = h.spacing ? h.spacing[0] / h.spacing[1] : 1;     // row / column spacing
  const w = cols, hh = rows * aspect;
  const sc = size / Math.max(w, hh);
  const out = document.createElement("canvas");
  out.width = Math.max(1, Math.round(w * sc));
  out.height = Math.max(1, Math.round(hh * sc));
  const ctx = out.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(full, 0, 0, out.width, out.height);
  return out;
}
