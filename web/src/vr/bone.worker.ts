/**
 * Bone worker: the 3D bone area is made here, so 3D opens at once and turns while it is made (a 0.625 mm head CT: seconds
 * on a phone, the page frozen until now). The grid values (and the plain scan's on the same grid) come transferred; the
 * bits go back transferred.
 */
import { boneBits, maskOnGrid, plainBoneSlabs, type BoneSource } from "./bone";
import { thinPlates } from "./venography";

export interface BoneRequest {
  id: number;
  values: Int16Array | Float32Array;
  /** The plain scan on the same grid, for source "plain". */
  plain: Int16Array | Float32Array | null;
  dims: [number, number, number];
  spacing: [number, number, number];
  hu: number;
  source: BoneSource;
  /** In-plane step of the grid of values. */
  f: number;
  /** For source "plain": the bone rule of the series without it, for the planes it does not line up with, and whether it
   *  lined up as a whole (else slab by slab, bone.ts slabPlain). */
  rule: BoneSource;
  whole: boolean;
  /** For source "thin": the volume pooled on venography.ts cavityGrid (mean and top, its size, steps and spacing). */
  pooled: { mean: Float32Array; top: Float32Array; d: [number, number, number]; f: number; fz: number; sp: [number, number, number] } | null;
}

const ctx = self as unknown as {
  onmessage: (e: MessageEvent<BoneRequest>) => void;
  postMessage: (m: unknown, transfer?: Transferable[]) => void;
};

ctx.onmessage = (e: MessageEvent<BoneRequest>) => {
  const { id, values, plain, dims, spacing, hu, source, f, pooled, rule, whole } = e.data;
  try {
    if (source === "plain" && plain) {
      const { bits, enhanced, plainPlanes } = plainBoneSlabs(values, plain, dims, spacing, hu, rule, whole, null);
      ctx.postMessage({ id, bits, enhanced, plainPlanes }, enhanced ? [bits.buffer, enhanced.buffer] : [bits.buffer]);
      return;
    }
    // a newborn's skull: its plates by shape, found here too (seconds on a phone)
    const plates = source === "thin" && pooled
      ? maskOnGrid(thinPlates(pooled.mean, pooled.top, pooled.d, pooled.sp), pooled.d, pooled.f, pooled.fz, dims, f) : null;
    const bits = boneBits(values, dims, spacing, hu, 3, 1.5, source === "cta", plates);
    ctx.postMessage({ id, bits }, [bits.buffer]);
  } catch (err) {
    ctx.postMessage({ id, error: String((err as Error)?.message ?? err) });
  }
};
