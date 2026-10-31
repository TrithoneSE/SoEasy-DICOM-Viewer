/** DICOM header fields the viewer needs, read with dicom-parser (same meaning as dcv/loader.py). */
import * as dicomParser from "dicom-parser";
import type { DataSet } from "dicom-parser";

export interface DsaHint {
  mask: number[] | null;        // 0-based frames of the scanner's mask (Mask Subtraction Sequence, AVG_SUB)
  shift: [number, number];      // Mask Sub-pixel Shift: (row, column) offset of the mask
  log: boolean;                 // Pixel Intensity Relationship LIN: subtract logarithms
  recommended: boolean;         // Recommended Viewing Mode = SUB
}

export interface Header {
  transferSyntax: string;
  sopClass: string;
  sopInstance: string;
  seriesUid: string;
  studyUid: string;
  /** Frame of Reference UID: series that share it share one patient coordinate system. */
  frameOfRef: string;
  modality: string;
  patientName: string;
  patientId: string;
  birthDate: string;
  studyDate: string;
  studyTime: string;
  studyDesc: string;
  seriesDesc: string;
  seriesNumber: number;
  instanceNumber: number;
  imageType: string[];
  rows: number;
  cols: number;
  frames: number;
  samples: number;
  photometric: string;
  bitsAllocated: number;
  bitsStored: number;
  pixelRep: number;
  planarConfig: number;
  slope: number;
  intercept: number;
  ww: number | null;
  wl: number | null;
  /** (row, column) spacing in mm; `imager` = Imager Pixel Spacing (XA / RF: at the detector, not corrected). */
  spacing: [number, number] | null;
  spacingSource: "pixel" | "imager" | null;
  sliceThickness: number | null;
  spacingBetweenSlices: number | null;
  iop: number[] | null;
  ipp: number[] | null;
  frameTimeMs: number | null;
  dsa: DsaHint;
}

export class TruncatedHeader extends Error {}

const T = {
  ts: "x00020010", sopClass: "x00080016", sopInstance: "x00080018", imageType: "x00080008", studyDate: "x00080020",
  studyTime: "x00080030", modality: "x00080060", studyDesc: "x00081030", seriesDesc: "x0008103e",
  patientName: "x00100010", patientId: "x00100020", birthDate: "x00100030", studyUid: "x0020000d",
  seriesUid: "x0020000e", frameOfRef: "x00200052", seriesNumber: "x00200011", instanceNumber: "x00200013", ipp: "x00200032",
  iop: "x00200037", samples: "x00280002", photometric: "x00280004", frames: "x00280008", planar: "x00280006",
  rows: "x00280010", cols: "x00280011", pixelSpacing: "x00280030", imagerSpacing: "x00181164",
  bitsAllocated: "x00280100", bitsStored: "x00280101", pixelRep: "x00280103", wc: "x00281050", ww: "x00281051",
  intercept: "x00281052", slope: "x00281053", sliceThickness: "x00180050", spacingBetween: "x00180088",
  frameTime: "x00181063", frameTimeVector: "x00181065", cineRate: "x00180040", recFrameRate: "x00082144",
  pixelRelation: "x00281040", recViewing: "x00281090", maskSeq: "x00286100", maskOp: "x00286101",
  maskFrames: "x00286110", maskShift: "x00286114", shared: "x52009229", perFrame: "x52009230",
  pixelMeasures: "x00289110", valueTransform: "x00289145", frameVoi: "x00289132", planePos: "x00209113",
  planeOri: "x00209116", pixelData: "x7fe00010",
};
export const PIXEL_DATA = T.pixelData;

function str(ds: DataSet, tag: string): string {
  try {
    return (ds.string(tag) ?? "").trim();
  } catch {
    return "";
  }
}

function num(ds: DataSet | undefined, tag: string, index = 0): number | null {
  if (!ds || !ds.elements[tag]) return null;
  try {
    const v = ds.floatString(tag, index);
    return v === undefined || Number.isNaN(v) ? null : v;
  } catch {
    return null;
  }
}

function nums(ds: DataSet | undefined, tag: string): number[] | null {
  if (!ds || !ds.elements[tag]) return null;
  const n = ds.numStringValues(tag) ?? 0;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = num(ds, tag, i);
    if (v === null) return null;
    out.push(v);
  }
  return out.length ? out : null;
}

function us(ds: DataSet, tag: string, dflt: number): number {
  const v = ds.elements[tag] ? ds.uint16(tag) : undefined;
  return v === undefined ? dflt : v;
}

function item(ds: DataSet | undefined, tag: string, i = 0): DataSet | undefined {
  return ds?.elements[tag]?.items?.[i]?.dataSet;
}

/** Functional group macro (enhanced multiframe): shared groups, else the first frame's. */
function group(ds: DataSet, seqTag: string): DataSet | undefined {
  return item(item(ds, T.shared), seqTag) ?? item(item(ds, T.perFrame), seqTag);
}

function frameTime(ds: DataSet): number | null {
  const ft = num(ds, T.frameTime);
  if (ft && ft > 0) return ft;
  const vec = nums(ds, T.frameTimeVector);
  if (vec && vec.length > 1) {
    const v = vec.slice(1).filter((x) => x > 0);           // the first entry is 0 by definition
    if (v.length) return v.reduce((a, b) => a + b, 0) / v.length;
  }
  for (const tag of [T.cineRate, T.recFrameRate]) {
    const fps = num(ds, tag);
    if (fps && fps > 0) return 1000 / fps;
  }
  return null;
}

/** What the header says about subtraction (dcv/dsa.py hint_from_dataset). */
function dsaHint(ds: DataSet, frames: number): DsaHint {
  let mask: number[] | null = null;
  let shift: [number, number] = [0, 0];
  for (const it of ds.elements[T.maskSeq]?.items ?? []) {
    const m = it.dataSet;
    if (!m || str(m, T.maskOp).toUpperCase() !== "AVG_SUB") continue;     // TID is not supported
    const el = m.elements[T.maskFrames];
    if (el) {
      const f: number[] = [];
      for (let i = 0; i < el.length / 2; i++) {
        const v = m.uint16(T.maskFrames, i);
        if (v !== undefined && v - 1 >= 0 && v - 1 < frames) f.push(v - 1);   // 1-based in DICOM
      }
      if (f.length) mask = f;
    }
    if (m.elements[T.maskShift] && m.elements[T.maskShift].length >= 8) {
      shift = [m.float(T.maskShift, 0) ?? 0, m.float(T.maskShift, 1) ?? 0];
    }
    break;
  }
  return {
    mask,
    shift,
    log: str(ds, T.pixelRelation).toUpperCase() === "LIN",
    recommended: str(ds, T.recViewing).toUpperCase() === "SUB",
  };
}

/** Parse a DICOM file (or its first bytes when `headerOnly`). Throws TruncatedHeader when the prefix was too short. */
export function parse(buf: ArrayBuffer, headerOnly: boolean): DataSet {
  const bytes = new Uint8Array(buf);
  const opts: dicomParser.ParseDicomOptions = headerOnly ? { untilTag: T.pixelData } : {};
  if (!readsDicm(bytes)) {
    // no "DICM" preamble: a raw implicit VR little endian data set (old exports) starts with group 0008
    if (!(bytes.length > 8 && bytes[0] === 0x08 && bytes[1] === 0x00)) throw new NotDicom();
    opts.TransferSyntaxUID = "1.2.840.10008.1.2";
  }
  try {
    return dicomParser.parseDicom(bytes, opts);
  } catch (e) {
    if (headerOnly) throw new TruncatedHeader(String((e as { exception?: string }).exception ?? e));
    // a file cut short after its pixel data still shows (dicom-parser hands back the partial data set)
    const partial = (e as { dataSet?: DataSet }).dataSet;
    if (partial?.elements[T.pixelData]) return partial;
    throw e;
  }
}

export class NotDicom extends Error {}

function readsDicm(b: Uint8Array): boolean {
  return b.length > 132 && b[128] === 0x44 && b[129] === 0x49 && b[130] === 0x43 && b[131] === 0x4d;
}

export function hasImage(ds: DataSet): boolean {
  return !!ds.elements[T.rows] && !!ds.elements[T.cols];
}

export function readHeader(ds: DataSet): Header {
  const frames = Math.max(1, Math.round(num(ds, T.frames) ?? 1));
  const pm = group(ds, T.pixelMeasures);
  const vt = group(ds, T.valueTransform);
  const voi = group(ds, T.frameVoi);
  let spacing: [number, number] | null = null;
  let spacingSource: Header["spacingSource"] = null;
  const ps = nums(pm, T.pixelSpacing) ?? nums(ds, T.pixelSpacing);
  const ips = nums(ds, T.imagerSpacing);
  if (ps && ps.length >= 2 && ps[0] > 0 && ps[1] > 0) {
    spacing = [ps[0], ps[1]];
    spacingSource = "pixel";
  } else if (ips && ips.length >= 2 && ips[0] > 0 && ips[1] > 0) {
    spacing = [ips[0], ips[1]];
    spacingSource = "imager";
  }
  const pos = item(item(ds, T.perFrame), T.planePos);
  const ori = group(ds, T.planeOri);
  const iop = nums(ds, T.iop) ?? nums(ori, T.iop);
  const ipp = nums(ds, T.ipp) ?? nums(pos, T.ipp);
  return {
    transferSyntax: str(ds, T.ts) || "1.2.840.10008.1.2",
    sopClass: str(ds, T.sopClass),
    sopInstance: str(ds, T.sopInstance),
    seriesUid: str(ds, T.seriesUid),
    studyUid: str(ds, T.studyUid),
    frameOfRef: str(ds, T.frameOfRef),
    modality: str(ds, T.modality),
    patientName: str(ds, T.patientName).replace(/\^+/g, " ").trim(),
    patientId: str(ds, T.patientId),
    birthDate: str(ds, T.birthDate),
    studyDate: str(ds, T.studyDate),
    studyTime: str(ds, T.studyTime),
    studyDesc: str(ds, T.studyDesc),
    seriesDesc: str(ds, T.seriesDesc),
    seriesNumber: Math.round(num(ds, T.seriesNumber) ?? 0),
    instanceNumber: Math.round(num(ds, T.instanceNumber) ?? 0),
    imageType: str(ds, T.imageType).split("\\").map((s) => s.trim().toUpperCase()),
    rows: us(ds, T.rows, 0),
    cols: us(ds, T.cols, 0),
    frames,
    samples: us(ds, T.samples, 1),
    photometric: str(ds, T.photometric).toUpperCase() || "MONOCHROME2",
    bitsAllocated: us(ds, T.bitsAllocated, 16),
    bitsStored: us(ds, T.bitsStored, us(ds, T.bitsAllocated, 16)),
    pixelRep: us(ds, T.pixelRep, 0),
    planarConfig: us(ds, T.planar, 0),
    slope: num(vt, T.slope) ?? num(ds, T.slope) ?? 1,
    intercept: num(vt, T.intercept) ?? num(ds, T.intercept) ?? 0,
    ww: num(voi, T.ww) ?? num(ds, T.ww),
    wl: num(voi, T.wc) ?? num(ds, T.wc),
    spacing,
    spacingSource,
    sliceThickness: num(pm, T.sliceThickness) ?? num(ds, T.sliceThickness),
    spacingBetweenSlices: num(pm, T.spacingBetween) ?? num(ds, T.spacingBetween),
    iop: iop && iop.length === 6 ? iop : null,
    ipp: ipp && ipp.length === 3 ? ipp : null,
    frameTimeMs: frames > 1 ? frameTime(ds) : null,
    dsa: dsaHint(ds, frames),
  };
}

/** ' A' / ' B' for the two planes of a biplane run, ' (ref)' for a reference image (Siemens Image Type). */
export function planeSuffix(h: Header): string {
  if (h.imageType.some((t) => t.includes("REFIMAGE"))) return " (ref)";
  for (const t of h.imageType) {
    for (const p of ["A", "B"]) if (t === `BIPLANE ${p}` || t === `SINGLE ${p}`) return ` ${p}`;
  }
  return "";
}

/** Position of a slice along its normal (mm); falls back to the instance number. */
export function sliceKey(h: Header): number {
  if (h.iop && h.ipp) {
    const n = normal(h.iop);
    return n[0] * h.ipp[0] + n[1] * h.ipp[1] + n[2] * h.ipp[2];
  }
  return h.instanceNumber;
}

export function normal(iop: number[]): [number, number, number] {
  const [a, b, c, d, e, f] = iop;
  return [b * f - c * e, c * d - a * f, a * e - b * d];
}
