/** Finding the DICOM images among picked files / archives and grouping them into series (dcv/loader.py scan_folder). */
import { hasImage, NotDicom, parse, planeSuffix, readHeader, sliceKey, TruncatedHeader, type Header } from "./header";
import { fileSource, readPrefix, type Source } from "./source";
import { isZipName, openZip } from "./zip";

export interface Instance {
  src: Source;
  h: Header;
  key: number;          // position along the slice normal (mm), else the instance number
}

export interface Series {
  id: string;
  /** A multiframe file (XA run, US loop) is an entry of its own: biplane A / B share one Series Instance UID. */
  multiframe: boolean;
  instances: Instance[];
  description: string;
  thumb: HTMLCanvasElement | null;
}

export function seriesCount(s: Series): number {
  return s.multiframe ? s.instances[0].h.frames : s.instances.length;
}

export function firstHeader(s: Series): Header {
  return s.instances[0].h;
}

// files that are never DICOM images (viewer programs on a CD, reports): not worth opening
const SKIP = /\.(dll|exe|ini|inf|txt|htm|html|xml|js|css|jpg|jpeg|png|gif|bmp|pdf|ico|bat|cab|msi|chm|cfg|log|sys|ocx|dat|jar|class|manifest|xsl|ttf|mp4|avi|wmv|db|json)$/i;

export type Progress = (done: number, total: number, text: string) => void;

/** Picked files → readable sources: .zip archives are opened in place (their entries read on demand). */
export async function expand(files: File[], progress?: Progress): Promise<{ sources: Source[]; errors: string[] }> {
  const sources: Source[] = [];
  const errors: string[] = [];
  for (const f of files) {
    const s = fileSource(f);
    if (isZipName(f.name)) {
      progress?.(0, 0, `Opening ${f.name}…`);
      try {
        sources.push(...(await openZip(f, s.name)));
      } catch (e) {
        errors.push(String((e as Error).message ?? e));
      }
    } else {
      sources.push(s);
    }
  }
  return { sources, errors };
}

async function headerOf(src: Source): Promise<Header | null> {
  if (SKIP.test(src.name) || /(^|\/)DICOMDIR$/i.test(src.name) || src.size < 132) return null;
  let ds;
  try {
    ds = parse(await readPrefix(src, 64 * 1024), true);
  } catch (e) {
    if (e instanceof NotDicom) return null;
    if (!(e instanceof TruncatedHeader) || src.size <= 64 * 1024) return null;
    try {
      ds = parse(await src.read(), true);          // a long header (big private sequences): read it all
    } catch {
      return null;
    }
  }
  if (!hasImage(ds)) return null;
  return readHeader(ds);
}

/** Read the headers (8 at a time) and group the images into series, sorted like the desktop app. */
export async function scan(sources: Source[], progress?: Progress, cancelled = () => false): Promise<Series[]> {
  const series = new Map<string, Series>();
  let done = 0;
  let next = 0;
  const work = async () => {
    while (next < sources.length && !cancelled()) {
      const src = sources[next++];
      const h = await headerOf(src).catch(() => null);
      done++;
      if (done % 10 === 0 || done === sources.length) progress?.(done, sources.length, "Scanning files…");
      if (!h || !h.rows || !h.cols) continue;
      let id = h.seriesUid || `nouid-${src.name.split("/").slice(0, -1).join("/")}`;
      const multi = h.frames > 1;
      if (multi) id = `${id}#${h.sopInstance || src.name}`;    // one entry per run (biplane A / B)
      let s = series.get(id);
      if (!s) {
        s = { id, multiframe: multi, instances: [], description: h.seriesDesc + (multi ? planeSuffix(h) : ""), thumb: null };
        series.set(id, s);
      }
      s.instances.push({ src, h, key: sliceKey(h) });
    }
  };
  await Promise.all(Array.from({ length: 8 }, work));
  return [...series.values()].sort((a, b) => {
    const ha = firstHeader(a), hb = firstHeader(b);
    return ha.studyDate.localeCompare(hb.studyDate) || ha.seriesNumber - hb.seriesNumber
      || minInstance(a) - minInstance(b);
  });
}

function minInstance(s: Series): number {
  return Math.min(...s.instances.map((i) => i.h.instanceNumber));
}

/** Patient → study → series, for the series list's header rows. */
export interface StudyGroup {
  patient: string;
  patientId: string;
  birthDate: string;
  studyUid: string;
  studyDesc: string;
  studyDate: string;
  series: Series[];
}

export function groupByStudy(list: Series[]): StudyGroup[] {
  const out = new Map<string, StudyGroup>();
  for (const s of list) {
    const h = firstHeader(s);
    const key = `${h.patientName}|${h.patientId}|${h.studyUid}`;
    let g = out.get(key);
    if (!g) {
      g = { patient: h.patientName, patientId: h.patientId, birthDate: h.birthDate, studyUid: h.studyUid,
        studyDesc: h.studyDesc, studyDate: h.studyDate, series: [] };
      out.set(key, g);
    }
    g.series.push(s);
  }
  return [...out.values()];
}
