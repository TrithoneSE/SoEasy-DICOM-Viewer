/** A file the viewer can read: a picked File or an entry inside a .zip (read lazily, never all at once). */
export interface Source {
  /** Path for display and sorting (folder/name.dcm or archive.zip/inner/path). */
  name: string;
  size: number;
  /** Bytes [start, end) — the whole file when no range is given. */
  read(start?: number, end?: number): Promise<ArrayBuffer>;
  /** Cheap reads anywhere (a File, a stored zip entry); false for deflated zip entries (inflated from the start). */
  random: boolean;
}

export function fileSource(f: File): Source {
  const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath;
  return {
    name: rel || f.name,
    size: f.size,
    random: true,
    read: (start = 0, end = f.size) => f.slice(start, Math.min(end, f.size)).arrayBuffer(),
  };
}

/** Prefix of a source, at least `n` bytes when the file is that long (for header-only parsing). */
export function readPrefix(src: Source, n: number): Promise<ArrayBuffer> {
  return src.read(0, Math.min(n, src.size));
}
