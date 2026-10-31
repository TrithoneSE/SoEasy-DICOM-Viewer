/** Decode worker: compressed frames are decoded here so the viewer stays smooth (JPEG lossless ~30–80 ms a frame). */
import charlsWasm from "@cornerstonejs/codec-charls/decodewasm?url";
import openjpegWasm from "@cornerstonejs/codec-openjpeg/decodewasm?url";
import jpegWasm from "@cornerstonejs/codec-libjpeg-turbo-8bit/decodewasm?url";
import { decodeFrame, type Codecs, type FrameInfo } from "./pixels";

const once = <T>(f: () => Promise<T>) => {
  let p: Promise<T> | null = null;
  return () => (p ??= f());
};
const quiet = () => undefined;            // OpenJPEG logs three [INFO] lines per frame otherwise
const at = (url: string) => ({ locateFile: (f: string) => (f.endsWith(".wasm") ? url : f), print: quiet, printErr: quiet });

const codecs: Codecs = {
  jpegLS: once(async () => (await import("@cornerstonejs/codec-charls/decodewasmjs")).default(at(charlsWasm))),
  j2k: once(async () => (await import("@cornerstonejs/codec-openjpeg/decodewasmjs")).default(at(openjpegWasm))),
  jpeg8: once(async () => (await import("@cornerstonejs/codec-libjpeg-turbo-8bit/decodewasmjs")).default(at(jpegWasm))),
};

export interface DecodeRequest {
  id: number;
  bytes: Uint8Array;
  info: FrameInfo;
}

const ctx = self as unknown as {
  onmessage: (e: MessageEvent<DecodeRequest>) => void;
  postMessage: (m: unknown, transfer?: Transferable[]) => void;
};

ctx.onmessage = async (e: MessageEvent<DecodeRequest>) => {
  const { id, bytes, info } = e.data;
  try {
    const d = await decodeFrame(bytes, info, codecs);
    const out = d.gray ?? d.rgb!;
    ctx.postMessage({ id, gray: d.gray, rgb: d.rgb }, [out.buffer]);
  } catch (err) {
    ctx.postMessage({ id, error: String((err as Error)?.message ?? err) });
  }
};
