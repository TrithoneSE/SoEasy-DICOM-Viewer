// Emscripten builds of the codecs (CommonJS factories, no typings of their own).
declare module "@cornerstonejs/codec-charls/decodewasmjs" {
  const factory: (opts?: object) => Promise<any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  export default factory;
}
declare module "@cornerstonejs/codec-openjpeg/decodewasmjs" {
  const factory: (opts?: object) => Promise<any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  export default factory;
}
declare module "@cornerstonejs/codec-libjpeg-turbo-8bit/decodewasmjs" {
  const factory: (opts?: object) => Promise<any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  export default factory;
}
declare module "jpeg-lossless-decoder-js" {
  export class Decoder {
    decode(buffer: ArrayBuffer, offset: number, length: number, numBytes: number): Uint8Array | Uint16Array;
  }
}
