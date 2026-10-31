/**
 * GPU volume rendering (WebGL2 ray casting) — what VTK's GPU ray cast mapper does in the desktop 3D window:
 * front-to-back compositing with the opacity corrected per 1 mm of ray (VTK's scalar opacity unit distance), a
 * colour / opacity table over value × gradient (the 2D transfer function that hides partial-volume rims), Phong
 * shading with a head light (ambient / diffuse / specular), and MIP / MinIP.
 *
 * The volume is one half-float 3D texture (2 bytes per voxel, uploaded a few slices at a time, so there is never a
 * second full copy in memory); it can be halved in-plane (¼ of the memory) for large volumes.
 */
import type { Volume } from "../dicom/volume";
import { BONE_OFFSET, CT_FLOOR, encodeVessel, VEIN_OFFSET, type BoneRegion, type VesselMasks } from "./bone";

const VS = `#version 300 es
out vec2 vNdc;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vNdc = p * 2.0 - 1.0;
  gl_Position = vec4(vNdc, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vNdc;
out vec4 frag;
uniform sampler3D uVol;
uniform sampler2D uTF;
uniform mat4 uInvVP;          // clip -> mm
uniform vec3 uFull;           // whole volume in mm (camera, crop box)
uniform vec3 uOrigin;         // the texture covers [uOrigin, uOrigin + uSize] (the content box, no air around)
uniform vec3 uSize;
uniform bool uFastGrad;       // while turning: one-sided differences (3 fetches instead of 6)
uniform vec3 uTexel;          // one voxel in texture coordinates
uniform vec3 uSpacing;        // mm per voxel of the texture
uniform float uStep;          // mm between samples
uniform vec2 uTFRange;        // value at the table's first / last column
uniform bool uTF2D;
uniform float uGradMax;       // gradient axis of the table (HU per smallest voxel)
uniform float uGradScale;     // smallest spacing (mm)
uniform int uBlend;           // 0 composite, 1 MIP, 2 MinIP
uniform bool uShade;
uniform vec4 uLight;          // ambient, diffuse, specular, power
uniform vec3 uBg;
uniform vec2 uWin;            // MIP / MinIP: value window (lo, hi)
uniform float uBoneOffset;
uniform vec3 uClipLo;         // crop box, fractions of the whole volume
uniform vec3 uClipHi;
uniform int uSegN;            // segments: opaque surfaces of their masks (each its own small texture)
uniform sampler3D uSeg0;
uniform sampler3D uSeg1;
uniform sampler3D uSeg2;
uniform sampler3D uSeg3;
uniform vec3 uSegLo[4];
uniform vec3 uSegSize[4];
uniform vec3 uSegTexel[4];
uniform vec3 uSegCol[4];

float decode(float v) { return v < -2048.0 ? -uBoneOffset - v : v > 10000.0 ? v - ${VEIN_OFFSET.toFixed(1)} : v; }

vec3 gradient(vec3 tc) {
  float dx = texture(uVol, tc + vec3(uTexel.x, 0, 0)).r - texture(uVol, tc - vec3(uTexel.x, 0, 0)).r;
  float dy = texture(uVol, tc + vec3(0, uTexel.y, 0)).r - texture(uVol, tc - vec3(0, uTexel.y, 0)).r;
  float dz = texture(uVol, tc + vec3(0, 0, uTexel.z)).r - texture(uVol, tc - vec3(0, 0, uTexel.z)).r;
  return vec3(dx, dy, dz) / (2.0 * uSpacing);          // value per mm
}

vec3 gradientFast(vec3 tc, float v) {
  return vec3(texture(uVol, tc + vec3(uTexel.x, 0, 0)).r - v, texture(uVol, tc + vec3(0, uTexel.y, 0)).r - v,
              texture(uVol, tc + vec3(0, 0, uTexel.z)).r - v) / uSpacing;
}

vec4 segShade(sampler3D s, int i, vec3 q, vec3 rd) {
  vec3 e = uSegTexel[i];
  vec3 g = vec3(texture(s, q + vec3(e.x, 0, 0)).r - texture(s, q - vec3(e.x, 0, 0)).r,
                texture(s, q + vec3(0, e.y, 0)).r - texture(s, q - vec3(0, e.y, 0)).r,
                texture(s, q + vec3(0, 0, e.z)).r - texture(s, q - vec3(0, 0, e.z)).r) / (uSegSize[i] * e);
  float nl = dot(g, g) > 1e-12 ? abs(dot(normalize(g), rd)) : 1.0;
  return vec4(uSegCol[i] * (uLight.x + uLight.y * nl) + uLight.z * pow(nl, uLight.w), 1.0);
}

#define SEG(i, S) if (uSegN > i) { vec3 q = (p - uSegLo[i]) / uSegSize[i]; \
  if (q.x >= 0.0 && q.y >= 0.0 && q.z >= 0.0 && q.x <= 1.0 && q.y <= 1.0 && q.z <= 1.0 && texture(S, q).r > 0.5) return segShade(S, i, q, rd); }

vec4 segHit(vec3 p, vec3 rd) {
  SEG(0, uSeg0)
  SEG(1, uSeg1)
  SEG(2, uSeg2)
  SEG(3, uSeg3)
  return vec4(0.0);
}

void main() {
  vec4 n4 = uInvVP * vec4(vNdc, -1.0, 1.0);
  vec4 f4 = uInvVP * vec4(vNdc, 1.0, 1.0);
  vec3 ro = n4.xyz / n4.w;
  vec3 rd = normalize(f4.xyz / f4.w - ro);
  vec3 bLo = max(uClipLo * uFull, uOrigin), bHi = min(uClipHi * uFull, uOrigin + uSize);
  vec3 inv = 1.0 / rd;
  vec3 t0 = (bLo - ro) * inv, t1 = (bHi - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  float tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  float tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tf <= tn) { frag = vec4(uBg, 1.0); return; }
  // jitter the first sample per pixel: no wood-grain rings
  float jit = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  float t = tn + jit * uStep;
  vec4 acc = vec4(0.0);
  float mx = -1e9, mn = 1e9;
  float span = uTFRange.y - uTFRange.x;
  for (int i = 0; i < 8192; i++) {
    if (t > tf) break;
    vec3 tc = (ro + rd * t - uOrigin) / uSize;
    if (uSegN > 0 && uBlend == 0) {
      vec4 sh = segHit(ro + rd * t, rd);
      if (sh.a > 0.5) {
        acc.rgb += (1.0 - acc.a) * sh.rgb;
        acc.a = 1.0;
        break;
      }
    }
    float v = texture(uVol, tc).r;
    if (uBlend == 0) {
      float x = (v - uTFRange.x) / span;
      vec4 c = texture(uTF, vec2(x, 0.0));
      if (c.a > 0.0005) {
        vec3 g = uFastGrad ? gradientFast(tc, v) : gradient(tc);
        float gm = length(g) * uGradScale;
        if (uTF2D) c = texture(uTF, vec2(x, clamp(gm / uGradMax, 0.0, 1.0)));
        if (c.a > 0.0005) {
          vec3 col = c.rgb;
          if (uShade) {
            float lit = uLight.x + uLight.y;
            float spec = 0.0;
            if (gm > 1e-3) {
              vec3 nrm = -normalize(g);
              float nl = abs(dot(nrm, -rd));            // head light, two-sided
              lit = uLight.x + uLight.y * nl;
              spec = uLight.z * pow(nl, uLight.w);
            }
            col = c.rgb * lit + spec;
          }
          float a = 1.0 - pow(1.0 - min(c.a, 0.9999), uStep);   // opacity per mm of ray
          acc.rgb += (1.0 - acc.a) * a * col;
          acc.a += (1.0 - acc.a) * a;
          if (acc.a > 0.985) break;
        }
      }
    } else {
      float hu = decode(v);
      mx = max(mx, hu);
      mn = min(mn, hu);
    }
    t += uStep;
  }
  if (uBlend == 0) {
    frag = vec4(acc.rgb + (1.0 - acc.a) * uBg, 1.0);
  } else {
    float val = uBlend == 1 ? mx : mn;
    float gy = clamp((val - uWin.x) / max(1.0, uWin.y - uWin.x), 0.0, 1.0);
    frag = vec4(vec3(gy), 1.0);
  }
}`;

export interface RenderParams {
  invVP: Float32Array;
  step: number;
  blend: 0 | 1 | 2;
  shade: boolean;
  light: [number, number, number, number];
  background: [number, number, number];
  window: [number, number];
  clipLo: [number, number, number];
  clipHi: [number, number, number];
  fastGradient: boolean;
}

/** Content box: voxel index ranges [x0, x1), [y0, y1), [z0, z1) of the full-resolution grid. */
export interface Box {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  z0: number;
  z1: number;
}

/**
 * The part of the volume that is not air (CT: above -600 HU; other data: above 2 % of the range), plus a margin.
 * Air around the head / body is a third or more of a CT: it is left out of the GPU texture (and of the rays).
 */
export function contentBox(vol: Volume, ct: boolean, margin = 3): Box {
  const [lo, hi] = vol.valueRange();
  const thr = ct ? -600 : lo + 0.02 * (hi - lo);
  const nx = vol.cols, ny = vol.rows, nz = vol.n, d = vol.data;
  let x0 = nx, x1 = -1, y0 = ny, y1 = -1, z0 = nz, z1 = -1;
  for (let z = 0; z < nz; z++) {
    let any = false;
    for (let y = 0; y < ny; y++) {
      const row = (z * ny + y) * nx;
      let a = -1, b = -1;
      for (let x = 0; x < nx; x++) if (d[row + x] > thr) { a = x; break; }
      if (a < 0) continue;
      for (let x = nx - 1; x >= a; x--) if (d[row + x] > thr) { b = x; break; }
      any = true;
      if (a < x0) x0 = a;
      if (b > x1) x1 = b;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    if (any) {
      if (z < z0) z0 = z;
      z1 = z;
    }
  }
  if (x1 < 0) return { x0: 0, x1: nx, y0: 0, y1: ny, z0: 0, z1: nz };
  return { x0: Math.max(0, x0 - margin), x1: Math.min(nx, x1 + 1 + margin), y0: Math.max(0, y0 - margin),
    y1: Math.min(ny, y1 + 1 + margin), z0: Math.max(0, z0 - margin), z1: Math.min(nz, z1 + 1 + margin) };
}

/** A segment for the renderer: its mask in full-resolution voxels (mm = voxel · spacing of the volume). */
export interface SegmentMask {
  id: number;
  rev: number;
  sub: Uint8Array;
  off: [number, number, number];      // (z, y, x)
  dims: [number, number, number];     // (nz, ny, nx)
  color: [number, number, number];
}

interface SegSlot {
  id: number;
  rev: number;
  smooth: boolean;
  tex: WebGLTexture;
  bytes: number;
  lo: [number, number, number];
  size: [number, number, number];
  texel: [number, number, number];
  color: [number, number, number];
}

export const MAX_SEGMENTS_3D = 4;
const SMOOTH_LIMIT = 8e6;               // voxels: larger boxes stay unsmoothed (they would cost too long)

/** The mask as a texture: padded so the surface closes at its box, smoothed once ([1 2 1] per axis) unless `fast`. */
export function maskTexture(m: SegmentMask, fast: boolean): { data: Uint8Array; nx: number; ny: number; nz: number; smooth: boolean } {
  const [d, h, w] = m.dims;
  const pad = 2;
  const nx = w + 2 * pad, ny = h + 2 * pad, nz = d + 2 * pad;
  let a = new Uint8Array(nx * ny * nz);
  for (let z = 0; z < d; z++) {
    for (let y = 0; y < h; y++) {
      const from = (z * h + y) * w, to = ((z + pad) * ny + y + pad) * nx + pad;
      for (let x = 0; x < w; x++) if (m.sub[from + x]) a[to + x] = 255;
    }
  }
  const smooth = !fast && a.length <= SMOOTH_LIMIT;
  if (!smooth) return { data: a, nx, ny, nz, smooth };
  let b = new Uint8Array(a.length);
  for (const [stride, n] of [[1, nx], [nx, ny], [nx * ny, nz]] as const) {
    const total = a.length;
    for (let i = 0; i < total; i++) {
      const v = a[i];
      const pos = Math.floor(i / stride) % n;
      const p = pos > 0 ? a[i - stride] : v, q = pos < n - 1 ? a[i + stride] : v;
      b[i] = (p + 2 * v + q + 2) >> 2;
    }
    [a, b] = [b, a];
  }
  return { data: a, nx, ny, nz, smooth };
}

export class VolumeRenderer {
  readonly gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private vol: WebGLTexture | null = null;
  private tf: WebGLTexture | null = null;
  private loc: Record<string, WebGLUniformLocation | null> = {};
  private vao: WebGLVertexArrayObject;
  dims: [number, number, number] = [1, 1, 1];
  spacing: [number, number, number] = [1, 1, 1];   // mm per texture voxel (x, y, z)
  /** The texture covers [origin, origin + size] mm of the whole volume (full = [0, full]). */
  size: [number, number, number] = [1, 1, 1];
  origin: [number, number, number] = [0, 0, 0];
  full: [number, number, number] = [1, 1, 1];
  /** First full-resolution voxel (x, y, z) of the texture, and the in-plane step. */
  start: [number, number, number] = [0, 0, 0];
  f: 1 | 2 = 1;
  gpuBytes = 0;
  /** Arteries / veins for the next fill (null = no vein layer). */
  vessels: VesselMasks | null = null;
  private segs: SegSlot[] = [];
  private segDummy: WebGLTexture | null = null;
  private tfRange: [number, number] = [0, 1];
  private tf2D = false;
  lost = false;
  /** The browser took the graphics context away (out of memory). */
  onLost: () => void = () => undefined;

  constructor(readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, depth: false, preserveDrawingBuffer: true, powerPreference: "high-performance" });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.lost = true;
      this.onLost();
    });
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
    this.prog = p;
    for (const n of ["uVol", "uTF", "uInvVP", "uSize", "uFull", "uOrigin", "uFastGrad", "uTexel", "uSpacing", "uStep", "uTFRange", "uTF2D", "uGradMax", "uGradScale",
      "uBlend", "uShade", "uLight", "uBg", "uWin", "uBoneOffset", "uClipLo", "uClipHi", "uSegN", "uSeg0", "uSeg1", "uSeg2", "uSeg3", "uSegLo", "uSegSize", "uSegTexel", "uSegCol"]) this.loc[n] = gl.getUniformLocation(p, n);
    this.vao = gl.createVertexArray()!;
  }

  get maxTexture2D(): number {
    return this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE) as number;
  }

  get maxTexture3D(): number {
    return this.gl.getParameter(this.gl.MAX_3D_TEXTURE_SIZE) as number;
  }

  /**
   * Upload the volume (in-plane step `f` = 1 or 2), CT floored at -1024, the bone area mirror-encoded and scalpel
   * cuts (`removed`, a bitset over the texture grid) set to air. Returns false when the GPU could not hold it (the
   * caller retries at half resolution).
   */
  upload(vol: Volume, f: 1 | 2, bone: BoneRegion | null, ct: boolean, removed: Uint32Array | null = null,
    box: Box | null = null): boolean {
    const gl = this.gl;
    this.release();
    const b = box ?? { x0: 0, x1: vol.cols, y0: 0, y1: vol.rows, z0: 0, z1: vol.n };
    const nx = Math.ceil((b.x1 - b.x0) / f), ny = Math.ceil((b.y1 - b.y0) / f), nz = b.z1 - b.z0;
    const [sz, sy, sx] = vol.spacing;
    this.f = f;
    this.start = [b.x0, b.y0, b.z0];
    this.dims = [nx, ny, nz];
    this.spacing = [sx * f, sy * f, sz];
    this.size = [nx * sx * f, ny * sy * f, nz * sz];
    this.origin = [b.x0 * sx, b.y0 * sy, b.z0 * sz];
    this.full = [vol.cols * sx, vol.rows * sy, vol.n * sz];
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_3D, tex);
    while (gl.getError() !== gl.NO_ERROR) { /* clear */ }
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R16F, nx, ny, nz);
    if (gl.getError() !== gl.NO_ERROR) {
      gl.deleteTexture(tex);
      return false;
    }
    this.vol = tex;
    this.fill(vol, bone, ct, removed, 0, nz);
    if (gl.getError() !== gl.NO_ERROR) {
      this.release();
      return false;
    }
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    this.gpuBytes = nx * ny * nz * 2;
    return true;
  }

  /** Write slices [z0, z1) of the texture again (after a cut or an undo), a few slices at a time. */
  fill(vol: Volume, bone: BoneRegion | null, ct: boolean, removed: Uint32Array | null, z0: number, z1: number): void {
    const gl = this.gl;
    if (!this.vol) return;
    const [nx, ny] = this.dims;
    const f = this.f;
    const [sx0, sy0, sz0] = this.start;
    const air = ct ? CT_FLOOR : vol.valueRange()[0];
    gl.bindTexture(gl.TEXTURE_3D, this.vol);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    const chunk = Math.max(1, Math.floor(1e6 / (nx * ny)));       // ~4 MB of Float32 staging at a time
    const buf = new Float32Array(nx * ny * Math.min(chunk, z1 - z0));
    const d = vol.data;
    for (let c0 = z0; c0 < z1; c0 += chunk) {
      const c1 = Math.min(z1, c0 + chunk);
      let o = 0;
      for (let z = c0; z < c1; z++) {
        for (let y = 0; y < ny; y++) {
          const Z = sz0 + z, Y = sy0 + y * f;
          const row = (Z * vol.rows + Y) * vol.cols + sx0;
          const by = bone ? (Z * bone.ny + Math.floor(Y / bone.f)) * bone.nx : 0;
          const vr = this.vessels ? (Z * this.vessels.arteries.ny + Math.floor(Y / this.vessels.arteries.f)) * this.vessels.arteries.nx : 0;
          const ri = (z * ny + y) * nx;
          for (let x = 0; x < nx; x++) {
            const i = ri + x;
            if (removed && removed[i >>> 5] & (1 << (i & 31))) {
              buf[o++] = air;
              continue;
            }
            let v = d[row + x * f];
            if (ct && v < CT_FLOOR) v = CT_FLOOR;
            if (bone) {
              const bi = by + Math.floor((sx0 + x * f) / bone.f);
              if (bone.bits[bi >>> 5] & (1 << (bi & 31))) {
                buf[o++] = -BONE_OFFSET - v;
                continue;
              }
            }
            if (this.vessels) {
              const a = this.vessels.arteries;
              const vi = vr + Math.floor((sx0 + x * f) / a.f);
              v = encodeVessel(v, (a.bits[vi >>> 5] & (1 << (vi & 31))) !== 0, (this.vessels.veins.bits[vi >>> 5] & (1 << (vi & 31))) !== 0);
            }
            buf[o++] = v;
          }
        }
      }
      gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, c0, nx, ny, c1 - c0, gl.RED, gl.FLOAT, buf, 0);
    }
  }

  /** Bring the segment textures up to date (at most MAX_SEGMENTS_3D); `fast` = while a range is being dragged. */
  setSegments(list: SegmentMask[], spacing: [number, number, number], fast: boolean): void {
    const gl = this.gl;
    if (this.lost) return;
    const wanted = list.slice(0, MAX_SEGMENTS_3D);
    const next: SegSlot[] = [];
    for (const old of this.segs) {
      if (!wanted.some((m) => m.id === old.id)) gl.deleteTexture(old.tex);
    }
    const [sz, sy, sx] = spacing;
    for (const m of wanted) {
      let slot = this.segs.find((s) => s.id === m.id);
      if (slot && slot.rev === m.rev && (slot.smooth || fast || m.sub.length > SMOOTH_LIMIT / 2)) {
        next.push(slot);
        continue;
      }
      const t = maskTexture(m, fast);
      const tex = slot?.tex ?? gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, t.nx, t.ny, t.nz, 0, gl.RED, gl.UNSIGNED_BYTE, t.data);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
      const pad = 2;
      slot = {
        id: m.id, rev: m.rev, smooth: t.smooth, tex, bytes: t.data.length,
        lo: [(m.off[2] - pad) * sx, (m.off[1] - pad) * sy, (m.off[0] - pad) * sz],
        size: [t.nx * sx, t.ny * sy, t.nz * sz], texel: [1 / t.nx, 1 / t.ny, 1 / t.nz], color: [m.color[0] / 255, m.color[1] / 255, m.color[2] / 255],
      };
      next.push(slot);
    }
    this.segs = next;
  }

  /** Graphics memory of the segments. */
  get segBytes(): number {
    return this.segs.reduce((a, s) => a + s.bytes, 0);
  }

  /** The colour / opacity table (tf.ts buildTable): width × rows RGBA over values [x0, x1]. */
  setTable(data: Float32Array, width: number, rows: number, x0: number, x1: number): void {
    const gl = this.gl;
    if (!this.tf) {
      this.tf = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, this.tf);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tf);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, rows, 0, gl.RGBA, gl.FLOAT, data);
    this.tfRange = [x0, x1];
    this.tf2D = rows > 1;
  }

  render(p: RenderParams, gradMax: number): void {
    const gl = this.gl;
    if (this.lost || !this.vol || !this.tf) return;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.vol);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.tf);
    const L = this.loc;
    gl.uniform1i(L.uVol, 0);
    gl.uniform1i(L.uTF, 1);
    gl.uniformMatrix4fv(L.uInvVP, false, p.invVP);
    gl.uniform3fv(L.uSize, this.size);
    gl.uniform3fv(L.uFull, this.full);
    gl.uniform3fv(L.uOrigin, this.origin);
    gl.uniform1i(L.uFastGrad, p.fastGradient ? 1 : 0);
    gl.uniform3f(L.uTexel, 1 / this.dims[0], 1 / this.dims[1], 1 / this.dims[2]);
    gl.uniform3fv(L.uSpacing, this.spacing);
    gl.uniform1f(L.uStep, p.step);
    gl.uniform2fv(L.uTFRange, this.tfRange);
    gl.uniform1i(L.uTF2D, this.tf2D ? 1 : 0);
    gl.uniform1f(L.uGradMax, gradMax);
    gl.uniform1f(L.uGradScale, Math.min(...this.spacing));
    gl.uniform1i(L.uBlend, p.blend);
    gl.uniform1i(L.uShade, p.shade ? 1 : 0);
    gl.uniform4fv(L.uLight, p.light);
    gl.uniform3fv(L.uBg, p.background);
    gl.uniform2fv(L.uWin, p.window);
    gl.uniform1f(L.uBoneOffset, BONE_OFFSET);
    gl.uniform3fv(L.uClipLo, p.clipLo);
    gl.uniform3fv(L.uClipHi, p.clipHi);
    if (!this.segDummy) {                  // samplers that are not in use still need a texture
      this.segDummy = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_3D, this.segDummy);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, 1, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1));
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    }
    gl.uniform1i(L.uSegN, this.segs.length);
    const lo: number[] = [], size: number[] = [], texel: number[] = [], col: number[] = [];
    for (let i = 0; i < MAX_SEGMENTS_3D; i++) {
      const s = this.segs[i];
      gl.activeTexture(gl.TEXTURE2 + i);
      gl.bindTexture(gl.TEXTURE_3D, s ? s.tex : this.segDummy);
      gl.uniform1i(L[`uSeg${i}`], 2 + i);
      lo.push(...(s?.lo ?? [0, 0, 0]));
      size.push(...(s?.size ?? [1, 1, 1]));
      texel.push(...(s?.texel ?? [1, 1, 1]));
      col.push(...(s?.color ?? [0, 0, 0]));
    }
    gl.uniform3fv(L.uSegLo, lo);
    gl.uniform3fv(L.uSegSize, size);
    gl.uniform3fv(L.uSegTexel, texel);
    gl.uniform3fv(L.uSegCol, col);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  release(): void {
    if (this.vol) this.gl.deleteTexture(this.vol);
    this.vol = null;
    this.gpuBytes = 0;
  }

  releaseSegments(): void {
    for (const s of this.segs) this.gl.deleteTexture(s.tex);
    this.segs = [];
  }

  dispose(): void {
    this.release();
    this.releaseSegments();
    if (this.tf) this.gl.deleteTexture(this.tf);
    this.tf = null;
    this.gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}
