# SoEasy DICOM Viewer — web (PWA)

A web version of the desktop viewer that installs on iPad / iPhone from Safari ("Add to Home Screen") and runs
offline. Phase 1: **2D viewing, cine and DSA**. Phase 2: **MPR** (oblique, slab). Phase 3: **3D volume rendering**
with the colour curves of `dcv/tf_model.py`, the bone lock and the rim threshold. DTI is unlikely to fit
in a browser tab.

> Personal visualisation aid — **not a certified diagnostic device** (shown on the start screen and in About).

## Privacy

All DICOM processing runs in the browser. There is no server code, no upload and no analytics:

- the built `index.html` carries a Content Security Policy with `connect-src 'self'` — the page cannot send data to
  any other origin, and the only same-origin requests are for the app's own files;
- the service worker caches **only the app's files** (`dist/sw.js` lists them); opened images are never stored —
  they live in memory while the tab is open and are gone when it closes;
- files are read where they are (a `File` from the picker, or entries inside a `.zip`, read lazily).

## Features

| | |
|---|---|
| Open | Files, a folder (desktop browsers), `.zip` archives (read in place: a 1.3 GB CD image is not loaded whole), drag & drop on desktop |
| Series | Grouped by patient → study → series like `dcv/loader.py`; a multiframe file (XA run, US loop) is one entry each (biplane A / B share a Series UID); stacks keep the majority orientation, sorted along the normal; thumbnails from the middle slice / one frame in the stored window |
| Decoding | Native (LE / BE), RLE, JPEG lossless (process 14 / SV1), JPEG baseline, JPEG-LS, JPEG 2000, in Web Workers; compressed runs open after frame 1 and the mask frames, the rest decodes in the background (a frame on screen jumps the queue) |
| 2D | W/L drag (same sensitivity as the desktop), presets 1–9 of the desktop, default / auto window, invert; pinch zoom + two-finger pan, double tap = fit; mouse: wheel = frames, Ctrl+wheel / trackpad pinch = zoom, middle = pan, right = zoom; scroll tool and frame slider |
| Layout | 1 × 1, 1 × 2, 1 × 3, 2 × 2 (side by side in landscape, stacked in portrait): tap a view, then a series to show it there; empty views are filled with series of the same study, planes not yet shown first (MRI: axial + sagittal + coronal). **Reference lines**: in each view, where the slice of every other view cuts it (Image Position / Orientation of each slice, same Frame of Reference only), clipped to the other image, in that view's colour (solid for the active view). **Locate** (3D cursor, C, or Shift + press with any tool): the point touched is found in the other series, which go to the slice through it and mark it with a cyan cross; Esc hides it. Series in the other views stay in memory while MPR / 3D is open when they are small together (¼ of the budget), else they reload on the way back |
| Collage | Port of `dcv/collage.py` (toolbar, Ctrl+G): the active series on one image of 5 × 5, 4 × 4, 4 × 3, 3 × 3, 3 × 2, 6 × 5 or up to 12 × 12 tiles, one slice every 5 mm by default, the run centred in the series; the Position slider moves it and Centre puts it back. Slices already that far apart are all kept; a run's frames are spread over the image. Tiles use the view's window, invert and DSA; optional header, image number and position. Tile width 256–512 px, narrowed when the page would pass 16 million pixels (a larger canvas stays blank on iOS). Save PNG / JPEG (share sheet on iOS), Copy (https only) |
| Seed | Port of `dcv/segment.py` `region_grow` / `auto_grow` (same voxels as the desktop on the sample CT): Seed tool or Ctrl + click in 2D / MPR; automatic range from around the seed, anti-leak, ±70 mm block. Shown in 2D, MPR, collage and as a solid surface in 3D, with the volume in ml. A histogram of the voxels around the seed has two handles: dragging one grows the segment again from the same seed, live. 3D shows up to 4 segments. Not ported: the desktop's "smooth noise first" (its smoothed block would cost ~90 MB on a phone), the brush and STL |
| Measure | Length, ellipse ROI (area, diameters, mean, SD, min, max; HU on CT), angle, Cobb angle, with the desktop's captions. Angle and Cobb: a drag, then a second drag or tap; tapping point by point works too |
| 3D measure | Length between two points of the visible surface (L): one ray marched on the CPU with the current curves, crop, cuts and bone lock, ~0.03 ms per pick on a PC, no extra graphics memory (a ~130 kB opacity table while measuring). The points are 3D, so the line follows every turn |
| Length | mm below 1 cm, cm above (`measure.fmt_len`); end points draggable, Delete / trash removes |
| Cine | Per series, frame rate from Frame Time / Frame Time Vector / Cine Rate (desktop rules), play on opening a run, ± fps, Space |
| DSA | Port of `dcv/dsa.py`: mask from the Mask Subtraction Sequence (AVG_SUB, sub-pixel shift), else frame 2; LOG data subtracted directly, LIN log-transformed; window = 0.6 × the run's window with "no change" at 64 % grey; displayed image as mask, reset, manual pixel-shift (drag, on-screen pad 1 / 0.1 px, Shift / Ctrl+Shift + arrows), auto pixel-shift (tap in the yellow square, Ctrl+M) |

| MPR | Port of `dcv/mpr_window.py`: sagittal / coronal / axial through one centre, radiological orientation letters; drag the line crossing = move the centre, drag a line = shift that plane, drag a circle = tilt the axes (oblique), wheel / scroll tool = move along the normal; slab MIP / MinIP / average 5–40 mm; one W/L for the three views, all 2D tools (zoom, pan, pinch, length) per view; three views side by side (landscape) or stacked (portrait), or one view alone. Planes are resampled trilinearly in JavaScript (~18 ms for an 829² plane on the PC), every other pixel while dragging |
| 3D | WebGL2 ray casting (own shader, see below): colour curves with the desktop's built-in presets and JSON format (export / import between the apps, own presets kept in the browser), Curve editor (drag points / curves, double tap = point / new curve / colour, square-root opacity axis, histogram, merged colour strip), opacity gain, **bone lock** (port of `segment.py` `bone_and_vessels`: of the voxels ≥ 500 HU, bright vessels and aneurysms are told from bone by density and shape; bone area closed 3 mm, rim 1.5 mm, draggable threshold line; bone voxels mirror-encoded), **arteries / veins** (preset "CT Angio - Arteries / Veins + Bone", or a curve named Vein… / Vena…: arteries = bright voxels and what connects to them, automatic or chosen core; veins in their own value domain. On a late or venous phase both are equally bright, so they cannot be separated), **rims hidden** curves through a value × gradient table with the log rim slider 5–200 HU, lighting (ambient 0.15 / diffuse 0.9 / specular 0.3 / 15), MIP / MinIP, six standard views, touch: one finger turns, two fingers zoom / pan / twist, double tap = anterior view; lighter rendering while turning, full when still; quality Fast / Normal / High; half resolution (¼ of the graphics memory) |
| Crop box | Like the desktop's box widget: a wireframe with six coloured face handles; drag a handle (finger or mouse) to move that face along its axis; on / off and reset; Shift+C. **Hide box** (B) hides the box and its handles and keeps the crop |
| Scalpel | Draw a lasso: everything whose projection falls inside is removed through the whole depth (option: keep the inside, cut the outside), then the tool returns to turning, like the desktop. Removed voxels are a bitset over the GPU grid (~10 MB for 300 × 512²); only the slices that changed are uploaded again. Undo (10 levels, each stored as the deflated change; Ctrl+Z), reset all cuts; X |
| Snapshots | Like the desktop's 3D snapshots (`dcv/snapshots.py`): camera, crop box, colour curves, opacity and mode with a thumbnail, in a strip under the view; tap = back to that view, × = delete, PgUp / PgDn, Insert = new. Kept in memory only (thumbnails are patient images) and dropped when the series is closed. PNG = save the still 3D view as an image |

Keys follow the desktop app: `/` DSA, `?` mask, `Ctrl+?` reset, `Shift+M` pixel-shift, `Ctrl+M` auto shift,
W / M / Z / S / L tools, A auto window, I invert.

## Why not cornerstone3D

The brief suggested cornerstone3D + its DICOM image loader. Phase 1 uses its codecs (`@cornerstonejs/codec-*`,
the same WASM builds) but not its viewport / tool layer:

- **DSA** works on whole runs: per-frame float maths, a mask image that changes with every drag, an auto-shift
  search over the raw frames. In cornerstone that is a custom image loader plus cache invalidation for every shift;
  here it is a typed-array function (`src/dsa.ts`, a line-by-line port of `dcv/dsa.py`).
- **iPad memory** is the hard limit. One typed array per series (Int16 when the values fit, like the desktop),
  frames read from the file one at a time, nothing cached twice — the numbers below are measured, not estimated.
- **Size and offline**: the whole app is ~1 MB (most of it the three WASM codecs) and precached by the service
  worker.
- **MPR and 3D** need what the desktop does with VTK, not what cornerstone's tools offer: the oblique MPR with
  the desktop's axis handles, and a value × gradient transfer function with the mirrored bone encoding. vtk.js's volume
  mapper has no 2D (value × gradient) table, so phase 3 uses its own WebGL2 ray caster (`src/vr/renderer.ts`,
  ~150 lines of shader): one half-float 3D texture, a table texture, opacity corrected per mm like VTK's unit
  distance. The MPR resampler is plain JavaScript over the same Int16 volume.

Plain Vite + TypeScript, no framework. Runtime dependencies: `dicom-parser`, `jpeg-lossless-decoder-js` and the
three cornerstone codec packages (all MIT / ISC).

## Develop

```bash
cd web
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests (Node): parser, RLE, zip, series grouping, storage, DSA, labels
npm run build      # type-check + static site in web/dist
npm run preview    # serve web/dist (service worker active)
```

Test data (synthetic, no patient data): `.venv/Scripts/python.exe web/tools/make_test_data.py <folder>` writes a
300-slice CT, a 121-frame 960² XA run, RLE / JPEG 2000 runs with a Mask Subtraction Sequence, a biplane pair and a
zip of `samples/head_ct`. In dev, `DCV_TESTDATA=<folder>[;<folder or file>…] npm run dev` serves them on localhost
and `http://localhost:5173/?testdata=0/xa_rle` opens one without the file picker (`window.__dcv` exposes the
viewer for automated checks). Neither exists in the production build. Icons: `web/tools/make_icons.py` (from
`assets/icon.png`).

## Host it

`web/dist` is a static site; any HTTPS host works (HTTPS is required for the service worker and for installing).
Paths are relative (`base: "./"`), so it works in a sub-folder such as `https://<user>.github.io/<repo>/`.

**GitHub Pages** (repository → Settings → Pages → Source: GitHub Actions), workflow
`.github/workflows/pages.yml`:

```yaml
name: web
on: { push: { branches: [main], paths: ["web/**"] }, workflow_dispatch: {} }
permissions: { contents: read, pages: write, id-token: write }
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: github-pages
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 24 }
      - run: npm ci && npm run build
        working-directory: web
      - uses: actions/upload-pages-artifact@v3
        with: { path: web/dist }
      - uses: actions/deploy-pages@v4
```

Pages from a private repository needs a paid GitHub plan; the site itself contains only the app code. Netlify /
Cloudflare Pages work the same way (build `npm run build` in `web`, publish `web/dist`), or drag `web/dist` into
Netlify Drop. Never put DICOM files in `web/public` — they would be published.

Updating: after a new deploy, the installed app picks it up on the next start and shows it from the start after
that (the service worker updates in the background).

## Install on iPad / iPhone

1. Open the site in **Safari** (iOS 16.4 or newer).
2. Share → **Add to Home Screen** → Add.
3. Start it once while online; from then on it opens offline.

Opening images: Open → *Files or .zip archive…* → the Files picker (iCloud Drive, On My iPad, USB-C drive, AirDrop
downloads). Select several files, or a `.zip`. To open a folder, compress it first in the Files app (long-press →
Compress) and open the `.zip`.

## iOS limits

- **Memory per tab.** iOS does not publish a limit; WebKit's content process is killed (the page reloads) at
  roughly 1–1.5 GB on recent iPads, less on older ones and on iPhones. The app keeps decoded series under 700 MB on
  iOS (2 GB elsewhere), dropping the least recently viewed; one 300-slice CT (~160 MB) or one 121-frame 960² run
  (~225 MB) fits easily, several large runs at once do not.
- **Storage eviction.** Safari may delete a site's storage after 7 days without use; Home Screen apps are exempt,
  and storage can still be evicted when the device is short of space. Only the app code is stored, so the worst case
  is that the app needs one online start again. Opened images are never stored, so a study has to be reopened after
  the app was closed or reloaded by iOS.
- **No CD / DVD, no DICOMDIR navigation.** iPads have no optical drive and the browser cannot browse a drive by
  itself: copy the CD to a USB-C stick or to Files (or zip it on a computer), then pick the files / zip. DICOMDIR is
  not needed — the app finds the images by reading every file; viewer programs on the CD (`.exe`, `.dll`) are skipped.
- **Folder picking** may not be offered by iOS's picker; use a `.zip`.
- Large zips are copied by iOS into the browser's temporary space when picked (a 1.3 GB CD takes a while).
- No "Open with…" from other apps into the installed web app (iOS has no file handlers for web apps).

## Measured memory and speed

Headless Chrome on the development PC (Windows, desktop CPU), dev server, files from local disk;
`performance.measureUserAgentSpecificMemory()` (JS heap incl. typed arrays and canvases) and the tab's renderer
process private memory. An iPad is slower; memory should be similar.

| Data | Decoded pixels | JS memory (UASM) | Renderer process | First image | All frames |
|---|---|---|---|---|---|
| CT 300 × 512², uncompressed (synthetic) | 157 MB (Int16) | 166 MB | 202 MB | 0.14 s | 0.64 s |
| XA 121 × 960², uncompressed (synthetic), DSA on + auto shift + 3 s cine | 223 MB (Int16) | 246 MB | 292 MB | 0.02 s | 0.58 s |
| XA 121 × 960², JPEG lossless, real Siemens CD in a 1.3 GB zip (deflated) | 223 MB (+61 MB of the first run opened) | 384 MB with both runs | — | 0.44 s | 4.8 s |
| CT 300 × 512² in **3D** (bone lock on) | 157 MB + 98 MB GPU texture (content box) | 174 MB; 180 MB after two cuts | — | 3D open 2.7 s (bone area + upload) | — |

- Empty app: ~6 MB JS memory.
- Before reading multiframe files frame by frame, the uncompressed run peaked at 511 MB in the renderer (the 213 MB
  file was read whole next to its decoded copy); now only one frame's bytes are read at a time. Deflated zip entries
  still have to be inflated whole while the run decodes, then that buffer is dropped.
- Real CD: 28 runs found in 0.53 s; every frame of three runs (121 + 104 + 2) is identical to pydicom /
  pylibjpeg. That check found that `jpeg-lossless-decoder-js` left the last pixel of one frame at 0 (129 in
  pydicom): it stops when the final code runs into the EOI marker, where libjpeg reads zero bits;
  `pixels.zeroPadBeforeEOI` gives it those bits.
- Auto pixel-shift on a 960² frame: ~0.25 s (synthetic run, found the known movement 1.5 / −2.3 px exactly) to
  0.47 s.

## 3D on iPad: what to expect

- The volume goes to the GPU as half floats, 2 bytes per voxel, but only its **content box**: air around the head /
  body (CT below -600 HU, 3-voxel margin) is left out of the texture and of the rays. The synthetic 300 × 512² CT
  needs 98 MB instead of 157 MB (375 × 437 × 300 voxels); the decoded Int16 copy in JavaScript is another 157 MB. iPads share that RAM between CPU and GPU.
- **Quality** (Auto / Full / Half, in the 3D panel, remembered in this browser). Half = every other voxel in-plane, ¼ of
  the graphics memory, a softer image. Auto stays full unless the texture would pass what the device is trusted with:
  350 MB on an iPhone, 600 MB on an iPad (iOS does not tell the memory, so this is an estimate that older 2–3 GB
  devices may not meet), by `navigator.deviceMemory` elsewhere.
- **When memory runs out.** The GPU refuses the texture: the app shows half resolution and says so. iOS takes the
  graphics context away: 3D closes with a message, the images stay, and Auto uses half when 3D is opened again. iOS
  reloads the whole page: the study has to be opened again, and Auto uses half for that visit. Full always tries full.
- Scalpel on a 300 × 512² CT: ~0.5 s per cut, ~0.3 s per undo on the PC (desktop: 0.2–1 s). Undo keeps only what
  each cut changed, deflated: ~60–110 kB per level instead of ~10 MB (10 levels).
- Bone area: 1 bit per voxel on the half grid (2.5 MB instead of 20 MB).
- Bone lock and arteries / veins on a real 512 × 512 × 569 CTA, PC with a GTX 1650: on the 37 M voxel grid that iPad uses, 3D
  opens in ~8 s and arteries / veins take ~5 s, renderer peak ~0.9–1.1 GB plus a 283 MB texture; on the full grid the masks
  equal the desktop's voxel for voxel (~23 s and ~8 s, ~2 GB). The grid is full up to 20 M voxels on phones and tablets,
  100 M elsewhere. An iPad will be slower, and an older one (2–3 GB) may not hold this study at full Quality.
- Seed: a search block of 1 byte per voxel while its range is being edited (±70 mm: ~23 MB on a 0.45 × 0.6 mm CT),
  released on Done; the mask itself takes 1 byte per voxel of its box, once more on the GPU. Growing takes about 40–220 ms
  for 0.5 M voxels on a PC (it follows the segment's size, not the block's).
- Smoothness: while turning, the share of pixels rendered adapts to the frame time (frames every ~25–45 ms),
  rays take double steps and a cheaper gradient (3 fetches instead of 6); a still frame is at most 2.2 M pixels on
  phones and tablets (3.2 M elsewhere): one frame at rest may take its time. Opening MPR or 3D drops the other decoded series from memory (they reload when chosen), and on
  iOS decoded series are kept under 450 MB.
- Bone lock: the bone area is computed on the main thread (~1–2 s on the PC for 300 × 512², more on an iPad; on a
  grid halved in-plane above 20 M voxels). Dragging the threshold line recomputes it on release.
- Leaving 3D (or opening another series) frees the texture. iOS can still drop the WebGL context under memory
  pressure; then reopen 3D.
- Synthetic phantoms with perfectly sharp bone edges show contour "terraces" of the voxel grid on shaded surfaces;
  real CT edges are smoother.

## Not done yet

DTI; snapshot movies (GIF fly-through) and snapshot files (.se3d) (desktop only so far); other measurements (rectangle, polygon, deviation, text…); segmentation brush and STL; 12-bit JPEG (process 2/4, `.4.51`), deflated
DICOM, HTJ2K, palette colour, native YBR 4:2:2; per-frame functional groups beyond the first frame (enhanced
CT / MR multiframe use the shared values); PACS; the DSA stage 2 / 3 tools the desktop does not have either.
Tested in desktop Chrome (headless and Electron); not yet on a real iPad — the touch handling uses Pointer Events
(Safari 13+), checked with emulated touch.
