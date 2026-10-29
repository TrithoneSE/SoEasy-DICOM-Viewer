import time
import numpy as np

nz, ny, nx = 300, 512, 512
sz, sy, sx = 0.625, 0.45, 0.45
w, h = 1600, 1500
lasso = np.zeros((h, w), bool)
lasso[400:1000, 500:1100] = True
# a perspective-like matrix
M = np.array([[0.004, 0.0003, 0.0001, -0.6], [0.0002, 0.0035, 0.003, -0.5], [0, 0, 0.001, 0.2],
              [0.00001, 0.00002, 0.00003, 1.0]])


def old():
    X, Y = np.meshgrid(np.arange(nx) * sx, np.arange(ny) * sy)
    X, Y = X.ravel(), Y.ravel()
    inside = np.zeros((nz, ny, nx), bool)
    for k in range(nz):
        Z = k * sz
        xs = M[0, 0] * X + M[0, 1] * Y + M[0, 2] * Z + M[0, 3]
        ys = M[1, 0] * X + M[1, 1] * Y + M[1, 2] * Z + M[1, 3]
        ws = M[3, 0] * X + M[3, 1] * Y + M[3, 2] * Z + M[3, 3]
        px = ((xs / ws + 1) * 0.5 * w).astype(np.int32)
        py = ((ys / ws + 1) * 0.5 * h).astype(np.int32)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        res = np.zeros(X.shape, bool)
        res[ok] = lasso[h - 1 - py[ok], px[ok]]
        inside[k] = res.reshape(ny, nx)
    return inside


def new(f=1):
    """separable float32 terms, padded lookup image (no bounds masks), optional stride f in-plane."""
    xi = (np.arange(0, nx, f) * sx).astype(np.float32)
    yi = (np.arange(0, ny, f) * sy).astype(np.float32)
    lut = np.zeros((h + 2, w + 2), bool)
    lut[1:-1, 1:-1] = lasso[::-1]                 # row 0 = bottom (VTK display y)
    ax, bx = (M[0, 0] * xi)[None, :], (M[0, 1] * yi)[:, None]
    ay, by = (M[1, 0] * xi)[None, :], (M[1, 1] * yi)[:, None]
    aw, bw = (M[3, 0] * xi)[None, :], (M[3, 1] * yi)[:, None]
    out = np.empty((nz, len(yi), len(xi)), bool)
    sxw, syh = np.float32(0.5 * w), np.float32(0.5 * h)
    for k in range(nz):
        Z = k * sz
        inv = 1.0 / (aw + bw + np.float32(M[3, 2] * Z + M[3, 3]))
        px = (ax + bx + np.float32(M[0, 2] * Z + M[0, 3])) * inv
        px += 1
        px *= sxw
        py = (ay + by + np.float32(M[1, 2] * Z + M[1, 3])) * inv
        py += 1
        py *= syh
        np.clip(px, -1, w, out=px)
        np.clip(py, -1, h, out=py)
        out[k] = lut[(py + 1).astype(np.int32), (px + 1).astype(np.int32)]
    return out


t = time.perf_counter(); a = old(); t1 = time.perf_counter() - t
t = time.perf_counter(); b = new(); t2 = time.perf_counter() - t
print(f"old {t1:.2f}s  new {t2:.2f}s  agree {np.mean(a == b):.5f}  inside {a.mean():.3f}")
rem = np.zeros((nz, ny, nx), np.uint8)
t = time.perf_counter(); pk = np.packbits(rem); print(f"packbits {time.perf_counter() - t:.2f}s")
rd = np.zeros((nz, ny, nx), np.int16)
t = time.perf_counter(); rem[b] = 1; rd[b] = -1024; print(f"apply incremental {time.perf_counter() - t:.2f}s")
t = time.perf_counter(); rd2 = np.array(rd, copy=True); rd2[rem.astype(bool)] = -1024; print(f"full re-apply {time.perf_counter() - t:.2f}s")


def fast(lasso_img, nthreads=None):
    """bbox culling with a coarse grid + parallel slices."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    hh, ww = lasso_img.shape
    rows = np.where(lasso_img.any(1))[0]
    cols = np.where(lasso_img.any(0))[0]
    # bbox in VTK display coordinates (y up)
    bx0, bx1 = cols[0], cols[-1]
    by0, by1 = hh - 1 - rows[-1], hh - 1 - rows[0]
    lut = np.zeros((hh + 2, ww + 2), bool)
    lut[1:-1, 1:-1] = lasso_img[::-1]
    xi = (np.arange(nx) * sx).astype(np.float32)
    yi = (np.arange(ny) * sy).astype(np.float32)
    A = M.astype(np.float32)
    out = np.zeros((nz, ny, nx), bool)
    s = 8

    def proj(ys, xs, Z):
        X, Y = xi[xs][None, :], yi[ys][:, None]
        inv = 1.0 / (A[3, 0] * X + A[3, 1] * Y + np.float32(M[3, 2] * Z + M[3, 3]))
        px = (A[0, 0] * X + A[0, 1] * Y + np.float32(M[0, 2] * Z + M[0, 3])) * inv
        py = (A[1, 0] * X + A[1, 1] * Y + np.float32(M[1, 2] * Z + M[1, 3])) * inv
        return (px + 1) * np.float32(0.5 * ww), (py + 1) * np.float32(0.5 * hh)

    cy, cx = np.arange(0, ny, s), np.arange(0, nx, s)

    def work(k):
        Z = k * sz
        px, py = proj(cy, cx, Z)
        pad = 40   # screen px: covers the gap between coarse samples
        hit = (px >= bx0 - pad) & (px <= bx1 + pad) & (py >= by0 - pad) & (py <= by1 + pad)
        if not hit.any():
            return
        r = np.where(hit.any(1))[0]
        c = np.where(hit.any(0))[0]
        y0, y1 = max(0, cy[r[0]] - s), min(ny, cy[r[-1]] + s + 1)
        x0, x1 = max(0, cx[c[0]] - s), min(nx, cx[c[-1]] + s + 1)
        px, py = proj(np.arange(y0, y1), np.arange(x0, x1), Z)
        np.clip(px, -1, ww, out=px)
        np.clip(py, -1, hh, out=py)
        out[k, y0:y1, x0:x1] = lut[(py + 1).astype(np.int32), (px + 1).astype(np.int32)]

    with ThreadPoolExecutor(nthreads or os.cpu_count()) as ex:
        list(ex.map(work, range(nz)))
    return out


import os
print("cpus", os.cpu_count())
t = time.perf_counter(); c = fast(lasso); t3 = time.perf_counter() - t
print(f"fast (big area) {t3:.2f}s agree {np.mean(a == c):.6f}")
small = np.zeros((h, w), bool); small[700:800, 750:850] = True
lasso = small
t = time.perf_counter(); a2 = old(); t1 = time.perf_counter() - t
t = time.perf_counter(); c2 = fast(small); t3 = time.perf_counter() - t
print(f"small area: old {t1:.2f}s fast {t3:.2f}s agree {np.mean(a2 == c2):.6f} inside {a2.mean():.4f}")
