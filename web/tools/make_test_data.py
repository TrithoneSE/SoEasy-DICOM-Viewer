"""Synthetic test data for the web viewer (no patient data):
  ct300/      300-slice head-like CT, 512 x 512, int16, explicit VR LE
  xa121/      one XA run, 121 frames 960 x 960, 10-bit LOG, uncompressed (the memory check)
  xa_rle/     12-frame 256 x 256 XA run, RLE lossless, with a Mask Subtraction Sequence (frames 3+4)
  xa_j2k/     the same run as JPEG 2000 lossless (needs pylibjpeg-openjpeg)
  biplane/    two 40-frame RLE runs (planes A and B) sharing one Series Instance UID
  kepala.zip  ../samples/head_ct as a .zip (deflated)
  cta/        CTA-like head: artery and aneurysm above the bone threshold, a vein (bone lock, arteries / veins)
  mri3/       MRI-like head in three separate series (axial 5 mm, sagittal 6 mm, coronal 6 mm), one Frame of
              Reference: for the linked layout (reference lines, 3D cursor)
Run from the repo folder: .venv/Scripts/python.exe web/tools/make_test_data.py <output folder> [names…]"""
import os
import sys
import zipfile

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, RLELossless, generate_uid

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
from dcv.dsa import shift_image  # noqa: E402

out = sys.argv[1]
only = set(sys.argv[2:])        # e.g. "mri3": only these sets
MOVE = (1.5, -2.3)


def base(sop_class, modality, series_uid, desc):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.PatientName, ds.PatientID, ds.PatientBirthDate = modality, "TEST^SYNTHETIC", "SYN1", "19700101"
    ds.StudyInstanceUID, ds.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.1", series_uid
    ds.StudyDate, ds.StudyDescription, ds.SeriesDescription = "20261029", "Synthetic test study", desc
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    return ds


def ct300(folder, n=300, size=512):
    os.makedirs(folder, exist_ok=True)
    uid = generate_uid()
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    for k in range(n):
        z = (k - n / 2) / (n / 2)
        r = np.hypot((yy - size / 2) / (size * 0.42), (xx - size / 2) / (size * 0.36)) / max(0.2, np.sqrt(1 - z * z))
        img = np.full((size, size), -1000, np.float32)
        img[r < 1.0] = 1200                       # skull
        img[r < 0.92] = 35                        # brain
        img[np.hypot(yy - size * 0.45, xx - size * 0.6) < 18 + 10 * np.cos(k / 9)] = 60      # a lesion
        img += np.random.default_rng(k).normal(0, 6, img.shape)
        ds = base("1.2.840.10008.5.1.4.1.1.2", "CT", uid, "Head 300 slices")
        ds.SeriesNumber, ds.InstanceNumber = 2, k + 1
        ds.Rows = ds.Columns = size
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.WindowCenter, ds.WindowWidth = 40, 80
        ds.PixelSpacing = [0.45, 0.45]
        ds.SliceThickness = 0.6
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-115.0, -115.0, k * 0.6]
        ds.PixelData = (np.clip(img + 1024, -32768, 32767)).astype(np.int16).tobytes()
        ds.save_as(os.path.join(folder, f"CT{k:04d}.dcm"), enforce_file_format=True)


def xa(folder, name, n=12, size=256, mask_seq=False, ts=None, plane="A", series_uid=None):
    os.makedirs(folder, exist_ok=True)
    s = size / 256
    yy, xx = np.mgrid[:size, :size]
    bone = (500 + 250 * ((np.hypot(yy - 128 * s, xx - 128 * s) // (20 * s)) % 2)
            + 150 * (np.hypot(yy - 100 * s, xx - 150 * s) < 25 * s)).astype(np.float32)
    vessel = -200.0 * (np.abs(xx - 0.6 * yy - 50 * s) < 3 * s)
    moved = shift_image(bone, *MOVE)
    rng = np.random.default_rng(3)
    ds = base("1.2.840.10008.5.1.4.1.1.12.1", "XA", series_uid or generate_uid(), "DSA run")
    ds.SeriesNumber, ds.InstanceNumber = 1, 1 if plane == "A" else 2
    ds.Manufacturer = "Siemens"
    ds.ImageType = ["ORIGINAL", "PRIMARY", f"BIPLANE {plane}"]
    ds.Rows = ds.Columns = size
    ds.NumberOfFrames = n
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 10, 9, 0
    ds.ImagerPixelSpacing = [0.3 / s, 0.3 / s]
    ds.FrameTime = 66.7
    ds.PixelIntensityRelationship = "LOG"
    ds.WindowCenter, ds.WindowWidth = 512, 1024
    if mask_seq:
        it = Dataset()
        it.MaskOperation = "AVG_SUB"
        it.MaskFrameNumbers = [3, 4]
        it.MaskSubPixelShift = [0.0, 0.0]
        ds.MaskSubtractionSequence = [it]
        ds.RecommendedViewingMode = "SUB"
    px = np.empty((n, size, size), np.uint16)
    for k in range(n):
        f = (moved if k >= 2 else bone) + (vessel if k >= 4 else 0)
        px[k] = np.clip(f + rng.normal(0, 2, f.shape), 0, 1023)
    ds.PixelData = px.tobytes()
    if ts is not None:
        ds.compress(ts)
    ds.save_as(os.path.join(folder, name), enforce_file_format=True)


def head_value(p):
    """Synthetic T2-ish head at patient points p (..., 3) in mm: scalp, brain, ventricles, a bright lesion."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.sqrt((x / 70) ** 2 + (y / 90) ** 2 + (z / 75) ** 2)
    v = np.where(r < 1, 300.0, 0.0)
    v = np.where(r < 0.93, 600.0, v)
    v = np.where(np.sqrt(((np.abs(x) - 8) / 6) ** 2 + (y / 25) ** 2 + ((z - 10) / 12) ** 2) < 1, 1400.0, v)
    v = np.where(np.sqrt((x - 30) ** 2 + (y + 20) ** 2 + (z - 25) ** 2) < 12, 1800.0, v)
    return v


def mri3(folder, size=256, px=0.9):
    """Three series of one head with their own orientations (like a routine MRI: no full-volume series)."""
    os.makedirs(folder, exist_ok=True)
    frame_of_ref = generate_uid()
    plans = [  # description, row direction, column direction, slice normal, n, spacing
        ("T2 AX", (1, 0, 0), (0, 1, 0), (0, 0, 1), 24, 5.0),
        ("T2 SAG", (0, 1, 0), (0, 0, -1), (1, 0, 0), 20, 6.0),
        ("T2 COR", (1, 0, 0), (0, 0, -1), (0, 1, 0), 20, 6.0),
    ]
    ii, jj = np.mgrid[:size, :size].astype(np.float64)
    for sn, (desc, r, c, nrm, n, step) in enumerate(plans):
        r, c, nrm = np.array(r, float), np.array(c, float), np.array(nrm, float)
        uid = generate_uid()
        for k in range(n):
            o = -(size / 2) * px * (r + c) + (k - (n - 1) / 2) * step * nrm
            pts = o + jj[..., None] * px * r + ii[..., None] * px * c
            img = head_value(pts) + np.random.default_rng(sn * 100 + k).normal(0, 15, (size, size))
            ds = base("1.2.840.10008.5.1.4.1.1.4", "MR", uid, desc)
            ds.FrameOfReferenceUID = frame_of_ref
            ds.SeriesNumber, ds.InstanceNumber = sn + 1, k + 1
            ds.Rows = ds.Columns = size
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
            ds.PixelSpacing = [px, px]
            ds.SliceThickness = step
            ds.ImageOrientationPatient = [*map(float, r), *map(float, c)]
            ds.ImagePositionPatient = [float(v) for v in o]
            ds.PixelData = np.clip(img, 0, 4095).astype(np.uint16).tobytes()
            ds.save_as(os.path.join(folder, f"{desc.replace(' ', '_')}_{k:03d}.dcm"), enforce_file_format=True)


def cta(folder):
    """The desktop tests' synthetic CTA: brain, skull 1400 HU, an artery (600 HU) from the skull to an aneurysm (650 HU),
    a vein (150 HU), a thin 600 HU plate on the skull; 1 mm voxels."""
    os.makedirs(folder, exist_ok=True)
    z, y, x = np.mgrid[0:100, 0:120, 0:120].astype(np.float32)
    r = np.sqrt((z - 50) ** 2 + (y - 60) ** 2 + (x - 60) ** 2)
    d = np.full(z.shape, -1000, np.int16)
    d[r < 48] = 40
    d[(r >= 44) & (r < 48)] = 1400
    d[(np.hypot(z - 50, y - 60) < 2.5) & (x > 16) & (x <= 60)] = 600
    d[np.sqrt((z - 50) ** 2 + (y - 60) ** 2 + (x - 62) ** 2) < 5] = 650
    d[(np.hypot(z - 50, x - 90) < 2.5) & (y > 40) & (y < 80)] = 150
    plate = (np.abs(x - 62) < 1) & (y >= 12) & (y <= 18) & (np.abs(z - 50) < 6)
    d[plate & (r < 48) & (r >= 42)] = 600
    uid = generate_uid()
    for k in range(d.shape[0]):
        ds = base("1.2.840.10008.5.1.4.1.1.2", "CT", uid, "Synthetic CTA")
        ds.SeriesNumber, ds.InstanceNumber = 5, k + 1
        ds.Rows, ds.Columns = d.shape[1], d.shape[2]
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.WindowCenter, ds.WindowWidth = 40, 400
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(k)]
        ds.PixelData = d[k].tobytes()
        ds.save_as(os.path.join(folder, f"CTA{k:03d}.dcm"), enforce_file_format=True)


if only:
    if "cta" in only:
        cta(os.path.join(out, "cta"))
    if "mri3" in only:
        mri3(os.path.join(out, "mri3"))
    print("done", out)
    sys.exit(0)

ct300(os.path.join(out, "ct300"))
xa(os.path.join(out, "xa121"), "XA121.dcm", n=121, size=960)
xa(os.path.join(out, "xa_rle"), "XA_RLE.dcm", mask_seq=True, ts=RLELossless)
try:
    xa(os.path.join(out, "xa_j2k"), "XA_J2K.dcm", ts=JPEG2000Lossless)
except Exception as ex:  # noqa: BLE001
    print("no JPEG 2000 encoder:", ex)
uid = generate_uid()
xa(os.path.join(out, "biplane"), "A.dcm", n=40, ts=RLELossless, plane="A", series_uid=uid)
xa(os.path.join(out, "biplane"), "B.dcm", n=40, ts=RLELossless, plane="B", series_uid=uid)
src = os.path.join(here, "..", "..", "samples", "head_ct")
with zipfile.ZipFile(os.path.join(out, "kepala.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(src)):
        z.write(os.path.join(src, f), f"contoh_kepala/{f}")
mri3(os.path.join(out, "mri3"))
cta(os.path.join(out, "cta"))
print("done", out)
