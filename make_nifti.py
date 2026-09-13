"""DICOM (single-file multiframe CBCT, explicit VR LE, uncompressed) -> NIfTI-1 for browser viewing."""
import gzip, numpy as np, os, struct, sys

SRC = sys.argv[1]                    # the .dcm from the Romexis export
NF, NR, NC = 705, 755, 755           # frames, rows, cols  (0028,0008/0010/0011)
SP, Z0, INTERCEPT = 0.2, 141.0, -650  # mm, first-frame IPP z (LPS), (0028,1052)
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 2   # 2 -> scan.nii.gz (0.4mm), 3 -> scan_lo.nii.gz (0.6mm)
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan.nii.gz" if STRIDE == 2 else "scan_lo.nii.gz")

nbytes = NF * NR * NC * 2
off = os.path.getsize(SRC) - nbytes
with open(SRC, "rb") as f:                     # verify we're really at (7FE0,0010) OW
    f.seek(off - 12)
    assert f.read(8) == b"\xe0\x7f\x10\x00OW\x00\x00", "PixelData tag not where expected"

vol = np.memmap(SRC, dtype="<i2", mode="r", offset=off, shape=(NF, NR, NC))
sub = np.ascontiguousarray(vol[::STRIDE, ::STRIDE, ::STRIDE]) + INTERCEPT   # -> HU
nk, nj, ni = sub.shape
s = SP * STRIDE
print("shape (k,j,i)", sub.shape, "HU range", sub.min(), sub.max())

hdr = bytearray(348)
struct.pack_into("<i", hdr, 0, 348)
struct.pack_into("<8h", hdr, 40, 3, ni, nj, nk, 1, 1, 1, 1)      # dim
struct.pack_into("<2h", hdr, 70, 4, 16)                          # datatype int16, bitpix
struct.pack_into("<8f", hdr, 76, 1, s, s, s, 0, 0, 0, 0)         # pixdim
struct.pack_into("<f", hdr, 108, 352.0)                          # vox_offset
struct.pack_into("<2f", hdr, 112, 1.0, 0.0)                      # scl_slope, scl_inter
hdr[123] = 2 | (1 << 3)                                          # xyzt_units: mm + sec
struct.pack_into("<2f", hdr, 124, float(sub.max()), float(sub.min()))  # cal_max, cal_min
struct.pack_into("<2h", hdr, 252, 0, 1)                          # qform_code, sform_code
# DICOM LPS -> NIfTI RAS. Rows +x(L), cols +y(P), frames descend in z.
struct.pack_into("<4f", hdr, 280, -s, 0, 0, 0.0)                 # srow_x
struct.pack_into("<4f", hdr, 296, 0, -s, 0, 0.0)                 # srow_y
struct.pack_into("<4f", hdr, 312, 0, 0, -s, Z0)                  # srow_z
hdr[344:348] = b"n+1\x00"

with gzip.open(DST, "wb") as f:
    f.write(hdr); f.write(b"\0" * 4); f.write(sub.tobytes())
print("wrote", DST, os.path.getsize(DST) // 2**20, "MB")
