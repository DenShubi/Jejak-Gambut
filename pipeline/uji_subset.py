"""
Uji jalur data asli tanpa jaringan: HDF5 tiruan berstruktur GCOV -> subset_gcov -> proses().
Prasyarat: jalankan uji_sintetis.py dulu (memakai GeoTIFF sintetis Riau di /tmp/jb_uji/riau/mentah).
"""
import os, sys, json, shutil
import numpy as np
import h5netcdf
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
import jejak_bara as jb

SRC = "/tmp/jb_uji/riau/mentah"
OUT = "/tmp/jb_subset"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
gagal = []
def cek(ok, pesan):
    print(("  LULUS  " if ok else "  GAGAL  ") + pesan)
    if not ok: gagal.append(pesan)

def buat_h5_tiruan(tgl, path):
    """Tulis HDF5 dengan grup, dimensi, dan dataset 'projection' seperti GCOV."""
    with rasterio.open(f"{SRC}/hv_{tgl}.tif") as s:
        hv, t, epsg = s.read(1), s.transform, s.crs.to_epsg()
    with rasterio.open(f"{SRC}/hh_{tgl}.tif") as s:
        hh = s.read(1)
    ny, nx = hv.shape
    xs = t.c + (np.arange(nx) + 0.5) * t.a          # pusat piksel
    ys = t.f + (np.arange(ny) + 0.5) * t.e          # menurun (utara ke selatan)
    with h5netcdf.File(path, "w") as f:
        g = f
        for nm in ["science", "LSAR", "GCOV", "grids", "frequencyA"]:
            g = g.create_group(nm)
        g.dimensions = {"yCoordinates": ny, "xCoordinates": nx}
        g.create_variable("xCoordinates", ("xCoordinates",), data=xs)
        g.create_variable("yCoordinates", ("yCoordinates",), data=ys)
        for nama, arr in [("HHHH", hh), ("HVHV", hv)]:
            g.create_variable(nama, ("yCoordinates", "xCoordinates"), data=arr.astype("float32"),
                              chunks=(256, 256), compression="gzip")
        g.create_variable("numberOfLooks", ("yCoordinates", "xCoordinates"),
                          data=np.full((ny, nx), 8, "uint8"), chunks=(256, 256))
        p = g.create_variable("projection", (), dtype="int32", data=np.int32(epsg))
        p.attrs["epsg_code"] = np.int32(epsg)
    return t, epsg

print("\n##### HDF5 tiruan -> subset_gcov #####")
t_pra, epsg = buat_h5_tiruan("2026-07-04", f"{OUT}/pra.h5")
buat_h5_tiruan("2026-10-12", f"{OUT}/pasca.h5")
mb_h5 = os.path.getsize(f"{OUT}/pra.h5") / 1e6

# AOI mencakup blob A (px 300,400) dan B (700,900), tapi tidak C (400,1200)
from pyproj import Transformer
ke_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
def px_ll(px, py): return ke_ll.transform(t_pra.c + (px + .5) * 20, t_pra.f - (py + .5) * 20)
w, n = px_ll(150, 250); e, s = px_ll(1000, 1100)
AOI = (w, s, e, n)

h = {}
for fase in ["pra", "pasca"]:
    h[fase] = jb.subset_gcov(f"{OUT}/{fase}.h5", AOI, f"{OUT}/subset", fase)

cek(set(h["pra"]) == {"HHHH", "HVHV"}, "subset menghasilkan HH dan HV")
with rasterio.open(h["pra"]["HVHV"]) as d:
    cek(d.crs.to_epsg() == epsg, f"CRS terbaca dari dataset 'projection' (EPSG:{d.crs.to_epsg()})")
    cek(abs(d.res[0] - 20) < 1e-6, f"resolusi 20 m terjaga ({d.res[0]})")
    sub = d.read(1); b = d.bounds
    mb_sub = os.path.getsize(h["pra"]["HVHV"]) / 1e6
with rasterio.open(f"{SRC}/hv_2026-07-04.tif") as s0:
    asli = s0.read(1, window=from_bounds(*b, s0.transform))
cek(sub.shape == asli.shape, f"ukuran subset cocok dengan potongan sumber {sub.shape}")
cek(np.allclose(sub, asli, equal_nan=True), "nilai piksel identik dengan sumber (tanpa geser setengah piksel)")
cek(mb_sub < mb_h5 / 2, f"subset jauh lebih kecil dari granule ({mb_sub:.1f} MB vs {mb_h5:.1f} MB per berkas)")

print("\n##### subset -> proses() #####")
hasil = jb.proses(
    aoi_nama="Uji jalur HDF5", aoi_bbox=AOI,
    tanggal_pra="2026-07-04", tanggal_pasca="2026-10-12",
    hv_pra=h["pra"]["HVHV"], hv_pasca=h["pasca"]["HVHV"],
    hh_pra=h["pra"]["HHHH"], hh_pasca=h["pasca"]["HHHH"],
    khg=f"{SRC}/khg.geojson", hotspot_csv=f"{SRC}/hotspot.csv",
    kabupaten=f"{SRC}/kab.geojson", aod_tif=f"{SRC}/aod.tif",
    folder_keluaran=f"{OUT}/keluaran/riau", folder_olahan=f"{OUT}/olahan", data_sintetis=True,
)
t = gpd.read_file(f"{OUT}/keluaran/riau/terbakar.geojson")
def kat(px, py):
    hit = t[t.contains(Point(*px_ll(px, py)))]
    return hit["kategori"].iloc[0] if len(hit) else None
cek(kat(300, 400) == "terkonfirmasi", f"blob A -> terkonfirmasi (dapat {kat(300,400)})")
cek(kat(735, 900) == "kandidat_tersembunyi", f"blob B -> kandidat_tersembunyi (dapat {kat(735,900)})")
cek(kat(400, 1200) is None, "blob C di luar AOI tidak muncul")

print(f"\n{'SEMUA LULUS' if not gagal else str(len(gagal)) + ' GAGAL'}")
sys.exit(1 if gagal else 0)
