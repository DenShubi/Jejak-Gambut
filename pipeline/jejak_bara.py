"""
jejak_bara.py — pipeline Jejak Bara, kontrak data v3.0 (15 Nov 2026)

Satu fungsi utama: proses(...) membaca GeoTIFF GCOV (HV dan HH, pra dan pasca, plus deret
tanggal lain bila ada), peta KHG, titik panas VIIRS, batas kabupaten, dan (opsional) raster AOD,
lalu menulis berkas statis ke keluaran/<aoi>/ sesuai kontrak v3.0 (superset v2).

Empat pilar v3:
  1. Asap + hotspot -> hektare per kategori (terkonfirmasi / kandidat_tersembunyi / perlu_verifikasi)
  2. Titik bakar mulai -> vektor arah rambatan (derajat, mata angin, meter)
  3. Computer vision: SLIC superpixel di dalam KHG + Multi-Otsu + rectangularity + deret 12-harian
  4. Keterhubungan muatan asap (AOD) & FRP terhadap hotspot dan kehilangan tajuk (-dHV)

Prinsip yang dipegang:
- Semua rata-rata dihitung di domain daya LINEAR, dB hanya di akhir.
- Filter speckle abai-NaN (normalized convolution), tidak menyebarkan NaN.
- Semua tanggal dibaca ke grid PRA (setara pasca.rio.reproject_match(pra)).
- Deteksi dibatasi ke lahan gambut (KHG) SEJAK AWAL, bukan di statistik akhir.
- Titik panas dipakai untuk MELABELI poligon, tidak untuk menyaring deteksi
  (kalau dipakai sebagai label latih, validasinya jadi melingkar).
- JSON keluaran tidak pernah berisi NaN (JSON.parse di browser akan gagal).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from scipy.ndimage import binary_closing, uniform_filter
from shapely.geometry import box, shape, LineString, Point
from skimage.filters import threshold_multiotsu
from skimage.segmentation import slic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize

VERSI_KONTRAK = "3.0"

# --------------------------------------------------------------------------
# Kandidat AOI — (W, S, E, N) dalam derajat, sesuai Spesifikasi v3.0 bagian 4.
# Kalteng dan Merauke di SELATAN khatulistiwa: S dan N NEGATIF. Salin dari sini, bukan dari PDF.
# --------------------------------------------------------------------------
AOI_KANDIDAT = {
    "riau":    {"nama": "Bengkalis-Siak-Pelalawan, Riau",       "bbox": (101.5,  0.0, 103.0,  1.8)},
    "kalteng": {"nama": "Sebangau-Pulang Pisau, Kalteng",       "bbox": (113.0, -3.2, 114.5, -1.8)},
    "merauke": {"nama": "Merauke, Papua Selatan",               "bbox": (139.3, -8.6, 140.8, -7.0)},
}
for _k, _v in AOI_KANDIDAT.items():
    _w, _s, _e, _n = _v["bbox"]
    assert _w < _e and _s < _n, f"bbox {_k} terbalik: {_v['bbox']}"
assert AOI_KANDIDAT["kalteng"]["bbox"][1] < 0 and AOI_KANDIDAT["merauke"]["bbox"][3] < 0

MATA_ANGIN = ["Utara", "Timur Laut", "Timur", "Tenggara", "Selatan", "Barat Daya", "Barat", "Barat Laut"]
MATA_ANGIN_EN = {"Utara": "North", "Timur Laut": "Northeast", "Timur": "East", "Tenggara": "Southeast",
                 "Selatan": "South", "Barat Daya": "Southwest", "Barat": "West", "Barat Laut": "Northwest"}


# --------------------------------------------------------------------------
# Parameter — semua yang mungkin perlu diubah ada di sini
# --------------------------------------------------------------------------
@dataclass
class Parameter:
    # --- grid & filter
    resolusi_m: float = 0.0           # 0 = resolusi asli GCOV (20 m). 40/80 untuk AOI besar (multilook rata linear)
    ukuran_filter: int = 5            # jendela speckle, piksel
    # --- CV: SLIC + Multi-Otsu
    n_superpixel: int = 3500          # minimum jumlah superpixel di dalam KHG
    luas_superpixel_maks_ha: float = 50.0   # kalau KHG luas, tambah superpixel supaya rerata <= ini
    kekompakan_slic: float = 0.3      # kecil = ikut batas radiometrik, besar = lebih kotak
    ambang_hv_db: float | None = None # None = otomatis Multi-Otsu. Isi angka untuk override manual
    ambang_hv_maks_db: float = -1.8   # ambang adaptif tidak boleh lebih longgar dari ini
    luas_min_ha: float = 1.0          # buang poligon lebih kecil dari ini
    # --- klasifikasi
    ambang_rasio_db: float = -1.0     # dRasio HV/HH < ini = kanopi hilang tapi permukaan tetap (organik)
    ambang_rectangularity: float = 0.78   # >= ini = bentuk kotak (pembukaan mekanis)
    buffer_hotspot_m: float = 375.0   # satu piksel VIIRS I-band
    confidence_hotspot: tuple = ("n", "h")   # VIIRS: l / n / h. 'l' dibuang (flaring, pantulan awan)
    # --- asap
    ambang_aod_pekat: float = 1.0     # AOD 550 nm di atas ini = asap pekat (kuadran)
    # --- tampilan
    simplify_derajat: float = 0.0002  # ~22 m, sekitar satu piksel GCOV 20 m
    alpha_png: float = 0.8
    sembunyikan_png_di_bawah_db: float = 1.5   # |perubahan| < ini dibuat transparan
    tampilkan_png_luar_gambut: bool = False


# --------------------------------------------------------------------------
# Utilitas raster
# --------------------------------------------------------------------------
def buat_grid(path_acuan: str, aoi_4326: tuple, resolusi_m: float = 0.0) -> dict:
    """Grid kerja = grid PRA: CRS berkas acuan, dipotong ke AOI, sejajar grid sumber."""
    with rasterio.open(path_acuan) as src:
        crs, t, res0 = src.crs, src.transform, src.res[0]
        w, s, e, n = transform_bounds("EPSG:4326", crs, *aoi_4326, densify_pts=21)
        sw, ss, se, sn = src.bounds
    res = max(res0, float(resolusi_m or 0))
    w, s, e, n = max(w, sw), max(s, ss), min(e, se), min(n, sn)
    if w >= e or s >= n:
        raise ValueError("AOI tidak beririsan dengan berkas acuan. Cek bbox atau granule.")
    x0 = t.c + np.floor((w - t.c) / res0) * res0
    y0 = t.f - np.floor((t.f - n) / res0) * res0
    lebar = int(np.ceil((e - x0) / res))
    tinggi = int(np.ceil((y0 - s) / res))
    return {"crs": crs, "transform": from_origin(x0, y0, res, res),
            "width": lebar, "height": tinggi, "res": res, "res_asli": res0}


def baca_ke_grid(path: str, grid: dict) -> np.ndarray:
    """
    Baca satu GeoTIFF ke grid kerja (= pasca.rio.reproject_match(pra)).
    Resolusi sama -> nearest (tanpa interpolasi speckle). Lebih kasar -> rata-rata daya linear (multilook).
    """
    rs = Resampling.average if grid["res"] > grid.get("res_asli", grid["res"]) * 1.01 else Resampling.nearest
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=grid["crs"], transform=grid["transform"],
                       width=grid["width"], height=grid["height"], resampling=rs,
                       src_nodata=src.nodata, nodata=np.nan, dtype="float32") as vrt:
            arr = vrt.read(1).astype("float32")
    arr[~(arr > 0)] = np.nan   # nol, negatif, dan NaN -> NaN
    return arr


def rerata_abai_nan(x: np.ndarray, size: int) -> np.ndarray:
    """Rata-rata jendela atas piksel valid saja. Tidak menyebarkan NaN, tidak membias."""
    valid = np.isfinite(x)
    num = uniform_filter(np.where(valid, x, 0.0), size=size)
    den = uniform_filter(valid.astype("float64"), size=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def ke_db(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(p)


def masker_vektor(gdf: gpd.GeoDataFrame, grid: dict) -> np.ndarray:
    g = gdf.to_crs(grid["crs"])
    g = g[~g.geometry.is_empty & g.geometry.notna()]
    if len(g) == 0:
        return np.zeros((grid["height"], grid["width"]), dtype=bool)
    return geometry_mask(g.geometry, out_shape=(grid["height"], grid["width"]),
                         transform=grid["transform"], invert=True)


def rata_linear_per_label(daya: np.ndarray, label: np.ndarray, n: int) -> np.ndarray:
    """Rata-rata daya LINEAR per label. Kembalian panjang n (indeks 0 = label 1)."""
    v = np.isfinite(daya) & (label > 0)
    s = np.bincount(label[v], weights=daya[v], minlength=n + 1)
    c = np.bincount(label[v], minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / c)[1:n + 1]


# --------------------------------------------------------------------------
# Geometri arah
# --------------------------------------------------------------------------
def ke_mata_angin(derajat: float) -> str | None:
    if derajat is None or not np.isfinite(derajat):
        return None
    return MATA_ANGIN[int(((derajat % 360) + 22.5) // 45) % 8]


def bearing(dx: float, dy: float) -> float:
    """Sudut kompas 0-360 (0 = utara, 90 = timur) dari vektor di CRS meter (y ke utara)."""
    return float(np.degrees(np.arctan2(dx, dy)) % 360)


def sumbu_utama(geom) -> tuple[float, float]:
    """(sudut 0-180, panjang sisi terpanjang m) dari kotak pembatas minimum berputar."""
    r = geom.minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        return float("nan"), 0.0
    c = np.asarray(r.exterior.coords)[:4]
    e1, e2 = c[1] - c[0], c[2] - c[1]
    v = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    return bearing(v[0], v[1]) % 180, float(max(np.hypot(*e1), np.hypot(*e2)))


def rerata_sudut(derajat: np.ndarray, bobot: np.ndarray) -> float:
    """Rerata melingkar berbobot (luas) dalam derajat kompas."""
    if len(derajat) == 0:
        return float("nan")
    r = np.radians(derajat)
    s, c = np.sum(bobot * np.sin(r)), np.sum(bobot * np.cos(r))
    if np.hypot(s, c) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(s, c)) % 360)


# --------------------------------------------------------------------------
# JSON aman untuk browser
# --------------------------------------------------------------------------
def _bersih(x):
    """NaN/inf -> None, numpy -> python. JSON.parse di browser gagal pada NaN."""
    if isinstance(x, dict):
        return {str(k): _bersih(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_bersih(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 3)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def tulis_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_bersih(obj), f, ensure_ascii=False, indent=1, allow_nan=False)


def tulis_geojson(gdf: gpd.GeoDataFrame, path: str):
    """GeoJSON EPSG:4326 dengan properti null (bukan NaN) — aman untuk JSON.parse."""
    g = gdf.to_crs("EPSG:4326")
    fitur = []
    kolom = [c for c in g.columns if c != "geometry"]
    for _, r in g.iterrows():
        fitur.append({"type": "Feature",
                      "properties": {c: _bersih(r[c]) if not (isinstance(r[c], float) and np.isnan(r[c])) else None
                                     for c in kolom},
                      "geometry": r.geometry.__geo_interface__ if r.geometry is not None else None})
    obj = {"type": "FeatureCollection", "features": fitur}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_bulatkan_koordinat(obj), f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _bulatkan_koordinat(o, nd=6):
    if isinstance(o, dict):
        return {k: (_bulatkan_koordinat(v, nd) if k in ("geometry", "coordinates", "features") else v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round(v, nd) if isinstance(v, float) else _bulatkan_koordinat(v, nd) for v in o]
    return o


# --------------------------------------------------------------------------
# Titik panas
# --------------------------------------------------------------------------
def baca_hotspot(path_csv: str, tanggal_pra: str, tanggal_pasca: str, p: Parameter) -> pd.DataFrame:
    """CSV FIRMS -> DataFrame terfilter jendela waktu dan confidence n/h."""
    hs = pd.read_csv(path_csv)
    hs["acq_date"] = pd.to_datetime(hs["acq_date"]).dt.strftime("%Y-%m-%d")
    n0 = len(hs)
    hs = hs[(hs["acq_date"] >= tanggal_pra) & (hs["acq_date"] <= tanggal_pasca)].copy()
    n1 = len(hs)
    if "confidence" in hs.columns:
        c = hs["confidence"]
        if pd.api.types.is_numeric_dtype(c):            # MODIS memakai 0-100
            hs = hs[c >= 50]
        else:
            ok = {s.lower() for s in p.confidence_hotspot} | {"nominal" if "n" in p.confidence_hotspot else "",
                                                               "high" if "h" in p.confidence_hotspot else ""}
            hs = hs[c.astype(str).str.strip().str.lower().isin(ok)]
    else:
        print("  [hotspot] PERINGATAN: tidak ada kolom confidence — semua titik dipakai")
    if "frp" not in hs.columns:
        hs["frp"] = np.nan
    print(f"  [hotspot] {n0:,} titik -> {n1:,} dalam jendela -> {len(hs):,} confidence {'/'.join(p.confidence_hotspot)}")
    return hs.reset_index(drop=True)


def sampel_raster(path: str | None, titik_4326: gpd.GeoSeries) -> np.ndarray:
    """Nilai raster (misal AOD rata-rata) di titik-titik. NaN kalau tak ada raster / di luar cakupan."""
    if not path or not os.path.exists(path) or len(titik_4326) == 0:
        return np.full(len(titik_4326), np.nan)
    with rasterio.open(path) as src:
        pts = titik_4326.to_crs(src.crs) if src.crs else titik_4326
        nod = src.nodata
        out = np.array([v[0] for v in src.sample([(g.x, g.y) for g in pts])], dtype="float64")
    if nod is not None:
        out[out == nod] = np.nan
    out[~np.isfinite(out) | (out < 0)] = np.nan
    return out


# --------------------------------------------------------------------------
# Pipeline utama v3
# --------------------------------------------------------------------------
def proses(
    *,
    aoi_nama: str,
    aoi_bbox: tuple,                     # (W, S, E, N) dalam derajat
    tanggal_pra: str,
    tanggal_pasca: str,
    hv_pra: str, hv_pasca: str,
    hh_pra: str, hh_pasca: str,
    khg: str,                            # GeoJSON KHG
    hotspot_csv: str,                    # CSV FIRMS (latitude, longitude, acq_date, confidence, frp)
    kabupaten: str,                      # GeoJSON batas kabupaten
    kolom_kabupaten: str = "nama_kabupaten",
    deret: list | None = None,           # [(tanggal, hv_path, hh_path), ...] seluruh lintasan 12-harian
    aod_tif: str | None = None,          # raster AOD rata-rata jendela (EPSG:4326), opsional
    sumber_aod: str = "",
    granule_pra: str = "", granule_pasca: str = "",
    produk: str = "NISAR_L2_GCOV_PROVISIONAL_V1",
    mode: str = "nisar",                 # nisar | sentinel1 | sintetis — ditulis ke meta & aoi.json
    folder_keluaran: str = "keluaran",
    folder_olahan: str = "olahan",
    param: Parameter | None = None,
    data_sintetis: bool = False,
) -> dict:
    p = param or Parameter()
    if data_sintetis:
        mode = "sintetis"
    os.makedirs(folder_keluaran, exist_ok=True)
    os.makedirs(folder_olahan, exist_ok=True)

    # ==== TAHAP 1: grid pra + reproject_match + speckle di domain linear =========
    grid = buat_grid(hv_pra, aoi_bbox, p.resolusi_m)
    px_ha = grid["res"] ** 2 / 1e4
    print(f"[1] grid {grid['width']}x{grid['height']} px @ {grid['res']:.0f} m, {grid['crs']}")
    raw = {k: baca_ke_grid(v, grid) for k, v in
           dict(hv_pra=hv_pra, hv_pasca=hv_pasca, hh_pra=hh_pra, hh_pasca=hh_pasca).items()}
    halus = {k: rerata_abai_nan(v, p.ukuran_filter) for k, v in raw.items()}
    db = {k: ke_db(v) for k, v in halus.items()}
    delta_hv = db["hv_pasca"] - db["hv_pra"]
    delta_hh = db["hh_pasca"] - db["hh_pra"]
    delta_rasio = (db["hv_pasca"] - db["hh_pasca"]) - (db["hv_pra"] - db["hh_pra"])
    print(f"[1] dHV median {np.nanmedian(delta_hv):+.2f} dB")

    gdf_khg = gpd.read_file(khg)
    gambut = masker_vektor(gdf_khg, grid)
    valid = gambut & np.isfinite(delta_hv) & np.isfinite(delta_hh)
    frac = gambut.mean()
    print(f"[1] {frac:.0%} grid berada di dalam KHG")
    if valid.sum() == 0:
        raise RuntimeError("AOI tidak menyentuh KHG (atau data kosong di KHG). Pilih AOI lain.")

    # ==== TAHAP 2: CV — tensor 3 kanal, SLIC di KHG, Multi-Otsu, shape =============
    tensor = np.dstack([-delta_hv, delta_hh, -delta_rasio]).astype("float32")
    tensor = np.clip(np.nan_to_num(tensor, nan=0.0), -12, 12) / 3.0       # ~1 per 3 dB
    luas_gambut_ha = valid.sum() * px_ha
    n_seg = int(max(p.n_superpixel, math.ceil(luas_gambut_ha / p.luas_superpixel_maks_ha)))
    n_seg = min(n_seg, int(valid.sum() // 16) or 1)
    sp = slic(tensor, n_segments=n_seg, compactness=p.kekompakan_slic, mask=valid,
              channel_axis=-1, convert2lab=False, start_label=1, enforce_connectivity=True)
    sp = sp.astype("int32")
    n_sp = int(sp.max())
    sp_px = np.bincount(sp.ravel(), minlength=n_sp + 1)[1:]
    print(f"[2] SLIC: {n_sp:,} superpixel di KHG, rerata {np.mean(sp_px[sp_px>0])*px_ha:.1f} ha")

    m_sp = {k: rata_linear_per_label(raw[k], sp, n_sp) for k in raw}
    dhv_sp = ke_db(m_sp["hv_pasca"] / m_sp["hv_pra"])
    ok_sp = np.isfinite(dhv_sp) & (sp_px > 0)

    # Multi-Otsu 3 kelas, histogram dibobot luas superpixel
    rentang = (-12.0, 8.0)
    hist, tepi = np.histogram(np.clip(dhv_sp[ok_sp], *rentang), bins=256, range=rentang, weights=sp_px[ok_sp])
    tengah = (tepi[:-1] + tepi[1:]) / 2
    try:
        t_otsu = threshold_multiotsu(classes=3, hist=(hist.astype("float64"), tengah))
    except Exception as ex:
        print(f"[2] Multi-Otsu gagal ({ex}); pakai persentil 5")
        t_otsu = np.array([np.percentile(dhv_sp[ok_sp], 5), np.percentile(dhv_sp[ok_sp], 50)])
    ambang_otsu = float(t_otsu[0])
    if p.ambang_hv_db is not None:
        ambang, sumber_ambang = float(p.ambang_hv_db), "manual"
    else:
        ambang, sumber_ambang = min(ambang_otsu, p.ambang_hv_maks_db), "multiotsu"
    print(f"[2] Multi-Otsu ambang {t_otsu.round(2).tolist()} dB -> dipakai {ambang:+.2f} dB ({sumber_ambang})")

    sp_bakar = np.zeros(n_sp + 1, dtype=bool)
    sp_bakar[1:] = ok_sp & (dhv_sp < ambang)
    terbakar = sp_bakar[sp]
    terbakar = binary_closing(terbakar, structure=np.ones((3, 3)), iterations=1) & valid

    _plot_histogram(delta_hv[valid], dhv_sp[ok_sp], sp_px[ok_sp] * px_ha, t_otsu, ambang, sumber_ambang,
                    os.path.join(folder_olahan, "histogram_multiotsu.png"))

    geoms = [shape(g) for g, val in shapes(terbakar.astype("uint8"), mask=terbakar,
                                           transform=grid["transform"]) if val == 1]
    if not geoms:
        raise RuntimeError("Tidak ada area terdeteksi. Cek histogram_multiotsu.png, pasangan tanggal, "
                           "atau isi Parameter(ambang_hv_db=...) manual.")
    pol = gpd.GeoDataFrame(geometry=geoms, crs=grid["crs"])
    pol["luas_ha"] = pol.area / 1e4                       # di CRS meter, SEBELUM reproject
    pol = pol[pol["luas_ha"] >= p.luas_min_ha].sort_values("luas_ha", ascending=False).reset_index(drop=True)
    if len(pol) == 0:
        raise RuntimeError(f"Semua poligon < {p.luas_min_ha} ha.")
    n = len(pol)
    pol["id"] = np.arange(1, n + 1)
    print(f"[2] {n} poligon, {pol['luas_ha'].sum():,.1f} ha")

    # rectangularity = luas / luas kotak pembatas minimum berputar
    mrr = pol.geometry.minimum_rotated_rectangle()
    pol["cv_rectangularity"] = (pol.area / mrr.area.replace(0, np.nan)).clip(0, 1)

    # statistik radar per poligon, rata daya linear
    label = rasterize(zip(pol.geometry, pol["id"]), out_shape=(grid["height"], grid["width"]),
                      transform=grid["transform"], fill=0, dtype="int32")
    m = {k: rata_linear_per_label(raw[k], label, n) for k in raw}
    pol["delta_hv_db"] = ke_db(m["hv_pasca"] / m["hv_pra"])
    pol["delta_hh_db"] = ke_db(m["hh_pasca"] / m["hh_pra"])
    pol["delta_rasio_db"] = ke_db((m["hv_pasca"] / m["hh_pasca"]) / (m["hv_pra"] / m["hh_pra"]))

    # ==== TAHAP 3: hotspot <= 375 m, AOD/FRP, 3 kategori ==========================
    hs = baca_hotspot(hotspot_csv, tanggal_pra, tanggal_pasca, p)
    ghs = gpd.GeoDataFrame(hs, geometry=gpd.points_from_xy(hs["longitude"], hs["latitude"]),
                           crs="EPSG:4326").to_crs(grid["crs"])
    kotak = box(*rasterio.transform.array_bounds(grid["height"], grid["width"], grid["transform"]))
    ghs = ghs[ghs.within(kotak)].copy().reset_index(drop=True)
    khg_utm = gdf_khg.to_crs(grid["crs"])
    ghs["di_gambut"] = ghs.within(khg_utm.union_all()) if len(ghs) else []

    buf = gpd.GeoDataFrame({"id": pol["id"]}, geometry=pol.buffer(p.buffer_hotspot_m), crs=grid["crs"])
    cocok = gpd.sjoin(ghs[["geometry", "acq_date", "frp"]].reset_index(names="hs_idx"), buf,
                      how="inner", predicate="within")
    # satu hotspot bisa jatuh di buffer dua poligon -> milik poligon terdekat
    if len(cocok):
        jarak = [pol.geometry.iloc[i - 1].distance(ghs.geometry.iloc[h]) for h, i in zip(cocok["hs_idx"], cocok["id"])]
        cocok = cocok.assign(_d=jarak).sort_values("_d").drop_duplicates("hs_idx")
    pol["terkonfirmasi_hotspot"] = pol["id"].isin(set(cocok["id"]))
    pol["jumlah_hotspot"] = pol["id"].map(cocok.groupby("id").size()).fillna(0).astype(int)
    pol["rerata_frp_mw"] = pol["id"].map(cocok.groupby("id")["frp"].mean())
    ghs["dalam_deteksi"] = ghs.index.isin(set(cocok["hs_idx"]))
    ghs["id_poligon"] = ghs.index.map(cocok.set_index("hs_idx")["id"]) if len(cocok) else np.nan
    hs_gambut = ghs[ghs["di_gambut"]] if len(ghs) else ghs
    validasi = (hs_gambut["dalam_deteksi"].mean() * 100) if len(hs_gambut) else np.nan

    titik_rep = gpd.GeoSeries(pol.representative_point(), crs=grid["crs"]).to_crs("EPSG:4326")
    pol["muatan_asap_aod"] = sampel_raster(aod_tif, titik_rep)

    organik = pol["cv_rectangularity"] < p.ambang_rectangularity
    rasio_turun = pol["delta_rasio_db"] < p.ambang_rasio_db
    pol["kategori"] = np.where(pol["terkonfirmasi_hotspot"], "terkonfirmasi",
                      np.where(rasio_turun & organik, "kandidat_tersembunyi", "perlu_verifikasi"))
    pol["alasan_kategori"] = np.where(pol["terkonfirmasi_hotspot"], "hotspot_375m",
                             np.where(rasio_turun & organik, "rasio_turun_organik",
                             np.where(~organik, "bentuk_kotak", "rasio_tidak_turun")))
    print(f"[3] {len(hs_gambut)} hotspot n/h di gambut, "
          + (f"{validasi:.0f}% jatuh di dalam deteksi" if len(hs_gambut) else "tidak ada"))
    print("[3] " + ", ".join(f"{k}: {v:,.1f} ha" for k, v in pol.groupby("kategori")["luas_ha"].sum().items()))

    # ==== TAHAP 4: titik bakar mulai -> vektor arah rambatan ======================
    cent = pol.geometry.centroid
    rows = []
    for i, r in pol.iterrows():
        sub = cocok[cocok["id"] == r["id"]]
        cx, cy = cent.iloc[i].x, cent.iloc[i].y
        if len(sub):
            t0 = sub["acq_date"].min()
            awal = ghs.loc[sub.loc[sub["acq_date"] == t0, "hs_idx"]]
            sx, sy = awal.geometry.x.mean(), awal.geometry.y.mean()
            dx, dy = cx - sx, cy - sy
            jarak = float(np.hypot(dx, dy))
            if jarak < grid["res"]:            # titik mulai tepat di pusat -> pakai sumbu utama
                deg, L = sumbu_utama(r.geometry)
                rows.append((t0, deg, L / 2, "sumbu_utama", cx - np.sin(np.radians(deg)) * L / 2,
                             cy - np.cos(np.radians(deg)) * L / 2))
            else:
                rows.append((t0, bearing(dx, dy), jarak, "titik_api", sx, sy))
        else:
            deg, L = sumbu_utama(r.geometry)
            # sumbu utama tidak punya arah (+-180). Panah digambar dari ujung ke ujung.
            rows.append(("Tertutup Asap", deg, L / 2, "sumbu_utama",
                         cx - np.sin(np.radians(deg)) * L / 2, cy - np.cos(np.radians(deg)) * L / 2))
    trk = pd.DataFrame(rows, columns=["tgl_titik_bakar_mulai", "arah_dampak_derajat", "jangkauan_rambatan_m",
                                      "arah_dampak_metode", "_sx", "_sy"])
    for c in trk.columns:
        pol[c] = trk[c].values
    pol["arah_dampak_kompas"] = [ke_mata_angin(d) for d in pol["arah_dampak_derajat"]]

    # garis panah: mulai -> pusat kerusakan (titik_api) atau ujung ke ujung sumbu (sumbu_utama)
    garis = []
    for i, r in pol.iterrows():
        sx, sy = r["_sx"], r["_sy"]
        cx, cy = cent.iloc[i].x, cent.iloc[i].y
        ex, ey = (cx, cy) if r["arah_dampak_metode"] == "titik_api" else (2 * cx - sx, 2 * cy - sy)
        garis.append(LineString([(sx, sy), (ex, ey)]))
    panah = gpd.GeoDataFrame({"id": pol["id"], "kategori": pol["kategori"],
                              "arah_dampak_derajat": pol["arah_dampak_derajat"],
                              "arah_dampak_kompas": pol["arah_dampak_kompas"],
                              "jangkauan_rambatan_m": pol["jangkauan_rambatan_m"],
                              "metode": pol["arah_dampak_metode"],
                              "tgl_titik_bakar_mulai": pol["tgl_titik_bakar_mulai"]},
                             geometry=garis, crs=grid["crs"])

    # ==== kabupaten: titik representatif (properti) + overlay presisi (rekap) =======
    kab = gpd.read_file(kabupaten).to_crs(grid["crs"])
    if kolom_kabupaten not in kab.columns:
        raise KeyError(f"Kolom '{kolom_kabupaten}' tidak ada. Kolom tersedia: {list(kab.columns)}")
    kab = kab[[kolom_kabupaten, "geometry"]].rename(columns={kolom_kabupaten: "_kab"})
    kab["geometry"] = kab.geometry.buffer(0)
    titik = gpd.GeoDataFrame({"id": pol["id"]}, geometry=pol.representative_point(), crs=grid["crs"])
    j = gpd.sjoin(titik, kab, how="left", predicate="within").drop_duplicates("id").set_index("id")["_kab"]
    pol["kabupaten"] = pol["id"].map(j).fillna("Tidak diketahui")
    potong = gpd.overlay(pol[["id", "kategori", "geometry"]], kab, how="intersection", keep_geom_type=True)
    potong["ha"] = potong.area / 1e4
    terdampak = potong[potong["ha"] >= 0.01].groupby("id")["_kab"].apply(lambda s: ", ".join(sorted(set(s))))
    pol["kabupaten_terdampak"] = pol["id"].map(terdampak).fillna(pol["kabupaten"])

    # ==== TAHAP 4b: deret waktu 12-harian (superpixel, kumulatif) ===================
    deret = deret or [(tanggal_pra, hv_pra, hh_pra), (tanggal_pasca, hv_pasca, hh_pasca)]
    deret = sorted({d[0]: d for d in deret}.values(), key=lambda d: d[0])
    seri_hv = {str(i): [] for i in pol["id"]}
    seri_hh = {str(i): [] for i in pol["id"]}
    seri_luas = {str(i): [] for i in pol["id"]}
    kumulatif = np.zeros(sp.shape, dtype=bool)
    dalam_pol = label > 0
    for tgl, phv, phh in deret:
        a_hv = raw["hv_pasca"] if phv == hv_pasca else (raw["hv_pra"] if phv == hv_pra else baca_ke_grid(phv, grid))
        a_hh = raw["hh_pasca"] if phh == hh_pasca else (raw["hh_pra"] if phh == hh_pra else baca_ke_grid(phh, grid))
        mhv = ke_db(rata_linear_per_label(a_hv, label, n))
        mhh = ke_db(rata_linear_per_label(a_hh, label, n))
        if tgl > tanggal_pra:
            d_sp = ke_db(rata_linear_per_label(a_hv, sp, n_sp) / m_sp["hv_pra"])
            b = np.zeros(n_sp + 1, dtype=bool)
            b[1:] = np.isfinite(d_sp) & (d_sp < ambang)
            kumulatif |= b[sp] & dalam_pol
        luas_t = np.bincount(label[kumulatif], minlength=n + 1)[1:] * px_ha
        for k, i in enumerate(pol["id"]):
            seri_hv[str(i)].append(mhv[k]); seri_hh[str(i)].append(mhh[k]); seri_luas[str(i)].append(luas_t[k])
    tanggal_deret = [d[0] for d in deret]
    tgl_det, laju = [], []
    for i in pol["id"]:
        s = np.asarray(seri_luas[str(i)], dtype=float)
        akhir = s[-1] if len(s) else 0
        if akhir <= 0:
            tgl_det.append(None); laju.append(np.nan); continue
        k0 = int(np.argmax(s >= 0.1 * akhir))
        k1 = int(np.argmax(s >= 0.9 * akhir))
        tgl_det.append(tanggal_deret[k0])
        kb = max(k0 - 1, 0)
        hari = (pd.Timestamp(tanggal_deret[k1]) - pd.Timestamp(tanggal_deret[kb])).days
        laju.append((s[k1] - s[kb]) / hari if hari > 0 else np.nan)
    pol["tgl_terdeteksi_nisar"] = tgl_det
    pol["laju_ekspansi_ha_per_hari"] = laju
    total_kum = np.sum([seri_luas[str(i)] for i in pol["id"]], axis=0).tolist()
    tulis_json({"tanggal": tanggal_deret, "hv_db": seri_hv, "hh_db": seri_hh,
                "luas_kumulatif_ha": seri_luas, "total_kumulatif_ha": total_kum},
               os.path.join(folder_keluaran, "deret_waktu.json"))
    print(f"[4] deret {len(deret)} tanggal: total kumulatif {[round(v) for v in total_kum]} ha")

    # ==== Pilar 4: keterhubungan asap-hotspot (kuadran) =============================
    luas_buf_km2 = buf.area / 1e6
    pol["kepadatan_hotspot_per_km2"] = pol["jumlah_hotspot"] / luas_buf_km2.values
    ada_aod = pol["muatan_asap_aod"].notna().any()
    kuadran = None
    if ada_aod:
        pekat = pol["muatan_asap_aod"] >= p.ambang_aod_pekat
        kuadran = {}
        for nm, msk in {"asap_pekat_ada_hotspot": pekat & pol["terkonfirmasi_hotspot"],
                        "asap_pekat_tanpa_hotspot": pekat & ~pol["terkonfirmasi_hotspot"],
                        "asap_tipis_ada_hotspot": ~pekat & pol["terkonfirmasi_hotspot"],
                        "asap_tipis_tanpa_hotspot": ~pekat & ~pol["terkonfirmasi_hotspot"]}.items():
            sub = pol[msk & pol["muatan_asap_aod"].notna()]
            kuadran[nm] = {"jumlah_poligon": int(len(sub)), "ha": float(sub["luas_ha"].sum()),
                           "rerata_minus_dhv_db": float(-sub["delta_hv_db"].mean()) if len(sub) else None}
    _plot_asap_hotspot(pol, p, os.path.join(folder_olahan, "plot_asap_hotspot.png"), ada_aod)

    # ==== EKSPOR =====================================================================
    kolom = ["id", "luas_ha", "kabupaten", "kabupaten_terdampak", "delta_hv_db", "delta_hh_db", "delta_rasio_db",
             "terkonfirmasi_hotspot", "kategori", "alasan_kategori", "cv_rectangularity",
             "tgl_titik_bakar_mulai", "tgl_terdeteksi_nisar", "arah_dampak_derajat", "arah_dampak_kompas",
             "arah_dampak_metode", "jangkauan_rambatan_m", "laju_ekspansi_ha_per_hari",
             "jumlah_hotspot", "kepadatan_hotspot_per_km2", "muatan_asap_aod", "rerata_frp_mw", "geometry"]
    out = pol[kolom].copy()
    out["geometry"] = out.to_crs("EPSG:4326").geometry.simplify(p.simplify_derajat, preserve_topology=True).to_crs(grid["crs"]) \
        if p.simplify_derajat else out.geometry
    for c in ["luas_ha", "delta_hv_db", "delta_hh_db", "delta_rasio_db", "arah_dampak_derajat",
              "laju_ekspansi_ha_per_hari", "kepadatan_hotspot_per_km2", "rerata_frp_mw", "muatan_asap_aod"]:
        out[c] = out[c].astype(float).round(2)
    out["cv_rectangularity"] = out["cv_rectangularity"].astype(float).round(3)
    out["jangkauan_rambatan_m"] = out["jangkauan_rambatan_m"].astype(float).round(0)
    tulis_geojson(out, os.path.join(folder_keluaran, "terbakar.geojson"))
    tulis_geojson(panah, os.path.join(folder_keluaran, "arah_rambatan.geojson"))

    aoi_poly = box(*aoi_bbox)
    g_out = gdf_khg.to_crs("EPSG:4326").clip(aoi_poly)
    g_out["geometry"] = g_out.geometry.simplify(max(p.simplify_derajat, 0.0001) * 2, preserve_topology=True)
    tulis_geojson(g_out[["geometry"]], os.path.join(folder_keluaran, "gambut.geojson"))

    h_cols = [c for c in ["acq_date", "frp", "confidence"] if c in ghs.columns]
    h_out = ghs[h_cols + ["di_gambut", "dalam_deteksi", "id_poligon", "geometry"]].copy()
    if "confidence" in h_out:
        h_out["confidence"] = h_out["confidence"].astype(str)
    h_out["frp"] = h_out["frp"].astype(float).round(1)
    tulis_geojson(h_out, os.path.join(folder_keluaran, "hotspot.geojson"))

    png_bounds = _tulis_png(delta_hv, gambut, grid, p, os.path.join(folder_keluaran, "perubahan.png"))
    _png_abu(db["hv_pra"], grid, os.path.join(folder_keluaran, "hv_pra.png"))
    _png_abu(db["hv_pasca"], grid, os.path.join(folder_keluaran, "hv_pasca.png"))

    # ---- statistik.json (kontrak v3) -------------------------------------------------
    ha_kat = pol.groupby("kategori")["luas_ha"].sum()
    k_ha = ha_kat.get("terkonfirmasi", 0.0)
    t_ha = ha_kat.get("kandidat_tersembunyi", 0.0)
    v_ha = ha_kat.get("perlu_verifikasi", 0.0)
    kom = k_ha + t_ha
    potong["tk"] = np.where(potong["kategori"] == "terkonfirmasi", potong["ha"], 0.0)
    potong["ta"] = np.where(potong["kategori"] == "kandidat_tersembunyi", potong["ha"], 0.0)
    potong["pv"] = np.where(potong["kategori"] == "perlu_verifikasi", potong["ha"], 0.0)
    per_kab = potong.groupby("_kab")[["ha", "tk", "ta", "pv"]].sum().sort_values("ha", ascending=False)
    dgn_api = pol[pol["arah_dampak_metode"] == "titik_api"]
    dom = rerata_sudut(dgn_api["arah_dampak_derajat"].values, dgn_api["luas_ha"].values)
    persen_luput_pekat = None
    if ada_aod:
        pk = pol[(pol["muatan_asap_aod"] >= p.ambang_aod_pekat) & pol["kategori"].isin(["terkonfirmasi", "kandidat_tersembunyi"])]
        if pk["luas_ha"].sum() > 0:
            persen_luput_pekat = float(pk.loc[pk["kategori"] == "kandidat_tersembunyi", "luas_ha"].sum() / pk["luas_ha"].sum() * 100)
    statistik = {
        "total_ha": pol["luas_ha"].sum(),
        "terkonfirmasi_ha": k_ha,
        "kandidat_tersembunyi_ha": t_ha,
        "perlu_verifikasi_ha": v_ha,
        "dampak_kombinasi_asap_dan_hotspot_ha": kom,
        "rasio_kebutaan_termal_akibat_asap_persen": (t_ha / kom * 100) if kom > 0 else np.nan,
        "persen_luput_di_bawah_asap_pekat": persen_luput_pekat,
        "validasi_hotspot_persen": validasi,
        "arah_rambatan_dampak_dominan": ke_mata_angin(dom),
        "arah_rambatan_dampak_dominan_derajat": dom,
        "ambang_multiotsu_db": ambang_otsu,
        "ambang_dipakai_db": ambang,
        "sumber_ambang": sumber_ambang,
        "ambang_multiotsu_semua_db": [float(x) for x in t_otsu],
        "jumlah_superpixel": n_sp,
        "jumlah_hotspot": int(len(hs_gambut)),
        "jumlah_poligon": int(n),
        "rerata_aod": float(pol["muatan_asap_aod"].mean()) if ada_aod else None,
        "ambang_aod_pekat": p.ambang_aod_pekat,
        "kuadran_asap_hotspot": kuadran,
        "luas_kumulatif_ha": total_kum,
        "per_kabupaten": [{"nama": k, "ha": r["ha"], "terkonfirmasi_ha": r["tk"],
                           "tertutup_asap_ha": r["ta"], "perlu_verifikasi_ha": r["pv"]}
                          for k, r in per_kab.iterrows()],
    }
    tulis_json(statistik, os.path.join(folder_keluaran, "statistik.json"))

    meta = {
        "versi_kontrak": VERSI_KONTRAK,
        "mode": mode,
        "data_sintetis": data_sintetis,
        "aoi_nama": aoi_nama, "aoi_bbox": list(aoi_bbox),
        "tanggal_pra": tanggal_pra, "tanggal_pasca": tanggal_pasca,
        "tanggal_deret": tanggal_deret,
        "produk": produk, "polarisasi": ["HV", "HH"],
        "granule_pra": granule_pra, "granule_pasca": granule_pasca,
        "sumber_aod": sumber_aod if ada_aod else "",
        "ukuran_piksel_m": grid["res"], "crs_olah": str(grid["crs"]),
        "deteksi_dibatasi_gambut": True,
        "metode": "SLIC superpixel (KHG) + Multi-Otsu 3 kelas + rectangularity + hotspot VIIRS n/h <= 375 m",
        "parameter": asdict(p),
        "png_bounds": png_bounds,   # format langsung untuk MapLibre image source
        "png": {"perubahan": "perubahan.png", "hv_pra": "hv_pra.png", "hv_pasca": "hv_pasca.png"},
        "berkas": ["terbakar.geojson", "arah_rambatan.geojson", "gambut.geojson", "hotspot.geojson",
                   "statistik.json", "deret_waktu.json", "perubahan.png", "hv_pra.png", "hv_pasca.png"],
    }
    tulis_json(meta, os.path.join(folder_keluaran, "meta.json"))
    print(f"[selesai] -> {folder_keluaran}/  (kombinasi {kom:,.1f} ha, kebutaan termal "
          f"{statistik['rasio_kebutaan_termal_akibat_asap_persen'] if kom else float('nan'):.1f}%, arah dominan {statistik['arah_rambatan_dampak_dominan']})")
    return {"statistik": statistik, "meta": meta}


# --------------------------------------------------------------------------
# Plot olahan (untuk slide)
# --------------------------------------------------------------------------
def _plot_histogram(dhv_px, dhv_sp, luas_sp, t_otsu, ambang, sumber, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(np.clip(dhv_px, -12, 8), bins=200, range=(-12, 8), density=True, color="0.82", label="pixels in KHG")
    ax.hist(np.clip(dhv_sp, -12, 8), bins=120, range=(-12, 8), weights=luas_sp, density=True,
            histtype="step", lw=2, color="0.15", label="superpixels (area-weighted)")
    for t in t_otsu:
        ax.axvline(t, color="0.5", ls=":", lw=1)
    ax.axvline(ambang, color="#c0392b", ls="--", lw=2, label=f"burn threshold {ambang:+.2f} dB ({sumber})")
    ax.set_xlabel("HV change (dB), peat only"); ax.set_ylabel("density"); ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper left"); ax.set_title("Multi-Otsu (3 classes) on SLIC superpixels inside KHG")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_asap_hotspot(pol, p, path, ada_aod):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = pol["muatan_asap_aod"] if ada_aod else pd.Series(np.zeros(len(pol)))
    y = np.log1p(pol["kepadatan_hotspot_per_km2"])
    sc = ax.scatter(x, y, s=np.clip(pol["luas_ha"], 5, 800) / 2, c=-pol["delta_hv_db"], cmap="inferno_r",
                    vmin=0, vmax=8, edgecolor="k", lw=.4)
    fig.colorbar(sc, ax=ax, label="canopy loss  −ΔHV (dB)")
    if ada_aod:
        ax.axvline(p.ambang_aod_pekat, color="0.3", ls="--")
        ax.text(p.ambang_aod_pekat, ax.get_ylim()[1] * .95, "  dense smoke →", va="top", fontsize=9)
        ax.set_xlabel("smoke load: aerosol optical depth (550 nm)")
    else:
        ax.set_xlabel("AOD not provided"); ax.set_xticks([])
    ax.axhline(0.05, color="0.6", ls=":")
    ax.set_ylabel("VIIRS hotspots per km²  (log1p)")
    ax.set_title("Smoke vs. hotspots per burn scar (size = ha)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------
# AOD MERRA-2 -> GeoTIFF rata-rata jendela (dipakai notebook)
# --------------------------------------------------------------------------
def buat_aod_rerata(berkas_nc: list, aoi_bbox: tuple, path_tif: str, variabel: str = "TOTEXTTAU",
                    tanggal: tuple | None = None) -> str:
    """
    Rata-rata AOD (MERRA-2 M2T1NXAER TOTEXTTAU, atau MCD19A2 yang sudah diregrid) di atas AOI
    -> GeoTIFF EPSG:4326 satu band. Grid MERRA-2 0.5° x 0.625°: kasar, tapi tidak pernah buta asap.
    """
    import xarray as xr
    import rioxarray  # noqa: F401
    w, s, e, n = aoi_bbox
    bagian = []
    for f in sorted(berkas_nc):
        with xr.open_dataset(f) as ds:
            da = ds[variabel].sel(lon=slice(w - 1, e + 1), lat=slice(s - 1, n + 1))
            if tanggal and "time" in da.dims:
                da = da.sel(time=slice(tanggal[0], tanggal[1]))
            if da.size:
                bagian.append(da.load())
    if not bagian:
        raise RuntimeError("AOD: tidak ada data MERRA-2 di AOI/tanggal ini")
    da = xr.concat(bagian, dim="time") if "time" in bagian[0].dims else xr.concat(bagian, dim="berkas")
    da = da.mean(da.dims[0]).sortby("lat", ascending=False)   # utara di atas
    da = da.rename({"lon": "x", "lat": "y"}).rio.write_crs("EPSG:4326").rio.write_nodata(np.nan)
    da.astype("float32").rio.to_raster(path_tif)
    print(f"[aod] {variabel} rerata {float(da.mean()):.2f} -> {path_tif}")
    return path_tif


def _crs_gcov(ds):
    """Ambil CRS dari dataset 'projection' di grup GCOV (atribut EPSG/WKT, atau nilai skalarnya)."""
    from rasterio.crs import CRS
    if "projection" in ds:
        p = ds["projection"]
        for k in ("epsg_code", "epsg"):
            if k in p.attrs:
                return CRS.from_epsg(int(np.asarray(p.attrs[k]).ravel()[0]))
        for k in ("spatial_ref", "crs_wkt"):
            if k in p.attrs:
                return CRS.from_wkt(str(p.attrs[k]))
        try:
            v = int(np.asarray(p.values).ravel()[0])
            if 1000 < v < 100000:
                return CRS.from_epsg(v)
        except Exception:
            pass
    raise ValueError("CRS tidak ditemukan. Cetak ds['projection'].attrs dan kirim isinya.")


def subset_gcov(sumber, aoi_bbox: tuple, folder: str, nama_dasar: str,
                frekuensi: str = "frequencyA", layer=("HHHH", "HVHV")) -> dict:
    """
    Ambil HANYA potongan AOI dari satu granule GCOV, simpan sebagai GeoTIFF kecil.

    sumber : path .h5 lokal, atau file-like dari earthaccess.open() (streaming HTTPS).
             Hanya chunk HDF5 yang beririsan AOI yang dibaca — bukan seluruh granule.
    Kembalian: {'HHHH': path_hh.tif, 'HVHV': path_hv.tif}

    Sengaja tidak memakai GDAL: GDAL < 3.13 (versi bawaan rasterio di Colab) tidak
    membaca georeferensi NISAR dari HDF5 secara otomatis.
    """
    import xarray as xr
    import rioxarray  # noqa: F401  (mendaftarkan accessor .rio)

    os.makedirs(folder, exist_ok=True)
    grup = f"/science/LSAR/GCOV/grids/{frekuensi}"
    # tanpa chunks: xarray tetap lazy, dan .sel() hanya membaca hyperslab AOI dari HDF5
    ds = xr.open_dataset(sumber, engine="h5netcdf", group=grup, phony_dims="access",
                         decode_timedelta=False)
    crs = _crs_gcov(ds)
    w, s, e, n = transform_bounds("EPSG:4326", crs, *aoi_bbox, densify_pts=21)
    x, y = ds["xCoordinates"].values, ds["yCoordinates"].values
    xsl = slice(w, e) if x[0] < x[-1] else slice(e, w)
    ysl = slice(n, s) if y[0] > y[-1] else slice(s, n)

    hasil = {}
    for L in layer:
        if L not in ds:
            print(f"  [subset] {L} tidak ada di granule ini (mode polarisasi berbeda?) — dilewati")
            continue
        da = ds[L].sel(xCoordinates=xsl, yCoordinates=ysl)
        if da.size == 0:
            raise ValueError(f"AOI tidak beririsan dengan granule ini ({L}). Pilih granule lain.")
        da = (da.rename({"xCoordinates": "x", "yCoordinates": "y"})
                .astype("float32")
                .rio.set_spatial_dims(x_dim="x", y_dim="y")
                .rio.write_crs(crs)
                .rio.write_nodata(np.nan))
        p = os.path.join(folder, f"{nama_dasar}_{L[:2].lower()}.tif")
        da.load().rio.to_raster(p, compress="deflate", tiled=True)
        hasil[L] = p
        print(f"  [subset] {L}: {da.shape[1]}x{da.shape[0]} px -> {p} ({os.path.getsize(p)/1e6:.1f} MB)")
    ds.close()
    return hasil


def tulis_indeks(folder_root: str, urutan=("riau", "kalteng", "merauke")) -> list:
    """
    Setiap AOI punya subfolder sendiri di keluaran/. Fungsi ini memindai subfolder dan menulis
    keluaran/aoi.json, satu-satunya berkas yang dibaca aplikasi saat pertama dibuka.
    """
    daftar = []
    ada = [d for d in os.listdir(folder_root) if os.path.isdir(os.path.join(folder_root, d))]
    for slug in [u for u in urutan if u in ada] + sorted(d for d in ada if d not in urutan):
        d = os.path.join(folder_root, slug)
        if not os.path.isfile(os.path.join(d, "meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        st = json.load(open(os.path.join(d, "statistik.json")))
        daftar.append({
            "slug": slug, "nama": meta["aoi_nama"], "bbox": meta["aoi_bbox"],
            "mode": meta.get("mode", "sintetis" if meta.get("data_sintetis") else "nisar"),
            "tanggal_pra": meta["tanggal_pra"], "tanggal_pasca": meta["tanggal_pasca"],
            "total_ha": st["total_ha"], "terkonfirmasi_ha": st["terkonfirmasi_ha"],
            "kandidat_tersembunyi_ha": st["kandidat_tersembunyi_ha"],
            "perlu_verifikasi_ha": st.get("perlu_verifikasi_ha"),
            "rasio_kebutaan_termal_akibat_asap_persen": st.get("rasio_kebutaan_termal_akibat_asap_persen"),
            "arah_rambatan_dampak_dominan": st.get("arah_rambatan_dampak_dominan"),
            "validasi_hotspot_persen": st["validasi_hotspot_persen"],
            "data_sintetis": meta.get("data_sintetis", False),
        })
    if not daftar:
        raise RuntimeError(f"Tidak ada subfolder AOI berisi meta.json di {folder_root}")
    tulis_json({"versi_kontrak": VERSI_KONTRAK, "aoi": daftar}, os.path.join(folder_root, "aoi.json"))
    print(f"[indeks] {len(daftar)} AOI -> {folder_root}/aoi.json  "
          f"(mode: {', '.join(a['slug'] + '=' + a['mode'] for a in daftar)})")
    return daftar


def _png_abu(arr_db, grid, path, vmin=-25.0, vmax=-5.0, langkah=2):
    """HV dalam dB sebagai PNG abu-abu di EPSG:4326, diperkecil 'langkah' kali supaya ringan."""
    src_t, h, w = grid["transform"], grid["height"], grid["width"]
    bnd = rasterio.transform.array_bounds(h, w, src_t)
    dst_t, dw, dh = calculate_default_transform(grid["crs"], "EPSG:4326", w, h, *bnd)
    out = np.full((dh, dw), np.nan, dtype="float32")
    reproject(arr_db.astype("float32"), out, src_transform=src_t, src_crs=grid["crs"],
              dst_transform=dst_t, dst_crs="EPSG:4326", src_nodata=np.nan,
              dst_nodata=np.nan, resampling=Resampling.nearest)
    out = out[::langkah, ::langkah]
    v = np.clip((np.nan_to_num(out, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    rgba = np.dstack([v, v, v, np.isfinite(out).astype("float32")])
    rgba[rgba[..., 3] == 0, :3] = 0.0
    plt.imsave(path, rgba)


def _tulis_png(delta_hv, gambut, grid, p: Parameter, path: str):
    """Reproject ke EPSG:4326 dulu supaya pas sebagai image source MapLibre, lalu RGBA."""
    src_t, h, w = grid["transform"], grid["height"], grid["width"]
    bnd = rasterio.transform.array_bounds(h, w, src_t)
    dst_t, dw, dh = calculate_default_transform(grid["crs"], "EPSG:4326", w, h, *bnd)

    def warp(arr, resampling):
        out = np.full((dh, dw), np.nan, dtype="float32")
        reproject(arr.astype("float32"), out, src_transform=src_t, src_crs=grid["crs"],
                  dst_transform=dst_t, dst_crs="EPSG:4326", src_nodata=np.nan,
                  dst_nodata=np.nan, resampling=resampling)
        return out

    d = warp(delta_hv, Resampling.nearest)
    g = warp(gambut.astype("float32"), Resampling.nearest) > 0.5

    rgba = colormaps["RdBu"](Normalize(vmin=-5, vmax=5)(np.nan_to_num(d, nan=0.0)))  # RENDAH = MERAH
    alpha = np.where(np.isfinite(d) & (np.abs(d) >= p.sembunyikan_png_di_bawah_db), p.alpha_png, 0.0)
    if not p.tampilkan_png_luar_gambut:
        alpha = np.where(g, alpha, 0.0)
    rgba[..., 3] = alpha
    rgba[alpha == 0, :3] = 0.0     # RGB seragam di piksel transparan -> PNG terkompresi jauh lebih kecil
    plt.imsave(path, rgba)

    W, N = dst_t.c, dst_t.f
    E, S = W + dst_t.a * dw, N + dst_t.e * dh
    return [[W, N], [E, N], [E, S], [W, S]]   # kiri-atas, kanan-atas, kanan-bawah, kiri-bawah
