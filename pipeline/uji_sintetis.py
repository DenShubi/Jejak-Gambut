"""
Uji pipeline v3 dengan data sintetis yang sengaja dibuat menjebak.

Skenario yang harus ditangani benar:
  A. lidah api organik + hotspot 'h', api mulai di sisi BARAT -> 'terkonfirmasi', arah ~Timur
     dan tumbuh: setengah pada 21 Agu, penuh pada 14 Sep
  B. lidah api organik di bawah asap pekat (AOD 2.5), tanpa hotspot sah
     (satu hotspot 'l' di dalamnya HARUS dibuang), rasio HV/HH turun -> 'kandidat_tersembunyi'
     B memotong batas dua kabupaten -> luasnya terbagi lewat overlay
  C. HV dan HH turun bersamaan (rasio tetap), organik             -> 'perlu_verifikasi' (rasio_tidak_turun)
  R. petak kotak miring 30° (ekskavator), rasio turun, tanpa hotspot -> 'perlu_verifikasi' (bentuk_kotak)
  D. sawah DIPANEN di LUAR gambut (HV anjlok)                      -> TIDAK BOLEH terdeteksi
  E. hotspot di luar jendela / luar gambut / confidence 'l'         -> tidak dihitung
Plus: granule pasca bergeser 10 piksel (reproject_match), piksel nol/NaN bertebaran.
"""
import json, os, sys, shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Point
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
import jejak_bara as jb

ROOT = os.environ.get("JB_UJI", "/tmp/jb_uji")
shutil.rmtree(ROOT, ignore_errors=True)
SEMUA_GAGAL = []


def bangun_dan_uji(slug, nama, lon0, lat0, CRS, kab_nama):
    rng = np.random.default_rng(7)
    DIR = f"{ROOT}/{slug}"
    os.makedirs(f"{DIR}/mentah", exist_ok=True)
    ke_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    ke_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    x0, y0 = ke_utm.transform(lon0, lat0)
    x0, y0 = np.floor(x0 / 20) * 20, np.floor(y0 / 20) * 20
    RES, N = 20.0, 1500                             # 30 km x 30 km

    def lin(db): return 10 ** (db / 10)

    yy, xx = np.mgrid[0:N, 0:N]

    def lidah(cx, cy, R, fase=0.0):
        """Bentuk organik bercuping (lidah api), bukan lingkaran."""
        th = np.arctan2(yy - cy, xx - cx)
        r = R * (1 + 0.28 * np.sin(3 * th + fase) + 0.16 * np.sin(5 * th + 2 * fase))
        return np.hypot(xx - cx, yy - cy) < r

    def kotak_miring(cx, cy, w, h, sudut):
        a = np.radians(sudut)
        u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
        v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
        return (np.abs(u) < w / 2) & (np.abs(v) < h / 2)

    blob_A = lidah(300, 400, 70, 0.3)
    A_barat = blob_A & (xx < 300)
    blob_B = lidah(735, 900, 60, 1.1)                # memotong batas kabupaten x=750
    blob_C = lidah(400, 1200, 50, 2.0)
    kotak_R = kotak_miring(840, 330, 130, 70, 30)
    sawah = (xx > 1150) & (xx < 1400) & (yy > 300) & (yy < 700)   # luar gambut

    def scene(tgl):
        hv = np.full((N, N), -12.0); hh = np.full((N, N), -7.0)          # hutan
        hv[sawah] = -13.0; hh[sawah] = -8.0
        if tgl >= "2026-08-21":
            hv[A_barat] = -18.0; hh[A_barat] = -8.0
            hv[blob_B] = -18.0; hh[blob_B] = -8.0
            hv[kotak_R] = -18.5; hh[kotak_R] = -7.5                      # tebang: kanopi hilang
        if tgl >= "2026-09-14":
            hv[blob_A] = -18.0; hh[blob_A] = -8.0
            hv[blob_C] = -18.0; hh[blob_C] = -13.0                       # dua-duanya turun
            hv[sawah] = -20.0; hh[sawah] = -14.0                         # PANEN
        return hv, hh

    def speckle(db, looks=4):
        return rng.gamma(looks, lin(db) / looks).astype("float32")

    def rusak(a):
        k = rng.random(a.shape)
        a[k < 0.001] = 0.0
        a[(k >= 0.001) & (k < 0.0015)] = np.nan
        return a

    def tulis(arr, path, geser=0):
        t = from_origin(x0 + geser * RES, y0 - geser * RES, RES, RES)
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype="float32", crs=CRS, transform=t, nodata=np.nan) as d:
            d.write(arr, 1)

    def potong_geser(arr, g):
        out = np.full_like(arr, np.nan)
        out[:-g, :-g] = arr[g:, g:]
        return out

    deret = []
    for tgl in ["2026-07-04", "2026-07-28", "2026-08-21", "2026-09-14", "2026-10-12"]:
        hv_db, hh_db = scene(tgl)
        hv, hh = rusak(speckle(hv_db)), rusak(speckle(hh_db))
        g = 10 if tgl > "2026-07-04" else 0
        if g:
            hv, hh = potong_geser(hv, g), potong_geser(hh, g)
        phv, phh = f"{DIR}/mentah/hv_{tgl}.tif", f"{DIR}/mentah/hh_{tgl}.tif"
        tulis(hv, phv, g); tulis(hh, phh, g)
        deret.append((tgl, phv, phh))

    def px_ke_ll(px, py):
        return ke_ll.transform(x0 + (px + 0.5) * RES, y0 - (py + 0.5) * RES)

    w, n_ = ke_ll.transform(x0 - 5000, y0 + 5000)
    e_khg, s_ = ke_ll.transform(x0 + 1000 * RES, y0 - N * RES - 5000)
    gpd.GeoDataFrame({"nama": ["KHG uji"]}, geometry=[box(w, s_, e_khg, n_)], crs="EPSG:4326") \
       .to_file(f"{DIR}/mentah/khg.geojson", driver="GeoJSON")
    xm, _ = ke_ll.transform(x0 + 750 * RES, y0)
    e_all, _ = ke_ll.transform(x0 + N * RES + 5000, y0)
    gpd.GeoDataFrame({"nama_kabupaten": kab_nama},
                     geometry=[box(w, s_, xm, n_), box(xm, s_, e_all, n_)], crs="EPSG:4326") \
       .to_file(f"{DIR}/mentah/kab.geojson", driver="GeoJSON")

    # AOD: asap pekat di paruh selatan (B, C), tipis di utara (A, R)
    yb = px_ke_ll(0, 650)[1]
    aod = np.where(np.linspace(n_, s_, 60)[:, None] < yb, 2.5, 0.4) * np.ones((60, 60))
    with rasterio.open(f"{DIR}/mentah/aod.tif", "w", driver="GTiff", height=60, width=60, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=rasterio.transform.from_bounds(w, s_, e_all, n_, 60, 60)) as d:
        d.write(aod.astype("float32"), 1)

    hs = []
    for _ in range(2):   # A: titik bakar mulai, sisi barat, 5 Agu
        lon, lat = px_ke_ll(245 + rng.integers(-6, 6), 400 + rng.integers(-15, 15))
        hs.append(dict(latitude=lat, longitude=lon, acq_date="2026-08-05", frp=55.0, confidence="h"))
    for _ in range(4):   # A: menyebar ke tengah-timur, 12 Agu
        lon, lat = px_ke_ll(330 + rng.integers(-15, 15), 400 + rng.integers(-20, 20))
        hs.append(dict(latitude=lat, longitude=lon, acq_date="2026-08-12", frp=30.0, confidence="h"))
    lon, lat = px_ke_ll(735, 900)       # B: confidence rendah -> dibuang
    hs.append(dict(latitude=lat, longitude=lon, acq_date="2026-08-15", frp=3.0, confidence="l"))
    for _ in range(2):   # B: luar jendela
        lon, lat = px_ke_ll(735, 900)
        hs.append(dict(latitude=lat, longitude=lon, acq_date="2026-06-01", frp=20.0, confidence="n"))
    for _ in range(3):   # sawah, luar gambut
        lon, lat = px_ke_ll(1250, 500)
        hs.append(dict(latitude=lat, longitude=lon, acq_date="2026-08-15", frp=10.0, confidence="n"))
    for px, py in [(150, 1350), (900, 1450)]:   # gambut tanpa bekas bakar
        lon, lat = px_ke_ll(px, py)
        hs.append(dict(latitude=lat, longitude=lon, acq_date="2026-09-01", frp=5.0, confidence="n"))
    pd.DataFrame(hs).to_csv(f"{DIR}/mentah/hotspot.csv", index=False)

    aw, an = px_ke_ll(20, 20); ae, as_ = px_ke_ll(N - 40, N - 40)
    jb.proses(
        aoi_nama=nama, aoi_bbox=(aw, as_, ae, an),
        tanggal_pra="2026-07-04", tanggal_pasca="2026-10-12",
        hv_pra=deret[0][1], hv_pasca=deret[-1][1], hh_pra=deret[0][2], hh_pasca=deret[-1][2],
        khg=f"{DIR}/mentah/khg.geojson", hotspot_csv=f"{DIR}/mentah/hotspot.csv",
        kabupaten=f"{DIR}/mentah/kab.geojson", deret=deret, aod_tif=f"{DIR}/mentah/aod.tif",
        sumber_aod="sintetis",
        folder_keluaran=f"{ROOT}/keluaran/{slug}", folder_olahan=f"{DIR}/olahan", data_sintetis=True,
    )

    # ============================== PEMERIKSAAN ==========================================
    K = f"{ROOT}/keluaran/{slug}"
    print(f"\n##### {nama} ({CRS}) #####")
    def cek(ok, pesan):
        print(("  LULUS  " if ok else "  GAGAL  ") + pesan)
        if not ok: SEMUA_GAGAL.append(f"{slug}: {pesan}")

    for f in ["meta.json", "statistik.json", "terbakar.geojson", "arah_rambatan.geojson", "gambut.geojson",
              "hotspot.geojson", "perubahan.png", "hv_pra.png", "hv_pasca.png", "deret_waktu.json"]:
        cek(os.path.exists(f"{K}/{f}"), f"berkas {f} ada")
    for f in ["histogram_multiotsu.png", "plot_asap_hotspot.png"]:
        cek(os.path.exists(f"{DIR}/olahan/{f}"), f"olahan/{f} ada")

    def tolak(c): raise ValueError(f"konstanta JSON tidak valid: {c}")
    for f in ["meta.json", "statistik.json", "deret_waktu.json", "terbakar.geojson", "hotspot.geojson",
              "arah_rambatan.geojson", "gambut.geojson"]:
        try:
            json.loads(open(f"{K}/{f}").read(), parse_constant=tolak); cek(True, f"{f} JSON ketat (tanpa NaN)")
        except Exception as ex:
            cek(False, f"{f} JSON ketat: {ex}")

    t = gpd.read_file(f"{K}/terbakar.geojson")
    wajib = {"id", "luas_ha", "kabupaten", "delta_hv_db", "delta_rasio_db", "terkonfirmasi_hotspot", "kategori",
             "cv_rectangularity", "tgl_titik_bakar_mulai", "arah_dampak_derajat", "arah_dampak_kompas",
             "jangkauan_rambatan_m", "muatan_asap_aod", "rerata_frp_mw"}
    cek(wajib <= set(t.columns), f"properti v3 lengkap {sorted(wajib - set(t.columns)) or ''}")
    cek(sorted(t["id"]) == list(range(1, len(t) + 1)), "id berurutan 1..n")

    def poli(px, py):
        hit = t[t.contains(Point(*px_ke_ll(px, py)))]
        return hit.iloc[0] if len(hit) else None

    A, B, C, R = poli(300, 400), poli(735, 900), poli(400, 1200), poli(840, 330)
    for nm, x, exp in [("A", A, "terkonfirmasi"), ("B", B, "kandidat_tersembunyi"),
                       ("C", C, "perlu_verifikasi"), ("R", R, "perlu_verifikasi")]:
        cek(x is not None and x["kategori"] == exp, f"{nm} -> {exp} (dapat: {None if x is None else x['kategori']})")
    cek(poli(1250, 500) is None, "D sawah dipanen di luar gambut TIDAK terdeteksi")
    if R is not None:
        cek(R["cv_rectangularity"] >= 0.78, f"R kotak: rectangularity >= 0.78 (dapat {R['cv_rectangularity']:.2f})")
        cek(R["alasan_kategori"] == "bentuk_kotak", f"R alasan bentuk_kotak (dapat {R['alasan_kategori']})")
    for nm, x in [("A", A), ("B", B)]:
        if x is not None:
            cek(x["cv_rectangularity"] < 0.78, f"{nm} organik: rectangularity < 0.78 (dapat {x['cv_rectangularity']:.2f})")
    if C is not None:
        cek(C["alasan_kategori"] == "rasio_tidak_turun", f"C alasan rasio_tidak_turun (dapat {C['alasan_kategori']})")

    if A is not None:
        cek(abs(A["delta_hv_db"] - (-6.0)) < 0.7, f"dHV A ~ -6 dB (dapat {A['delta_hv_db']:+.2f})")
        luas_teori = blob_A.sum() * RES**2 / 1e4
        cek(abs(A["luas_ha"] - luas_teori) / luas_teori < 0.15, f"luas A ~ {luas_teori:.0f} ha (dapat {A['luas_ha']:.0f})")
        cek(str(A["tgl_titik_bakar_mulai"])[:10] == "2026-08-05", f"A titik bakar mulai 2026-08-05 (dapat {A['tgl_titik_bakar_mulai']})")
        cek(45 <= A["arah_dampak_derajat"] <= 135, f"A merambat ke timur (dapat {A['arah_dampak_derajat']:.0f}° {A['arah_dampak_kompas']})")
        cek(A["arah_dampak_metode"] == "titik_api", "A arah dari titik api")
        cek(500 < A["jangkauan_rambatan_m"] < 3000, f"A jangkauan masuk akal (dapat {A['jangkauan_rambatan_m']:.0f} m)")
        cek(abs(A["muatan_asap_aod"] - 0.4) < 0.01, f"A AOD tipis 0.4 (dapat {A['muatan_asap_aod']})")
        cek(abs(A["rerata_frp_mw"] - (2 * 55 + 4 * 30) / 6) < 0.5, f"A FRP rerata ~38.3 MW (dapat {A['rerata_frp_mw']})")
        cek(A["kabupaten"] == kab_nama[0], f"A di {kab_nama[0]} (dapat {A['kabupaten']})")
    if B is not None:
        cek(B["tgl_titik_bakar_mulai"] == "Tertutup Asap", f"B 'Tertutup Asap' (dapat {B['tgl_titik_bakar_mulai']})")
        cek(B["arah_dampak_metode"] == "sumbu_utama", "B arah dari sumbu utama")
        cek(abs(B["muatan_asap_aod"] - 2.5) < 0.01, f"B AOD pekat 2.5 (dapat {B['muatan_asap_aod']})")
        cek(B["rerata_frp_mw"] is None or not np.isfinite(B["rerata_frp_mw"]), "B FRP null")
        cek("," in B["kabupaten_terdampak"], f"B terdampak dua kabupaten ({B['kabupaten_terdampak']})")

    s = json.load(open(f"{K}/statistik.json"))
    cek(s["jumlah_hotspot"] == 8, f"penyebut validasi = 8 hotspot n/h gambut dalam jendela (dapat {s['jumlah_hotspot']})")
    cek(abs(s["validasi_hotspot_persen"] - 6 / 8 * 100) < 1, f"validasi 75% (dapat {s['validasi_hotspot_persen']})")
    tot = s["terkonfirmasi_ha"] + s["kandidat_tersembunyi_ha"] + s["perlu_verifikasi_ha"]
    cek(abs(tot - s["total_ha"]) < 0.1, "jumlah tiga kategori = total")
    cek(abs(s["dampak_kombinasi_asap_dan_hotspot_ha"] - s["terkonfirmasi_ha"] - s["kandidat_tersembunyi_ha"]) < 0.1,
        "kombinasi = terkonfirmasi + tertutup asap")
    r_exp = s["kandidat_tersembunyi_ha"] / s["dampak_kombinasi_asap_dan_hotspot_ha"] * 100
    cek(abs(s["rasio_kebutaan_termal_akibat_asap_persen"] - r_exp) < 0.1, f"rasio kebutaan termal {s['rasio_kebutaan_termal_akibat_asap_persen']}%")
    cek(s["arah_rambatan_dampak_dominan"] == "Timur", f"arah dominan Timur (dapat {s['arah_rambatan_dampak_dominan']})")
    cek(-6 < s["ambang_dipakai_db"] <= -1.8, f"ambang Multi-Otsu masuk akal (dapat {s['ambang_dipakai_db']}, otsu {s['ambang_multiotsu_semua_db']})")
    pk = sum(k["ha"] for k in s["per_kabupaten"])
    cek(abs(pk - s["total_ha"]) / s["total_ha"] < 0.01, f"overlay kabupaten: jumlah {pk:.1f} = total {s['total_ha']:.1f}")
    cek(sum(1 for k in s["per_kabupaten"] if k["tertutup_asap_ha"] > 0) == 2, "B terbagi ke dua kabupaten di rekap")
    q = s["kuadran_asap_hotspot"]
    cek(q and q["asap_pekat_tanpa_hotspot"]["ha"] > 0 and q["asap_tipis_ada_hotspot"]["ha"] > 0 and
        q["asap_pekat_ada_hotspot"]["ha"] == 0, "kuadran: pekat->tanpa hotspot, tipis->ada hotspot")
    cek(s["persen_luput_di_bawah_asap_pekat"] == 100.0, f"100% luka organik di asap pekat luput VIIRS (dapat {s['persen_luput_di_bawah_asap_pekat']})")

    d = json.load(open(f"{K}/deret_waktu.json"))
    cek(len(d["tanggal"]) == 5, "deret 5 tanggal")
    cek(set(d["hv_db"]) == {str(i) for i in t["id"]} == set(d["luas_kumulatif_ha"]), "kunci deret = id poligon")
    if A is not None:
        la = d["luas_kumulatif_ha"][str(int(A["id"]))]
        cek(la[0] == 0 and la[1] < 0.1 * la[-1], f"A belum terbakar Juli ({[round(v) for v in la]})")
        cek(0.3 < la[2] / la[-1] < 0.7, f"A ~setengah pada 21 Agu ({la[2]/la[-1]:.2f})")
        cek(la[3] / la[-1] > 0.9, "A penuh pada 14 Sep")
        cek(all(b >= a for a, b in zip(la, la[1:])), "luas kumulatif monoton naik")
        cek(str(A["tgl_terdeteksi_nisar"])[:10] == "2026-08-21", f"A pertama terlihat NISAR 21 Agu (dapat {A['tgl_terdeteksi_nisar']})")
        cek(A["laju_ekspansi_ha_per_hari"] > 0, f"laju ekspansi A {A['laju_ekspansi_ha_per_hari']} ha/hari")
        sa = d["hv_db"][str(int(A["id"]))]
        cek(sa[1] - sa[4] > 4, f"deret HV A anjlok ({[round(v,1) for v in sa]})")
    if C is not None:
        cek(str(C["tgl_terdeteksi_nisar"])[:10] == "2026-09-14", f"C pertama terlihat 14 Sep (dapat {C['tgl_terdeteksi_nisar']})")

    pa = gpd.read_file(f"{K}/arah_rambatan.geojson")
    cek(len(pa) == len(t) and set(pa.geom_type) == {"LineString"}, "satu panah per poligon")
    hsj = gpd.read_file(f"{K}/hotspot.geojson")
    cek((hsj["confidence"] == "l").sum() == 0, "hotspot confidence 'l' tidak diekspor")

    m = json.load(open(f"{K}/meta.json"))
    cek(m["versi_kontrak"] == "3.0" and m["mode"] == "sintetis", "meta v3.0, mode sintetis")
    (W, N_), (E, _), (_, S), _ = m["png_bounds"]
    cek(W < E and S < N_, "png_bounds berurutan kiri-atas, kanan-atas, kanan-bawah, kiri-bawah")
    import matplotlib.image as mpimg
    img = mpimg.imread(f"{K}/perubahan.png")
    def piksel(img, px, py):
        lon, lat = px_ke_ll(px, py)
        c = int((lon - W) / (E - W) * img.shape[1]); r = int((N_ - lat) / (N_ - S) * img.shape[0])
        return img[r, c]
    pa_, pd_, pb_ = piksel(img, 300, 400), piksel(img, 1250, 500), piksel(img, 150, 150)
    cek(pa_[3] > 0 and pa_[0] > pa_[2], f"PNG: blob A MERAH dan terlihat")
    cek(pd_[3] == 0, "PNG: sawah luar gambut transparan")
    cek(pb_[3] == 0, "PNG: hutan tak berubah transparan")
    ipra, ipas = mpimg.imread(f"{K}/hv_pra.png"), mpimg.imread(f"{K}/hv_pasca.png")
    cek(piksel(ipra, 300, 400)[0] - piksel(ipas, 300, 400)[0] > 0.15, "blob A lebih gelap sesudah kebakaran")


LOKASI = [
    ("riau",    "Bengkalis, Riau (SYNTHETIC)",            101.95,  1.05, "EPSG:32648", ["Bengkalis", "Siak"]),
    ("kalteng", "Pulang Pisau, Central Kalimantan (SYNTHETIC)", 113.85, -2.20, "EPSG:32749", ["Pulang Pisau", "Katingan"]),
    ("merauke", "Merauke, South Papua (SYNTHETIC)",       140.30, -7.80, "EPSG:32754", ["Merauke", "Boven Digoel"]),
]
pilih = sys.argv[1:] or [l[0] for l in LOKASI]
for args in LOKASI:
    if args[0] in pilih:
        bangun_dan_uji(*args)

print("\n##### INDEKS MULTI-AOI #####")
jb.tulis_indeks(f"{ROOT}/keluaran")
idx = json.loads(open(f"{ROOT}/keluaran/aoi.json").read(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
def cek2(ok, pesan):
    print(("  LULUS  " if ok else "  GAGAL  ") + pesan)
    if not ok: SEMUA_GAGAL.append("indeks: " + pesan)
cek2(idx["versi_kontrak"] == "3.0", "aoi.json versi 3.0")
cek2(all(a["mode"] == "sintetis" for a in idx["aoi"]), "semua mode sintetis")
if len(idx["aoi"]) == 3:
    lat = {a["slug"]: (a["bbox"][1] + a["bbox"][3]) / 2 for a in idx["aoi"]}
    cek2(lat["riau"] > 0 > lat["kalteng"] > lat["merauke"], "lintang benar: riau utara, merauke paling selatan")

total = len(SEMUA_GAGAL)
print(f"\n{'SEMUA LULUS' if not total else str(total) + ' GAGAL:'}")
for g in SEMUA_GAGAL: print("   -", g)
sys.exit(1 if total else 0)
