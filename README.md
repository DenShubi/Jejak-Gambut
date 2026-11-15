# Tilas

*Tilas*, from the Indonesian *napak tilas*: retracing where something went.

Peat burn-scar mapping for Indonesia's 2026 fire season from NISAR L-band GCOV.
VIIRS reports fire *points*; this pipeline turns 12-day NISAR passes into mapped *hectares* per regency, with growth over time,
spread direction, and a confidence class per scar.

**App:** https://denshubi.github.io/Jejak-Gambut/ (`?aoi=kalteng|merauke|riau`)
NASA Space Apps Challenge 2026 — *Dancing with the SARs* — Yogyakarta.

## Results

Detection is restricted to mapped peat (KHG). Areas are computed in UTM.

| AOI | NISAR track / frames | Passes | Canopy loss on peat | VIIRS-confirmed | Hidden¹ | Verify² | Lift³ |
|---|---|---|---|---|---|---|---|
| Central Kalimantan (Sebangau–Pulang Pisau) | 061 D / 089–090 | 8, 24 Jul – 28 Oct | 82,817 ha | 74,179 ha | 1,197 ha | 7,441 ha | 3.5× |
| South Papua (Merauke) | 124 A / 173–174 | 8, 16 Jul – 1 Nov | 22,012 ha | 20,565 ha | 628 ha | 819 ha | 2.1× |
| Riau (Bengkalis–Siak–Pelalawan) | 148 D / 087–089 | 5, 18 Jul – 16 Sep | 36,269 ha | 330 ha | 1,608 ha | 34,332 ha | n/a⁴ |

¹ No VIIRS hotspot within 375 m of the scar, but a VIIRS fire within 3 km or AOD ≥ 1.
² Box-shaped, HV/HH ratio unchanged, or no fire evidence nearby.
³ Share of random points inside scars within 375 m of a VIIRS fire, divided by the same share for random peat points outside scars
(Kalteng 73% vs 21%, Merauke 60% vs 28%). Hotspots are never used for detection, only for labelling.
⁴ 0.6% vs 0.2%: too few fires for the test to mean anything.

- **Kalteng:** burned area tripled in October (29,164 ha on 22 Sep → 81,994 ha on 28 Oct; +23,886 ha between 4 and 16 Oct).
  Largest regency Pulang Pisau (34,679 ha). Area-weighted spread direction 302° (NW).
- **Smoke:** MERRA-2 monthly AOD stayed at 0.2–0.54. VIIRS missed 1.6% (Kalteng) and 3.0% (Merauke) of fire-scar area.
  The smoke-blindness hypothesis we started with is not supported at these sites in 2026.
- **Riau** is a negative control: 36k ha of canopy loss, 433 peat hotspots. 17,909 ha show HV and HH dropping together (stem
  removal); 14,810 ha have no fire within 3 km. Consistent with acacia harvest and replanting. The pipeline flags it instead of
  calling it fire.
- **Merauke:** one box-shaped scar (25 ha) inside KHG. Food-estate clearing lies mostly outside the peat map, so it is out of scope.
- About half of peat hotspots fall outside detected scars (Kalteng 43% inside, Merauke 49%): surface and shrub fires that do
  not remove canopy are not detected.

National figures (2.88 Mha Nusantara Atlas, 202,004 ha official, July) cover all land cover nationwide and are not comparable.

## Data

| Dataset | Source | Use |
|---|---|---|
| NISAR L2 GCOV, HHHH + HVHV, dual-pol | NASA/ASF DAAC, `NISAR_L2_GCOV_PROVISIONAL_V1` (search also tries `_V1`, `_BETA_V1`) | detection, time series |
| VIIRS 375 m active fire, confidence n/h | NASA FIRMS: `VIIRS_SNPP_SP`, falling back to `VIIRS_SNPP_NRT` / `VIIRS_NOAA20_NRT` | labels, ignition point, FRP |
| MERRA-2 aerosol, `TOTEXTTAU` | NASA GES DISC, `M2TMNXAER` (monthly) | smoke load per scar |
| VIIRS NOAA-20 corrected reflectance | NASA GIBS / Worldview | optical context in the app |
| Peat Hydrological Units (KHG) | Ministry of Environment and Forestry, Indonesia | detection domain |
| Regency boundaries (ADM2) | geoBoundaries gbOpen | per-regency split |

NISAR provisional products are not rigorously validated. Hotspots: 1 May – 14 Nov 2026. All data retrieved 15 Nov 2026.

## Method

1. **Subset.** Download each GCOV granule, cut the AOI from `/science/LSAR/GCOV/grids/frequencyA`, multilook to 80 m by averaging
   linear power, mosaic frames of the same pass. Tracks are ranked by AOI coverage from CMR footprints; only one track and
   direction is used per AOI.
2. **Co-register.** Every date is resampled onto the pre-fire grid.
3. **Speckle.** NaN-aware 3×3 mean in linear power, then dB.
4. **Change tensor.** −ΔHV, ΔHH, −Δ(HV−HH). Multi-temporal: per pixel, the deepest HV drop across all passes after the pre date,
   so scars that regrow or get wet before the last pass are kept.
5. **Segmentation.** SLIC superpixels (~10 ha, compactness 0.1) on the full grid, clipped to KHG.
6. **Threshold.** 3-class Multi-Otsu on area-weighted superpixel ΔHV; the lower threshold is used, capped at −1.8 dB.
   Applied: Kalteng −1.80 (cap), Merauke −3.21, Riau −1.96. Morphological closing; polygons < 1 ha dropped.
7. **Shape.** Rectangularity = area / minimum rotated rectangle. Only applied to polygons ≥ 10 ha and ≥ 3 superpixels.
8. **Classes.**
   - *confirmed*: a VIIRS hotspot (n/h) within 375 m during the pre–post window.
   - *hidden*: no hotspot; HV/HH ratio fell > 1 dB; rectangularity < 0.78; fire context (VIIRS within 3 km or AOD ≥ 1).
   - *verify*: everything else.
9. **Tracking.** The ignition point is the centroid of the hotspots on the earliest date. Bearing and distance run from there to
   the scar centroid. Scars without hotspots get their long axis (±180°). Cumulative area per pass is built from superpixels
   crossing the threshold, first-seen date and expansion rate (ha/day) included.
10. **Accounting.** Per-regency hectares use `gpd.overlay`, so scars crossing a boundary are split.

No supervised model: there is no 2026 ground truth, and training on VIIRS would make the hotspot check circular.

## Limitations

- Peat only. Mineral-soil and savanna burns are out of scope.
- 80 m grid. Scars under a few hectares are missed or merged.
- Canopy-loss proxy. Surface fires that leave canopy intact are not detected; harvest and clearing are detected and must be
  separated by the class rules.
- AOD is monthly MERRA-2 (~0.5°), which averages out daily smoke peaks. October 2026 may not have been published at retrieval time.
- Merauke's last pass (1 Nov) has frame 174 only.
- Hidden and verify classes are candidates, not confirmed fires. No field validation.

## Repository

```
index.html                 app (MapLibre GL JS, no build step)
keluaran/aoi.json          AOI index, the only file the app reads first
keluaran/<aoi>/            terbakar.geojson, arah_rambatan.geojson, hotspot.geojson, gambut.geojson,
                           statistik.json, deret_waktu.json, meta.json, perubahan.png, hv_pra.png, hv_pasca.png
pipeline/jejak_bara.py     processing (single module)
pipeline/jejak_bara_colab.ipynb   end-to-end Colab notebook
pipeline/uji_sintetis.py   synthetic scenario tests (fire, smoke-hidden, flood, excavator box, regrowth, harvest, rice harvest)
pipeline/uji_subset.py     GCOV HDF5 subset path test
```

## Run

```
# tests, no network
python pipeline/uji_sintetis.py && python pipeline/uji_subset.py

# app
python -m http.server   # then open http://localhost:8000
```

Processing: open `pipeline/jejak_bara_colab.ipynb` in Colab with `jejak_bara.py` in `MyDrive/spaceapps_nisar/`.
Needs an Earthdata login (NISAR EULA accepted in ASF Vertex, GES DISC enabled) and a FIRMS MAP_KEY. One AOI takes about
25 min to download (16 granules) and under a minute to process.

## Use of AI

Code for the pipeline, tests, notebook and app was written with Claude (Anthropic). The team chose the problem, AOIs and data,
ran all processing, and reviewed the results. Several method changes came from checking real outputs: multi-temporal
detection, the shape-size rule, and the fire-context rule for the hidden class.

## License

MIT
