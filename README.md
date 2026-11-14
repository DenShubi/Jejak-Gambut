# Jejak Bara — peat fire scars through the smoke, with NISAR

NASA Space Apps Challenge 2026 · *Dancing with the SARs* · Yogyakarta

**Live app:** https://<your-github-user>.github.io/jejak-bara/

During Indonesia's 2026 fire season, haze blinded optical satellites for weeks, and VIIRS thermal
sensors only catch fires that are burning — and not hidden under dense smoke — at the moment of
overpass. Jejak Bara uses NISAR's dual-polarization L-band radar, which sees through smoke, to map
burn scars on peat: the loss of HV volume scattering, cross-checked against the HV/HH ratio.
Detection is restricted to mapped peat hydrological units (KHG) from the start. Object-based computer vision
(SLIC superpixels, Multi-Otsu, shape descriptors) labels every scar as **confirmed** by VIIRS hotspots,
**hidden under smoke**, or **needs verification** (e.g. box-shaped excavator clearing), tracks which way it spread
from the first ignition, and measures how much burned area VIIRS missed under dense smoke.

![screenshot](screenshot.png)

## Data (all NASA unless noted)
- **NISAR L2 GCOV** (`NISAR_L2_GCOV_V1`, falling back to `_PROVISIONAL_V1` / `_BETA_V1`), layers HHHH and HVHV — NASA/ASF DAAC
- **VIIRS 375 m active fire** (`VIIRS_SNPP_SP`, `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`) — NASA FIRMS; nominal/high confidence only
- **MERRA-2 aerosol** (`M2TMNXAER`, `TOTEXTTAU`, AOD 550 nm) — NASA GES DISC
- **Optical context**: NASA GIBS / Worldview, VIIRS NOAA-20 corrected reflectance
- Peat Hydrological Units (KHG), Ministry of Environment and Forestry, Indonesia · regency boundaries, geoBoundaries

## Method (contract v3.0)
1. Stream only the AOI window of each GCOV granule. Every date is resampled onto the pre-fire grid
   (same track and frame), the equivalent of `post.rio.reproject_match(pre)`.
2. NaN-aware 5×5 speckle filter in **linear power**, then dB (avoids the −2.5 dB log-averaging bias).
3. **Computer vision, peat only:** a 3-channel change tensor (−ΔHV, ΔHH, −ΔHV/HH) is segmented into SLIC
   superpixels inside KHG; a 3-class **Multi-Otsu** threshold on area-weighted superpixel ΔHV (capped at −1.8 dB)
   separates canopy loss; morphological closing; polygons < 1 ha dropped.
4. **Shape:** rectangularity = area / minimum rotated rectangle. ≥ 0.78 = box-like (mechanical clearing).
5. **Three classes:** VIIRS hotspot within 375 m → *confirmed*; otherwise HV/HH ratio fell > 1 dB and organic shape →
   *hidden under smoke*; otherwise → *needs verification*.
6. **Tracking:** the centroid of the earliest-day hotspots is the ignition point; the vector to the scar centroid gives
   bearing, compass direction and reach. Smoke-hidden scars use the long axis (±180°). Every 12-day pass gives a
   cumulative burned-area curve and an expansion rate (ha/day).
7. **Smoke–hotspot coupling:** MERRA-2 AOD and mean FRP per scar; thermal blindness = hidden / (confirmed + hidden);
   a smoke × hotspot quadrant shows where VIIRS went blind.

No deep learning: there are no ground-truth labels for 2026, and training on VIIRS would make validation circular.

**Limitations.** Detection covers peat only (most 2026 burned area nationally was on mineral land), so our totals are
not comparable with national figures. "Hidden" scars are radar evidence, not proven fires. MERRA-2 AOD is ≈50 km
reanalysis. A compact ellipse already scores ≈0.785 rectangularity, so the 0.78 cut can misread round organic scars;
check every *verify* polygon visually. Grass and savanna burns give a weaker HV signal and are likely under-detected.

## Output files (`keluaran/<aoi>/`)
`terbakar.geojson` (scars, all v3 properties) · `arah_rambatan.geojson` (spread vectors) · `hotspot.geojson` ·
`gambut.geojson` · `statistik.json` · `deret_waktu.json` · `perubahan.png`, `hv_pra.png`, `hv_pasca.png` · `meta.json`.
`keluaran/aoi.json` indexes the areas; `meta.json` → `"mode": "nisar"` marks real results (`"sintetis"` = demo).

## Run it
- Processing: `pipeline/jejak_bara_colab.ipynb` in Google Colab (needs an Earthdata login and a free FIRMS MAP_KEY).
- Tests (no network): `python pipeline/uji_sintetis.py` then `python pipeline/uji_subset.py`.
- App: any static server, e.g. `python -m http.server`, then open http://localhost:8000 (`?aoi=kalteng` to deep-link).

## Use of AI
Parts of the processing code, tests and app were written with the help of Claude (Anthropic) and
reviewed by the team.

## License
MIT
