#!/usr/bin/env python3
"""
ibtracs_to_custom_format.py
============================
Converts IBTrACS v04r01 NetCDF to the 12-column ASCII format
expected by getTrajectories_direct() in plotting_TCs_custom_memory.py.

Output column layout (space-separated, 12 columns):
  [0]  storm_id   : unique storm identifier  (e.g. 1979001N10120)
  [1]  year       : 4-digit year
  [2]  month      : month (1-12)
  [3]  day        : day (1-31)
  [4]  hour       : hour (0-23)
  [5]  minute     : minute (always 0 for IBTrACS)
  [6]  step       : time-step index within this storm (0,1,2,...)
  [7]  lon        : longitude [0,360)  ← reader converts >180 internally
  [8]  lat        : latitude  (-90,90)
  [9]  slp        : minimum central pressure [Pa]  (mb × 100)
                    NOTE: getTrajectories_direct divides by 100 to get hPa
  [10] wind       : maximum sustained wind speed [kts] (wmo_wind, fallback usa_wind)
  [11] sshs       : Saffir-Simpson category from IBTrACS usa_sshs (wind-based)
                    Values: -5=unknown -4=post-tropical -3=disturbance
                            -2=subtropical -1=TD 0=TS 1=Cat1 ... 5=Cat5
                    -15 if missing

Filter applied:
  - year in [start_year, end_year]   (default 1979-2014)
  - at least MIN_VALID_STEPS valid timesteps with position + (wind or pressure)
  - nature flag: all natures kept by default (FILTER_NATURE=False)
    → set FILTER_NATURE=True to keep only TS and DS

Usage:
  python ibtracs_to_custom_format.py \
      --input  /path/to/IBTrACS.ALL.v04r01.nc \
      --output /path/to/ibtracs_1979_2014_custom.txt \
      --start  1979 --end 2014

Requirements:
  pip install netCDF4 numpy
"""

import argparse
import sys
import os
import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("ERROR: netCDF4 not installed.  Run:  pip install netCDF4")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR   = 2014

# Keep only purely tropical systems?
# True  → drop ET / EX / SS / NR timesteps (nature != 'TS' and != 'DS')
# False → keep every timestep (consistent with TE which detects all lows)
FILTER_NATURE = False

# Minimum number of valid timesteps a storm must have to be written
MIN_VALID_STEPS = 4   # ~12 hours at 3-hourly resolution

# Fill values used in IBTrACS
IBTRACS_FILL_WIND = -9999
IBTRACS_FILL_PRES = -9999
IBTRACS_FILL_SSHS = -15


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def decode_bytes(arr):
    """Convert char array (bytes) from netCDF4 to a list of Python strings."""
    if arr.ndim == 1:
        return b"".join(arr.filled(b'')).decode('utf-8', errors='replace').strip()
    out = []
    for row in arr:
        out.append(b"".join(row.filled(b'')).decode('utf-8', errors='replace').strip())
    return out


def parse_iso_time(iso_bytes):
    """
    Parse iso_time variable (shape: (ndatetimes, niso)).
    Returns list of (year, month, day, hour) tuples or None for missing.
    ISO format: 'YYYY-MM-DD HH:MM:SS'
    """
    times = []
    for t in iso_bytes:
        s = b"".join(t.filled(b' ')).decode('utf-8', errors='replace').strip()
        if not s or s.startswith('\x00') or len(s) < 10:
            times.append(None)
        else:
            try:
                year  = int(s[0:4])
                month = int(s[5:7])
                day   = int(s[8:10])
                hour  = int(s[11:13]) if len(s) > 10 else 0
                times.append((year, month, day, hour))
            except (ValueError, IndexError):
                times.append(None)
    return times


# ─────────────────────────────────────────────
# MAIN CONVERTER
# ─────────────────────────────────────────────
def convert(input_nc, output_txt, start_year, end_year):
    print(f"\n{'='*60}")
    print(f"IBTrACS → custom format converter")
    print(f"  Input  : {input_nc}")
    print(f"  Output : {output_txt}")
    print(f"  Period : {start_year}–{end_year}")
    print(f"{'='*60}\n")

    ds = nc.Dataset(input_nc, 'r')

    # ── dimensions ──────────────────────────────────────────
    nstorms    = ds.dimensions['storm'].size
    ndatetimes = ds.dimensions['date_time'].size
    print(f"Dataset: {nstorms} storms × {ndatetimes} time steps")

    # ── load variables ───────────────────────────────────────
    print("Loading variables …")
    iso_time_var = ds.variables['iso_time']    # (storm, date_time, niso)
    lat_var      = ds.variables['lat']          # (storm, date_time)
    lon_var      = ds.variables['lon']          # (storm, date_time)
    nature_var   = ds.variables['nature']       # (storm, date_time, char2)

    # Wind: prefer wmo_wind, fall back to usa_wind
    if 'wmo_wind' in ds.variables:
        wind_var = ds.variables['wmo_wind']
        wind_src = 'wmo_wind'
    else:
        wind_var = ds.variables['usa_wind']
        wind_src = 'usa_wind'

    # Pressure: prefer wmo_pres, fall back to usa_pres
    if 'wmo_pres' in ds.variables:
        pres_var = ds.variables['wmo_pres']
        pres_src = 'wmo_pres'
    else:
        pres_var = ds.variables['usa_pres']
        pres_src = 'usa_pres'

    # Saffir-Simpson: usa_sshs (wind-based, official NOAA definition)
    if 'usa_sshs' in ds.variables:
        sshs_var = ds.variables['usa_sshs']
        sshs_available = True
        print(f"  SSHS source : usa_sshs (wind-based, official)")
    else:
        sshs_available = False
        print(f"  SSHS source : NOT AVAILABLE → will write -15 for all timesteps")

    print(f"  Wind source : {wind_src}")
    print(f"  Pres source : {pres_src}")

    # ── write output ─────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_txt)), exist_ok=True)

    storms_written        = 0
    points_written        = 0
    storms_skipped_year   = 0
    storms_skipped_nodata = 0

    with open(output_txt, 'w') as fout:
        # Header — skipped by getTrajectories_direct (contains 'track_id')
        fout.write("# track_id year month day hour minute step "
                   "lon lat slp_Pa wind_kts sshs\n")

        for i in range(nstorms):

            if i % 2000 == 0:
                print(f"  Processing storm {i}/{nstorms} …", flush=True)

            # ── parse time for this storm ──────────────────
            iso_bytes_storm = iso_time_var[i]        # (ndatetimes, niso)
            times = parse_iso_time(iso_bytes_storm)  # list of (y,m,d,h) or None

            # Quick year filter on first valid timestep
            first_valid_year = None
            for t in times:
                if t is not None:
                    first_valid_year = t[0]
                    break
            if first_valid_year is None:
                storms_skipped_nodata += 1
                continue
            if not (start_year <= first_valid_year <= end_year):
                storms_skipped_year += 1
                continue

            # ── load per-storm arrays ──────────────────────
            lat_s  = lat_var[i, :].filled(-9999.0)
            lon_s  = lon_var[i, :].filled(-9999.0)
            wind_s = wind_var[i, :].filled(IBTRACS_FILL_WIND)
            pres_s = pres_var[i, :].filled(IBTRACS_FILL_PRES)

            if sshs_available:
                sshs_s = sshs_var[i, :].filled(IBTRACS_FILL_SSHS)
            else:
                sshs_s = np.full(ndatetimes, IBTRACS_FILL_SSHS)

            if FILTER_NATURE:
                nature_s_raw = nature_var[i, :, :]  # (ndatetimes, 2)

            # ── collect valid rows ─────────────────────────
            rows = []
            for j in range(ndatetimes):
                t = times[j]
                if t is None:
                    continue
                year, month, day, hour = t

                # Per-timestep year guard (storms spanning Jan boundary)
                if not (start_year <= year <= end_year):
                    continue

                # Missing position → skip
                if lat_s[j] <= -9000 or lon_s[j] <= -9000:
                    continue

                lat_j = float(lat_s[j])
                lon_j = float(lon_s[j])

                # ── nature filter (optional) ───────────────
                if FILTER_NATURE:
                    nat = b"".join(
                        nature_s_raw[j].filled(b' ')
                    ).decode('utf-8', errors='replace').strip()
                    if nat not in ('TS', 'DS', ''):
                        continue

                # ── wind ──────────────────────────────────
                wind_j = float(wind_s[j])
                if wind_j <= 0 or wind_j == IBTRACS_FILL_WIND:
                    wind_j = -9999.0

                # ── pressure (mb → Pa) ─────────────────────
                pres_j = float(pres_s[j])
                if pres_j <= 0 or pres_j == IBTRACS_FILL_PRES:
                    pres_j = -9999.0
                else:
                    pres_j = pres_j * 100.0  # mb → Pa

                # Require at least wind OR pressure to be valid
                if wind_j == -9999.0 and pres_j == -9999.0:
                    continue

                # ── Saffir-Simpson (wind-based, from IBTrACS) ──
                sshs_j = int(sshs_s[j])
                # -15 = missing in IBTrACS fill convention

                # Longitude: keep [0, 360) — reader handles >180 internally
                lon_out = lon_j if lon_j >= 0 else lon_j + 360.0

                rows.append((year, month, day, hour,
                             lon_out, lat_j, pres_j, wind_j, sshs_j))

            if len(rows) < MIN_VALID_STEPS:
                storms_skipped_nodata += 1
                continue

            # ── resolve storm_id from dataset ─────────────
            storm_id_final = None
            for sid_name in ('sid', 'serial_id'):
                if sid_name in ds.variables:
                    sid_chars = ds.variables[sid_name][i, :]
                    sid_str = b"".join(sid_chars.filled(b'')).decode(
                        'utf-8', errors='replace').strip().replace(' ', '_')
                    if sid_str:
                        storm_id_final = sid_str
                        break
            if storm_id_final is None:
                storm_id_final = f"IBT{i:06d}"

            # ── write rows ─────────────────────────────────
            for step, row in enumerate(rows):
                (year, month, day, hour,
                 lon_out, lat_j, pres_j, wind_j, sshs_j) = row

                fout.write(
                    f"{storm_id_final} "
                    f"{year:4d} {month:2d} {day:2d} {hour:2d} 0 {step:4d} "
                    f"{lon_out:9.4f} {lat_j:8.4f} "
                    f"{pres_j:10.1f} {wind_j:7.1f} "
                    f"{sshs_j}\n"
                )
                points_written += 1

            storms_written += 1

    ds.close()

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Storms written          : {storms_written:,}")
    print(f"  Points written          : {points_written:,}")
    print(f"  Storms skipped (year)   : {storms_skipped_year:,}")
    print(f"  Storms skipped (no data): {storms_skipped_nodata:,}")
    print(f"  Output file             : {output_txt}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert IBTrACS NetCDF to TempestExtremes-compatible ASCII format"
    )
    parser.add_argument('--input',  required=True,
                        help='Path to IBTrACS .nc file (e.g. IBTrACS.ALL.v04r01.nc)')
    parser.add_argument('--output', required=True,
                        help='Path for output ASCII file')
    parser.add_argument('--start',  type=int, default=DEFAULT_START_YEAR,
                        help=f'Start year (default {DEFAULT_START_YEAR})')
    parser.add_argument('--end',    type=int, default=DEFAULT_END_YEAR,
                        help=f'End year (default {DEFAULT_END_YEAR})')
    args = parser.parse_args()

    convert(args.input, args.output, args.start, args.end)


if __name__ == '__main__':
    main()