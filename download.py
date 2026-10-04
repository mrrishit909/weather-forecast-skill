"""Step 1: daily maximum and minimum temperature and precipitation at twelve US airport weather stations, from NOAA's Global Historical Climatology Network - Daily (GHCN-Daily).

    ./venv/bin/python download.py   -> data/<station>.csv.gz (12)  data/ghcnd-stations.txt, results/pull.json

NOAA National Centers for Environmental Information, https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/<station>.csv.gz and ghcnd-stations.txt (version 3.35 of the readme). US government data; the readme asks for citation: Menne, M.J., I. Durre, R.S. Vose, B.E. Gleason and T.G. Houston, 2012,
"An overview of the Global Historical Climatology Network-Daily Database", Journal of Atmospheric and Oceanic Technology 29, 897-910, doi:10.1175/JTECH-D-11-00103.1; and the dataset itself, doi:10.7289/V5D21VHZ. One row per station, day and element
(TMAX, TMIN in tenths of a degree C, PRCP in tenths of a mm) with measurement, quality and source flags. NOAA rebuilds these files every day, so a later download can differ in the most recent months and in rare quality revisions; the analysis uses only 1991 to 2025,
and results/pull.json records the checksum of the files used. The files are 1.5 to 1.8 MB each and are fetched, never committed.
"""
import hashlib
import json
import ssl
import urllib.request
from datetime import date
from pathlib import Path

import certifi

HERE = Path(__file__).parent
BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
STATIONS = ["USW00094728", "USW00094846", "USW00024233", "USW00023183", "USW00012839", "USW00013874", "USW00014739", "USW00014922", "USW00023174", "USW00003927", "USW00022521", "USW00023050"]
FILES = {f"{s}.csv.gz": BASE + f"by_station/{s}.csv.gz" for s in STATIONS}
FILES["ghcnd-stations.txt"] = BASE + "ghcnd-stations.txt"
CTX = ssl.create_default_context(cafile=certifi.where())


def main():
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    info = {"pulled": date.today().isoformat(), "files": {}}
    for name, url in FILES.items():
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-portfolio (Rishit Raj Mathur)"}), context=CTX, timeout=600).read()
        (HERE / "data" / name).write_bytes(raw)
        info["files"][name] = {"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (HERE / "results" / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v["bytes"] for k, v in info["files"].items()}))


if __name__ == "__main__":
    main()
