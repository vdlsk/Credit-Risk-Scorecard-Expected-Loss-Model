"""
Step 0 - Download the raw Lending Club loan files (2007-2014).

Source: Lending Club's public issued-loan data, as packaged in the GitHub repo
kuhnrl30/LendingClubData (one R .rda file per issue year). About 60 MB in total.

    python src/get_data.py            -> data/raw/LendingClubData_2007.rda ... _2014.rda
"""
import subprocess
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://raw.githubusercontent.com/kuhnrl30/LendingClubData/master/data/LendingClubData_{y}.rda"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for y in range(2007, 2015):
        out = RAW / f"LendingClubData_{y}.rda"
        if out.exists():
            continue
        try:
            urllib.request.urlretrieve(URL.format(y=y), out)
        except Exception:
            # large files are stored with Git LFS on some mirrors; fall back to git
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/kuhnrl30/LendingClubData",
                            str(RAW / "_repo")], check=True)
            for f in (RAW / "_repo" / "data").glob("LendingClubData_20*.rda"):
                if 2007 <= int(f.stem[-4:]) <= 2014:
                    f.replace(RAW / f.name)
            break
        print("downloaded", out.name)


if __name__ == "__main__":
    main()
