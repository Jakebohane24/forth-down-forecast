"""Download team logo assets from URLs maintained in nflverse metadata."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import nfl_data_py as nfl


OUTPUT_DIR = Path("web/public/team-logos")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    teams = nfl.import_team_desc().drop_duplicates("team_abbr")
    for row in teams.itertuples():
        destination = OUTPUT_DIR / f"{row.team_abbr}.png"
        with urlopen(row.team_logo_espn, timeout=30) as response:
            destination.write_bytes(response.read())
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
