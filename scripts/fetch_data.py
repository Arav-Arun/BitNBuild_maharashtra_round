"""Download and unpack the recorded dataset archive into a data directory.

Used by the Dockerfile so the deployed image carries the (git-ignored) data/.
Usage: python scripts/fetch_data.py <url> <dest-dir>
"""

import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def main(url: str, dest: str) -> None:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        print(f"downloading {url}", flush=True)
        with urllib.request.urlopen(url) as response:
            while chunk := response.read(1 << 20):
                tmp.write(chunk)
        tmp.flush()
        with tarfile.open(tmp.name) as archive:
            archive.extractall(out, filter="data")
    print(f"unpacked into {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
