"""Pack a data directory into the archive scripts/fetch_data.py unpacks for deployment.

SQLite databases go through the backup API, so the archive is consistent even while the
API is running. Per-fork regression exports are local artifacts and are left out.
Usage: python scripts/pack_data.py <data-dir> <archive.tar.gz>
"""

import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

SKIP_TOP = {"exports"}
SKIP_SUFFIXES = (".db-wal", ".db-shm", ".db-journal")


def main(src: str, dest: str) -> None:
    root = Path(src)
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(dest, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if not path.is_file() or rel.parts[0] in SKIP_TOP or path.name.endswith(SKIP_SUFFIXES):
                continue
            source = path
            if path.suffix == ".db":
                source = Path(tmp) / "_".join(rel.parts)
                reader, writer = sqlite3.connect(path), sqlite3.connect(source)
                try:
                    reader.backup(writer)
                finally:
                    reader.close()
                    writer.close()
            archive.add(source, arcname=f"./{rel.as_posix()}")
    size = Path(dest).stat().st_size / 1e6
    print(f"packed {root} into {dest} ({size:.1f} MB)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
