#!/usr/bin/env python3
"""Extract only images referenced by the official split files from the archive."""

import argparse
import sqlite3
import tarfile
from pathlib import Path, PurePosixPath

SPLIT_NAMES = {"train_F.txt", "train_M.txt", "test_F.txt", "test_M.txt"}


def safe_destination(root: Path, member_name: str) -> Path:
    relative = Path(*PurePosixPath(member_name).parts)
    destination = (root / relative).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"unsafe archive member: {member_name}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    split_members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive:
            if Path(member.name).name in SPLIT_NAMES:
                split_members[Path(member.name).name] = member
    missing = SPLIT_NAMES - split_members.keys()
    if missing:
        raise RuntimeError(f"archive is missing split files: {sorted(missing)}")

    selected: set[str] = set()
    prefixes = {str(PurePosixPath(member.name).parent) for member in split_members.values()}
    if len(prefixes) != 1:
        raise RuntimeError(f"split files have inconsistent roots: {sorted(prefixes)}")
    archive_prefix = prefixes.pop()
    dataset_root = safe_destination(args.output, archive_prefix)
    with tarfile.open(args.archive, "r:gz") as archive:
        for split_name, member in split_members.items():
            prefix = str(PurePosixPath(member.name).parent)
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read {member.name}")
            destination = safe_destination(args.output, member.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(handle.read())
            for raw in destination.read_text().splitlines():
                # Female lists include an undocumented trailing integer; the
                # first whitespace-delimited field is the image path.
                relative = raw.strip().split()[0].replace("\\", "/").lstrip("./")
                if relative:
                    selected.add(str(PurePosixPath(prefix) / relative))

    database_path = dataset_root / "portraits.sqlite3"
    database_path.unlink(missing_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE images (path TEXT PRIMARY KEY, data BLOB NOT NULL)")
    extracted = 0
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive:
            normalized = str(PurePosixPath(member.name))
            if normalized not in selected:
                continue
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            relative = str(PurePosixPath(member.name).relative_to(archive_prefix))
            connection.execute(
                "INSERT INTO images(path, data) VALUES (?, ?)",
                (relative, source.read()),
            )
            extracted += 1
            if extracted % 1000 == 0:
                connection.commit()
                print(f"Stored {extracted}/{len(selected)} images", flush=True)
    connection.commit()
    connection.close()
    if extracted != len(selected):
        raise RuntimeError(f"extracted {extracted} of {len(selected)} referenced images")
    print(f"Stored {extracted} official images in {database_path}")


if __name__ == "__main__":
    main()
