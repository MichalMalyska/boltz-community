"""Download and extract a pinned commit of uplifting-biomolecular-modeling.

Used as an image-build step for torch_cuda_benchmark.py. Deliberately avoids
needing `git` (or `curl`/`tar`) installed in the container — some Modal
workspaces are still on an older/legacy Image Builder version pinned to an EOL
Debian snapshot whose `bullseye-security` packages (git included) have since
rotated out of the mirror and 404 on `apt-get install`. Python's stdlib
(urllib + tarfile) needs nothing from apt.

Usage: python3 _fetch_opt_core.py <commit-sha> <dest-dir>
"""

import pathlib
import sys
import tarfile
import urllib.request

REPO = "anthropics/uplifting-biomolecular-modeling"


def main() -> None:
    commit, dest = sys.argv[1], sys.argv[2]
    dest_path = pathlib.Path(dest)
    url = f"https://github.com/{REPO}/archive/{commit}.tar.gz"
    extract_root = dest_path.resolve().parent

    with (
        urllib.request.urlopen(url) as resp,
        tarfile.open(fileobj=resp, mode="r|gz") as tar,
    ):
        tar.extractall(extract_root, filter="data")

    repo_name = REPO.split("/")[1]
    extracted_dir = extract_root / f"{repo_name}-{commit}"
    extracted_dir.rename(dest_path)
    print(f"Extracted {url} to {dest_path}")


if __name__ == "__main__":
    main()
