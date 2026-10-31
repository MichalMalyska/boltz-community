"""Download one subdirectory of a pinned commit of uplifting-biomolecular-modeling.

Used as an image-build step for torch_cuda_benchmark.py. Deliberately avoids
needing `git` (or `curl`/`tar`) installed in the container — some Modal
workspaces are still on an older/legacy Image Builder version pinned to an EOL
Debian snapshot whose `bullseye-security` packages (git included) have since
rotated out of the mirror and 404 on `apt-get install`. Python's stdlib
(urllib + tarfile) needs nothing from apt.

Only the requested subdirectory is extracted (streamed, member by member, with
the repo-root/subdir prefix stripped) rather than the whole archive — the
full `uplifting-biomolecular-modeling` repo is ~1.1GB at this commit, while
`common/opt_core` alone (everything actually needed) is ~300MB, so this keeps
every built image a few hundred MB smaller and a bit faster to build.

Usage: python3 _fetch_opt_core.py <commit-sha> <subdir> <dest-dir>
e.g.:  python3 _fetch_opt_core.py f4f62fa... common/opt_core /root/opt_core
"""

import pathlib
import sys
import tarfile
import urllib.request

REPO = "anthropics/uplifting-biomolecular-modeling"


def main() -> None:
    commit, subdir, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    dest_path = pathlib.Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    repo_name = REPO.split("/")[1]
    prefix = f"{repo_name}-{commit}/{subdir.strip('/')}/"
    url = f"https://github.com/{REPO}/archive/{commit}.tar.gz"

    extracted = 0
    with (
        urllib.request.urlopen(url) as resp,
        tarfile.open(fileobj=resp, mode="r|gz") as tar,
    ):
        # Streaming mode (r|gz) can't seek, so each member must be inspected
        # and extracted (or skipped) before the next one is read off the wire.
        for member in tar:
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix) :]
            if not member.name:
                continue
            tar.extract(member, dest_path, filter="data")
            extracted += 1

    if extracted == 0:
        msg = f"No files found under {subdir!r} in {url} — bad commit or subdir?"
        raise RuntimeError(msg)

    print(f"Extracted {extracted} files from {subdir!r} ({url}) to {dest_path}")


if __name__ == "__main__":
    main()
