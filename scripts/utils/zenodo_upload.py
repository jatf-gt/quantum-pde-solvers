"""
Uploads a prepared deposit to Zenodo and leaves it unpublished for review.

Scope, deliberately
-------------------
This module creates a deposition, attaches its metadata and uploads its files.
It does **not** publish. Publication mints a DOI and is irreversible: a published
record's files can never be altered, only superseded by a new version. The final
step is therefore left to a person, in the Zenodo web interface, after reading
what was uploaded. The module prints the URL to review.

The token
---------
Read from the environment, never from an argument: a command line is recorded in
the shell history, in the process table and in this session's transcript, and a
Zenodo personal access token grants write access to every record its owner has.

    export ZENODO_TOKEN="..."            # bash
    $env:ZENODO_TOKEN = "..."            # PowerShell

The token is not written to disk, not echoed, and not included in any error this
module raises. Revoke it at zenodo.org/account/settings/applications once the
deposit is published.

Usage
-----
    python scripts/utils/zenodo_upload.py --package <dir> --sandbox
    python scripts/utils/zenodo_upload.py --package <dir>
    python scripts/utils/zenodo_upload.py --package <dir> --deposition 1234567

`--sandbox` targets sandbox.zenodo.org, which is a full copy of the service with
throwaway records and a separate token; it is the way to rehearse an upload
whose real counterpart cannot be undone. `--deposition` resumes an existing
draft rather than creating a second one — the correct response to an interrupted
upload, since a new invocation would otherwise leave an orphan draft behind.

References
----------
Zenodo REST API documentation: https://developers.zenodo.org/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]

LIVE_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"

# Generous: a 137 MB file over a domestic uplink is minutes, and a timeout that
# fires mid-upload leaves a partial file on the bucket that must be re-uploaded.
UPLOAD_TIMEOUT_S = 1800
API_TIMEOUT_S = 60

log = logging.getLogger("zenodo")


def _token() -> str:
    """
    Read the Zenodo personal access token from the environment.

    Returns
    -------
    str
        The token.

    Raises
    ------
    SystemExit
        Where `ZENODO_TOKEN` is unset or empty, with the instruction for setting
        it. Raised rather than returned so that no caller can proceed to an
        unauthenticated request that would fail obscurely.
    """
    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "ZENODO_TOKEN is not set. Create a personal access token at\n"
            "  https://zenodo.org/account/settings/applications/tokens/new/\n"
            "with the scopes deposit:write and deposit:actions, then export it:\n"
            '  export ZENODO_TOKEN="..."')
    return token


def _check(response: requests.Response, what: str) -> dict:
    """
    Raise on an unsuccessful API call, quoting the service's own message.

    Parameters
    ----------
    response : requests.Response
        The response to check.
    what : str
        Short description of the attempted operation, for the error.

    Returns
    -------
    dict
        The decoded JSON body, or an empty dict where the body is empty.

    Raises
    ------
    SystemExit
        On any non-2xx status. Zenodo reports field-level validation errors in
        the body, which are far more useful than the status alone, so the body
        is quoted in full.
    """
    if not response.ok:
        raise SystemExit(f"{what} failed: HTTP {response.status_code}\n"
                         f"{response.text[:2000]}")
    return response.json() if response.content else {}


def create_or_fetch(api: str, token: str, deposition_id: int | None) -> dict:
    """
    Create a new deposition, or fetch an existing draft to resume.

    Parameters
    ----------
    api : str
        API root, live or sandbox.
    token : str
        Personal access token.
    deposition_id : int or None
        Draft to resume; None creates one.

    Returns
    -------
    dict
        The deposition record, carrying `id` and `links`.
    """
    params = {"access_token": token}
    if deposition_id is not None:
        log.info("  Resuming deposition %d", deposition_id)
        return _check(requests.get(f"{api}/deposit/depositions/{deposition_id}",
                                   params=params, timeout=API_TIMEOUT_S),
                      "fetching the deposition")
    log.info("  Creating a new deposition")
    return _check(requests.post(f"{api}/deposit/depositions", params=params,
                                json={}, timeout=API_TIMEOUT_S),
                  "creating the deposition")


def set_metadata(api: str, token: str, deposition: dict, metadata: dict) -> dict:
    """
    Attach the deposit's metadata to the draft.

    Parameters
    ----------
    api : str
        API root.
    token : str
        Personal access token.
    deposition : dict
        The deposition record.
    metadata : dict
        Contents of `.zenodo.json`.

    Returns
    -------
    dict
        The updated deposition record.
    """
    return _check(requests.put(f"{api}/deposit/depositions/{deposition['id']}",
                               params={"access_token": token},
                               json={"metadata": metadata},
                               timeout=API_TIMEOUT_S),
                  "setting the metadata")


def upload_files(deposition: dict, token: str, files: list[Path]) -> None:
    """
    Upload each file to the deposition's bucket.

    The bucket API is used rather than the older `/files` endpoint: it streams a
    plain PUT of the file's bytes, has no 100 MB limit, and re-uploading the same
    filename replaces it, which makes an interrupted run safe to repeat.

    Parameters
    ----------
    deposition : dict
        The deposition record; its `links.bucket` is the upload target.
    token : str
        Personal access token.
    files : list of Path
        Files to upload, in order.

    Raises
    ------
    SystemExit
        Where the record carries no bucket link, or any upload fails.
    """
    bucket = (deposition.get("links") or {}).get("bucket")
    if not bucket:
        raise SystemExit("The deposition carries no bucket link; it may already "
                         "be published, which makes its files immutable.")
    for path in files:
        size_mb = path.stat().st_size / 1048576
        log.info("    %-28s %7.1f MB ...", path.name, size_mb)
        with open(path, "rb") as fh:
            _check(requests.put(f"{bucket}/{path.name}", data=fh,
                                params={"access_token": token},
                                timeout=UPLOAD_TIMEOUT_S),
                   f"uploading {path.name}")


def main() -> int:
    """
    Upload a prepared package, leaving the deposition unpublished.

    Returns
    -------
    int
        0 on success; 1 where the package directory is absent or empty.
    """
    ap = argparse.ArgumentParser(
        description="Upload a prepared deposit to Zenodo, without publishing it.")
    ap.add_argument("--package", type=Path, required=True,
                    help="Directory built by make_zenodo_package.py.")
    ap.add_argument("--metadata", type=Path, default=REPO_ROOT / ".zenodo.json",
                    help="Deposit metadata (default: the repository's .zenodo.json).")
    ap.add_argument("--sandbox", action="store_true",
                    help="Target sandbox.zenodo.org. Needs a sandbox token.")
    ap.add_argument("--deposition", type=int, default=None,
                    help="Resume this draft instead of creating a new one.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be uploaded and contact nothing.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.package.is_dir():
        log.error("  No package directory at %s. Run make_zenodo_package.py first.",
                  args.package)
        return 1
    files = sorted(p for p in args.package.iterdir() if p.is_file())
    if not files:
        log.error("  %s is empty.", args.package)
        return 1
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    api = SANDBOX_API if args.sandbox else LIVE_API
    total_mb = sum(p.stat().st_size for p in files) / 1048576

    log.info("=" * 78)
    log.info("  ZENODO UPLOAD%s%s", "  (sandbox)" if args.sandbox else "",
             "  (dry run)" if args.dry_run else "")
    log.info("=" * 78)
    log.info("  endpoint  : %s", api)
    log.info("  package   : %s", args.package)
    log.info("  files     : %d, %.1f MB", len(files), total_mb)
    log.info("  title     : %s", metadata.get("title", "<none>"))
    log.info("  licence   : %s", metadata.get("license", "<none>"))
    for path in files:
        log.info("    %-28s %7.1f MB", path.name, path.stat().st_size / 1048576)

    if args.dry_run:
        log.info("-" * 78)
        log.info("  Dry run: nothing uploaded, no token read.")
        return 0

    token = _token()
    log.info("-" * 78)
    deposition = create_or_fetch(api, token, args.deposition)
    log.info("  Deposition: %s", deposition["id"])
    deposition = set_metadata(api, token, deposition, metadata)
    log.info("  Metadata attached")
    log.info("  Uploading:")
    upload_files(deposition, token, files)

    html = (deposition.get("links") or {}).get("html", "")
    log.info("-" * 78)
    log.info("  Uploaded, NOT published. Review and publish here:")
    log.info("    %s", html or f"{api}/deposit/depositions/{deposition['id']}")
    log.info("")
    log.info("  Publishing mints the DOI and freezes the files permanently.")
    log.info("  Afterwards: put the DOI in results/README.md and CITATION.cff,")
    log.info("  and revoke the token at zenodo.org/account/settings/applications.")
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
