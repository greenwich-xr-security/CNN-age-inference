import argparse
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_BASE = "https://api.github.com"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sha256_file(onnx_path: Path, out_path: Path) -> Path:
    digest = _sha256_file(onnx_path)
    out_path.write_text(f"{digest}  {onnx_path.name}\n", encoding="utf-8")
    return out_path


def _infer_repo_from_git() -> str:
    try:
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Could not infer GitHub repo from git remote. Pass --repo owner/name.") from exc

    patterns = [
        r"^https://github\.com/([^/]+/[^/]+?)(?:\.git)?$",
        r"^git@github\.com:([^/]+/[^/]+?)(?:\.git)?$",
        r"^ssh://git@github\.com/([^/]+/[^/]+?)(?:\.git)?$",
    ]
    for pat in patterns:
        m = re.match(pat, remote)
        if m:
            return m.group(1)
    raise RuntimeError(
        f"Remote origin '{remote}' is not a recognized GitHub URL. Pass --repo owner/name."
    )


def _api_request(
    method: str,
    url: str,
    token: str,
    *,
    json_payload: dict | None = None,
    raw_body: bytes | None = None,
    content_type: str | None = None,
    accept: str = "application/vnd.github+json",
) -> tuple[int, dict | list | None]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "User-Agent": "cnn-age-inference-release-uploader",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body: bytes | None = None
    if json_payload is not None:
        body = json.dumps(json_payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw_body is not None:
        body = raw_body
        headers["Content-Type"] = content_type or "application/octet-stream"

    req = urllib.request.Request(url=url, method=method, headers=headers, data=body)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.getcode()
            text = resp.read().decode("utf-8")
            if not text:
                return status, None
            return status, json.loads(text)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {url} failed: {exc.code} {detail}") from exc


def _get_or_create_release(
    repo: str,
    tag: str,
    token: str,
    *,
    release_name: str | None = None,
    notes: str | None = None,
    target: str | None = None,
    draft: bool = False,
    prerelease: bool = False,
) -> dict:
    get_url = f"{API_BASE}/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}"
    try:
        _, data = _api_request("GET", get_url, token)
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected response while fetching release.")
        return data
    except RuntimeError as exc:
        if " 404 " not in str(exc):
            raise

    payload = {
        "tag_name": tag,
        "name": release_name or tag,
        "body": notes or "",
        "draft": draft,
        "prerelease": prerelease,
    }
    if target:
        payload["target_commitish"] = target
    create_url = f"{API_BASE}/repos/{repo}/releases"
    _, created = _api_request("POST", create_url, token, json_payload=payload)
    if not isinstance(created, dict):
        raise RuntimeError("Unexpected response while creating release.")
    return created


def _delete_asset(repo: str, asset_id: int, token: str) -> None:
    url = f"{API_BASE}/repos/{repo}/releases/assets/{asset_id}"
    _api_request("DELETE", url, token)


def _upload_asset(
    release: dict,
    asset_path: Path,
    token: str,
) -> dict:
    upload_url_tmpl = str(release.get("upload_url", ""))
    if not upload_url_tmpl:
        raise RuntimeError("Release response did not include upload_url.")
    upload_url = upload_url_tmpl.split("{", 1)[0]
    query = urllib.parse.urlencode({"name": asset_path.name})
    url = f"{upload_url}?{query}"
    ctype = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
    payload = asset_path.read_bytes()
    _, data = _api_request(
        "POST",
        url,
        token,
        raw_body=payload,
        content_type=ctype,
        accept="application/vnd.github+json",
    )
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected upload response for {asset_path.name}.")
    return data


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload ONNX and model documentation assets to a GitHub release."
    )
    p.add_argument("--repo", type=str, default=None, help="GitHub repo as owner/name. Defaults to origin remote.")
    p.add_argument("--tag", type=str, required=True, help="Release tag (created if missing).")
    p.add_argument("--release-name", type=str, default=None, help="Release title (default: tag).")
    p.add_argument("--notes", type=str, default=None, help="Release body text when creating a release.")
    p.add_argument("--target", type=str, default=None, help="Target commitish/branch when creating a release.")
    p.add_argument("--draft", action="store_true", help="Create release as draft when new.")
    p.add_argument("--prerelease", action="store_true", help="Create release as prerelease when new.")
    p.add_argument("--token", type=str, default=None, help="GitHub token (fallback: GITHUB_TOKEN env var).")
    p.add_argument("--onnx-path", type=str, required=True, help="Path to ONNX model to upload.")
    p.add_argument("--model-card-path", type=str, default=None, help="Path to model_card.md (default: ONNX folder).")
    p.add_argument(
        "--deployment-sheet-path",
        type=str,
        default=None,
        help="Path to deployment_sheet.json (default: ONNX folder).",
    )
    p.add_argument("--sha256-path", type=str, default=None, help="Path for checksum file (default: <onnx>.sha256).")
    p.add_argument(
        "--skip-generate-sha256",
        action="store_true",
        help="Do not generate checksum file automatically.",
    )
    p.add_argument(
        "--overwrite-assets",
        action="store_true",
        help="Delete and replace release assets with same filenames.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    token = args.token or os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Missing token. Provide --token or set GITHUB_TOKEN.")

    repo = args.repo or _infer_repo_from_git()
    onnx_path = Path(args.onnx_path).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    model_card_path = (
        Path(args.model_card_path).expanduser().resolve()
        if args.model_card_path
        else onnx_path.parent / "model_card.md"
    )
    deployment_sheet_path = (
        Path(args.deployment_sheet_path).expanduser().resolve()
        if args.deployment_sheet_path
        else onnx_path.parent / "deployment_sheet.json"
    )

    missing = [p for p in (model_card_path, deployment_sheet_path) if not p.is_file()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Required model docs missing: {names}")

    sha256_path = (
        Path(args.sha256_path).expanduser().resolve()
        if args.sha256_path
        else Path(f"{onnx_path}.sha256")
    )
    if not args.skip_generate_sha256:
        _write_sha256_file(onnx_path, sha256_path)
    elif not sha256_path.is_file():
        raise FileNotFoundError(f"SHA256 file not found (and generation disabled): {sha256_path}")

    release = _get_or_create_release(
        repo=repo,
        tag=args.tag,
        token=token,
        release_name=args.release_name,
        notes=args.notes,
        target=args.target,
        draft=args.draft,
        prerelease=args.prerelease,
    )
    print(f"[release] Using release: {release.get('html_url')}")

    existing_assets = {asset.get("name"): asset for asset in release.get("assets", [])}
    to_upload = [onnx_path, model_card_path, deployment_sheet_path, sha256_path]
    for asset_path in to_upload:
        existing = existing_assets.get(asset_path.name)
        if existing:
            if not args.overwrite_assets:
                raise RuntimeError(
                    f"Asset '{asset_path.name}' already exists. Re-run with --overwrite-assets."
                )
            _delete_asset(repo, int(existing["id"]), token)
            print(f"[release] Deleted existing asset: {asset_path.name}")
        uploaded = _upload_asset(release, asset_path, token)
        print(f"[release] Uploaded: {asset_path.name} -> {uploaded.get('browser_download_url')}")

    print("[release] Completed asset upload.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[release] ERROR: {exc}", file=sys.stderr)
        raise
