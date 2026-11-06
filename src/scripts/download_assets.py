#!/usr/bin/env python3
"""Download a bounded, resumable public StreamGaze/Qwen pilot into this workspace.

Without --execute, print an offline plan only. No login, paid API, repository
code execution, automatic archive extraction, or whole-dataset download.
"""
import argparse
import hashlib
import json
import re
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data" / "external" / "download_state"
DATASET_REPO = "daeunni/StreamGaze"
MODEL_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
QA_FILES = [
    "past_gaze_sequence_matching.json",
    "past_non_fixated_object_identification.json",
    "past_object_transition_prediction.json",
    "past_scene_recall.json",
    "present_future_action_prediction.json",
    "present_object_attribute_recognition.json",
    "present_object_identification_easy.json",
    "present_object_identification_hard.json",
    "proactive_gaze_triggered_alert.json",
    "proactive_object_appearance_alert.json",
]
MODEL_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "chat_template.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "model.safetensors.index.json",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    "LICENSE", "README.md",
]
VIDEO_FILES = ["videos_egtea_original.tar", "videos_egtea_viz.tar"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def asset_groups(selection):
    """Explicit filenames: never a wildcard over this 215 GB dataset."""
    groups = []
    if selection in {"pilot", "annotations", "videos"}:
        files = ["README.md", "metadata/egtea.csv"] + ["qa/" + f for f in QA_FILES]
        if selection in {"pilot", "videos"}:
            files += VIDEO_FILES
        groups.append({"repo_id": DATASET_REPO, "repo_type": "dataset",
                       "destination": "data/external/streamgaze", "files": files})
    if selection in {"pilot", "model"}:
        groups.append({"repo_id": MODEL_REPO, "repo_type": "model",
                       "destination": "data/models/Qwen2.5-VL-3B-Instruct",
                       "files": MODEL_FILES.copy()})
    return groups


def resolve_group(group, info):
    siblings = {item.rfilename: item for item in info.siblings}
    records = []
    for filename in group["files"]:
        if filename not in siblings:
            raise ValueError(f"Publisher changed file layout: missing {filename}; review before downloading")
        item = siblings[filename]
        if item.size is None:
            raise ValueError(f"No size available for {filename}; cannot enforce download bound")
        records.append({"filename": filename, "size": item.size,
                        "publisher_sha256": item.lfs.sha256 if item.lfs else None})
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise ValueError("Hub did not return a full immutable revision")
    return {**group, "revision": info.sha, "records": records}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, record):
    if path.stat().st_size != record["size"]:
        raise ValueError(f"Size mismatch for {record['filename']}")
    digest = sha256(path)
    if record["publisher_sha256"] and digest != record["publisher_sha256"]:
        raise ValueError(f"Checksum mismatch for {record['filename']}; retain file for inspection")
    return {"size": path.stat().st_size, "sha256": digest}


def check_budget(groups, max_gb):
    total = sum(r["size"] for g in groups for r in g["records"])
    if total > max_gb * 1e9:
        raise ValueError(f"Selected assets total {total / 1e9:.2f} GB, above --max-gb {max_gb:g}")
    # Conservative allowance: complete selected assets plus 5 GB headroom.
    missing = sum(r["size"] for g in groups for r in g["records"]
                  if not (ROOT / g["destination"] / r["filename"]).exists())
    if shutil.disk_usage(ROOT).free < missing + 5e9:
        raise ValueError(f"Insufficient disk space; need at least {missing / 1e9 + 5:.2f} GB free")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", choices=["pilot", "annotations", "model", "videos"], default="pilot")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Actually download selected files")
    action.add_argument("--plan", action="store_true", help="Print offline plan only (default)")
    parser.add_argument("--max-gb", type=float, default=32, help="Hard upper bound on selected remote bytes")
    args = parser.parse_args()
    if args.max_gb <= 0:
        parser.error("--max-gb must be positive")
    groups = asset_groups(args.assets)
    estimates = {"pilot": 25.81, "annotations": 0.0062, "model": 7.52, "videos": 18.29}
    if not args.execute:
        print(json.dumps({"status": "offline_plan_no_download", "assets": args.assets,
                          "estimated_gb": estimates[args.assets], "groups": groups,
                          "note": "Estimates from publisher file pages, not a transfer measurement. "
                                  "Full EGTEA archives are needed; individual clips are not separate Hub files. "
                                  "No raw gaze coordinates are promised by this bundle."}, indent=2))
        return 0

    status_path = STATE / f"{args.assets}_status.json"
    lock_path = STATE / f"{args.assets}_lock.json"
    status = {"started_utc": utc_now(), "assets": args.assets, "status": "starting", "files": []}
    write_json(status_path, status)
    try:
        # Fast, explicit failure inside network-restricted agent sessions.
        socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)
        from huggingface_hub import HfApi, hf_hub_download
        # Public files only; do not use cached credentials implicitly.
        api = HfApi(endpoint="https://huggingface.co", token=False)
        if lock_path.exists():
            locked = json.loads(lock_path.read_text())
            if locked["requested_groups"] != groups:
                raise ValueError("Existing lock does not match asset selection; retain it and review")
            resolved = locked["resolved_groups"]
        else:
            resolved = [resolve_group(g, api.repo_info(g["repo_id"], repo_type=g["repo_type"],
                                                       files_metadata=True, timeout=15)) for g in groups]
            check_budget(resolved, args.max_gb)
            write_json(lock_path, {"created_utc": utc_now(), "requested_groups": groups,
                                   "resolved_groups": resolved})
        total = check_budget(resolved, args.max_gb)
        print(f"Selected {total / 1e9:.2f} GB at pinned revisions; interrupted downloads can be resumed.", flush=True)
        status["status"] = "downloading"
        status["selected_bytes"] = total
        write_json(status_path, status)
        for group in resolved:
            for record in group["records"]:
                filename = record["filename"]
                print(f"Downloading/checking {group['repo_id']}/{filename}", flush=True)
                local = Path(hf_hub_download(
                    repo_id=group["repo_id"], repo_type=group["repo_type"], filename=filename,
                    revision=group["revision"], local_dir=ROOT / group["destination"],
                    endpoint="https://huggingface.co", token=False, etag_timeout=15))
                checked = check_file(local, record)
                status["files"].append({"repo_id": group["repo_id"], "revision": group["revision"],
                                        "filename": filename, "path": str(local), **checked})
                write_json(status_path, status)
        status["status"] = "downloaded_and_checked"
        status["finished_utc"] = utc_now()
        status["note"] = "Assets downloaded only. Archives remain unextracted; no VideoLLM accuracy run yet."
        write_json(status_path, status)
        print(f"Download complete. Provenance: {status_path}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        status["status"] = "network_blocked" if isinstance(exc, socket.gaierror) else "incomplete"
        status["error_type"] = type(exc).__name__
        # Do not persist HTTP exceptions/headers/URLs that may contain credentials.
        status["finished_utc"] = utc_now()
        write_json(status_path, status)
        if isinstance(exc, socket.gaierror):
            print("Cannot resolve huggingface.co. This process has no working DNS/network access. "
                  "Run the same command from a network-enabled Terminal.", file=sys.stderr)
        elif isinstance(exc, ValueError):
            print(str(exc), file=sys.stderr)
        else:
            print(f"Download stopped ({type(exc).__name__}); completed/partial files are retained. "
                  "Re-run to resume. Check network, access, and disk space.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
