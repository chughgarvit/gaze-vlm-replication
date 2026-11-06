#!/usr/bin/env python3
"""Audit licensed Ego4D metadata before selecting a gaze/annotation video pilot.

No network, credentials, license acceptance, chronological splitting, or model
inference. UID presence is NOT proof that an annotation supplies suitable labels.
Empty/insufficient selections never produce a UID file that could trigger an
unfiltered official-CLI download.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def annotation_video_refs(data, known):
    """Support video_uid fields and narration dictionaries keyed by video UID."""
    found = set()
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            value = item.get("video_uid")
            if isinstance(value, str) and value in known:
                found.add(value)
            found.update(key for key in item if key in known)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


def inspect_metadata(metadata, annotations, max_videos=12, min_subjects=3):
    videos = metadata.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Expected publisher ego4d.json with a nonempty videos list")
    known = {v["video_uid"] for v in videos}
    if len(known) != len(videos):
        raise ValueError("Duplicate canonical video UIDs in metadata")
    intersections = {name: annotation_video_refs(data, known) for name, data in annotations.items()}
    referenced = set().union(*intersections.values()) if intersections else set()
    gaze = [v for v in videos if v.get("has_gaze") is True]
    rows, grouped = [], defaultdict(list)
    unknown_identity = 0
    for video in gaze:
        participant = video.get("fb_participant_id")
        identified = type(participant) is int and participant >= 0
        if not identified:
            unknown_identity += 1
        source, origin = video.get("video_source"), video.get("origin_video_id")
        distinct_origin_known = bool(source and origin)
        uid = video["video_uid"]
        row = {"video_uid": uid, "participant_id": participant if identified else "",
               "video_source": source or "", "origin_video_id": origin or "",
               "duration_sec": video.get("duration_sec"),
               "annotation_reference_present": uid in referenced,
               "origin_recording_known": distinct_origin_known}
        rows.append(row)
        if identified and distinct_origin_known and uid in referenced:
            grouped[participant].append(video)
    # Redacted pieces/components of one origin cannot become independent support
    # and query recordings just because their canonical video UIDs differ.
    repeat_candidates = {}
    for participant, entries in grouped.items():
        origins = defaultdict(list)
        for v in entries:
            origins[(v["video_source"], v["origin_video_id"])].append(v)
        if len(origins) >= 2:
            representatives = [min(group, key=lambda x: x["video_uid"]) for group in origins.values()]
            repeat_candidates[participant] = sorted(representatives, key=lambda x: x["video_uid"])
    selected = []
    for participant in sorted(repeat_candidates):
        if len(selected) + 2 > max_videos:
            break
        selected.extend(v["video_uid"] for v in repeat_candidates[participant][:2])
    selected_subjects = len(selected) // 2
    selection_ready = bool(annotations) and selected_subjects >= min_subjects
    report = {
        "status": "candidate_subset_selected" if selection_ready else "insufficient_audited_candidates",
        "dataset_version": metadata.get("version"), "total_videos": len(videos),
        "gaze_videos": len(gaze), "gaze_videos_without_known_participant": unknown_identity,
        "annotation_files": list(annotations),
        "gaze_annotation_reference_counts": {name: sum(v["video_uid"] in ids for v in gaze)
                                              for name, ids in intersections.items()},
        "repeat_origin_candidate_wearers": len(repeat_candidates),
        "candidate_video_uids": selected if selection_ready else [],
        "selected_candidate_wearers": selected_subjects if selection_ready else 0,
        "research_ready": False, "chronology_verified": False, "gaze_alignment_verified": False,
        "labels_suitable_verified": False,
        "notes": ["Video-UID annotation references are coverage candidates, not validated task labels.",
                  "Distinct publisher origins are only recording candidates; verify capture grouping before splitting.",
                  "Lexicographic UID selection is deterministic, NOT chronological.",
                  "Same capture/event, concurrent views, and overlapping components must not cross calibration/query.",
                  "Use the official downloader's displayed byte estimate; do not bypass its confirmation prompt.",
                  "If no action-label intersection/repeated wearers remain, use another dataset or explicitly revise the task."]}
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--annotation-files", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=12)
    parser.add_argument("--min-subjects", type=int, default=3)
    parser.add_argument("--max-json-mb", type=float, default=512)
    args = parser.parse_args()
    if args.max_videos < 2 or args.min_subjects < 3 or args.max_json_mb <= 0:
        parser.error("Need a positive file bound, >=2 videos, and >=3 train/dev/test wearer candidates")
    args.output.mkdir(parents=True, exist_ok=False)
    files = [args.metadata, *args.annotation_files]
    if any(not p.is_file() for p in files):
        report = {"status": "blocked_missing_licensed_metadata_or_annotations", "research_ready": False,
                  "note": "Obtain licensed Ego4D files with the official CLI; no video download command was generated."}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(report["status"])
        return 1
    if any(p.stat().st_size > args.max_json_mb * 1e6 for p in files):
        parser.error("JSON exceeds parsing bound; select smaller annotation files or explicitly revise --max-json-mb")
    metadata = json.loads(args.metadata.read_text())
    annotations = {str(p.resolve()): json.loads(p.read_text()) for p in args.annotation_files}
    report, rows = inspect_metadata(metadata, annotations, args.max_videos, args.min_subjects)
    report["source_sha256"] = {str(p.resolve()): file_hash(p) for p in files}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        with (args.output / "gaze_inventory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    selected = report["candidate_video_uids"]
    if selected:
        uid_path = args.output / "video_uids.txt"
        uid_path.write_text("\n".join(selected) + "\n")
        print(f"Candidate pilot: {len(selected)} videos, {report['selected_candidate_wearers']} wearers.")
        print("Inspect report.json and verify suitable labels/capture grouping before downloading videos.")
    else:
        print("No UID file emitted: insufficient joint gaze/annotation/repeat-wearer coverage.")
    print("Not yet research-ready: gaze alignment, labels, and recording chronology remain unaudited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
