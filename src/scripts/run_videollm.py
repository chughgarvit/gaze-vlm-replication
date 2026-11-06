#!/usr/bin/env python3
"""Evaluate local Qwen2.5-VL on aligned frames; --dry-run requires no model."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from gaze_memory.core import save_json
from gaze_memory.frames import MODES, prepare_frame, select_past_frames, usable_gaze
import numpy as np


def load_sample(sample, manifest_dir, mode="full"):
    key = "overlay_frames_manifest" if mode == "publisher_overlay" else "frames_manifest"
    if key not in sample:
        raise ValueError(f"{mode} requires {key}; no synthetic gaze substitute is allowed")
    path = (manifest_dir/sample[key]).resolve()
    data = json.loads(path.read_text())
    selected = select_past_frames(data["frames"],sample.get("start_s",0),sample["cutoff_s"],sample.get("frame_count",8))
    if mode == "publisher_overlay":
        original_path = (manifest_dir/sample["frames_manifest"]).resolve()
        original = json.loads(original_path.read_text())
        expected = select_past_frames(original["frames"], sample.get("start_s",0),
                                      sample["cutoff_s"], sample.get("frame_count",8))
        if len(expected) != len(selected) or any(abs(a["timestamp_s"]-b["timestamp_s"]) > .001
                                               for a,b in zip(expected,selected)):
            raise ValueError("Original/overlay frame timestamps do not match within 1 ms")
    images = []
    for frame in selected:
        with Image.open((path.parent/frame["path"]).resolve()) as image:
            images.append(image.convert("RGB"))
    return selected,images


def make_messages(sample, frames, mode):
    content = []
    for f in frames:
        content.extend([{"type":"text","text":f"Frame at {f['timestamp_s']:.3f} seconds."},
                        {"type":"image","image":"placeholder"}])
    hint = ""
    if sample.get("comparison") == "publisher_overlay":
        # Identical text in both paired conditions; only the pixels differ.
        hint = ("Some videos contain annotations: a green point indicates the wearer's gaze "
                "and a red circle the field of view. Use such annotations only when visible. ")
    elif mode == "gaze_overlay":
        hint = "The red ring indicates measured gaze when available. "
    elif mode in {"gaze_crop","shuffled_gaze"}:
        hint = "Each image shows a region selected around the gaze point when available. "
    elif mode == "gaze_context":
        hint = "Each image has a full-scene view on the left and gaze-region view on the right when gaze is available. "
    content.append({"type":"text","text":hint+sample["question"]+"\n"+"\n".join(sample["options"])+
                    "\nReturn only the option letter. If uncertain, choose the best supported option."})
    return [{"role":"user","content":content}]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--model",type=Path)
    p.add_argument("--modes",nargs="+",choices=MODES+["publisher_overlay"],default=MODES)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    if args.output.exists():
        p.error("Output already exists; choose a new directory")
    if not args.dry_run and (args.model is None or not args.model.is_dir()):
        p.error("Inference requires a local Qwen2.5-VL model directory")
    samples=json.loads(args.manifest.read_text())["samples"]
    if not samples:
        p.error("Manifest has no samples")
    model=processor=None
    if not args.dry_run:
        os.environ.setdefault("HF_HUB_OFFLINE","1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
        import torch
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        torch.set_num_threads(4)
        processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,
                                               min_pixels=448*448,max_pixels=448*448)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,local_files_only=True,
            torch_dtype=torch.float32 if args.device=="cpu" else torch.bfloat16,
            attn_implementation="eager").to(args.device).eval()
    rows=[]
    for sample in samples:
        # Prompt construction reads only allowed question fields. Answers and
        # benchmark metadata such as gazed_objects never enter model inputs.
        base_frames,base_images=load_sample(sample,args.manifest.parent)
        gaze=[f.get("gaze") for f in base_frames]
        shuffled=list(gaze)
        if len(shuffled)>1:
            # Causal within the already-observed prefix; deterministic control.
            shuffled=[shuffled[i] for i in np.random.default_rng(42).permutation(len(gaze))]
        for mode in args.modes:
            frames,images = (load_sample(sample,args.manifest.parent,mode)
                             if mode == "publisher_overlay" else (base_frames,base_images))
            actual_gaze=shuffled if mode=="shuffled_gaze" else gaze
            treated=[prepare_frame(im,g,"full" if mode == "publisher_overlay" else mode)
                     for im,g in zip(images,actual_gaze)]
            messages=make_messages(sample,frames,mode)
            row={"sample_id":sample["id"],"mode":mode,"frames":len(frames),
                 "valid_gaze_frames":sum(usable_gaze(g) for g in actual_gaze),
                 "cutoff_s":sample["cutoff_s"],"max_frame_time":max(f["timestamp_s"] for f in frames),
                 "status":"dry_run" if args.dry_run else "inference"}
            if mode == "publisher_overlay":
                row["gaze_source"] = "publisher_burned_in_overlay_not_raw_coordinates"
                row["valid_gaze_frames"] = 0
            if args.dry_run:
                folder=args.output/sample["id"]/mode
                folder.mkdir(parents=True)
                for i,im in enumerate(treated):
                    im.save(folder/f"frame-{i:02d}.png")
                save_json(folder/"prompt.json",messages)
            else:
                text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs=processor(text=[text],images=treated,padding=True,return_tensors="pt").to(args.device)
                # Qwen merges 2x2 visual patches; count actual input grid tokens.
                row["visual_tokens"]=int(inputs["image_grid_thw"].prod(dim=1).sum().item()//4)
                row["input_tokens"]=int(inputs["attention_mask"].sum().item())
                t=time.perf_counter()
                with torch.inference_mode():
                    out=model.generate(**inputs,max_new_tokens=8,do_sample=False)
                response=processor.batch_decode(out[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip()
                prediction=re.search(r"\b([A-Z])\b",response)
                row.update(response=response,prediction=prediction[1] if prediction else None,
                           correct=bool(prediction and prediction[1]==sample["answer"]),
                           latency_s=time.perf_counter()-t)
            rows.append(row)
            save_json(args.output/"predictions.json",rows)
    summary={"status":"preprocessing_verified_no_model" if args.dry_run else "model_evaluated",
             "samples":len(samples),"rows":len(rows),"model":str(args.model),
             "note":"Frame-sequence input; not native video decoding. Accuracy absent in dry-run mode."}
    if not args.dry_run:
        summary["accuracy"]={mode:float(np.mean([r["correct"] for r in rows if r["mode"]==mode])) for mode in args.modes}
        summary["visual_tokens"]={mode:sorted({r["visual_tokens"] for r in rows if r["mode"]==mode}) for mode in args.modes}
    save_json(args.output/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
