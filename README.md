# Gaze-conditioned VLM pilot: replication package

This repository contains the source snapshot, trained adapters, development predictions, protocol, notebook, and visual report for the HoloAssist pilot. **It does not contain HoloAssist data or the 2 GB pretrained weights.**

## Model and artifact sizes

The student base is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; install it from Hugging Face with the standard Transformers APIs and place it at `~/Datasets/HoloAssist/gaze_vlm_study/models/student`. See `artifacts/model_provenance.json` for file hashes. Fine-tuned LoRA and gaze adapters for B1–B5 and T0–T2 (three seeds each) are included under `artifacts/checkpoints/`; base weights are needed to use them.

## Dataset

Indrajeet must obtain the official HoloAssist release and accept its terms independently. The download script in `src/scripts/download_holoassist_full.py` downloads released RGB video, gaze, camera calibration, annotations, and split files; it enforces the 400 GB cap and does not auto-extract. Run outside OneDrive, then use `src/scripts/prepare_gaze_vlm_study.py` to prepare windows. Protocol details are in `src/GAZE_VLM_STUDY.md` and `artifacts/protocol.json`. Do not commit downloaded or extracted dataset files.

## Environment and reproduction

Use Python 3.12 and `requirements-gaze-vlm-study.lock.txt`. For CUDA 12.8 PyTorch, install the pinned torch/torchvision pair from the PyTorch cu128 index first, then install the remaining lock requirements. Commands and source listings are in `notebooks/Gaze_VLM_Complete_Experiment_Notebook.ipynb`.

The pipeline expects `~/Datasets/HoloAssist/study_data/subset` for extracted recordings and `~/Datasets/HoloAssist/gaze_vlm_study` for manifests, models, runs, teacher cache, and flow maps. Obtain the dataset, prepare manifests and flow maps, fetch the pinned model revisions, then run notebook commands. Existing adapters are not standalone models.

## Results and limits

`docs/GAZE_VLM_PILOT_REPORT.html` is the visual report. `artifacts/predictions/` contains saved development predictions. Seed-0 results are preliminary; B5 development evaluation was interrupted. The locked test was not evaluated. T0/T1/T2 trained across three seeds but their development results were not evaluated. Personalization, cross-dataset testing, mobile energy, and verified wearer-disjoint evaluation remain unfinished.

Teacher extraction processed 33,011 training windows; 14,813 outputs passed schema checks. Schema validity does not establish factual accuracy. Raw data, credentials, and server private keys are excluded.

## Citation and provenance

See `src/GAZE_VLM_STUDY.md` for references and protocol. Model revisions and hashes are in `artifacts/model_provenance.json`.
