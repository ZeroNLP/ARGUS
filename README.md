# ARGUS

Official code release for **ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior**, accepted as a **CVPR 2026** paper.

ARGUS studies multimodal indirect prompt injection, where malicious instructions are hidden inside images, videos, or audio. This repository provides reproducible dataset construction, baseline evaluation, and the ARGUS defense pipeline.

If this repo helps your research, please consider giving it a star and citing our paper.

## Setup

Create an environment with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

For audio dataset construction, make sure `ffmpeg` and `ffprobe` are available on your `PATH`.

## Configuration

There are two config files.

`configs/dataset_baselines.yaml` is used for dataset construction and baseline evaluation across all three modalities. Fill the paths under `modalities.image`, `modalities.video`, and `modalities.audio` as needed:

- `model.path`: local path to the model used by that modality.
- `data.alpaca_path`: The default path is `./data/alpaca`. Please download the dataset from [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) and move the `data` folder to `./data/alpaca`
- `data.vtqa_path`: VTQA2023 source. The default is `CalfKing/vtqa2023` with `data.vtqa_config: en-image`.
- `data.msrvtt_path`:  The default path is `./data/MSRVTT`. Please download the dataset from [VLM2Vec/MSR-VTT](https://huggingface.co/datasets/VLM2Vec/MSR-VTT) and move the `raw_videos` folder to `./data/MSRVTT`.
- `data.clotho_*`: Clotho-AQA csv files and audio directory for audio. Please download the dataset from [Zenodo](https://zenodo.org/records/6473207).

`configs/argus.yaml` is used for the ARGUS image pipeline. Fill:

- `model.path`: local Qwen2-VL path.
- `run.device`: default CUDA device or device list.

## 1. Build Datasets

Build one modality at a time:

```bash
python scripts/build_dataset.py --config configs/dataset_baselines.yaml --modality image
python scripts/build_dataset.py --config configs/dataset_baselines.yaml --modality video
python scripts/build_dataset.py --config configs/dataset_baselines.yaml --modality audio
```

Useful flags:

- `--force`: rebuild outputs that already exist.
- `--output-dir /path/to/output`: override the config output directory.

Expected output layout:

```text
outputs/
  data/
    train/
    val/
    test/
```

For video and audio, the default config writes to `outputs_video/` and `outputs_audio/`.

## 2. Run Baselines

After building a dataset, run all configured baselines:

```bash
python scripts/run_baselines.py --config configs/dataset_baselines.yaml --modality image
python scripts/run_baselines.py --config configs/dataset_baselines.yaml --modality video
python scripts/run_baselines.py --config configs/dataset_baselines.yaml --modality audio
```

Run on multiple GPUs for faster MLLM inference:

```bash
python scripts/run_baselines.py --config configs/dataset_baselines.yaml --modality image --gpu-id 0,1,2,3
```

Run a subset of baselines:

```bash
python scripts/run_baselines.py \
  --config configs/dataset_baselines.yaml \
  --modality image \
  --baselines none,system_prompt,ignore,noise
```

Baseline results are saved under:

```text
<output_dir>/results/baselines/
```

## 3. Run ARGUS

This project provides a demonstration for the image modality only. ARGUS assumes the image dataset has already been built. Then run:

```bash
python scripts/run_argus.py --config configs/argus.yaml
```

For multi-GPU runs:

```bash
python scripts/run_argus.py --config configs/argus.yaml --gpu-id 0,1,2,3
```

## Citation

Please cite ARGUS if you find the code, dataset construction, or baselines useful:

```bibtex
@inproceedings{lu2026argus,
  title={ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior},
  author={Lu, W and Zeng, Z and Zhang, K and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={31--40},
  year={2026}
}
```
