# VideoMAE Feature Extractor

## Install

Create and activate a new Conda environment:

```
conda create --name videomaefe python=3.8
conda activate videomaefe
```

Install VideoMAE dependencies:

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install timm==0.4.12
DS_BUILD_OPS=1 pip install deepspeed==0.7.3
pip install tensorboardX
pip install decord==0.6.0
pip install einops==0.4.1
```

If installation of `deepspeed` fails, try without `DS_BUILD_OPS=1`.

According to VideoMAE, [PyTorch 1.8.0 and 1.6.0 are recommended](https://github.com/MCG-NJU/VideoMAE/blob/main/INSTALL.md). But it seems PyTorch 1.10.0 works fine after comparing the features extracted from a few videos under these different settings (max difference 0.0213, mean abs difference 0.0004). Since the reproducibility is not yet verified, use it at your own risk:

```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```

Finally, recursively clone this repository.

## Configure and run

Modify data, device, model, and sampling variables in `feature_extraction.py` according to you setting and run this script. Videos are read from `DATA_ROOT/VIDEO_LIST_PATH_i+EXT_NAME`, and features are written to `OUTPUT_DIR/VIDEO_LIST_PATH_i.pkl`, with shape `clip_num * features_dim`.

Example:

```
DATA_ROOT
├── 001.mp4
├── 002.mp4
├── 003.mp4
└── 004.mp4
```

In text file `VIDEO_LIST_PATH` (trailing newline is ok):
```
001
002
003
004
```

This script splits the video list into equal parts and feed into each device respectively. `tqdm` progress bar is printed to the terminal only for the first process for brevity.
