## Installation

### Requirements

```
git clone https://github.com/UlinduP/Seg-TTO.git
cd Seg-TTO/clip-dinoiser
conda create -n cdtto python=3.9
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=[your CUDA version] -c pytorch
pip install -r requirements.txt
```

Also install MMCV by running

```
pip install "mmcv-full>=1.3.13,<1.7.0" -f https://download.openmmlab.com/mmcv/dist/[your CUDA version]/torch1.12.0/index.html
```

Example installation for CUDA 11.3

```
pip install "mmcv-full>=1.3.13,<1.7.0" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
```
