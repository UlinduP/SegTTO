# Instructions to Run the Script

## 1. Prepare the Configuration File
Ensure you have a configuration file, e.g., `configs/vitb_r101_384.yaml`, available. You can modify the `--config-file` argument to point to your specific configuration file.

## 2.Set the DETECTRON2_DATASETS environment variable
```bash
export DETECTRON2_DATASETS="/path/to/datasets/directory"
```

# Define dataset name (replace with actual dataset name)
```bash
DATASET='suim_sem_seg_test'   # .json file name
```

## 3. Running the Script
Run the script with an input image:

```bash
python demo.py \
    --config-file configs/ade20k-150/maskformer_R50_bs16_160k.yaml \
    --input path/to/image1.jpg \
    --output path/to/output_directory \
    --opts MODEL.WEIGHTS path/to/model_weights.pth
```



Happy segmenting!