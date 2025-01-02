# sh eval_multi.sh --config-file configs/seg_tto_l.yaml --datasets 'cryonuseg_sem_seg_test,cwfid_sem_seg_test'
# sh eval_multi.sh --config-file config_file_path --datasets 'dataset1,dataset2,...'
#!/bin/sh

# # MESS datasets
# BDD100K = 'bdd100k_sem_seg_val'
# Dark_Zurich = 'dark_zurich_sem_seg_val'
# MHP_v1 = 'mhp_v1_sem_seg_test'
# FoodSeg103 = 'foodseg103_sem_seg_test'
# ATLANTIS = 'atlantis_sem_seg_test'
# DRAM = 'dram_sem_seg_test'
# iSAID = 'isaid_sem_seg_val'
# ISPRS_Potsdam_irrg = 'isprs_potsdam_sem_seg_test_irrg'
# ISPRS_Potsdam_rgb = 'isprs_potsdam_sem_seg_test_rgb'
# WorldFloods_irrg = 'worldfloods_sem_seg_test_irrg'
# WorldFloods_rgb = 'worldfloods_sem_seg_test_rgb'
# FloodNet = 'floodnet_sem_seg_test'
# UAVid = 'uavid_sem_seg_val'
# Kvasir-Instrument = 'kvasir_instrument_sem_seg_test'
# CHASE_BD1 = 'chase_db1_sem_seg_test'
# CryoNuSeg = 'cryonuseg_sem_seg_test'
# PAXRay-4_lungs = 'paxray_sem_seg_test_lungs'
# PAXRay-4_bones = 'paxray_sem_seg_test_bones'
# PAXRay-4_mediastinum = 'paxray_sem_seg_test_mediastinum'
# PAXRay-4_diaphragm = 'paxray_sem_seg_test_diaphragm'
# Corrosion_CS = 'corrosion_cs_sem_seg_test'
# DeepCrack = 'deepcrack_sem_seg_test'
# PST900 = 'pst900_sem_seg_test'
# PST900_rgb = 'pst900_sem_seg_test_rgb'
# PST900_pseudo = 'pst900_sem_seg_test_pseudo'
# ZeroWaste-f = 'zerowaste_sem_seg_test'
# SUIM = 'suim_sem_seg_test'
# CUB-200 = 'cub_200_sem_seg_test'
# CWFID = 'cwfid_sem_seg_test'

# Default values
DEFAULT_DATASET="cwfid_sem_seg_test"
DEFAULT_CONFIG_FILE="configs/seg_tto_l.yaml"
DEFAULT_MODEL_WEIGHTS="/home/ulindu/checkpoints/model_final_large.pth"
DEFAULT_OUTPUT_DIR="output/CAT-Seg_large"

# Help function
show_help() {
    echo "Usage: $0 [--config-file <config_file>] [--datasets <dataset1,dataset2,...>]"
    echo ""
    echo "Options:"
    echo "  --config-file     Path to the config file (default: $DEFAULT_CONFIG_FILE)"
    echo "  --datasets        Comma-separated list of dataset names (default: $DEFAULT_DATASET)"
    exit 1
}

# Parse arguments
CONFIG_FILE=$DEFAULT_CONFIG_FILE
DATASETS=$DEFAULT_DATASET
while [ "$#" -gt 0 ]; do
    case $1 in
        --config-file) CONFIG_FILE="$2"; shift ;;
        --datasets) DATASETS="$2"; shift ;;
        --help) show_help ;;
        *) echo "Unknown parameter passed: $1"; show_help ;;
    esac
    shift
done

# Ensure the environment variable for datasets is set
export DETECTRON2_DATASETS="/home/ulindu/datasets"

# Define other parameters
NUM_GPUS=1
MODEL_WEIGHTS=$DEFAULT_MODEL_WEIGHTS
SLIDING_WINDOW="True"
POOLING_SIZES="[1,1]"
DEVICE="cuda:0"

IFS=',' 
set -- $DATASETS
DATASET_ARRAY="$@"

echo "DATASET_ARRAY: $DATASET_ARRAY"

# Iterate over each dataset
for DATASET_NAME in $DATASET_ARRAY; do
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR/$DATASET_NAME"

    echo "Running evaluation for dataset: $DATASET_NAME"
    echo "Config file: $CONFIG_FILE"

    # Run the command
    python train_net.py --num-gpus $NUM_GPUS --eval-only \
        --config-file "$CONFIG_FILE" \
        DATASETS.TEST "('$DATASET_NAME',)" \
        MODEL.WEIGHTS "$MODEL_WEIGHTS" \
        OUTPUT_DIR "$OUTPUT_DIR" \
        TEST.SLIDING_WINDOW "$SLIDING_WINDOW" \
        MODEL.SEM_SEG_HEAD.POOLING_SIZES "$POOLING_SIZES" \
        MODEL.DEVICE "$DEVICE"
done
