#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate catseg

# Benchmark
export DETECTRON2_DATASETS="datasets"
TEST_DATASETS="bdd100k_sem_seg_val dark_zurich_sem_seg_val mhp_v1_sem_seg_test foodseg103_sem_seg_test atlantis_sem_seg_test dram_sem_seg_test isaid_sem_seg_val isprs_potsdam_sem_seg_test_irrg worldfloods_sem_seg_test_irrg floodnet_sem_seg_test uavid_sem_seg_val kvasir_instrument_sem_seg_test chase_db1_sem_seg_test cryonuseg_sem_seg_test paxray_sem_seg_test_lungs paxray_sem_seg_test_bones paxray_sem_seg_test_mediastinum paxray_sem_seg_test_diaphragm corrosion_cs_sem_seg_test deepcrack_sem_seg_test pst900_sem_seg_test zerowaste_sem_seg_test suim_sem_seg_test cub_200_sem_seg_test cwfid_sem_seg_test"

# Run experiments with CAT-Seg
for DATASET in $TEST_DATASETS
do
 # Base model:
 python train_net.py --num-gpus 1 --eval-only --config-file configs/vitb_r101_384.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS weights/model_final_base.pth OUTPUT_DIR output/CAT-Seg_base/$DATASET TEST.SLIDING_WINDOW True MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]"
 # Large model:
 python train_net.py --num-gpus 1 --eval-only --config-file configs/vitl_swinb_384.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS weights/model_final_large.pth OUTPUT_DIR output/CAT-Seg_large/$DATASET TEST.SLIDING_WINDOW True MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]"
 # Huge model:
 python train_net.py --num-gpus 1 --eval-only --config-file configs/vitl_swinb_384.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS weights/model_final_huge.pth OUTPUT_DIR output/CAT-Seg_huge/$DATASET TEST.SLIDING_WINDOW True MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED "ViT-H" MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM 1024
done

# Combine results
python mess/evaluation/mess_evaluation.py --model_outputs output/CAT-Seg_base output/CAT-Seg_large output/CAT-Seg_huge


# Run evaluation with:
# nohup bash mess/eval.sh > eval.log &
# tail -f eval.log
