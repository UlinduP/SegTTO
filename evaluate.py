import subprocess

# Set environment variable for datasets
import os
os.environ['DETECTRON2_DATASETS'] = "/home/ulindu/datasets"

# Define dataset name
dataset_name = ['bdd100k_sem_seg_val']#,'corrosion_cs_sem_seg_test','cryonuseg_sem_seg_test','suim_sem_seg_test','kvasir_instrument_sem_seg_test','worldfloods_sem_seg_test_irrg','pst900_sem_seg_test']

# voc_2012_sem_seg_val, ade20k_sem_seg_val
# 'pst900_sem_seg_test','deepcrack_sem_seg_test','corrosion_cs_sem_seg_test',dark_zurich_sem_seg_val
#                 'cryonuseg_sem_seg_test','kvasir_instrument_sem_seg_test','dram_sem_seg_test','isprs_potsdam_sem_seg_test_irrg',
#                 'worldfloods_sem_seg_test_irrg','paxray_sem_seg_test_lungs','paxray_sem_seg_test_bones',paxray_sem_seg_test_diaphragm, paxray_sem_seg_test_mediastinum

#                 'cwfid_sem_seg_test','suim_sem_seg_test','chase_db1_sem_seg_test', mhp_v1_sem_seg_test, bdd100k_sem_seg_val
#  cub_200_sem_seg_test


for dataset in dataset_name:
    # Construct the command to be run
    command = [
        "python", "train_net.py", "--num-gpus", "1", "--eval-only",
        "--config-file", "configs/seg_tto_l.yaml",
        "DATASETS.TEST", f"('{dataset}',)",
        "MODEL.WEIGHTS", "/home/ulindu/checkpoints/model_final_large.pth",
        "OUTPUT_DIR", f"output/CAT-Seg_large/{dataset}",
        "TEST.SLIDING_WINDOW", "True",
        "MODEL.SEM_SEG_HEAD.POOLING_SIZES", "[1,1]",
        "MODEL.DEVICE", "cuda:1",
    ]

    # Run the command
    subprocess.run(command)
