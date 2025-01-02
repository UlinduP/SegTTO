# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology
# ----------------------------------------------------------------------------------------------------
# GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
custom_imports = dict(
    imports=["segmentation.datasets.coco_object", "segmentation.datasets.pascal_voc",
             "segmentation.datasets.pascal_voc20", "segmentation.datasets.corrosion_cs", "segmentation.datasets.cwfid",
             "segmentation.datasets.cryonuseg", "segmentation.datasets.atlantis", "segmentation.datasets.bdd100k",
             "segmentation.datasets.cub", "segmentation.datasets.deepcrack",
             "segmentation.datasets.dram", "segmentation.datasets.floodnet", "segmentation.datasets.foodseg103",
             "segmentation.datasets.kvasir", "segmentation.datasets.mhpv1",
             "segmentation.datasets.paxray", "segmentation.datasets.pst900",
             "segmentation.datasets.suim", "segmentation.datasets.uavid", "segmentation.datasets.worldfloods",
             "segmentation.datasets.zerowaste", 
             ],
    allow_failed_imports=False,
)
