from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class PST900Dataset(CustomDataset):
    """DeepCrack dataset.

    This dataset is designed for crack segmentation, where the labels represent
    concrete/asphalt or other surfaces and crack areas.

    In the segmentation map annotation for DeepCrack, label 0 is the background,
    which is included in 2 categories. ``reduce_zero_label`` is set to False to
    ensure that background labels are not reduced.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('background', 'fire extinguisher', 'backpack', 'drill', 'human')

    PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]

    def __init__(self, **kwargs):
        super(PST900Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
