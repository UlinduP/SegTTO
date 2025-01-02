from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class DeepCrackDataset(CustomDataset):
    """DeepCrack dataset.

    This dataset is designed for crack segmentation, where the labels represent
    concrete/asphalt or other surfaces and crack areas.

    In the segmentation map annotation for DeepCrack, label 0 is the background,
    which is included in 2 categories. ``reduce_zero_label`` is set to False to
    ensure that background labels are not reduced.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('concrete or asphalt or others', 'crack')

    PALETTE = [[0, 0, 0], [1, 1, 1]]

    def __init__(self, **kwargs):
        super(DeepCrackDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
