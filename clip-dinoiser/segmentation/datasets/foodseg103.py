from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class MHPDataset(CustomDataset):
    """DeepCrack dataset.

    This dataset is designed for crack segmentation, where the labels represent
    concrete/asphalt or other surfaces and crack areas.

    In the segmentation map annotation for DeepCrack, label 0 is the background,
    which is included in 2 categories. ``reduce_zero_label`` is set to False to
    ensure that background labels are not reduced.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('others', 'hat', 'hair', 'sunglasses', 'upper clothes', 'skirt', 'pants', 'dress', 'belt', 'left shoe', 'right shoe',
           'face', 'left leg', 'right leg', 'left arm', 'right arm', 'bag', 'scarf', 'torso skin')

    PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10],
              [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18]]

    def __init__(self, **kwargs):
        super(MHPDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
