from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class ZeroWasteDataset(CustomDataset):
    """Chase_db1 dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_1stHO.png'.
    """

    CLASSES = ("background or trash", "rigid plastic", "cardboard", "metal", "soft plastic")

    PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]

    def __init__(self, **kwargs):
        super(ZeroWasteDataset, self).__init__(
            img_suffix='.PNG',
            seg_map_suffix='.PNG',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
