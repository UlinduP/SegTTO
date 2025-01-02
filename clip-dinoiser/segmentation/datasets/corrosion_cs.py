from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class CorrosionCSDataset(CustomDataset):
    """CorrosionCS dataset.

    The color palette is defined according to detectron2 library standard.
    """

    CLASSES = ('good (background)', 'fair corrosion', 'poor corrosion', 'severe corrosion')

    PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]

    def __init__(self, **kwargs):
        super(CorrosionCSDataset, self).__init__(
            img_suffix='.jpeg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
