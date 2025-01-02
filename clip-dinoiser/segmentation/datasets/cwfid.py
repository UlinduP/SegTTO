from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class CwfidDataset(CustomDataset):
    """CWFID dataset.

    The color palette is defined according to detectron2 library standard.
    """

    CLASSES = ('ground', 'crop_seedling', 'weed')

    PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    def __init__(self, **kwargs):
        super(CwfidDataset, self).__init__(
            img_suffix='_image.png',
            seg_map_suffix='_annotation.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
