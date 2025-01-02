from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class CryoNuSegDataset(CustomDataset):
    """cryonuseg dataset.

    The color palette is defined according to detectron2 library standard.
    """

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [1, 1, 1]]

    def __init__(self, **kwargs):
        super(CryoNuSegDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
