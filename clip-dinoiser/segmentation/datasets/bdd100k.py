from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class BDD100KDataset(CustomDataset):
    """Chase_db1 dataset.

    The color palette is defined according to detectron2 library standard.
    """

    CLASSES = ('road','sidewalk','building','wall','fence','pole','traffic light','traffic sign', 'vegetation','terrain','sky', 'person',
    'rider','car','truck','bus','train','motorcycle','bicycle')

    PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10],
               [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18], [19, 19, 19]]

    def __init__(self, **kwargs):
        super(BDD100KDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
