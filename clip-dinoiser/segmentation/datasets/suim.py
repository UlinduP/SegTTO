from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class SuimDataset(CustomDataset):
    """Chase_db1 dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_1stHO.png'.
    """

    CLASSES = ('background (waterbody)', 'human diver', 'aquatic plants and sea-grass', 
        'wrecks and ruins', 'robots (auvs/rovs/instruments)', 'reefs and invertebrates', 
        'fish and vertebrates', 'sea-floor and rocks')

    PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]

    def __init__(self, **kwargs):
        super(SuimDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
