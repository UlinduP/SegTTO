from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class CUBDataset(CustomDataset):
    """Chase_db1 dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_1stHO.png'.
    """

    CLASSES = ('background', 'Black-footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani',
        'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird',
        'Red winged Blackbird', 'Rusty Blackbird', 'Yellow-headed Blackbird', 'Bobolink', 'Indigo Bunting',
        'Lazuli Bunting', 'Painted Bunting', 'Cardinal', 'Spotted Catbird', 'Gray Catbird',
        'Yellow-breasted Chat', 'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 'Red-faced Cormorant',
        'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow',
        'Black-billed Cuckoo', 'Mangrove Cuckoo', 'Yellow-billed Cuckoo', 'Gray-crowned Rosy-Finch', 'Purple Finch',
        'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive-sided Flycatcher',
        'Scissor-tailed Flycatcher', 'Vermilion Flycatcher', 'Yellow-bellied Flycatcher', 'Frigatebird', 'Northern Fulmar',
        'Gadwall', 'American Goldfinch', 'European Goldfinch', 'Boat-tailed Grackle', 'Eared Grebe', 'Horned Grebe',
        'Pied-billed Grebe', 'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak', 'Pine Grosbeak', 'Rose-breasted Grosbeak',
        'Pigeon Guillemot', 'California Gull', 'Glaucous-winged Gull', 'Heermann Gull', 'Herring Gull', 'Ivory Gull',
        'Ring-billed Gull', 'Slaty-backed Gull', 'Western Gull', 'Anna Hummingbird', 'Ruby-throated Hummingbird',
        'Rufous Hummingbird', 'Green Violetear', 'Long-tailed Jaeger', 'Pomarine Jaeger', 'Blue Jay', 'Florida Jay',
        'Green Jay', 'Dark-eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher',
        'Pied Kingfisher', 'Ringed Kingfisher', 'White-breasted Kingfisher', 'Red-legged Kittiwake', 'Horned Lark',
        'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red-breasted Merganser', 'Mockingbird',
        'Nighthawk', 'Clark Nutcracker', 'White-breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole',
        'Scott Oriole', 'Ovenbird', 'Brown Pelican', 'White Pelican', 'Western Wood-Pewee', 'Sayornis', 'American Pipit',
        'Whip-poor-Will', 'Horned Puffin', 'Common Raven', 'White-necked Raven', 'American Redstart', 'Geococcyx',
        'Loggerhead Shrike', 'Great Grey Shrike', 'Baird Sparrow', 'Black-throated Sparrow', 'Brewer Sparrow',
        'Chipping Sparrow', 'Clay-colored Sparrow', 'House Sparrow', 'Field Sparrow', 'Fox Sparrow', 'Grasshopper Sparrow',
        'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 'Nelson Sharp-tailed Sparrow',
        'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 'White-crowned Sparrow',
        'White-throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 'Tree Swallow',
        'Scarlet Tanager', 'Summer Tanager', 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern',
        'Forsters Tern', 'Least Tern', 'Green-tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black-capped Vireo',
        'Blue-headed Vireo', 'Philadelphia Vireo', 'Red-eyed Vireo', 'Warbling Vireo', 'White-eyed Vireo',
        'Yellow-throated Vireo', 'Bay-breasted Warbler', 'Black-and-white Warbler', 'Black-throated Blue Warbler',
        'Blue-winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 'Chestnut-sided Warbler',
        'Golden-winged Warbler', 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler',
        'Myrtle Warbler', 'Nashville Warbler', 'Orange-crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler',
        'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm-eating Warbler',
        'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing',
        'American Three-toed Woodpecker', 'Pileated Woodpecker', 'Red-bellied Woodpecker', 'Red-cockaded Woodpecker',
        'Red-headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 'Carolina Wren', 'House Wren',
        'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat')


    PALETTE = [
        [0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10],
        [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18], [19, 19, 19], 
        [20, 20, 20], [21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24], [25, 25, 25], [26, 26, 26], [27, 27, 27], [28, 28, 28], 
        [29, 29, 29], [30, 30, 30], [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37], 
        [38, 38, 38], [39, 39, 39], [40, 40, 40], [41, 41, 41], [42, 42, 42], [43, 43, 43], [44, 44, 44], [45, 45, 45], [46, 46, 46], 
        [47, 47, 47], [48, 48, 48], [49, 49, 49], [50, 50, 50], [51, 51, 51], [52, 52, 52], [53, 53, 53], [54, 54, 54], [55, 55, 55], 
        [56, 56, 56], [57, 57, 57], [58, 58, 58], [59, 59, 59], [60, 60, 60], [61, 61, 61], [62, 62, 62], [63, 63, 63], [64, 64, 64], 
        [65, 65, 65], [66, 66, 66], [67, 67, 67], [68, 68, 68], [69, 69, 69], [70, 70, 70], [71, 71, 71], [72, 72, 72], [73, 73, 73], 
        [74, 74, 74], [75, 75, 75], [76, 76, 76], [77, 77, 77], [78, 78, 78], [79, 79, 79], [80, 80, 80], [81, 81, 81], [82, 82, 82], 
        [83, 83, 83], [84, 84, 84], [85, 85, 85], [86, 86, 86], [87, 87, 87], [88, 88, 88], [89, 89, 89], [90, 90, 90], [91, 91, 91], 
        [92, 92, 92], [93, 93, 93], [94, 94, 94], [95, 95, 95], [96, 96, 96], [97, 97, 97], [98, 98, 98], [99, 99, 99], [100, 100, 100],
        [101, 101, 101], [102, 102, 102], [103, 103, 103], [104, 104, 104], [105, 105, 105], [106, 106, 106], [107, 107, 107], [108, 108, 108], 
        [109, 109, 109], [110, 110, 110], [111, 111, 111], [112, 112, 112], [113, 113, 113], [114, 114, 114], [115, 115, 115], [116, 116, 116], 
        [117, 117, 117], [118, 118, 118], [119, 119, 119], [120, 120, 120], [121, 121, 121], [122, 122, 122], [123, 123, 123], [124, 124, 124], 
        [125, 125, 125], [126, 126, 126], [127, 127, 127], [128, 128, 128], [129, 129, 129], [130, 130, 130], [131, 131, 131], [132, 132, 132], 
        [133, 133, 133], [134, 134, 134], [135, 135, 135], [136, 136, 136], [137, 137, 137], [138, 138, 138], [139, 139, 139], [140, 140, 140], 
        [141, 141, 141], [142, 142, 142], [143, 143, 143], [144, 144, 144], [145, 145, 145], [146, 146, 146], [147, 147, 147], [148, 148, 148], 
        [149, 149, 149], [150, 150, 150], [151, 151, 151], [152, 152, 152], [153, 153, 153], [154, 154, 154], [155, 155, 155], [156, 156, 156], 
        [157, 157, 157], [158, 158, 158], [159, 159, 159], [160, 160, 160], [161, 161, 161], [162, 162, 162], [163, 163, 163], [164, 164, 164], 
        [165, 165, 165], [166, 166, 166], [167, 167, 167], [168, 168, 168], [169, 169, 169], [170, 170, 170], [171, 171, 171], [172, 172, 172], 
        [173, 173, 173], [174, 174, 174], [175, 175, 175], [176, 176, 176], [177, 177, 177], [178, 178, 178], [179, 179, 179], [180, 180, 180], 
        [181, 181, 181], [182, 182, 182], [183, 183, 183], [184, 184, 184], [185, 185, 185], [186, 186, 186], [187, 187, 187], [188, 188, 188], 
        [189, 189, 189], [190, 190, 190], [191, 191, 191], [192, 192, 192], [193, 193, 193], [194, 194, 194], [195, 195, 195], [196, 196, 196], 
        [197, 197, 197], [198, 198, 198], [199, 199, 199], [200, 200, 200]
    ]

    def __init__(self, **kwargs):
        super(CUBDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
