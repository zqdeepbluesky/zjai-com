# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco

# Set up voc_<year>_<split>
def get_sets(pk_name):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}'.format(year, split)
            t = (lambda split=split, year=year: pascal_voc(split, year))
            # t2 =  pascal_voc(split, year)
            __sets[name] = (lambda split=split, year=year: pascal_voc(split, year,pk_name))

    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_diff'.format(year, split)
            __sets[name] = (lambda split=split, year=year: pascal_voc(split, year,pk_name, use_diff=True))
    return __sets


def get_imdb(name,package_name):
    """Get an imdb (image database) by name."""
    __sets=get_sets(package_name)
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())

print(list_imdbs())