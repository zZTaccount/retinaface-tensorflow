import numpy as np
from ..config import config
from ..dataset import *


def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path,
                  flip=False):
    """ load ground truth roidb """
    print('utils/load_data: dataset:{}, image_set:{}, root_path:{}, dataset_path:{}\n'.format(dataset_name,image_set_name,root_path,dataset_path))
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path)
    roidb = imdb.gt_roidb()
    print('utils/load_data: roidb size', len(roidb))
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    print('utils/load_data: flipped roidb size', len(roidb))
    return roidb


def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path,
                        proposal='rpn', append_gt=True, flip=False):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path)
    gt_roidb = imdb.gt_roidb()
    roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb, append_gt)
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb):
    """ remove roidb entries without usable rois """

    def is_valid(entry):
        """ valid images have at least 1 fg or bg roi """
        overlaps = entry['max_overlaps']
        fg_inds = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
        bg_inds = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        #valid = len(fg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)

    return filtered_roidb
