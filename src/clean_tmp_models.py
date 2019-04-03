#!/usr/bin/env python

import os
import os.path as osp

import sys


def get_sorted_model_list(root_dir, prefix='model'):
    dir_list = os.listdir(root_dir)

    list_dict = {}
    idx_list = []

    for fn in dir_list:
        if fn.startswith(prefix) and 'slim' not in fn:
            splits = fn.rsplit('-', 1)
            if splits[1].startswith('symbol'):
                continue

            epoch = int(splits[1].split('.')[0])

            list_dict[str(epoch)] = fn
            idx_list.append(epoch)

    list_fn = []
    idx_list.sort()

    for idx in idx_list:
        list_fn.append(list_dict[str(idx)])

    return list_fn


def clean_dir(root_dir, prefix='model', keep_n=3):
    model_list = get_sorted_model_list(root_dir, prefix)
    print('===> model list: \n', model_list)
    if len(model_list) > keep_n:
        for fn in model_list[0: -keep_n]:
            full_fn = osp.join(root_dir, fn)
            print('---> deleting ', full_fn)
            os.remove(full_fn)


def clean_all_subdir(root_dir, model_prefix='model', subdir_keyword=''):
    dir_list = os.listdir(root_dir)

    for fn in dir_list:
        if subdir_keyword and subdir_keyword not in fn:
            continue

        sub_dir = osp.join(root_dir, fn)
        if not osp.isdir(sub_dir):
            continue
        else:
            print('===> clean subdir: ', sub_dir)
            clean_dir(sub_dir, prefix=model_prefix, keep_n=3)


if __name__ == '__main__':
    root_dir = './'
    prefix = 'model'
    keyword = ''

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    if len(sys.argv) > 2:
        prefix = sys.argv[2]

    if len(sys.argv) > 3:
        keyword = sys.argv[3]

    model_list = get_sorted_model_list(root_dir, prefix)
    print('===> model list: \n', model_list)

    clean_dir(root_dir, prefix)

    # clean_all_subdir(root_dir, prefix, keyword)
