from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import pandas as pd
import random
from collections import Counter

from tqdm import tqdm

from utils import mkdir_if_missing, write_json, read_json
from video_loader import read_image
import transforms as T

'''
  ------------------------------
  subset   | # ids | # tracklets
  ------------------------------
  train    |   625 |     8298
  query    |   626 |     1980
  gallery  |   622 |     9330
  ------------------------------
  total    |  1251 |    19608
  number of images per tracklet: 2 ~ 920, average 59.5
  ------------------------------
'''







class Mars_event_vid(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """

    def __init__(self, root, min_seq_len=0, **kwargs):

##########################################################################
# img
        self.root = osp.join(root, 'MARS')
        self.train_name_path = osp.join(self.root, 'info/train_name.txt')
        self.test_name_path = osp.join(self.root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.root, 'info/query_IDX.mat')
# event
        self.root_event = osp.join(root, 'MARS')
        self.train_name_path_event = osp.join(self.root_event, 'info_event/train_name.txt')
        self.test_name_path_event = osp.join(self.root_event, 'info_event/test_name.txt')
        self.track_train_info_path_event = osp.join(self.root_event, 'info_event/tracks_train_info.mat')
        self.track_test_info_path_event = osp.join(self.root_event, 'info_event/tracks_test_info.mat')
        self.query_IDX_path_event = osp.join(self.root_event, 'info_event/query_IDX.mat')
##########################################################################



        self._check_before_run()
        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        train_names_event = self._get_names(self.train_name_path_event)
        test_names_event = self._get_names(self.test_name_path_event)


# img
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]
# event
        track_train_event = loadmat(self.track_train_info_path_event)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test_event = loadmat(self.track_test_info_path_event)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX_event = loadmat(self.query_IDX_path_event)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX_event -= 1 # index from 0
        track_query_event = track_test_event[query_IDX_event,:]
        gallery_IDX_event = [i for i in range(track_test_event.shape[0]) if i not in query_IDX_event]
        track_gallery_event = track_test_event[gallery_IDX_event,:]

#########################################################################################
# img
        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)
        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)
# event
        train_event, num_train_tracklets_event, num_train_pids_event, num_train_imgs_event = \
          self._process_data_event(train_names_event, track_train_event, home_dir='train_origin_mat_divide', relabel=True, min_seq_len=min_seq_len)
        query_event, num_query_tracklets_event, num_query_pids_event, num_query_imgs_event = \
          self._process_data_event(test_names_event, track_query_event, home_dir='test_origin_mat_divide2', relabel=False, min_seq_len=min_seq_len)
        gallery_event, num_gallery_tracklets_event, num_gallery_pids_event, num_gallery_imgs_event = \
          self._process_data_event(test_names_event, track_gallery_event, home_dir='test_origin_mat_divide2', relabel=False, min_seq_len=min_seq_len)




        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets


        num_imgs_per_tracklet_event = num_train_imgs_event + num_query_imgs_event + num_gallery_imgs_event


        num_total_pids_event = num_train_pids_event + num_query_pids_event
        num_total_tracklets_event = num_train_tracklets_event + num_query_tracklets_event + num_gallery_tracklets_event

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        print("=> MARS_event loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids_event, num_train_tracklets_event))
        print("  query    | {:5d} | {:8d}".format(num_query_pids_event, num_query_tracklets_event))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids_event, num_gallery_tracklets_event))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids_event, num_total_tracklets_event))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")


# img
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

# event
        self.train_event = train_event
        self.query_event = query_event
        self.gallery_event = gallery_event

        self.num_train_pids_event = num_train_pids_event
        self.num_query_pids_event = num_query_pids_event
        self.num_gallery_pids_event = num_gallery_pids_event

        # sys.exit()








    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, ):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            img_names = names[start_index-1:end_index]
            assert 1 <= camid <= 6

            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera*
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


    def _process_data_event(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, ):
        assert home_dir in ['train_origin_mat_divide', 'test_origin_mat_divide2']




        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            img_names = names[start_index-1:end_index]
            assert 1 <= camid <= 6

            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera*
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root_event, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet






