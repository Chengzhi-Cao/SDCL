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



class PRID_dark_event_vid(object):

    # root_event
    # root = '/gdata/caocz/Event_Re_ID/prid_event_v2/prid_mat/prid'
    root = '/gdata/caocz/Event_Re_ID/prid_2011'
    dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    split_path = osp.join(root, 'splits_prid2011.json')
    cam_a_path = osp.join(root, 'prid_event_divide', 'multi_shot', 'cam_a')
    cam_b_path = osp.join(root, 'prid_event_divide', 'multi_shot', 'cam_b')

    # root_img
    root_img = '/gdata/caocz/Event_Re_ID/prid_2011/prid_2011_dark2'
    dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    img_split_path = osp.join(root_img, 'splits_prid2011.json')
    img_cam_a_path = osp.join(root_img, 'prid_2011', 'multi_shot', 'cam_a')
    img_cam_b_path = osp.join(root_img, 'prid_2011', 'multi_shot', 'cam_b')



    def __init__(self, split_id=0, min_seq_len=0):
        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        # train, num_train_tracklets, num_train_pids, num_imgs_train = \
        #   self._process_data(train_dirs, cam1=True, cam2=True)
        # query, num_query_tracklets, num_query_pids, num_imgs_query = \
        #   self._process_data(test_dirs, cam1=True, cam2=False)
        # gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
        #   self._process_data(test_dirs, cam1=False, cam2=True)

        train, num_train_tracklets, num_train_pids, num_imgs_train, event_train,num_event_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query, event_query,num_event_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery, event_gallery,num_event_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)
          

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
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

        self.train = train  # 存储的是mat或png的路径
        self.query = query
        self.gallery = gallery

        self.train_event = event_train
        self.query_event = event_query
        self.gallery_event = event_gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}

        img_tracklets = []
        img_num_imgs_per_tracklet = []
    
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.mat'))
                img_names.sort()
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))
                #fid.write("cama_" + dirname + '\n')
            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.mat'))
                img_names.sort()
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))
                #fid.write("camb_" + dirname + '\n')

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.img_cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                img_names.sort()
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                img_tracklets.append((img_names, pid, 0))
                img_num_imgs_per_tracklet.append(len(img_names))
                #fid.write("cama_" + dirname + '\n')
            if cam2:
                person_dir = osp.join(self.img_cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                img_names.sort()
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                img_tracklets.append((img_names, pid, 1))
                img_num_imgs_per_tracklet.append(len(img_names))
                #fid.write("camb_" + dirname + '\n')

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        img_tracklets.sort()
        tracklets.sort()

        # print('img_tracklets=',img_tracklets[0])
        # print('tracklets=',tracklets[0])
        # sys.exit()
        return img_tracklets, num_tracklets, num_pids,img_num_imgs_per_tracklet,tracklets, num_imgs_per_tracklet
        # tracklets 有178个序列，存储的是mat的路径
        # num_tracklets = 178
        # num_pids = 89
        # num_imgs_per_tracklet是一个list,里面存储的是照片的数量[30,30,30,20,...]
