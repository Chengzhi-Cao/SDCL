from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import h5py
# from scipy.misc import imsave

from .iotools import mkdir_if_missing, write_json, read_json
from .bases import BaseVideoDataset


class iLIDSVID_dark_event(BaseVideoDataset):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.

    URL: http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2
    """
    dataset_dir = '/gdata/caocz/Event_Re_ID/ilids-vid'

    event_dataset_dir = '/gdata/caocz/Event_Re_ID/ilids-vid_sim'

    def __init__(self, root='data', split_id=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.data_dir = osp.join(self.dataset_dir, 'i-LIDS-VID_dark_2')
        self.split_dir = osp.join(self.dataset_dir, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.cam_1_path = osp.join(self.dataset_dir, 'i-LIDS-VID_dark_2/sequences/cam1')
        self.cam_2_path = osp.join(self.dataset_dir, 'i-LIDS-VID_dark_2/sequences/cam2')


        self.dataset_dir_event = osp.join(root, self.event_dataset_dir)
        self.dataset_url_event = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.data_dir_event = osp.join(self.dataset_dir_event, 'i-LIDS-VID_mat_divide')
        self.split_dir_event = osp.join(self.dataset_dir_event, 'train-test people splits')
        self.split_mat_path_event = osp.join(self.split_dir_event, 'train_test_splits_ilidsvid.mat')
        self.split_path_event = osp.join(self.dataset_dir_event, 'splits.json')
        self.cam_1_path_event = osp.join(self.dataset_dir_event, 'i-LIDS-VID_mat_divide/sequences/cam1')
        self.cam_2_path_event = osp.join(self.dataset_dir_event, 'i-LIDS-VID_mat_divide/sequences/cam2')





        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))
####################################################################
        splits_event = read_json(self.split_path_event)

        split_event = splits_event[split_id]
        train_dirs_event, test_dirs_event = split_event['train'], split_event['test']


        train = self._process_data(train_dirs, cam1=True, cam2=True)
        query = self._process_data(test_dirs, cam1=True, cam2=False)
        gallery = self._process_data(test_dirs, cam1=False, cam2=True)

        train_event = self._process_data_event(train_dirs_event, cam1=True, cam2=True)
        query_event = self._process_data_event(test_dirs_event, cam1=True, cam2=False)
        gallery_event = self._process_data_event(test_dirs_event, cam1=False, cam2=True)




        if verbose:
            print("=> iLIDS-VID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.train_event = train_event
        self.query_event = query_event
        self.gallery_event = gallery_event



        self.num_train_pids, _, self.num_train_cams = self.get_videodata_info(self.train)
        self.num_query_pids, _, self.num_query_cams = self.get_videodata_info(self.query)
        self.num_gallery_pids, _, self.num_gallery_cams = self.get_videodata_info(self.gallery)


        self.num_train_pids_event, _, self.num_train_cams_event = self.get_videodata_info(self.train_event)
        self.num_query_pids_event, _, self.num_query_cams_event = self.get_videodata_info(self.query_event)
        self.num_gallery_pids_event, _, self.num_gallery_cams_event = self.get_videodata_info(self.gallery_event)



    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        urllib.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.dataset_dir)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits ...")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids // 2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = sorted(glob.glob(osp.join(self.cam_1_path, '*')))
            person_cam2_dirs = sorted(glob.glob(osp.join(self.cam_2_path, '*')))

            person_cam1_dirs = [osp.basename(item) for item in person_cam1_dirs]
            person_cam2_dirs = [osp.basename(item) for item in person_cam2_dirs]

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split, num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split, :num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets


    def _process_data_event(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path_event, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.mat'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_2_path_event, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.mat'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets
    