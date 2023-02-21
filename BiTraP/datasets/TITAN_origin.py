import sys
import pickle
import cv2

import numpy as np
import xml.etree.ElementTree as ET

from os.path import join, abspath, exists
from os import listdir, makedirs
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


class TITAN(object):
    def __init__(self, data_path='', regen_pkl=False):
        """
        Constructor of the titan class
        :param data_path: Path to the folder of the dataset
        :param regen_pkl: Whether to regenerate the database
        """
        self._year = '2020'
        self._name = 'TITAN'
        self._regen_pkl = regen_pkl

        # Paths
        self._titan_path = data_path if data_path else self._get_default_path()
        assert exists(self._titan_path), \
            'TITAN path does not exist: {}'.format(self._titan_path)

        self._annotation_path = join(self._titan_path, 'annotations')
        self._data_split_ids_path = join(self._titan_path, 'splits')

    def _get_video_ids_split(self, image_set="all") -> list:
        """
            Returns a list of video ids for a given data split
            :param:  split_vids_path: path of TITAN split
                    image_set: Data split, train, test, val
            :return: The list of video ids
            """
        assert image_set in ["train", "test", "val", "all"]
        vid_ids = []
        sets = [image_set + '_set'] if image_set != 'all' else ['train_set', 'test_set', 'val_set']
        for s in sets:
            # vid_id_file = os.path.join(split_vids_path, s + '.txt')
            vid_id_file = os.path.join(self._data_split_ids_path, s + '.txt')
            with open(vid_id_file, 'rt') as fid:
                vid_ids.extend([x.strip() for x in fid.readlines()])
        return vid_ids

    # Path generators
    @property
    def cache_path(self):
        """
        Generate a path to save cache files
        :return: Cache file folder path
        """
        cache_path = abspath(join(self._titan_path, 'data_cache'))
        if not exists(cache_path):
            makedirs(cache_path)
        return cache_path

    def _get_default_path(self):
        """
        Return the default path where titan_raw files are expected to be placed.
        :return: the default path to the dataset folder
        """
        return 'dataset/titan'

    def _get_center(self, box):
        """
        Calculates the center coordinate of a bounding box
        :param box: Bounding box coordinates
        :return: The center coordinate
        """
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def read_csv_titan(self, vid):
        """
        Column number: title
        0: frames
        1: label
        2: obj_track_id
        3: top
        4: left
        5: height
        6: width
        7: attributes.Trunk Open (vehicle 4 wheeled only)
        8: attributes.Motion Status (vehicle 2 wheeled and vehicle 4 wheeled)
        9: attributes.Doors Open (vehicle 4 wheeled only)
        10: attributes.Communicative (person only)
        11: attributes.Complex Contextual (person only)
        12: attributes.Atomic Actions (person only)
        13: attributes.Simple Context (person only)
        14: attributes.Transporting (person only)
        15: attributes.Age (person only)
        """
        video_number = int(vid.split("_")[1])
        df = pd.read_csv(os.path.join(self._annotation_path, vid + '.csv'))
        veh_rows = df[df['label'] != "person"].index
        df.drop(veh_rows, inplace=True)

        # Drops columns that do not have information relevant to pedestrians
        df.drop([df.columns[1], df.columns[7], df.columns[8], df.columns[9]], axis='columns', inplace=True)
        df.sort_values(by=['obj_track_id', 'frames'], inplace=True)
        ped_info_raw = df.values.tolist()
        pids = df['obj_track_id'].values.tolist()
        pids = list(set(list(map(int, pids))))

        return video_number, ped_info_raw, pids

    def generate_database(self) -> dict:
        """
            Generate a database of jaad dataset by integrating all annotations
            Dictionary structure:
            'vid_id'(str): {
                'width': int
                'height': int
                'ped_annotations'(str): {
                    'ped_id'(str): {
                        'old_id': str
                        'frames: list(int)
                        'bbox': list([x1, y1, x2, y2])
                        'behavior'(str): {
                            'communicative': str
                            'complex_context': str
                            'atomic_actions': str
                            'simple_context': str
                            'transport': str
                        'age'(str):
            :return: A database dictionary
            """
        print('---------------------------------------------------------')
        print("Generating database for TITAN")

        cache_file = join(self.cache_path, 'titan_database.pkl')
        if exists(cache_file) and not self._regen_pkl:
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('titan database loaded from {}'.format(cache_file))
            return database

        vids = self._get_video_ids_split()
        ped_info = {}
        for vid in vids:
            video_number, ped_info_raw, pids = self.read_csv_titan(vid)
            n = len(pids)
            ped_info[vid] = {}
            flag = 0
            ped_info[vid]['width'] = 1920
            ped_info[vid]['height'] = 1200
            ped_info[vid]['ped_annotations'] = {}
            for i in range(n):
                idx = f"ped_{video_number}_{i + 1}"
                ped_info[vid]['ped_annotations'][idx] = {}
                ped_info[vid]['ped_annotations'][idx]["frames"] = []
                ped_info[vid]['ped_annotations'][idx]["bbox"] = []
                ped_info[vid]['ped_annotations'][idx]['behavior'] = {}
                ped_info[vid]['ped_annotations'][idx]['behavior']['communicative'] = []
                ped_info[vid]['ped_annotations'][idx]['behavior']['complex_context'] = []
                ped_info[vid]['ped_annotations'][idx]['behavior']['atomic_actions'] = []
                ped_info[vid]['ped_annotations'][idx]['behavior']['simple_context'] = []
                ped_info[vid]['ped_annotations'][idx]['behavior']['transport'] = []
                ped_info[vid]['ped_annotations'][idx]['age'] = ''
                for j in range(flag, len(ped_info_raw)):
                    if ped_info_raw[j][1] == pids[i]:
                        ele = ped_info_raw[j]
                        t = int(ele[0].split('.')[0])
                        # box = list([ele[3], ele[2], ele[3] + ele[5], ele[2] + ele[4]])
                        box = list(map(round, [ele[3], ele[2], ele[3] + ele[5], ele[2] + ele[4]]))
                        box = list(map(float, box))
                        action = 1 if ele[6] == "walking" else 0
                        ped_info[vid]['ped_annotations'][idx]['frames'].append(t)
                        ped_info[vid]['ped_annotations'][idx]['bbox'].append(box)
                        ped_info[vid]['ped_annotations'][idx]['behavior']['communicative'].append(ele[6])
                        ped_info[vid]['ped_annotations'][idx]['behavior']['complex_context'].append(ele[7])
                        ped_info[vid]['ped_annotations'][idx]['behavior']['atomic_actions'].append(ele[8])
                        ped_info[vid]['ped_annotations'][idx]['behavior']['simple_context'].append(ele[9])
                        ped_info[vid]['ped_annotations'][idx]['behavior']['transport'].append(ele[10])
                        # The annotation in TITAN for elderly is 'senior over 65'
                        # We change it to 'elderly' to stay consistent with JAAD and PIE age annotations
                        if ele[11] == 'senior over 65':
                            ped_info[vid]['ped_annotations'][idx]['age'] = 'elderly'
                        else:
                            ped_info[vid]['ped_annotations'][idx]['age'] = ele[11]
                    else:
                        flag += len(ped_info[vid]['ped_annotations'][idx]["frames"])
                        break
                ped_info[vid]['ped_annotations'][idx]['old_id'] = vid + f'_{pids[i]}'

        with open(cache_file, 'wb') as fid:
            pickle.dump(ped_info, fid, pickle.HIGHEST_PROTOCOL)
        print('The database is written to {}'.format(cache_file))

        return ped_info

    def generate_data_trajectory_sequence(self, image_set, age_type, **opts):
        """
        Generates pedestrian tracks
        :param image_set: the split set to produce for. Options are train, test, val.
        :param opts:
                'fstride': Frequency of sampling from the data.
                'sample_type': Whether to use 'all' pedestrian annotations or the ones
                                    with 'beh'avior only.
                'subset': The subset of data annotations to use. Options are: 'default': Includes high resolution and
                                                                                         high visibility videos
                                                                           'high_visibility': Only videos with high
                                                                                             visibility (include low
                                                                                              resolution videos)
                                                                           'all': Uses all videos
                'height_rng': The height range of pedestrians to use.
                'squarify_ratio': The width/height ratio of bounding boxes. A value between (0,1]. 0 the original
                                        ratio is used.
                'data_split_type': How to split the data. Options: 'default', predefined sets, 'random', randomly split the data,
                                        and 'kfold', k-fold data split (NOTE: only train/test splits).
                'seq_type': Sequence type to generate. Options: 'trajectory', generates tracks, 'crossing', generates
                                  tracks up to 'crossing_point', 'intention' generates tracks similar to human experiments
                'min_track_size': Min track length allowable.
                'random_params: Parameters for random data split generation. (see _get_random_pedestrian_ids)
                'kfold_params: Parameters for kfold split generation. (see _get_kfold_pedestrian_ids)
        :return: Sequence data
        """
        params = {'fstride': 1,
                  'sample_type': 'all',  # 'beh'
                  'subset': 'default',
                  'height_rng': [0, float('inf')],
                  'squarify_ratio': 0,
                  'data_split_type': 'default',  # kfold, random, default
                  'seq_type': 'intention',
                  'min_track_size': 15,
                  'random_params': {'ratios': None,
                                    'val_data': True,
                                    'regen_data': False},
                  'kfold_params': {'num_folds': 5, 'fold': 1}}
        assert all(k in params for k in opts.keys()), "Wrong option(s)." \
                                                      "Choose one of the following: {}".format(list(params.keys()))
        params.update(opts)

        print('---------------------------------------------------------')
        print("Generating action sequence data")
        # self._print_dict(params)
        annot_database = self.generate_database()
        if params['seq_type'] == 'trajectory':
            sequence = self._get_trajectories(image_set, age_type, annot_database, **params)

        return sequence

    def _height_check(self, height_rng, frame_ids, boxes):
        """
        Checks whether the bounding boxes are within a given height limit. If not, it
        will adjust the length of data sequences accordingly
        :param height_rng: Height limit [lower, higher]
        :param frame_ids: List of frame ids
        :param boxes: List of bounding boxes
        :param images: List of images
        :param occlusion: List of occlusions
        :return: The adjusted data sequences
        """
        box, frames = [], []
        for i, b in enumerate(boxes):
            bbox_height = abs(b[0] - b[2])
            if height_rng[0] <= bbox_height <= height_rng[1]:
                box.append(b)

                frames.append(frame_ids[i])

        return box, frames

    def _get_data_ids(self, image_set, params):
        """
        A helper function to generate set id and ped ids (if needed) for processing
        :param image_set: Image-set to generate data
        :param params: Data generation params
        :return: Set and pedestrian ids
        """
        _pids = None

        if params['data_split_type'] == 'default':
            return self._get_video_ids_split(image_set), _pids

        video_ids = self._get_video_ids_split('all', params['subset'])
        if params['data_split_type'] == 'random':
            params['random_params']['sample_type'] = params['sample_type']
            _pids = self._get_random_pedestrian_ids(image_set, **params['random_params'])
        elif params['data_split_type'] == 'kfold':
            params['kfold_params']['sample_type'] = params['sample_type']
            _pids = self._get_kfold_pedestrian_ids(image_set, **params['kfold_params'])

        return video_ids, _pids

    def _get_trajectories(self, image_set, age_type, annotations, **params):
        """
        Generates trajectory data.
        :param params: Parameters for generating trajectories
        :param annotations: The annotations database
        :return: A dictionary of trajectories
        """
        print('---------------------------------------------------------')
        print("Generating trajectory data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        resolution_seq = []
        # _pids is not None if params['data_split_type'] != 'default
        video_ids, _pids = self._get_data_ids(image_set, params)

        for vid in sorted(video_ids):
            img_width = annotations[vid]['width']
            img_height = annotations[vid]['height']
            pid_annots = annotations[vid]['ped_annotations']


            for pid in sorted(annotations[vid]['ped_annotations']):
                if age_type != 'all':
                    if pid_annots[pid]['age'] != age_type:
                        continue

                num_pedestrians += 1
                frame_ids = pid_annots[pid]['frames']
                boxes = pid_annots[pid]['bbox']

                if height_rng[0] > 0 or height_rng[1] < float('inf'):
                    boxes, frame_ids= self._height_check(height_rng, frame_ids, boxes)

                if len(boxes) / seq_stride < params['min_track_size']:
                    continue

                ped_ids = [[pid]] * len(boxes)

                center = [self._get_center(b) for b in boxes]

                box_seq.append(boxes[::seq_stride])
                center_seq.append(center[::seq_stride])
                pids_seq.append(ped_ids[::seq_stride])
                resolutions = [[img_width, img_height]] * len(boxes)
                resolution_seq.append(resolutions[::seq_stride])

        print('Split: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of used pedestrians: %d ' % len(resolution_seq))

        return {'image': image_seq,
                'resolution': resolution_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq}
