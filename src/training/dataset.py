# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import copy
from typing import List, Dict
import zipfile
import json
import random
from typing import Tuple

import numpy as np
import cv2
import PIL.Image
import torch
from src import dnnlib
from omegaconf import DictConfig, OmegaConf

from src.training.layers import sample_frames

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

NUMPY_INTEGER_TYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
NUMPY_FLOAT_TYPES = [np.float16, np.float32, np.float64, np.single, np.double]

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        **kwargs,
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        return {
            'image': image.copy(),
            'label': self.get_label(idx),
        }

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1, f"Labels must be 1-dimensional: {self.label_shape} to use `.label_dim`"
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.res = resolution
        self._path = path
        self.res = resolution
        self._zipfile = None

        if 's3:' in self._path:
            self._path = self._path[5:-4]

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        elif self._path.startswith('s3:'):
            self._type = 'ceph'
            import json
            with open(super_kwargs['dataset_jsonpath'], 'r') as f:
                self._video_dir2frames = json.load(f)
            self._all_fnames = sum(self._video_dir2frames.values(), [])
            from petrel_client.client import Client
            conf_path = os.path.join(super_kwargs['confpath'], "petreloss.conf")
            client = Client(conf_path)
            self.client = client
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if self._type == 'ceph':
            self._image_fnames = sorted(self._all_fnames)
        else:
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError(f'Image files do not match the specified resolution. Resolution is {resolution}, shape is {raw_shape}')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]

        if self._type =='ceph':
            img_bytes = self.client.get(fname)
            assert img_bytes is not None
            img_mem_view = memoryview(img_bytes)
            img_array = np.frombuffer(img_mem_view, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = img[...,::-1]
            if img.shape[0] != img.shape[1]:
                 img= cv2.resize(img, dsize=(self.res,self.res), interpolation=cv2.INTER_CUBIC)
            image = np.array(img).transpose(2, 0, 1)
        else:
            with self._open_file(fname) as f:
                use_pyspng = pyspng is not None and self._file_ext(fname) == '.png'
                image = load_image_from_buffer(f, use_pyspng=use_pyspng, sz=self.res)

        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        labels_files = [f for f in self._all_fnames if f.endswith(fname)]
        if len(labels_files) == 0:
            return None
        assert len(labels_files) == 1, f"There can be only a single {fname} file"
        with self._open_file(labels_files[0]) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[remove_root(fname, self._name).replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)

        if labels.dtype in NUMPY_INTEGER_TYPES:
            labels = labels.astype(np.int64)
        elif labels.dtype in NUMPY_FLOAT_TYPES:
            labels = labels.astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported label dtype: {labels.dtype}")

        return labels

#----------------------------------------------------------------------------

class VideoFramesFolderDataset(Dataset):
    def __init__(self,
        path,                                           # Path to directory or zip.
        cfg: DictConfig,                                # Config
        resolution=None,                                # Unused arg for backward compatibility
        load_n_consecutive: int=None,                   # Should we load first N frames for each video?
        load_n_consecutive_random_offset: bool=True,    # Should we use a random offset when loading consecutive frames?
        subsample_factor: int=1,                        # Sampling factor, i.e. decreasing the temporal resolution
        discard_short_videos: bool=False,               # Should we discard videos that are shorter than `load_n_consecutive`?
        **super_kwargs,                                 # Additional arguments for the Dataset base class.
    ):
        self.res = cfg.resolution

        self.cfg = cfg
        self.sampling_dict = OmegaConf.to_container(OmegaConf.create({**cfg.sampling}))
        self.max_num_frames = cfg.max_num_frames
        self._path = path
        self._zipfile = None
        self.load_n_consecutive = load_n_consecutive
        self.load_n_consecutive_random_offset = load_n_consecutive_random_offset
        self.subsample_factor = subsample_factor
        self.discard_short_videos = discard_short_videos

        if self.subsample_factor > 1 and self.load_n_consecutive is None:
            raise NotImplementedError("Can do subsampling only when loading consecutive frames.")

        listdir_full_paths = lambda d: sorted([os.path.join(d, x) for x in os.listdir(d)])
        name = os.path.splitext(os.path.basename(self._path))[0]
     
        if 's3:' in self._path:
            self._path = self._path[5:-4]
        if os.path.isdir(self._path):
            self._type = 'dir'
            # We assume that the depth is 2
            self._all_objects = {o for d in listdir_full_paths(self._path) for o in (([d] + listdir_full_paths(d)) if os.path.isdir(d) else [d])}
            self._all_objects = {os.path.relpath(o, start=os.path.dirname(self._path)) for o in {self._path}.union(self._all_objects)}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_objects = set(self._get_zipfile().namelist())
        elif self._path.startswith('s3:'):
            self._type = 'ceph'
        else:
            raise IOError('Path must be either a directory or point to a zip archive')

        PIL.Image.init()
        self._video_dir2frames = {}
        if self._type == 'ceph':
            import json
            with open(cfg.dataset_jsonpath, 'r') as f:
                self._video_dir2frames = json.load(f)
            from petrel_client.client import Client
            conf_path = os.path.join(cfg.confpath, "petreloss.conf")
            client = Client(conf_path)
            self.client = client
            # dir_names = client.list(self._path)
            # print('Ceph starts loading!', flush=True)   
            # for dir_name in dir_names:
            #     cur_path = os.path.join(self._path, dir_name)
            #     self._video_dir2frames[cur_path] = []
            #     frames = client.list(cur_path)
            #     for frame in frames:
            #         self._video_dir2frames[cur_path].append(os.path.join(cur_path, frame))
            # print('Ceph loaded!', flush=True)   
        else:         
            objects = sorted([d for d in self._all_objects])
            first_o = objects[0]
            first_o_type = os.path.splitext(first_o)[-1].lower() in PIL.Image.EXTENSION
            if not first_o_type:
                for last_o, o in zip(objects[:-1], objects[1:]):
                    last_o_type = os.path.splitext(last_o)[-1].lower() in PIL.Image.EXTENSION
                    o_type = os.path.splitext(o)[-1].lower() in PIL.Image.EXTENSION
                    if not last_o_type and o_type:
                        curr_d = last_o
                        self._video_dir2frames[curr_d] = []
                
                    if o_type:
                        self._video_dir2frames[curr_d].append(o)
            else:
                last_dir_name = None
                for o in objects:
                    cur_dir_name = o[o.find('-')+1: o.find('/')]
                    if last_dir_name is None or cur_dir_name != last_dir_name:
                        self._video_dir2frames[cur_dir_name] = []
                        last_dir_name = cur_dir_name
                    self._video_dir2frames[cur_dir_name].append(o)
            try:
                self._video_dir2frames = {k:sorted(v, key=lambda s:int(s.split('.')[0].split('/')[-1])) for k,v in self._video_dir2frames.items()}
            except:
                pass

        # chunk long video into small pieces
        chunk = lambda l,s:[l[i*s:min(len(l), i*s+s)] for i in range(len(l)//s+(len(l)%s!=0))]

        if self.discard_short_videos:
            self._video_dir2frames = {d: fs for d, fs in self._video_dir2frames.items() if len(fs) >= self.load_n_consecutive * self.subsample_factor}

        self._video_idx2frames = []
        tmp_idx2frames = [frames for frames in self._video_dir2frames.values()]

        if cfg.chunk:
            for frames in tmp_idx2frames:
                chunk_frames = chunk(frames, cfg.chunk)
                self._video_idx2frames.extend([fs for fs in chunk_frames if len(fs) >= 128])
        else:
            self._video_idx2frames =  [frames for frames in self._video_dir2frames.values()]
                # self._video_idx2frames.extend(chunk_frames)

        if len(self._video_idx2frames) == 0:
            raise IOError('No videos found in the specified archive')

        raw_shape = [len(self._video_idx2frames)] + list(self._load_raw_frames(0)[0][0].shape)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(os.path.dirname(self._path), fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_labels(self):
        """
        We leave the `dataset.json` file in the same format as in the original SG2-ADA repo:
        it's `labels` field is a hashmap of filename-label pairs.
        """
        if self._type == 'ceph':
            return None
        fname = 'dataset.json'
        labels_files = [f for f in self._all_objects if f.endswith(fname)]
        if len(labels_files) == 0:
            return None
        assert len(labels_files) == 1, f"There can be only a single {fname} file"
        with self._open_file(labels_files[0]) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None

        labels = dict(labels)
        # The `dataset.json` file defines a label for each image and
        # For the video dataset, this is both inconvenient and redundant.
        # So let's redefine this
        video_labels = {}
        for filename, label in labels.items():
            dirname = os.path.dirname(filename)
            if dirname in video_labels:
                assert video_labels[dirname] == label
            else:
                video_labels[dirname] = label
        labels = video_labels
        labels = [labels[os.path.normpath(dname).split(os.path.sep)[-1]] for dname in self._video_dir2frames]
        labels = np.array(labels)

        if labels.dtype in NUMPY_INTEGER_TYPES:
            labels = labels.astype(np.int64)
        elif labels.dtype in NUMPY_FLOAT_TYPES:
            labels = labels.astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported label dtype: {labels.dtype}")

        return labels

    def __getitem__(self, idx: int) -> Dict:
        if self.load_n_consecutive:
            num_frames_available = len(self._video_idx2frames[self._raw_idx[idx]])
            assert num_frames_available - self.load_n_consecutive * self.subsample_factor >= 0, f"We have only {num_frames_available} frames available, cannot load {self.load_n_consecutive} frames."

            if self.load_n_consecutive_random_offset:
                random_offset = random.randint(0, num_frames_available - self.load_n_consecutive * self.subsample_factor + self.subsample_factor - 1)
            else:
                random_offset = 0
            frames_idx = np.arange(0, self.load_n_consecutive * self.subsample_factor, self.subsample_factor) + random_offset
        else:
            frames_idx = None

        frames, times = self._load_raw_frames(self._raw_idx[idx], frames_idx=frames_idx)

        assert isinstance(frames, np.ndarray)
        assert list(frames[0].shape) == self.image_shape
        assert frames.dtype == np.uint8
        assert len(frames) == len(times)

        if self._xflip[idx]:
            assert frames.ndim == 4 # TCHW
            frames = frames[:, :, :, ::-1]

        return {
            'image': frames.copy(),
            'label': self.get_label(idx),
            'times': times,
            'video_len': self.get_video_len(idx),
        }

    def get_video_len(self, idx: int) -> int:
        return min(self.max_num_frames, len(self._video_idx2frames[self._raw_idx[idx]]))

    def _load_raw_frames(self, raw_idx: int, frames_idx: List[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        frame_paths = self._video_idx2frames[raw_idx]
        total_len = len(frame_paths)
        offset = 0
        images = []

        if frames_idx is None:
            if total_len > self.max_num_frames:
                offset = random.randint(0, total_len - self.max_num_frames)
            frames_idx = sample_frames(self.sampling_dict, total_video_len=min(total_len, self.max_num_frames)) + offset
        else:
            frames_idx = np.array(frames_idx)

        for frame_idx in frames_idx:
            if self._type == 'ceph':
                img_bytes = self.client.get(frame_paths[frame_idx])
                assert img_bytes is not None
                img_mem_view = memoryview(img_bytes)
                img_array = np.frombuffer(img_mem_view, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = img[..., ::-1]
                if img.shape[0] != img.shape[1]:
                    img = cv2.resize(img, dsize=(self.res,self.res), interpolation=cv2.INTER_CUBIC)
                img = np.array(img).transpose(2, 0, 1)
                images.append(img)
            else:
                with self._open_file(frame_paths[frame_idx]) as f:
                    images.append(load_image_from_buffer(f, sz=self.res))

        return np.array(images), frames_idx - offset

    def compute_max_num_frames(self) -> int:
        return max(len(frames) for frames in self._video_idx2frames)

#----------------------------------------------------------------------------

def load_image_from_buffer(f, use_pyspng: bool=False, sz=256) -> np.ndarray:
    if use_pyspng:
        image = pyspng.load(f.read())
    else:
        image = np.array(PIL.Image.open(f))
    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    
    if image.shape[0] != image.shape[1]:
        image = cv2.resize(image, dsize=(sz,sz), interpolation=cv2.INTER_CUBIC)
    image = image.transpose(2, 0, 1) # HWC => CHW

    return image

#----------------------------------------------------------------------------

def video_to_image_dataset_kwargs(video_dataset_kwargs: dnnlib.EasyDict) -> dnnlib.EasyDict:
    """Converts video dataset kwargs to image dataset kwargs"""
    return dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=video_dataset_kwargs.path,
        use_labels=video_dataset_kwargs.use_labels,
        xflip=video_dataset_kwargs.xflip,
        resolution=video_dataset_kwargs.resolution,
        random_seed=video_dataset_kwargs.get('random_seed'),
        dataset_jsonpath=video_dataset_kwargs.cfg.get('dataset_jsonpath', None),
        confpath=video_dataset_kwargs.cfg.get('confpath', None),
        # Explicitly ignoring the max size, since we are now interested
        # in the number of images instead of the number of videos
        # max_size=video_dataset_kwargs.max_size,
    )

#----------------------------------------------------------------------------

def remove_root(fname: os.PathLike, root_name: os.PathLike):
    """`root_name` should NOT start with '/'"""
    if fname == root_name or fname == ('/' + root_name):
        return ''
    elif fname.startswith(root_name + '/'):
        return fname[len(root_name) + 1:]
    elif fname.startswith('/' + root_name + '/'):
        return fname[len(root_name) + 2:]
    else:
        return fname

#----------------------------------------------------------------------------
