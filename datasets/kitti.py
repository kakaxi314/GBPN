import random
import os
import numpy as np
import glob
from PIL import Image
import torch
import augs


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


class KITTI(torch.utils.data.Dataset):
    """
    kitti depth completion dataset: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """

    def __init__(self, path='datas/kitti', mode='train', height=256, width=1216, npoints=20480, line=64,
                 mean=(90.9950, 96.2278, 94.3213), rand_scale=0.,
                 std=(79.2382, 80.5267, 82.1483), mul_factor=1.,
                 RandCrop=False, tp_min=50):
        self.base_dir = path
        self.height = height
        self.width = width
        self.mode = mode
        self.line = line
        self.npoints = npoints
        self.mul_factor = mul_factor
        if mode == 'train':
            if rand_scale > 0:
                self.transform = augs.Compose([
                    augs.Jitter(),
                    augs.Flip(),
                    augs.RandScale(rand_scale),
                    augs.Norm(mean=mean, std=std),
                ])
            else:
                self.transform = augs.Compose([
                    augs.Jitter(),
                    augs.Flip(),
                    augs.Norm(mean=mean, std=std),
                ])
        else:
            self.transform = augs.Compose([
                augs.Norm(mean=mean, std=std),
            ])
        self.RandCrop = RandCrop and mode == 'train'
        self.tp_min = tp_min
        if mode in ['train', 'val']:
            self.depth_path = os.path.join(self.base_dir, 'data_depth_annotated', mode)
            self.lidar_path = os.path.join(self.base_dir, 'data_depth_velodyne', mode)
            self.depths = list(sorted(glob.iglob(self.depth_path + "/**/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/**/*.png", recursive=True)))
        elif mode == 'selval':
            self.depth_path = os.path.join(self.base_dir, 'val_selection_cropped', 'groundtruth_depth')
            self.lidar_path = os.path.join(self.base_dir, 'val_selection_cropped', 'velodyne_raw')
            self.image_path = os.path.join(self.base_dir, 'val_selection_cropped', 'image')
            self.depths = list(sorted(glob.iglob(self.depth_path + "/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            self.images = list(sorted(glob.iglob(self.image_path + "/*.png", recursive=True)))
        elif mode == 'test':
            self.lidar_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
            self.image_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'image')
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            self.images = list(sorted(glob.iglob(self.image_path + "/*.png", recursive=True)))
            self.depths = self.lidars
        else:
            raise ValueError("Unknown mode: {}".format(mode))
        assert (len(self.depths) == len(self.lidars))

    def __len__(self):
        return len(self.depths)

    def get_item(self, index):
        depth_path = self.depths[index]
        lidar_path = self.lidars[index]
        depth = self.pull_DEPTH(depth_path)
        depth = np.expand_dims(depth, axis=2)
        lidar = self.pull_DEPTH(lidar_path)
        lidar = np.expand_dims(lidar, axis=2)
        K_cam = self.pull_K_cam(index).astype(np.float32)
        file_names = self.depths[index].split('/')
        if self.mode in ['train', 'val']:
            rgb_path = os.path.join(*file_names[:-7], 'raw', file_names[-5].split('_drive')[0], file_names[-5],
                                    file_names[-2], 'data', file_names[-1])
        elif self.mode in ['selval', 'test']:
            rgb_path = self.images[index]
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))
        rgb = self.pull_RGB(rgb_path)
        rgb = rgb.astype(np.float32)
        lidar = self.mul_factor * lidar.astype(np.float32)
        depth = self.mul_factor * depth.astype(np.float32)

        if self.transform:
            rgb, lidar, depth, K_cam = self.transform(rgb, lidar, depth, K_cam)
        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        lidar = lidar.transpose(2, 0, 1).astype(np.float32)
        depth = depth.transpose(2, 0, 1).astype(np.float32)
        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        if self.RandCrop:
            tp = random.randint(self.tp_min, tp)
            lp = random.randint(0, rgb.shape[2] - self.width)
        rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        lidar = lidar[:, tp:tp + self.height, lp:lp + self.width]
        depth = depth[:, tp:tp + self.height, lp:lp + self.width]
        K_cam[0, 2] -= lp
        K_cam[1, 2] -= tp
        lidar = self.sample_lidar_lines(lidar, K_cam)
        return rgb, lidar, K_cam, depth

    def sample_lidar_lines(self, depth_map, intrinsics):
        """
        Takes in input a depth map generated by a 64 line lidar and sparsify the number of
        lines used, returning a sparse depth map with less lidar lines.
        Parameters
        ----------
        depth_map: array like
            sparse depth map of shape H x W x 1
        intrinsics: array like
            the intrinsic parameters of shape 3 x 3
        keep_ratio: float, default 1.0
            the sparsification parameter, 1.0 is 64 lines, 0.50 roughly 32 lines and so on.
        Returns
        -------
        sparse_depth_map: array like
            the sparsified depth map of shape H x W x 1
        """
        if self.line == 64:
            return depth_map
        depth_map = depth_map[0]
        inter = 64 // self.line
        v, u = np.nonzero(depth_map)
        z = depth_map[v, u]
        points = np.linalg.inv(intrinsics) @ (np.vstack([u, v, np.ones_like(u)]) * z)
        points = points.transpose([1, 0])

        scan_y = points[:, 1]
        distance = np.linalg.norm(points, 2, axis=1)
        pitch = np.arcsin(scan_y / distance)

        max_pitch = np.max(pitch)
        min_pitch = np.min(pitch)
        angle_interval = (max_pitch - min_pitch) / 64.0
        angle_label = np.round((pitch - min_pitch) / angle_interval)
        sampling_mask = angle_label % inter == (inter // 2)

        final_mask = np.zeros_like(depth_map, dtype=bool)
        final_mask[depth_map > 0] = sampling_mask
        sampled_depth = np.zeros_like(final_mask, dtype=np.float32)
        sampled_depth[final_mask] = depth_map[final_mask]
        return sampled_depth[None, ...]

    def __getitem__(self, index):
        return self.get_item(index)

    def pull_RGB(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img

    def pull_DEPTH(self, path):
        depth_png = np.array(Image.open(path), dtype=int)
        assert (np.max(depth_png) > 255)
        depth_image = (depth_png / 256.).astype(np.float32)
        return depth_image

    def pull_K_cam(self, index):
        file_names = self.depths[index].split('/')
        if self.mode in ['train', 'val', 'trainval']:
            calib_path = os.path.join(*file_names[:-7], 'raw', file_names[-5].split('_drive')[0],
                                      'calib_cam_to_cam.txt')
            filedata = read_calib_file(calib_path)
            P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
            P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))
            if file_names[-2] == 'image_02':
                K_cam = P_rect_20[0:3, 0:3]
            elif file_names[-2] == 'image_03':
                K_cam = P_rect_30[0:3, 0:3]
            else:
                raise ValueError("Unknown mode: {}".format(file_names[-2]))

        elif self.mode in ['selval', 'test']:
            file_names = self.images[index].split('/')
            calib_path = os.path.join(*file_names[:-2], 'intrinsics', file_names[-1][:-3] + 'txt')
            with open(calib_path, 'r') as f:
                K_cam = f.read().split()
            K_cam = np.array(K_cam, dtype=np.float32).reshape(3, 3)
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))
        return K_cam
