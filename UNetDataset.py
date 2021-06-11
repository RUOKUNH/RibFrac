import SimpleITK as stk
from scipy import ndimage
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage.measure import regionprops
import torch
from itertools import product
import nibabel as nib

class UNetTrainDataset(Dataset):
    def __init__(self, data_root, label_root, crop_size = 64, 
                 num_sample = 4, transforms = None, train = True):
        # self.mode = mode
        self.data_root = data_root
        self.label_root = label_root
        self.data_file = sorted(os.listdir(data_root))
        self.label_file = sorted(os.listdir(label_root))
        self.crop_size = crop_size
        self.num_samples = num_sample
        self.transforms = transforms
        self.train = train
    
    def _get_pos_centroids(self, label):
        centroids = [tuple([round(x) for x in prop.centroid])
            for prop in regionprops(label)]

        return centroids
    
    def _get_symmetric_neg_centroids(self, pos_centroids, shape):
        sym_neg_centroids = [(shape[0]-x-1, shape[1]-y-1, shape[2]-z-1) for x, y, z in pos_centroids]

        return sym_neg_centroids
    
    def _get_spine_neg_centroids(self, shape, crop_size, num_samples):
        x_min, x_max = shape[0] // 2 - 40, shape[0] // 2 + 40
        y_min, y_max = shape[1] // 2 - 40, shape[1] // 2 + 40
        z_min, z_max = crop_size // 2, shape[2] - crop_size // 2
        spine_neg_centroids = [(
            np.random.randint(x_min, x_max),
            np.random.randint(y_min, y_max),
            np.random.randint(z_min, z_max)
        ) for _ in range(num_samples)]

        return spine_neg_centroids
    
    def _get_neg_centroids(self, pos_centroids, image_shape):
        num_pos = len(pos_centroids)
        sym_neg_centroids = self._get_symmetric_neg_centroids(
            pos_centroids, image_shape)

        if num_pos < self.num_samples // 2:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, self.num_samples - 2 * num_pos)
        else:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, num_pos)

        return sym_neg_centroids + spine_neg_centroids
    
    def _get_roi_centroids(self, label):
        if self.train:
            # generate positive samples' centroids
            pos_centroids = self._get_pos_centroids(label)

            # generate negative samples' centroids
            neg_centroids = self._get_neg_centroids(pos_centroids,
                label.shape)
            
            pops = []
            
            for i in range(len(pos_centroids)-1, -1, -1):
                centroid = pos_centroids[i]
                for k in range(len(centroid)):
                    if centroid[k] > label.shape[k]:
                        print('pos ', centroid)
                        pos_centroids.pop(i)
                        break
                        
            for i in range(len(neg_centroids)-1, -1, -1):
                centroid = neg_centroids[i]
                for k in range(len(centroid)):
                    if centroid[k] > label.shape[k]:
                        print('neg ', centroid)
                        neg_centroids.pop(i)
                        break
                        

            # sample positives and negatives when necessary
            num_pos = len(pos_centroids)
            num_neg = len(neg_centroids)
            if num_pos >= self.num_samples:
                num_pos = self.num_samples
                num_neg = 0
            elif num_pos >= self.num_samples // 2:
                num_neg = self.num_samples - num_pos

            if num_pos < len(pos_centroids):
                pos_centroids = [pos_centroids[i] for i in np.random.choice(
                    range(0, len(pos_centroids)), size=num_pos, replace=False)]
            if num_neg < len(neg_centroids):
                neg_centroids = [neg_centroids[i] for i in np.random.choice(
                    range(0, len(neg_centroids)), size=num_neg, replace=False)]

            roi_centroids = pos_centroids + neg_centroids
        else:
            roi_centroids = [list(range(0, x, y // 2))[1:-1] + [x - y // 2]
                for x, y in zip(label.shape, self.crop_size)]
            roi_centroids = list(product(*roi_centroids))

        roi_centroids = [tuple([int(x) for x in centroid])
            for centroid in roi_centroids]

        return roi_centroids
    
    def _crop_roi(self, arr, centroid):
        roi = np.ones(tuple([self.crop_size] * 3)) * (-1024)

        src_beg = [max(0, centroid[i] - self.crop_size // 2)
            for i in range(len(centroid))]
        src_end = [min(arr.shape[i], centroid[i] + self.crop_size // 2)
            for i in range(len(centroid))]
        dst_beg = [max(0, self.crop_size // 2 - centroid[i])
            for i in range(len(centroid))]
        dst_end = [min(arr.shape[i] - (centroid[i] - self.crop_size // 2),
            self.crop_size) for i in range(len(centroid))]
        
        roi[
            dst_beg[0]:dst_end[0],
            dst_beg[1]:dst_end[1],
            dst_beg[2]:dst_end[2],
        ] = arr[
            src_beg[0]:src_end[0],
            src_beg[1]:src_end[1],
            src_beg[2]:src_end[2],
        ]

        return roi
    
    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)

        return image
    
    def apply_transforms(self, image_roi):
        for t in self.transforms:
            image_roi = t(image_roi)
        return image_roi
    
    def __getitem__(self, index):
        imgfile = os.path.join(self.data_root, self.data_file[index])
        labelfile = os.path.join(self.label_root, self.label_file[index])
        img = stk.ReadImage(imgfile)
        img_arr = stk.GetArrayFromImage(img)
        label = stk.ReadImage(labelfile)
        label_arr = stk.GetArrayFromImage(label)
        # label_arr = (label_arr > 0).astype(np.int16)
        
        roi_centroids = self._get_roi_centroids(label_arr)

        # crop rois
        img_rois = []
        label_rois = []
        for centroid in roi_centroids:
            try:
                img_roi = self._crop_roi(img_arr, centroid)
                label_roi = self._crop_roi(label_arr, centroid)
            except:
                print('arr ',img_arr.shape, label_arr.shape)
                print(imgfile, labelfile)
                print('centers ',centroid)
                continue
            img_rois.append(img_roi)
            label_rois.append(label_roi)
                
        # img_rois = [self._crop_roi(img_arr, centroid)
        #     for centroid in roi_centroids]
        
        # label_rois = [self._crop_roi(label_arr, centroid)
        #     for centroid in roi_centroids]

        if self.transforms is not None:
            img_rois = [self._apply_transforms(img_roi)
                for img_roi in img_rois]
            
        img_rois = torch.tensor(np.stack(img_rois)[:, np.newaxis],
            dtype=torch.float)
        
        label_rois = (np.stack(label_rois) > 0).astype(np.float)
        label_rois = torch.tensor(label_rois[:, np.newaxis],
            dtype=torch.float)
        
        return img_rois, label_rois
    
    def __len__(self):
        return len(self.data_file)
    
class UNetTestDataset(Dataset):

    def __init__(self, image_path, crop_size=64, transforms=None):
        image = nib.load(image_path)
        self.affine = image.affine
        self.image = image.get_fdata().transpose(2,1,0)
        self.crop_size = crop_size
        self.transforms = transforms
        self.centers = self._get_centers()

    def _get_centers(self):
        dim_coords = [list(range(0, dim, self.crop_size // 2))[1:-1]\
            + [dim - self.crop_size // 2] for dim in self.image.shape]
        centers = list(product(*dim_coords))

        return centers

    def __len__(self):
        return len(self.centers)

    def _crop_patch(self, idx):
        center_x, center_y, center_z = self.centers[idx]
        patch = self.image[
            center_x - self.crop_size // 2:center_x + self.crop_size // 2,
            center_y - self.crop_size // 2:center_y + self.crop_size // 2,
            center_z - self.crop_size // 2:center_z + self.crop_size // 2
        ]

        return patch

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)

        return image

    def __getitem__(self, idx):
        image = self._crop_patch(idx)
        center = self.centers[idx]

        if self.transforms is not None:
            image = self._apply_transforms(image)

        image = torch.tensor(image[np.newaxis], dtype=torch.float)

        return image, center

    @staticmethod
    def _collate_fn(samples):
        images = torch.stack([x[0] for x in samples])
        centers = [x[1] for x in samples]

        return images, centers

    @staticmethod
    def get_dataloader(dataset, batch_size, num_workers=0):
        return DataLoader(dataset, batch_size, num_workers=num_workers,
            collate_fn=UNetTestDataset._collate_fn)
