import os
# import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
# import torch.nn as nn
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm
import SimpleITK as stk
from UNetDataset import UNetTestDataset
import transforms as tsfm
# from model import UNet


def _remove_low_probs(pred, prob_thresh):
    pred = np.where(pred > prob_thresh, pred, 0)

    return pred


def _remove_spine_fp(pred, image, bone_thresh):
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0

    return np.where(image_spine[..., np.newaxis], 0, pred)


def _remove_small_objects(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def _post_process(pred, image, prob_thresh, bone_thresh, size_thresh):

    # remove connected regions with low confidence
    pred = _remove_low_probs(pred, prob_thresh)

    # remove spine false positives
    pred = _remove_spine_fp(pred, image, bone_thresh)

    # remove small connected regions
    pred = _remove_small_objects(pred, size_thresh)

    return pred


def _predict_single_image(model, dataloader, postprocess, prob_thresh,
        bone_thresh, size_thresh):
    pred = np.zeros(dataloader.dataset.image.shape)
    crop_size = dataloader.dataset.crop_size
    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images, centers = sample
            images = images.cuda(non_blocking=True)
            output = model(images).sigmoid().cpu().numpy().squeeze(axis=1)

            for i in range(output.shape[0]):
                center_x, center_y, center_z = centers[0][i], centers[1][i], centers[2][i]
                cur_pred_patch = pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ]
                pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ] = np.where(cur_pred_patch > 0, np.mean((output[i],
                    cur_pred_patch), axis=0), output[i])

    if postprocess:
        pred = _post_process(pred, dataloader.dataset.image, prob_thresh,
            bone_thresh, size_thresh)

    return pred


def _make_submission_files(pred, image_id):
    pred_label = (pred > 0.5).astype(np.int16)
    if pred_label.sum() > 0:
        print('detection succeed')
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
    # placeholder for label class since classifaction isn't included
    pred_label_code = [0] + [1] * int(pred_label.max())
    pred_image = stk.GetImageFromArray(pred_label)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info


def predict(model, image_dir, pred_dir, unique, model_path = None, prob_thresh = 0.1, bone_thresh = 300,
            size_thresh = 100, postprocess = True):
    batch_size = 8
    postprocess = True if postprocess == "True" else False


    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]

    image_path_list = sorted([os.path.join(image_dir, file)
        for file in os.listdir(image_dir) if "nii" in file])
    if unique is not None:
        image_path_list = [os.path.join(image_dir, unique)]
    
    image_id_list = [os.path.basename(path).split("-")[0]
        for path in image_path_list]

    progress = tqdm(total=len(image_id_list))
    pred_info_list = []
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = UNetTestDataset(image_path, transforms=transforms)
        datasampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=datasampler)
        # dataloader = data.DataLoader(dataset, batch_size = batch_size, collate_fn=UNetTestDataset._collate_fn)
        pred_arr = _predict_single_image(model, dataloader, postprocess,
            prob_thresh, bone_thresh, size_thresh)
        pred_image, pred_info = _make_submission_files(pred_arr, image_id)
        print(image_id, " finished")
        pred_info_list.append(pred_info)
        pred_path = os.path.join(pred_dir, f"{image_id}_pred.nii.gz")
        stk.WriteImage(pred_image, pred_path)

        progress.update()

    pred_info = pd.concat(pred_info_list, ignore_index=True)
    pred_info.to_csv(os.path.join(pred_dir, "pred_info.csv"),
        index=False)