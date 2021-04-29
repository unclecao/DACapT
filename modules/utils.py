from scipy.misc import imresize
import numpy as np
import torch
import scipy.io as sio
import os

def overlap_ratio(rect1, rect2):
    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def extract_regions(image, samples, resize_size): 
    # image [H,W,3]
    regions = np.zeros((len(samples),resize_size,resize_size,3),dtype='uint8')
    for i, sample in enumerate(samples):
        regions[i] = crop_image(image, sample, resize_size)    
    regions = regions.transpose(0,3,1,2)
    regions = regions.astype('float32') - 128.
    return regions

def crop_image(img, bbox, region_size, valid=False):
    
    x,y,w,h = np.array(bbox,dtype='float32')

    half_w, half_h = w/2, h/2
    center_x, center_y = x + half_w, y + half_h
        
    img_h, img_w, _ = img.shape
    min_x = int(x) #int(center_x - half_w + 0.5)
    min_y = int(y) #int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)
    
    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >=0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)
        
        cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='uint8') 
        cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
    
    scaled = imresize(cropped, (region_size, region_size))
    return scaled

def pad_image(img, final_w, final_h, ori_wh): # img=[1,3,H,W]
    padded = np.zeros((1, 3, final_h, final_w), dtype='uint8') 
    padded[:, :, :ori_wh[1], :ori_wh[0]] = img
    return padded

def squash(sj, dim=2):
    sj_mag_sq = torch.sum(sj**2, dim, keepdim=True)
    sj_mag = torch.sqrt(sj_mag_sq)
    v_j = (sj_mag_sq / (1.0 + sj_mag_sq)) * (sj / sj_mag)
    return v_j
def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    
def compute_success_overlap(gt_bb, result_bb, idx):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = overlap_ratio(gt_bb, result_bb)
    overlap_tmp = -np.ones(len(idx))
    overlap_tmp[idx]=iou
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success

def compute_success_error(gt_center, result_center,idx):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    dist[(1-idx).astype(np.bool)] = -1
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success

def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

def statistic_result(gt, bbox, success_overlap_all, success_error_all, video_id):# To deal with NaN situation
    index = gt>0
    idx=(np.sum(index,1)==4)
    gt_cen = convert_bb_to_center(gt)
    bbox_center = convert_bb_to_center(bbox)
    success_error_all[video_id,:] = compute_success_error(gt_cen,bbox_center,idx)
    success_overlap_all[video_id,:]= compute_success_overlap(gt[idx,:], bbox[idx,:], idx)
    return success_overlap_all, success_error_all

def featureVisualize(fea):
    fea=fea.detach().cpu()
    fea=fea.numpy()
    conv3={}
    conv3['conv3'] = fea[0]
    result_path = os.path.join('../visualize/conv3'+ '.mat')
    sio.savemat(result_path, conv3)