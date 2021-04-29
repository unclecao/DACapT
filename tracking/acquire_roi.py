import sys
import numpy as np
#import matplotlib.pyplot as plt
sys.path.insert(0,'../modules')
from sample_generator import gen_samples
from utils import crop_image
#from scipy.misc import imresize

def acquire_rois_bb(Generator, image_np, opts, target_bbox_, n, overlap=None, scale=None):
    target_bbox = target_bbox_.copy()
    sample_length = np.round(np.sqrt(np.square(target_bbox[2])+np.square(target_bbox[3]))) #sqrt(w**2+h**2)
    tem = opts['extend']*sample_length
    gt_ctr = np.array(target_bbox[:2]+target_bbox[2:]/2) # center of the original image
    sample_region = np.array([gt_ctr[0]-tem, gt_ctr[1]-tem, gt_ctr[0]+tem, gt_ctr[1]+tem])
    sample_region[2:] = sample_region[2:]-sample_region[:2]#[x_min, y_min, w, h]
    region = np.zeros((1,opts['crop_size'],opts['crop_size'],3),dtype='uint8')
    region[0] = crop_image(image_np, sample_region, opts['crop_size'])       
    region = region.transpose(0,3,1,2)
    region = region.astype('float32') - 128.    
    coe = sample_region[2:]*1./np.array(opts['crop_size'])
    crop_region_sz = sample_region[2:]
    target_bbox[:2] =np.round(sample_region[2:]/2-target_bbox[2:]/2) 
    target_bbox_crop = target_bbox # position of the target in the cropped region 
    target_bbox_roi = np.array([target_bbox_crop[0]*1./coe[0],target_bbox_crop[1]*1./coe[1],target_bbox_crop[2]*1./coe[0],target_bbox_crop[3]*1./coe[1]])
    examples = gen_samples(Generator, target_bbox_crop, n, crop_region_sz, overlap_range=overlap, scale_range = scale)    
    bb_examples_roi = (np.hstack(((examples[:,0]*1./coe[0]).reshape(-1,1), (examples[:,1]*1./coe[1]).reshape(-1,1),
                                           ((examples[:,0]+examples[:,2])*1./coe[0]).reshape(-1,1), ((examples[:,1]+examples[:,3])*1./coe[1]).reshape(-1,1)))).astype('float32')  
    bb_im_index = np.tile(0, len(bb_examples_roi))
    return region, crop_region_sz, bb_examples_roi, bb_im_index, target_bbox_roi, target_bbox_crop, coe

def acquire_roi_samples(Generator, coe, target_bbox_crop, crop_region_sz, n, overlap=None, scale=None):
    target_bbox_crop_ = target_bbox_crop.copy()
    examples = gen_samples(Generator,target_bbox_crop_, n, crop_region_sz, overlap_range=overlap, scale_range = scale)
    examples_roi = (np.hstack(((examples[:,0]*1./coe[0]).reshape(-1,1), (examples[:,1]*1./coe[1]).reshape(-1,1),
                                           ((examples[:,0]+examples[:,2])*1./coe[0]).reshape(-1,1), ((examples[:,1]+examples[:,3])*1./coe[1]).reshape(-1,1)))).astype('float32')
    im_index = np.tile(0, len(examples_roi))
    
    return examples, examples_roi, im_index
    
def acquire_region(image_np, target_bbox_, opts):
    target_bbox = target_bbox_.copy()
    sample_length = np.round(np.sqrt(np.square(target_bbox[2])+np.square(target_bbox[3]))) #sqrt(w**2+h**2)
    tem = opts['extend']*sample_length
    gt_ctr = np.array(target_bbox[:2]+target_bbox[2:]/2) # center of the original image
    sample_region = np.array([gt_ctr[0]-tem, gt_ctr[1]-tem, gt_ctr[0]+tem, gt_ctr[1]+tem])
    sample_region[2:] = sample_region[2:]-sample_region[:2]#[x_min, y_min, w, h]
    region = np.zeros((1,opts['crop_size'],opts['crop_size'],3),dtype='uint8')
    region[0] = crop_image(image_np, sample_region, opts['crop_size'])       
    region = region.transpose(0,3,1,2)
    region = region.astype('float32') - 128.    
    coe = sample_region[2:]*1./np.array(opts['crop_size'])
    crop_region_sz = sample_region[2:]
    target_bbox[:2] =np.round(sample_region[2:]/2-target_bbox[2:]/2) 
    target_bbox_crop = target_bbox # position of the target in the cropped region 

    return region, crop_region_sz, coe, target_bbox_crop