# The main process flow is referenced from MDNet (https://github.com/hyeonseobnam/py-MDNet)
import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
sys.path.insert(0,'../modules')
from sample_generator import SampleGenerator, gen_samples
from data_prov import RegionExtractor
from cap_model import capMDNet, MarginLoss, one_hot_encode
from bbreg import BBRegressor
from options import opts
from gen_config import gen_config_uav
from acquire_roi import acquire_roi_samples, acquire_region, acquire_rois_bb
from utils import overlap_ratio, statistic_result

def extract_feat(model, region, samples, im_index, out_layer='conv3', fea_view = False):
    model.eval()
    region = torch.from_numpy(region)
    samples = torch.from_numpy(samples)
    im_index =torch.from_numpy(im_index)
    region = Variable(region)
    if opts['use_gpu']:
        region = region.cuda()
    if out_layer == 'conv3':
        feats_ = model(region, samples, im_index, out_layer=out_layer,fea_view = fea_view)
        feats = feats_.data.clone()
        return feats
    else:
        feats_, sta_g_weight, sta_penalty, atten_map, conv3_fea_ = model(region, samples, im_index, out_layer=out_layer,fea_view = fea_view)
        feats = feats_.data.clone()
        conv3_fea = conv3_fea_.data.clone() 
        return feats, sta_g_weight, sta_penalty, atten_map, conv3_fea


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train(model, criterion, target, optimizer, pos_feats, neg_feats, maxiter, in_layer='capPrimary'):
    
    model.train()
    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score, sta_g_weight, sta_penalty, atten_map, conv3_fea = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
        
		# forward
        model.train()
        batch_feats = torch.cat((batch_pos_feats, batch_neg_feats), 0)
        score, sta_g_weight, sta_penalty, atten_map, conv3_fea = model(batch_feats, in_layer = in_layer)

        # optimize
        loss = criterion(score, target)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()

        print "Iter %d, Loss %.4f" % (iter, loss.data[0])


def run_(model, criterion, target, init_optimizer, img_files, init_rect, video, gt=None):
    # Init bbox
    box_ = np.array(init_rect)
    result = np.zeros((len(img_files),4))
    result_bb = np.zeros((len(img_files),4))
    result[0] = box_
    bbreg_bbox = box_
    result_bb[0] = bbreg_bbox

    tic = time.time()
    # Load first image
    image = Image.open(img_files[0]).convert('RGB') #[W,H] RGB
    image_np = np.asarray(image) #[H, W, 3]
	
    # Init bbox regressor
    # Give the cropped region
	# bbreg_examples_roi is [x_min,y_min,x_max,y_max]
    region, crop_region_sz, bbreg_examples_roi, bbreg_im_index, box__roi, box__crop, coe = acquire_rois_bb(SampleGenerator('uniform', 0.3, 1.5, 1.1, True),
                                                                                   image_np, opts, box_, opts['n_bbreg'], 
                                                                                   opts['overlap_bbreg'], opts['scale_bbreg'])
	# bbreg_examples_reg is [x_min,y_min,w,h]  
    bbreg_examples_reg = np.hstack(((bbreg_examples_roi[:,0]).reshape(-1,1), (bbreg_examples_roi[:,1]).reshape(-1,1),
                                      (bbreg_examples_roi[:,2:]-bbreg_examples_roi[:,:2])))
    bbreg_feats = extract_feat(model, region, bbreg_examples_roi, bbreg_im_index, fea_view = True)
    bbreg = BBRegressor((np.array(region.shape[2:])).reshape(-1,2), overlap = opts['overlap_bbreg'], scale=opts['scale_bbreg'])
    bbreg.train(bbreg_feats, bbreg_examples_reg, box__roi)
    
    # Draw pos/neg samples
    pos_examples, pos_examples_roi, pos_im_index = acquire_roi_samples(SampleGenerator('gaussian', 0.1, 1.2, valid=True), coe,
                                                                     box__crop, crop_region_sz, opts['n_pos_init'], opts['overlap_pos_init'])
    
    neg_examples, neg_examples_roi, neg_im_index  = acquire_roi_samples(SampleGenerator('uniform', 1, 2, 1.1, valid=True), coe, 
                                                                     box__crop, crop_region_sz, opts['n_neg_init']//2, opts['overlap_neg_init'])
    
    neg_examples_whole, neg_examples_roi_whole, neg_im_index_whole  = acquire_roi_samples(SampleGenerator('whole', 0, 1.2, 1.1, valid=True), coe, 
                                                                     box__crop, crop_region_sz, opts['n_neg_init']//2, opts['overlap_neg_init'])
    
    neg_examples_roi = np.concatenate((neg_examples_roi,neg_examples_roi_whole), axis = 0)
    neg_examples_roi = np.random.permutation(neg_examples_roi)
    neg_im_index = np.concatenate((neg_im_index,neg_im_index_whole), axis = 0)
    
	# Extract pos/neg features
    pos_feats = extract_feat(model, region, pos_examples_roi, pos_im_index)
    neg_feats = extract_feat(model, region, neg_examples_roi, neg_im_index)
    feat_dim = pos_feats.size(-1)
    channel_dim = pos_feats.size(-3)

    # Initial training
    train(model, criterion, target, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    model.stop_learnable_params(opts['stop_layers'])
    update_optimizer = set_optimizer(model, opts['lr_update'])
    
	# Init sample generators
    sample_generator = SampleGenerator('gaussian', opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', 0.1, 1.2, valid=True)
    neg_generator = SampleGenerator('uniform', 1.5, 1.2, valid = True)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]
    
    spf_total = time.time()-tic
    
    # Start tracking
    unsuccess_num =0
    for i in range(1,len(img_files)):

        tic = time.time()
        # Load image
        image = Image.open(img_files[i]).convert('RGB')
        image_np = np.asarray(image) #[H, W, 3]
		
        # Cropping
        region, crop_region_sz, coe, box__crop = acquire_region(image_np, box_, opts)
        samples, samples_roi, samples_im_index = acquire_roi_samples(sample_generator, coe,
                                                                     box__crop, crop_region_sz, opts['n_samples'])        
        sample_scores, sta_g_weight, sta_penalty, atten_map, conv3_fea = extract_feat(model, region, samples_roi, samples_im_index, out_layer = 'capsule')
        top_scores, top_idx = sample_scores[:,1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()        
        samples_topk = samples[top_idx]
        samples_topk[:,:2] = samples_topk[:,:2]-box__crop[:2].reshape(-1,2)+box_[:2].reshape(-1,2)
        box__copy = box_.copy()
        box_ = samples_topk.mean(axis=0)#Take the mean value of top 5 as the tracking result
        success = target_score > opts['success_thr']
        
        # Expand search area when failure occurs
        if success:
            unsuccess_num =0
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            unsuccess_num+=1
            sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples_roi = samples_roi[top_idx]
            bbreg_samples_reg = np.hstack(((bbreg_samples_roi[:,0]).reshape(-1,1), (bbreg_samples_roi[:,1]).reshape(-1,1),
                                      (bbreg_samples_roi[:,2:]-bbreg_samples_roi[:,:2])))
            bbreg_feats = extract_feat(model, region, bbreg_samples_roi, samples_im_index[top_idx], fea_view = True)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples_reg)
            bbreg_bbox = bbreg_samples.mean(axis=0)
            
            bbreg_bbox = np.array([bbreg_bbox[0]*coe[0], bbreg_bbox[1]*coe[1], bbreg_bbox[2]*coe[0],bbreg_bbox[3]*coe[1]])
            bbreg_bbox[:2] = np.array(bbreg_bbox[:2]-box__crop[:2] + box__copy[:2])
        else:
            bbreg_bbox = box_
        
        # Copy previous result at failure
        if not success:
            box_ = result[i-1]
            bbreg_bbox = result_bb[i-1]
        
        # Save result
        result[i] = box_
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            # Draw pos/neg samples
            region, crop_region_sz, coe, box__crop = acquire_region(image_np, box_, opts)
            pos_examples, pos_examples_roi, pos_im_index = acquire_roi_samples(pos_generator, coe,
                                                                         box__crop, crop_region_sz, opts['n_pos_update'], opts['overlap_pos_update'])
            neg_examples, neg_examples_roi, neg_im_index  = acquire_roi_samples(neg_generator, coe, 
                                                                     box__crop, crop_region_sz, opts['n_neg_update'], opts['overlap_neg_update'])
            # Extract pos/neg features
            pos_feats = extract_feat(model, region, pos_examples_roi, pos_im_index)
            neg_feats = extract_feat(model, region, neg_examples_roi, neg_im_index)
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > opts['n_frames_long']: # Accumulate updating features
                del pos_feats_all[1] # Keep the information of the first frame 1 or 0 
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0] # Keep the information of the first frame, but it will hurt ironman

        # Short term update
        if (not success) & (unsuccess_num <15):
            nframes = min(opts['n_frames_short'],len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:],0).view(-1,channel_dim, feat_dim, feat_dim) # [20*50, 512,7,7]
            neg_data = torch.stack(neg_feats_all,0).view(-1,channel_dim, feat_dim, feat_dim)# [20 or less *200, 512,7,7]
            train(model, criterion, target, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all,0).view(-1,channel_dim, feat_dim, feat_dim)
            neg_data = torch.stack(neg_feats_all,0).view(-1,channel_dim, feat_dim, feat_dim)
            train(model, criterion, target, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
        spf = time.time()-tic
        spf_total += spf

    fps = len(img_files) / spf_total
    print ("Speed: %.3f" % (fps))
    return result, result_bb, fps, spf_total


def mod_opts(video, opts):
	if video == 'S0308':
		opts['overlap_pos_init'] = [0.65, 1]
	elif video == 'person21':
		opts['overlap_pos_init'] = [0.55, 1]
		opts['overlap_bbreg'] = [0.5, 1]
	elif video == 'uav2':
		opts['overlap_pos_init'] = [0.45, 1]
		opts['overlap_neg_init'] = [0, 0.3]
		opts['overlap_bbreg'] = [0.4, 1]
		opts['scale_bbreg'] = [1, 2.5]
	else:
		opts['overlap_pos_init'] = [0.7, 1]
		opts['overlap_neg_init'] = [0, 0.5]
		opts['overlap_bbreg'] = [0.6, 1]
		opts['scale_bbreg'] = [1, 2]
	return opts

if __name__ == "__main__":
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)
    parser = argparse.ArgumentParser(description='Tracking on datasets')
    parser.add_argument('--dataset', default = 'UAVDT',
                        choices = ['UAV123','UAVDT','DTB70'], help = 'choose the dataset')
    args = parser.parse_args()
    dataset = args.dataset
    json_path = os.path.join('../dataset', dataset+ '.json')
    videos = json.load(open(json_path, 'r'))
    video_order = sorted(videos.keys())
    
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    thresholds_error = np.arange(0, 51, 1)
    tracking_results_path = 'tracking_results.txt'
    if not os.path.exists(tracking_results_path):
        os.mknod(tracking_results_path)
    
	# Initialize statistics
	success_overlap_all = np.zeros((len(video_order), len(thresholds_overlap)))
	prec_error_all = np.zeros((len(video_order), len(thresholds_error)))
	auc = np.zeros(len(video_order))
	prec = np.zeros(len(video_order))
	
	# Start tracking
	for video_id, video in enumerate(video_order):
		np.random.seed(123)
		torch.manual_seed(456)
		torch.cuda.manual_seed(789)
		print(video)
		# modify opts
		opts = mod_opts(video, opts)

		# Init model
		model = capMDNet(opts['roi_size'], opts['spatial_scale'], opts['num_input_primary_capsule_channel'],opts['num_output_primary_capsule_channel'], 
						 opts['num_primary_unit'], opts['primary_unit_size'], opts['num_predictions'], opts['output_unit_size'], opts['num_routing'], 
						 opts['model_path'], network_status = 'Tracking', regrouping_type = opts['regrouping_type'], group_attention =opts['group_attention'], 
						 high_cap_conv = opts['high_cap_conv'], single_conv = opts['single_conv'], 
						 noTM = opts['noTM'], penalty_attention =opts['penalty_attention'], fc = opts['fc'], all_fc = opts['all_fc'])
		
		if opts['use_gpu']:
			model = model.cuda()
		model.set_learnable_params(opts['ft_layers'])
		
		# Initialize criterion and optimizer 
		criterion = MarginLoss()
		target = one_hot_encode(opts['batch_pos'],opts['batch_neg'], 2)
		init_optimizer = set_optimizer(model, opts['lr_init'])
		
		# Generate sequence config
		init_rect, img_files, gt, result_path = config_uav(videos, args, video, video_order,opts)
		if os.path.exists(result_path):
			continue

		# Run tracker
		result, result_bb, fps, spf_total = run_(model, criterion,target, init_optimizer, img_files, init_rect, video, gt)
		
		# Compute statistic results
		success_overlap_all, prec_error_all = statistic_result(gt, result_bb, success_overlap_all, prec_error_all, video_id)
		auc[video_id] = success_overlap_all[video_id,:].mean()
		prec[video_id] = prec_error_all[video_id,20]
		print("%s auc = %.4f, prec = %.4f" % (video, auc[video_id], prec[video_id]))
		
		# Save .mat result
		res_ = {}
		res_['res'] = result_bb.tolist()
		res_['type'] = 'rect'
		res_['fps'] = fps
		res_['sumT'] = spf_total
		res_['sumF'] = len(img_files)
		res_['len'] = len(img_files)
		res_['annoBegin'] = 1
		res_['startFrame'] = 1
		res_['anno'] = gt
		sio.savemat(result_path, res_)
		
		# Save per video result
		if video_id == 0:
			with open(tracking_results_path, 'w') as f:
				f.write(video)
				f.write(',')
				f.write(str(round(auc[video_id],4)))
				f.write('/')
				f.write(str(round(prec[video_id],4)))
				f.write('\n')
		else:
			with open(tracking_results_path, 'a') as f:
				f.write(video)
				f.write(',')
				f.write(str(round(auc[video_id],4)))
				f.write('/')
				f.write(str(round(prec[video_id],4)))
				f.write('\n')
		print('saved')
	print("Total results auc = %.4f, prec = %.4f" % (auc.mean(), prec.mean()))