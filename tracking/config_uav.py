import os
import numpy as np

def config_uav(videos, args, video, video_order, opts):
 
    # Parse the video
    init_rect = np.array(videos[video]['init_rect']).astype(np.float32)
    img_files = [im_f for im_f in videos[video]['image_files']]
    gt_rect = np.array(videos[video]['gt_rect'])
    result_dir = os.path.join('../result',args.dataset,opts['regrouping_type'],opts['type']) 
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, video + '.mat')

    return init_rect, img_files, gt_rect, result_path
