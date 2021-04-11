from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True
opts['model_path'] = '../models/DACapT_NET.pth' # final
opts['crop_size'] = 256
opts['padding'] = 16
opts['extend']=3

# MDNet settings
opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand'] = 1024
opts['batch_test'] = 256
opts['n_samples'] = 256
opts['trans_f'] = 0.6
opts['scale_f'] = 1.05
opts['trans_f_expand'] = 1.5
opts['n_bbreg'] = 1000
opts['overlap_bbreg'] = [0.6, 1]
opts['scale_bbreg'] = [1, 2]
opts['lr_init'] = 0.0001
opts['maxiter_init'] = 30
opts['n_pos_init'] = 500
opts['n_neg_init'] = 5000
opts['overlap_pos_init'] = [0.7, 1]
opts['overlap_neg_init'] = [0, 0.5]
opts['lr_update'] = 0.0002 
opts['maxiter_update'] = 10 # Updated
opts['n_pos_update'] = 50
opts['n_neg_update'] = 200
opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]
opts['success_thr'] = 0.4 # Updated
opts['n_frames_short'] = 20
opts['n_frames_long'] = 100
opts['long_interval'] = 10
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10

# DACapT special settings
opts['ft_layers'] = ['cap']
                        # attention weight       attention bias            TM
opts['stop_layers'] = ['capPrimary_weight10', 'capPrimary_bias11','cap_0_prediction_weight'] # standard, after the first frame, stop updating these parameters
opts['roi_size'] = 7
opts['spatial_scale'] = 1./4
opts['num_input_primary_capsule_channel'] = 512 # number of channels
opts['num_output_primary_capsule_channel'] =512
opts['num_primary_unit'] = 4 # number of primary unit
opts['primary_unit_size']=1152
opts['num_predictions']=2 # positive and negative sample
opts['output_unit_size']=4 # output unit size
opts['num_routing']=3 # number of routing iteration
opts['regrouping_type'] = 'shuffle' # 'local','adjacent','shuffle'
opts['type'] = '4-4-512-3' # The path for saving the results
opts['group_attention'] = True
opts['high_cap_conv'] =False # The settings for weight_conv are different
opts['single_conv'] = None # If true, only one conv is used for TM, if False, several convs will be used for TM, where each conv is computing one element in high capsule
opts['noTM'] = False # If false, do not use Transformation matrix
opts['penalty_attention'] = True
opts['fc'] = False
opts['all_fc'] = False
