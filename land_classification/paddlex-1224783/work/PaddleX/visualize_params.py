import os
import os.path as osp
import pickle
import six
from visualdl import LogWriter


model_save_dir = 'output/deeplabv3p_mobilenetv3_large_ssld/'
vdl_save_dir = 'output/deeplabv3p_mobilenetv3_large_ssld//vdl_out/params_histogram'
vis_var_names = ['conv10_expand_weights', 'conv11_expand_weights', 'conv12_expand_weights']

epoch_paths = list()
for model_dir in os.listdir(model_save_dir):
    if 'epoch' in model_dir:
        epoch_paths.append(model_dir)
epoch_paths = sorted(epoch_paths, key=lambda k: int(k[6:]))

vdl_writer = LogWriter(vdl_save_dir)
for epoch_path in epoch_paths:
    epoch_id = epoch_path.split('_')[-1]
    model_dir = osp.join(model_save_dir, epoch_path)
    params_file = osp.join(model_dir, 'model.pdparams')
    print(params_file)    
    with open(params_file, 'rb') as f:
        params_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
    for var_name, var_param in params_dict.items():
        if var_name in vis_var_names:
            vdl_writer.add_histogram(tag='deeplabv3p_output/{}'.format(var_name),
                                     values=var_param.reshape(-1),
                                     step=int(epoch_id),
                                     buckets=10)
