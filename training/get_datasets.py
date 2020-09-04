import numpy as np
import glob
import os

def get_data_for_dataset(dataset_name, mode):
        if dataset_name == 'imagenet_video':
                datadir = os.path.join(
                        os.path.dirname(__file__),
                        'datasets',
                        'imagenet_video')
                gt = np.load(datadir + '/labels/' + mode + '/labels.npy')
                image_paths = [line.strip() for line in open(datadir + '/labels/' + mode + '/image_names.txt')]
        elif dataset_name == "MOT":
                datadir =  "/home/pats/Documents/datasets/completed_dataset"
                gt = np.load(os.path.join(datadir, mode, "labels.npy"))
                image_paths = [line.strip() for line in open(datadir + '/' + mode + '/image_names.txt')]
                
        return {
            'gt' : gt,
            'image_paths' : image_paths
            }

