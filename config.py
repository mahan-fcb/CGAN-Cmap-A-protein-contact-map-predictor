import argparse
import json
import shutil
import os
from utils import ensure_dirs
class ConfigGAN(object):
    def __init__(self):

        # parse command line arguments
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.data_root = 'data/'
        self.exp_dir = args.exp_name+'/'
        self.log_dir = os.path.join(self.exp_dir, 'images')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.pred_dir = os.path.join(self.exp_dir, 'prediction')
        self.pretrain_model = os.path.join(self.model_dir, args.Premodel_name)
            
        self.test_dir = os.path.join(self.exp_dir, args.test_data)
        if args.test_data == 'initial':
            self.feature_3d =  os.path.join(self.test_dir,'LxLx5_test_sort.npy')
            self.feature_1d =  os.path.join(self.test_dir,'Lx54_test_sort.npy')
            self.contact_map =  os.path.join(self.test_dir,'contact_sort_test.npy')
        elif args.test_data == 'CAMEO':
            self.feature_3d =  os.path.join(self.test_dir,'LxLx5_CAMEO.npy')
            self.feature_1d =  os.path.join(self.test_dir,'Lx54_CAMEO.npy')
            self.contact_map =  os.path.join(self.test_dir,'contact_CAMEO.npy')
        else:
            self.feature_3d =  os.path.join(self.test_dir,'LxLx5_'+ args.test_data+'.npy')
            self.feature_1d =  os.path.join(self.test_dir,'Lx54_'+ args.test_data+'.npy')
            self.contact_map =  os.path.join(self.test_dir,'contact_'+ args.test_data+'.npy')

        ensure_dirs([self.log_dir, self.model_dir, self.pred_dir])

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()

        parser.add_argument('--exp_name', type=str, default="CGAN_CMAP", help="name of this experiment")
        parser.add_argument('--Premodel_name', type=str, default="CGAN_Cmap.h5", help="Pretrain model name")
        parser.add_argument('--traintest', type=str, default="traintest", help="options: [traintest, test]")

        parser.add_argument('--test_data', type=str, default="initial", help="options: [initial, CAMEO, casp12,casp13,casp14]")

        parser.add_argument('--batch_size', type=int, default=4, help="batch size")
        #parser.add_argument('--load_point', type=int, default=500, help="The model load point")
        
        parser.add_argument('--n_testsamples', type=int, default=10, help="number of test samples to save")

        parser.add_argument('--n_epoch', type=int, default=500, help="total number of epochs to train")
        parser.add_argument('--save_step', type=int, default=50, help="save models every x epoch")
        parser.add_argument('--lr', type=float, default=2e-4, help="initial learning rate")
        
        parser.add_argument('--SE_concat', type=int, default=3, help="number of SE concat block")
        args = parser.parse_args()
        return parser, args
