#ltu
#Jun 14, 2022
#Compare the KLs for:
#   VAE_refit_enc v.s. SVAE         <=> Use a static fitted decoder from SVAE, to train an encoder of VAE using VAE loss
#   SOS-VAE_refit_enc v.s. SVAE     <=> Use a static fitted decoder from SVAE, to train an encoder of SOS-VAE using VAE loss


import os
import pickle
import numpy as np
import torch
from sklearn import preprocessing
import torchvision.transforms as transforms
from sklearn import decomposition as dp
from utils import my_dataset, print_cv_avg_std, save_fold_indexes, split_train_test
import argparse
from sklearn.model_selection import GroupKFold, KFold    
from dset_loaders.prepare_dataset import get_dataset
from train_model import svae, vae_refit_enc
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--decoder_fn', type=str, required=True)  #mlp to use the models defined in mlp.py, nmf for nmf.py, cnn for cnn.py
parser.add_argument('--out_path', type=str, required=True)  #location where the intermidate output of the trained models be stored. e.g., output/models/mlp-6
parser.add_argument('--dec_model_pathname', type=str, required=True)
################################################################
parser.add_argument('--dataset_name', type=str, default='TST')
parser.add_argument('--dataDir', type=str, default='/datacommons/carlsonlab/lt187/data')
parser.add_argument('--img_width', default=1, type=int)
parser.add_argument('--img_height', default=3696, type=int)
parser.add_argument('--out_features', default=30, type=int)
parser.add_argument('--num_labels', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--step_size', default=30, type=int)
parser.add_argument('--total_epochs', default=80, type=int)
parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), type=str)

parser.add_argument('--comp', default=None)
parser.add_argument('--invmu', default=100, type=float)
parser.add_argument('--rec_kl_scale', default=1e-4, type=float)
parser.add_argument('--sim_loss_scale', default=10, type=float)
parser.add_argument('--n_fold', default=10, type=int)


args = parser.parse_args()

#read data
X, y, groups = get_dataset(args.dataset_name, args.dataDir) 

if args.dataset_name == 'SEED' or args.dataset_name == 'SEED_PSD_COH':
    args.num_labels = 3    
elif args.dataset_name == 'MNIST':
    args.num_labels = 10 
else:
    args.num_labels = 3

args.img_height = X.shape[1]
print('args.img_height: ', args.img_height)

if(args.decoder_fn == 'decoder_nmf'):
#    args.out_features = 20
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)    
    fn = os.path.join(args.out_path, args.decoder_fn+'_comp.pt')
    #Get an NMF model (will be used to initialize the NMF decoder)
    print('NMF decomposition...')
    mod_nmf = dp.NMF(args.out_features)
    SS = mod_nmf.fit_transform(X) #the decomposition applied to the whole tst dataset
    comp = mod_nmf.components_.astype(np.float32)
    print(comp)

    #print(SS.shape) #(70000, 30)
    
    print('Save the NMF comp matrix to: ', fn)
    torch.save(comp, fn)
    
    #to save time, use an existing one
    # print('Load a prior NMF components...')
    # comp = torch.load(fn)
    print('nmf comp shape:', comp.shape) #(out_features, 3696)
    
    args.comp = comp
    

#args.invmu = 100
#args.rec_kl_scale = 1e-4
#args.sim_loss_scale = 10           
dataset = my_dataset(X, y)

#the split is identical with multiple runs, this is important for retraining using previous model, e.g. svae_refit
if args.dataset_name == 'MNIST': #MNIST does not has group info, thus we do KFold cv.
    print('KFold CV...')
    #kfold = KFold(n_splits=args.n_fold, random_state=42, shuffle=True) #MNIST train_index: (63000,), test_index: (7000,)
    kfold = KFold(n_splits=args.n_fold)
    splits = kfold.split(dataset) 
elif args.dataset_name == 'TST_18_8':
    print('Random 18 for training, the rest 8 for test...(repeat args.n_fold times)')
    #generate different splits  
    splits = split_train_test(args.dataDir) 
else:
    print('GroupKFold CV...')
    gkfold = GroupKFold(n_splits=args.n_fold)
    splits = gkfold.split(dataset, groups=groups)    

#Training the models with cross validation    
for fold, (train_index, test_index) in enumerate(splits):
    save_fold_indexes(args.out_path, train_index, test_index, fold)

    print('>>Train the ', fold, ' fold...')
    #dividing data into folds
    train = torch.utils.data.Subset(dataset, train_index)
    test = torch.utils.data.Subset(dataset, test_index)
    
    #load data 
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=True)
    
    vae_refit_enc(args, train_loader, test_loader, fold, dec_model_pathname, True) 
    