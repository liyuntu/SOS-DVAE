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
from train_model import svae, svae_refit, svae_2enc, meta_2enc_vae, meta_1enc_vae, vae, vae_refit
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--decoder_fn', type=str, required=True)  #mlp to use the models defined in mlp.py, nmf for nmf.py, cnn for cnn.py
parser.add_argument('--out_path', type=str, required=True)  #location where the intermidate output of the trained models be stored. e.g., output/models/mlp-6
################################################################
parser.add_argument('--dataset_name', type=str, default='TST')
parser.add_argument('--dataDir', type=str, default='/datacommons/carlsonlab/lt187/data')
parser.add_argument('--img_width', default=1, type=int)     #TST=1, MNIST=28
parser.add_argument('--img_height', default=3696, type=int) #L = 3696 #length of the input. E.g., TST=3696, MNIST=28, SEED=1900
parser.add_argument('--out_features', default=20, type=int)  #dimension of the latent space
parser.add_argument('--num_labels', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--step_size', default=30, type=int)
parser.add_argument('--total_epochs', default=80, type=int)
parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), type=str)
#parser.add_argument('--device', default=torch.cuda.set_device(0)) #for cluster???

parser.add_argument('--comp', default=None)
parser.add_argument('--invmu', default=100, type=float)
parser.add_argument('--rec_kl_scale', default=1e-4, type=float)
parser.add_argument('--sim_loss_scale', default=10, type=float)
parser.add_argument('--n_fold', default=10, type=int)

args = parser.parse_args()

#read data
X, y, groups = get_dataset(args.dataset_name, args.dataDir) 

if args.dataset_name == 'SEED' or args.dataset_name == 'SEED_PSD_COH': #SEED dataset, X: (30375, 62, 1000)
    args.num_labels = 3    
elif args.dataset_name == 'MNIST': #SEED dataset, X: (30375, 62, 1000)
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
    kfold = KFold(n_splits=args.n_fold, random_state=42) #MNIST train_index: (63000,), test_index: (7000,)
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
svae_perf = np.zeros([args.n_fold,10], dtype=np.float32)
svae_refit_perf = np.zeros([args.n_fold,16], dtype=np.float32)
svae_2enc_perf = np.zeros([args.n_fold,10], dtype=np.float32)
meta_2enc_vae_perf = np.zeros([args.n_fold,10], dtype=np.float32)
meta_1enc_vae_perf = np.zeros([args.n_fold,10], dtype=np.float32)
vae_perf = np.zeros([args.n_fold,10], dtype=np.float32)
vae_refit_perf = np.zeros([args.n_fold,10], dtype=np.float32)
    
for fold, (train_index, test_index) in enumerate(splits):
    save_fold_indexes(args.out_path, train_index, test_index, fold)

    print('>>Train the ', fold, ' fold...')
    #dividing data into folds
    train = torch.utils.data.Subset(dataset, train_index)
    test = torch.utils.data.Subset(dataset, test_index)
    
    #load data 
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=True)
    
    #train the models
    ##vae_refit is based on the saved model from vae, so vae should have been run first before vae_refit
    vae_perf[fold] = vae(args, train_loader, test_loader, fold)
    vae_refit_perf[fold] = vae_refit(args, train_loader, test_loader, fold, True)
    
    svae_perf[fold] = svae(args, train_loader, test_loader, fold)    
    svae_refit_perf[fold] = svae_refit(args, train_loader, test_loader, fold, True)
    meta_1enc_vae_perf[fold] = meta_1enc_vae(args, train_loader, test_loader, fold, True)
        
    svae_2enc_perf[fold] = svae_2enc(args, train_loader, test_loader, fold, True)    
    meta_2enc_vae_perf[fold] = meta_2enc_vae(args, train_loader, test_loader, fold, True)

    


##print average and std
print_cv_avg_std(vae_perf, 'VAE')              # a standard vae
print_cv_avg_std(vae_refit_perf, 'VAE_refit') 

print_cv_avg_std(svae_perf, 'SVAE')
print_cv_avg_std(svae_refit_perf, 'SVAE_refit')
print_cv_avg_std(meta_1enc_vae_perf, 'META_1enc_vae')            #meta + single encoder

print_cv_avg_std(svae_2enc_perf, 'SVAE_2enc')
print_cv_avg_std(meta_2enc_vae_perf, 'meta_2enc_vae')  #meta + double encoder

    
