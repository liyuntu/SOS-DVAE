import os
import pickle
import numpy as np
import numpy.random as rand
import torch

from torch.utils.data.dataset import Dataset
import re
 
#normalize data to have 0 mean and 1 variance
class my_dataset(Dataset):
    def __init__(self,data,label, transform=None):
        self.data=data
        self.label=label  
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        
        # Normalize the data here
        if self.transform:
            x = self.transform(x)
            
        return x,y
    
    def __len__(self):
        return len(self.data)
        
#M is a 1*10 or 1*16 array
def print_metrics(M, flag, epoch, t):
    if len(M) == 16:
        print('{}: {:03d} {:.2f}s || TRAIN LOSS: [REC|CLS|KL] {:.4f} {:.4f} {:.4f} KL-dist {:.4f} ACC {:.4f} AUC {:.4f} || '
                  'TEST LOSS: [REC|CLS] {:.4f} {:.4f} ACC {:.4f} AUC {:.4f} || With 1st enc: TRAIN [CLS LOSS|ACC|AUC] {:.4f} {:.4f} {:.4f} TEST [CLS LOSS|ACC|AUC] {:.4f} {:.4f} {:.4f}'
                  .format(flag, epoch, t, M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8], M[9],M[10], M[11], M[12],M[13], M[14], M[15]))
    else:
        print('{}: {:03d} {:.2f}s || TRAIN LOSS: [REC|CLS|KL] {:.4f} {:.4f} {:.4f} KL-dist {:.4f} ACC {:.4f} AUC {:.4f} || '
                  'TEST LOSS: [REC|CLS] {:.4f} {:.4f} ACC {:.4f} AUC {:.4f}'
                  .format(flag, epoch, t, M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8], M[9]))
    
              
def print_cv_avg_std(A, flag):
    A_avg = np.mean(A, 0) #mean over all folds
    A_std = np.std(A, 0)
    
    if len(A_avg) == 16:
        print('{} AVG +/- STD | TRAIN LOSS: [REC|CLS|KL] {:.6f} +/- {:.6f}, {:.6f} +/- {:.6f}, {:.6f} +/- {:.6f} KL-dist {:.6f} +/- {:.6f} ACC {:.6f} +/- {:.6f} AUC {:.6f} +/- {:.6f} || '
            ' TEST LOSS: [REC|CLS] {:.6f} +/- {:.6f} {:.6f} +/- {:.6f} ACC {:.6f} +/- {:.6f} AUC {:.6f} +/- {:.6f} '
            '|| With 1st enc: TRAIN [CLS LOSS|ACC|AUC] {:.6f} +/- {:.6f}, {:.6f} +/- {:.6f}, {:.6f} +/- {:.6f} TEST [CLS LOSS|ACC|AUC] {:.6f} +/- {:.6f}, {:.6f} +/- {:.6f}, {:.6f} +/- {:.6f}'
            .format(flag, A_avg[0],A_std[0], A_avg[1],A_std[1], A_avg[2],A_std[2], A_avg[3],A_std[3], A_avg[4],A_std[4],
            A_avg[5],A_std[5], A_avg[6],A_std[6], A_avg[7],A_std[7],A_avg[8],A_std[8],A_avg[9],A_std[9],A_avg[10],A_std[10],A_avg[11],A_std[11],A_avg[12],A_std[12],
            A_avg[13],A_std[13],A_avg[14],A_std[14],A_avg[15],A_std[15]))
    else:    
        print('{} AVG +/- STD | TRAIN LOSS: [REC|CLS|KL] {:.6f} +/- {:.6f}, {:.6f} +/- {:.6f}, {:.6f} +/- {:.6f} KL-dist {:.6f} +/- {:.6f} ACC {:.6f} +/- {:.6f} AUC {:.6f} +/- {:.6f} || '
                ' TEST LOSS: [REC|CLS] {:.6f} +/- {:.6f} {:.6f} +/- {:.6f} ACC {:.6f} +/- {:.6f} AUC {:.6f} +/- {:.6f}'
                .format(flag, A_avg[0],A_std[0], A_avg[1],A_std[1], A_avg[2],A_std[2], A_avg[3],A_std[3], A_avg[4],A_std[4],
                A_avg[5],A_std[5], A_avg[6],A_std[6], A_avg[7],A_std[7],A_avg[8],A_std[8],A_avg[9],A_std[9]))    
            

def kl_distance(mu1, log_var1, mu2, log_var2):
    std1 = torch.exp(log_var1/2)
    std2 = torch.exp(log_var2/2)

    batch_size = mu1.size(0)
    dist = 0
    for i in range(batch_size):
        p = torch.distributions.Normal(mu1[i], std1[i])
        q = torch.distributions.Normal(mu2[i], std2[i])
        dist += torch.distributions.kl_divergence(p, q).mean()
   
        # dist += torch.sum(p * torch.log(p / q))

    return dist / batch_size


#mu: mean
#log_var: log variance    
def kl_divergence(mu, log_var):
    return 0.5 * torch.mean(torch.exp(log_var) + mu**2 - 1.0 - log_var) # KL divergence loss
    #return 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1.0 - log_var) # KL divergence loss
        

def save_single_model(model, out_path, fn):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    model_fn = os.path.join(out_path, fn)    
    print('Save the model to: ', model_fn)
    
    #save the model reference
    torch.save(model.state_dict(), model_fn)
    
def save_four_models(enc0, enc1, dec, cls, out_path, fn_prefix):
    save_single_model(enc0, out_path, fn_prefix+'enc0.pt')
    save_single_model(enc1, out_path, fn_prefix+'enc1.pt')
    save_single_model(dec, out_path, fn_prefix+'dec.pt')
    save_single_model(cls, out_path, fn_prefix+'cls.pt')

def save_three_models(enc, dec, cls, out_path, fn_prefix):
    save_two_models(enc, dec, out_path, fn_prefix)
    save_single_model(cls, out_path, fn_prefix+'cls.pt')

def save_two_models(enc, dec, out_path, fn_prefix):
    save_single_model(enc, out_path, fn_prefix+'enc.pt')
    save_single_model(dec, out_path, fn_prefix+'dec.pt')

def get_best_model_filename(model_dir, dataset_name):
    prefix  = 'svae_best_auc_'  #TST SEED dataset, use AUC metric by default
    postfix = '_model_'
    if dataset_name == 'MNIST': #MNIST dataset conventionally use ACC metric
        prefix = 'svae_best_acc_'
    
    best_model_fn = get_fn(model_dir, prefix, postfix)
    
    if not best_model_fn: #if no file of 'svae_best_auc_*', use the 'svae_best_acc_*' instead, since a model only saved once if the acc and auc both higher in the same epoch. Look at "svae" function.        
        prefix = 'svae_best_acc_'
        best_model_fn = get_fn(model_dir, prefix, postfix)
    
    print('The filename of the pretrained best SVAE model start with: ', best_model_fn)
    return best_model_fn

def get_fn(model_dir, prefix, postfix):
    best_index = 0
    best_model_fn = ''
    for i in os.listdir(model_dir):
        if os.path.isfile(os.path.join(model_dir,i)) and i.startswith(prefix):
            #print(i)
            
            #extract the number in the file name (that is the index of epoch of the best model) 
            pattern = prefix + "(.*?)" + postfix
            index = int(re.search(pattern, i).group(1))
            
            #get the file with the highest index value of epoch (which is the best model)
            if best_index < index:
                best_index = index
                best_model_fn = prefix + str(index) + postfix          
    return best_model_fn
    
def load_pretrained_models(model_dir, dataset_name):    
    #get the file name of the best model
    best_model_fn = get_best_model_filename(model_dir, dataset_name)            
            
    #file name of the encoder and classifier
    fn_enc = os.path.join(model_dir, best_model_fn+"enc.pt")
    fn_cls = os.path.join(model_dir, best_model_fn+"cls.pt")
                  
    print('Load pretrained encoder ', fn_enc)
    pretrained_dict_enc = torch.load(fn_enc)
    
    print('Load pretrained classifier ', fn_cls)
    pretrained_dict_cls = torch.load(fn_cls)

    return pretrained_dict_enc, pretrained_dict_cls  

def load_pretrained_model_v2(best_model_fn, postfix):    

    fn = best_model_fn+postfix+".pt"
                  
    print('Load pretrained model: ', fn)
    pretrained_dict = torch.load(fn)

    return pretrained_dict
    
def save_fold_indexes(out_path, train_index, test_index, fold):
    out_dir = os.path.join(out_path, 'fold_index')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('Folder created: ', out_dir)
    
    print('Save the fold train and test indexes to ', out_dir)
    with open(os.path.join(out_dir, str(fold) +'_train.pickle'), 'wb') as myfile:
        pickle.dump(train_index, myfile)
    with open(os.path.join(out_dir, str(fold) +'_test.pickle'), 'wb') as myfile:
        pickle.dump(test_index, myfile)

#Split 18 mice for train, 8 for test, repeate 5 times with given seeds (reproducible)       
def split_train_test(dataDir, n_splits=5, seeds=[42,35,10,20,11]):

    print('Reading TST data...')
    myDict = pickle.load(open(os.path.join(dataDir,'TST_Data.p'),'rb'))
    mm = myDict['mouse']
    mice_ids = np.unique(mm)
    print(mice_ids)
        
    splits = []
    for s in seeds:
        #choose 18 mice
        rand.seed(s)
        train_mice = rand.choice(mice_ids,size=18,replace=False)
        print("Chosen training mice:{}".format(train_mice))
    
        # Get training and test examples
        # train_indx_temp = np.zeros(len(mm))
        # for i in range(len(train_mice)):
            # train_indx_temp[mm==train_mice[i]] = 15
        
        # print(train_indx_temp)
        # train_indx = np.where(train_indx_temp == 15)
        # print(train_indx)
        
        
        train_indx = []
        test_indx  = []
        i = 0
        while i < len(mm):
            if mm[i] in train_mice:
                train_indx.append(i)
            else:
                test_indx.append(i)
            i += 1
            
        print('Train/test indexes: ')
        print(len(train_indx))
        print(len(test_indx))
        
        splits.append((train_indx, test_indx))
        
    return splits
        
    