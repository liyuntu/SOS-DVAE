import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms
from sklearn import preprocessing

    
def get_dataset(dataset_name, dataDir, target='task'):
    # LFP dataset
    if dataset_name == 'TST':
        print('Reading TST data...')
        myDict = pickle.load(open(os.path.join(dataDir,'TST_Data.p'),'rb'))
        power = myDict['power']
        coh = myDict['coherence']
        mm = myDict['mouse'] 
        genotype = myDict['genotype']
        task = myDict['task']

        print('power:', power.shape)
        print('coherence:', coh.shape)
        print('mouse: ', mm.shape, '\tNumber of mice: ', np.size(np.unique(mm)))
        print('genotype: ', genotype.shape, '\tNumber of genotypes: ', np.size(np.unique(genotype)))

        all_features=np.hstack((10*power,coh))
        
        # Normalization
        unitnorm=preprocessing.Normalizer()
        all_features=unitnorm.fit(X=all_features).transform(X=all_features)

        # Prepare data and labels
        X = torch.from_numpy(all_features).float()
        
        if target=='task':
            y = task
        elif target=='genotype':
            y = genotype
        else:
            raise ValueError('Dataset %s not found!' %(dataset_name))
        
        groups = mm
    
    # EEG dataset (SEED)  
    elif dataset_name == 'SEED':
        from dset_loaders.extract_eeg_features import extract_feats, get_data
        
        # Load exsiting feature from file
        data_dir = os.path.join(os.path.join(dataDir,'SEED'), 'Result_zscore_1000')
        
        if os.path.exists(os.path.join(data_dir,'seed_feats.npy')):
            print('Loading the existing features...')
            X = np.load(os.path.join(data_dir,'seed_feats.npy'))
            y = np.load(os.path.join(data_dir,'seed_y.npy'))
            groups = np.load(os.path.join(data_dir,'seed_groups.npy'))

        # Calculate the feature, and save it future use
        else:
            highFreq = 50
            lowFreq = 1 #0.3
            X, y, groups = extract_feats(highFreq, lowFreq, data_dir)    
        
        # Convert numpy.ndarray from double to float
        X = X.astype(np.float32)
    
    # MNIST
    elif dataset_name == 'MNIST':
        print('Reading MNIST data...')
        
        mnist_train = datasets.MNIST(root=dataDir, train=True, transform=transforms.ToTensor(), download=True)
        mnist_test = datasets.MNIST(root=dataDir, train=False, transform=transforms.ToTensor(), download=True)
        
        # Concatnate the train and test images
        X = torch.cat((mnist_train.data, mnist_test.data), 0)
        X = X.view(-1, 28*28)
        print('Data: ', X.shape) 
        X = X.float() #convert torch.uint8 to torch.float

        y = torch.cat((mnist_train.targets, mnist_test.targets), 0)
        print('Label: ', y.shape)
        
        groups = np.array([])
        
    else:
        raise ValueError('Dataset %s not support yet!' %(dataset_name))
            
    return X, y, groups 