#Extract power and coherence features from SEED dataset
import os
import numpy as np
from dset_loaders.psd_features import get_psd_features
import scipy.io as sio
import mne
from mne.connectivity import seed_target_indices, spectral_connectivity

def get_data(mat, prefix):
    # Train data
    neg_data = mat[prefix+'negative'].astype('float32')
    neu_data = mat[prefix+'neutral'].astype('float32')
    pos_data = mat[prefix+'positive'].astype('float32')
            
    neg_label = np.ones(neg_data.shape[0]) - 1
    neu_label = np.ones(neu_data.shape[0])
    pos_label = np.ones(pos_data.shape[0]) + 1

    case_data = np.concatenate([neg_data, neu_data])
    case_data = np.concatenate([case_data, pos_data])

    case_label = np.concatenate([neg_label, neu_label])
    case_label = np.concatenate([case_label, pos_label])
    
    if prefix == 'test_':
        print('Test set: ')
    else:
        print('Train set: ')                
    print("Data: ", case_data.shape)
    print('Lable: ', case_label.shape)
    
    return case_data, case_label
    
    
def extract_feats(highFreq, lowFreq, data_dir):
    print('Reading SEED data...')
    
    X = []
    y = []
    groups = []

    # Walk through the folder and read the data from all subjects
    print('Reading SEED data from: ', data_dir)
    
    subjects = ['dujingcheng','jianglin','jingjing','liuqiujun','liuye','mahaiwei','penghuiling','sunxiangyu','wangkui','weiwei','wusifan','wuyangwei','xiayulu','yansheng','zhujiayi']

    # Loop each subject
    for subject_index, subject_name in enumerate(subjects):
        print('--Reading subject: ', subject_index, subject_name)
        for dirName, subdirList, fileList in os.walk(data_dir):
            #print('Found directory: %s' % dirName)

            for fname in fileList:
                #print('\t%s' % fname)
                
                # For the three files of this subject
                index = fname.find(subject_name)
                
                if index != -1:
                    print(fname)
                    
                    # Read each file of this subject
                    mat = sio.loadmat(os.path.join(data_dir, fname))
                    
                    tr_data, tr_label = get_data(mat, '')
                    te_data, te_label = get_data(mat, 'test_')
                    
                    # Data
                    X.append(tr_data)
                    X.append(te_data)
                    
                    # Label
                    y.append(tr_label)
                    y.append(te_label)
                    
                    # Group
                    num_items = tr_label.shape[0] + te_label.shape[0]
                    print('num_items: ', num_items)
                    # Creat an array with the same size, holding the index of this subject
                    temp = [subject_index for i in range(num_items)]
                    groups.append(temp)
        
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    groups = np.concatenate(groups, axis=0)

    print('The whole dataset: ')
    print('X: ', X.shape) #X:  (N, 62, 1000)
    print('y:', y.shape)
    print('groups:', groups.shape)  
    
    # Select a subset of 19 channels
    channel_sub = [x-1 for x in [1,3,6,8,10,12,14,24,26,28,30,32,42,44,46,48,50,59,61]]
    X_sub = X[:,channel_sub,:]
    print(X_sub.shape)  #(N, 19, 1000)
    del X
    
    # Calculate power spectral densities features
    print('Calculate the PSD feature...')
    samp_rate = 200.0 # The sampling frequency
    freq_bands = np.array([[1.0,4.0], [4.0, 8.0], [8.0, 12.0], [13,30], [30,50]])
       
    psd_feat = np.zeros([X_sub.shape[0], X_sub.shape[1]*len(freq_bands)], dtype=np.float32)
    for i in range(X_sub.shape[0]):
        psd_feat[i,:] = get_psd_features(X_sub[i], samp_rate, freq_bands)
    
    print('psd_feat: ', psd_feat.shape) #[N, 19*5 = 95]
    
    
    print('Calculate coherence features...')    
    #use some specific frequency bands, e.g., Gamma >30(Hz), BETA (13-30Hz), ALPHA (8-12 Hz), THETA (4-8 Hz), and DELTA(< 4 Hz)
    fmin = np.array([1, 4, 8, 13, 30])
    fmax = np.array([4, 8, 12, 30, 50])
    
    coh_feat = None
    for i in range(X_sub.shape[0]):
        win_i = X_sub[[i],:]
        #print('Input: ', win_i.shape)   #(1, 19, 1000)
        coh, freqs, times, n_epochs, n_tapers = spectral_connectivity(win_i, method='coh', sfreq=samp_rate, fmin=fmin, fmax=fmax, faverage=True, n_jobs=1)
        
        coh = np.array(coh)
        # print(coh.shape)                #(19, 19, 5)
        # print(n_epochs)                 #1, Number of epochs used for computation.
        
        coh_vals = np.reshape(coh, newshape=(1, -1))
        # print(coh_vals.shape)           #(1, 1805)
         
        if coh_feat is None:
            coh_feat = coh_vals
        else:
            coh_feat = np.concatenate((coh_feat, coh_vals), axis=0)
    
    print('coh_feat: ', coh_feat.shape) #[N, 1805]
    
    del X_sub
    
    feats = np.concatenate((psd_feat, coh_feat), axis=1)
    print('feats: ', feats.shape) #(N, 95+1805=1900)
    
    #save for future use
    print('Save features to: ', data_dir)
    np.save(os.path.join(data_dir,'seed_feats.npy'), feats)
    np.save(os.path.join(data_dir,'seed_y.npy'), y)
    np.save(os.path.join(data_dir,'seed_groups.npy'), groups)
    
    print('Done!')
    
    return feats, y, groups


