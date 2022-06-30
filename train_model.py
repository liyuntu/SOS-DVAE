import os
#import pickle
import numpy as np
import torch
import torch.nn as nn
import time
import importlib
from collections import OrderedDict
from sklearn.metrics import roc_curve,auc, roc_auc_score
from torch.autograd import grad
from utils import *
from sklearn.preprocessing import LabelBinarizer

def import_model(model_filename):
    if model_filename == 'cnn':
        #the encoder, decoder, and classifier are in the same file
        mod_ = importlib.import_module(model_filename)
        
        model_enc = getattr(mod_, 'encoder')
        model_dec = getattr(mod_, 'decoder')
        model_cls = getattr(mod_, 'classifier')
        
    elif model_filename == 'decoder_mlp' or model_filename == 'decoder_nmf':
        enc_mod = importlib.import_module('encoder')    
        dec_mod = importlib.import_module(model_filename)
        cls_mod = importlib.import_module('classifier')
        
        model_enc = getattr(enc_mod, 'encoder')
        model_dec = getattr(dec_mod, 'decoder')
        model_cls = getattr(cls_mod, 'classifier')
        
    else:
        raise ValueError('The input model not support! ')
        
    return model_enc, model_dec, model_cls

############################variational autoencoder##########################
# Train a standard SVAE using L(x,x') + L(y,y') + KL(q1||p) => distribution q1. p is the true distribution of the oberservations (input).
def svae(args, train_loader, test_loader, fold_index):
    print('\nTrain a standard SVAE model...')
    print(args)
    
    t1 = time.time()
    
    #import the model
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    enc = model_enc(args).to(args.device)
    dec = model_dec(args).to(args.device)
    cls = model_cls(args).to(args.device)

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()

    # optimizers
    optimizer_enc = torch.optim.Adam(enc.parameters(), lr=args.lr)
    optimizer_cls = torch.optim.Adam(cls.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr=args.lr)
    
    enc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=args.step_size, gamma=0.5)
    cls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=args.step_size, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=args.step_size, gamma=0.5)

    # Iterate through train set minibatchs
    best_acc = 0
    best_auc = 0
    best_metrics = np.zeros(10, dtype=np.float32)
    out_path = os.path.join(args.out_path, str(fold_index))
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(10, dtype=np.float32)
        t0 = time.time()
        model_saved = False
        enc.train()
        dec.train()
        cls.train()
        for instances, labels in train_loader:
            x, y = instances.to(args.device), labels.long().to(args.device)

            # Forward pass
            z_mu, z_var, x_sample = enc(x)
            x_ = dec(x_sample)
            y_  = cls(x_sample)
            
            # Zero out the gradients
            optimizer_enc.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_dec.zero_grad()
            
            ## Loss and updates
            cls_loss = criterion_cls(y_,y)
            rec_loss = criterion_rec(x_, x)
            kl_loss = kl_divergence(z_mu, z_var)
            
            loss = cls_loss + args.invmu*rec_loss + args.rec_kl_scale*kl_loss
            loss.backward()

            # Gradient Steps
            optimizer_enc.step()
            optimizer_cls.step()
            optimizer_dec.step()   

            perf_mat[0] += rec_loss.item()
            perf_mat[1] += cls_loss.item()
            perf_mat[2] += kl_loss.item()
            perf_mat[3] += 0                    #place holder of kl_dist, just for uniform print
            perf_mat[4] += compute_acc(y_, y)
            perf_mat[5] += compute_auc(y_, y)
        
        enc_scheduler.step()
        cls_scheduler.step()
        dec_scheduler.step()
        
        ## Testing
        enc.eval()
        dec.eval()
        cls.eval()
        with torch.no_grad():
            # Iterate through test set minibatchs 
            for x, y in test_loader:
                x, y = x.to(args.device), y.long().to(args.device)

                z_mu, z_var, x_sample = enc(x)
                x_ = dec(x_sample)
                y_ = cls(x_sample)              

                perf_mat[6] += criterion_rec(x_, x).item()
                perf_mat[7] += criterion_cls(y_,y).item()
                perf_mat[8] += compute_acc(y_, y)   #prediction acc
                perf_mat[9] += compute_auc(y_, y)   #prediction auc
        
        #to save space, only save models of the end half epochs
        start_save_epo = args.total_epochs * 0.5
        if(epoch > start_save_epo and best_acc<perf_mat[8] and not np.isnan(perf_mat[5])):
            best_acc = perf_mat[8]
            best_metrics = perf_mat
            #save the model            
            save_three_models(enc, dec, cls, out_path, 'svae_best_acc_'+str(epoch)+'_model_') #if change this file name, should also update the function load_pretrained_models.
            model_saved = True
        if(epoch > start_save_epo and best_auc<perf_mat[9] and not np.isnan(perf_mat[5])):
            best_auc = perf_mat[9]  

            if not model_saved:
                #save the model
                save_three_models(enc, dec, cls, out_path, 'svae_best_auc_'+str(epoch)+'_model_')
        
        perf_mat[0:6] /= len(train_loader)
        perf_mat[6:10] /= len(test_loader)

        print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)

    print('Best test ACC|AUC: {:.4f} {:.4f}'.format(best_acc/len(test_loader), best_auc/len(test_loader)))
    
    #print losses and acc of this fold (selected from an approparate epoch)
    print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics

#Fit the SVAE, then do the refit only with the decoder, and compare the two encoders.
#Fix the cls and enc (encoder1, E1) obtained from svae.
#Train a VAE using L(x,x') + KL(q2||p) + KL(q2||q1). q2 is the distribution of encoder2 (E2). The KL(q2||q1) is the descrepancy between q2 and q1.
#Compute the kl_dist between the two encoders.
#Report the acc and auc using cls.
def svae_refit(args, train_loader, test_loader, fold_index, save_model=False):
    print('\nTrain a SVAE_refit model...')
    print(args)
    
    t1 = time.time()
    
    #import the model
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    #define models
    enc0 = model_enc(args).to(args.device)  #this will be the pretrained encoder obtained from SVAE.    <=> encoder1
    enc1 = model_enc(args).to(args.device)  #this will be the second encoder trains here using VAE.     <=> encoder2
    dec = model_dec(args).to(args.device)
    cls = model_cls(args).to(args.device)
    
    out_path = os.path.join(args.out_path, str(fold_index))
    
    #load the pretrained/fixed enc and cls (that were obtained from the standard SVAE)    
    pretrained_dict_enc, pretrained_dict_cls = load_pretrained_models(out_path, args.dataset_name)
    
    #set model paras to be the pretrained parameters in the dictionary
    enc0.load_state_dict(pretrained_dict_enc)
    cls.load_state_dict(pretrained_dict_cls)            

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()

    # optimizers
    optimizer_enc = torch.optim.Adam(enc1.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr=args.lr)
    
    enc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=args.step_size, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=args.step_size, gamma=0.5)

    # Iterate through train set minibatchs
    best_acc = 0
    best_auc = 0
    best_metrics = np.zeros(16, dtype=np.float32)
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(16, dtype=np.float32)
        t0 = time.time()
        model_saved = False
        enc0.eval()
        enc1.train()
        dec.train()
        cls.eval()
        for instances, labels in train_loader:
            x, y = instances.to(args.device), labels.long().to(args.device)
            
            #the distribution of the pretrained/fixed encoder obtained from SVAE
            z_mu0, z_var0, z0 = enc0(x)

            #the second encoder to be trained here
            z_mu1, z_var1, z1 = enc1(x)
            x_ = dec(z1)
            
            # Zero out the gradients
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            
            #train a standard VAE wrt the similarity between the two encoders
            rec_loss = criterion_rec(x_, x)
            kl_loss = kl_divergence(z_mu1, z_var1)
            sim_loss = kl_distance(z_mu0, z_var0, z_mu1, z_var1)            
            loss = args.invmu*rec_loss + args.rec_kl_scale*kl_loss                                 #this loss only gives acc of ~11%
            #loss = args.invmu*rec_loss + args.rec_kl_scale*kl_loss + args.sim_loss_scale*sim_loss   #this will give acc of ~95% (--invmu 0.001 --rec_kl_scale 0.001 --sim_loss_scale 1)
            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
            
            #classification using the trained encoder and the pretrained/fixed classifier  
            y1_ = cls(z1)
            cls_loss1 = criterion_cls(y1_,y)          
            
            

            perf_mat[0] += rec_loss.item()
            perf_mat[1] += cls_loss1.item()
            perf_mat[2] += kl_loss.item()
            perf_mat[3] += sim_loss.item()
            perf_mat[4] += compute_acc(y1_, y)
            perf_mat[5] += compute_auc(y1_, y)            
            
            #classification using the pretrained/fixed encoder and pretrained/fixed classifier
            y0_ = cls(z0)
            cls_loss0 = criterion_cls(y0_,y)
            #metrics using the pretrained/fixed encoder, only svae_refit has these value.
            perf_mat[10] += cls_loss0.item()
            perf_mat[11] += compute_acc(y0_, y)
            perf_mat[12] += compute_auc(y0_, y)            
                    
        enc_scheduler.step()
        dec_scheduler.step()
        
        ## Testing
        enc0.eval() #the pretrained/fixed encoder (encoder1)
        enc1.eval() #the training encoder (encoder2)
        dec.eval()
        cls.eval()
        with torch.no_grad():
            # Iterate through test set minibatchs 
            for x, y in test_loader:
                x, y = x.to(args.device), y.long().to(args.device)

                _, _, z1 = enc1(x)
                x_ = dec(z1)
                y1_ = cls(z1)              

                perf_mat[6] += criterion_rec(x_, x).item()
                perf_mat[7] += criterion_cls(y1_,y).item()
                perf_mat[8] += compute_acc(y1_, y)   #prediction acc
                perf_mat[9] += compute_auc(y1_, y)   #prediction auc
                
                #classification using the pretrained/fixed encoder and pretrained/fixed classifier
                _, _, z0 = enc0(x)
                y0_ = cls(z0)
                #metrics using the pretrained/fixed encoder, only svae_refit has these value.
                perf_mat[13] += criterion_cls(y0_,y).item()
                perf_mat[14] += compute_acc(y0_, y)
                perf_mat[15] += compute_auc(y0_, y)     
        
        #to save space, only save models of the end half epochs
        start_save_epo = args.total_epochs * 0.5
        if(epoch > start_save_epo and best_acc<perf_mat[8] and not np.isnan(perf_mat[5])):
            best_acc = perf_mat[8]
            best_metrics = perf_mat
            
            if save_model:
                #save the model
                save_two_models(enc1, dec, out_path, 'svae_refit_best_acc_'+str(epoch)+'_model_') #if change this file name, should also update the function load_pretrained_models.
                model_saved = True
        if(epoch > start_save_epo and best_auc<perf_mat[9] and not np.isnan(perf_mat[5])):
            best_auc = perf_mat[9]  

            if save_model and not model_saved:
                #save the model
                save_two_models(enc1, dec, out_path, 'svae_refit_best_auc_'+str(epoch)+'_model_')
        
        perf_mat[0:6] /= len(train_loader)
        perf_mat[6:10] /= len(test_loader)
        
        perf_mat[10:13] /= len(train_loader)
        perf_mat[13:16] /= len(test_loader)

        print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)

    print('Best test ACC|AUC: {:.4f} {:.4f}'.format(best_acc/len(test_loader), best_auc/len(test_loader)))

    #print losses and acc of this fold (selected from an approparate epoch)
    print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics

# Train a supervised VAE with double encoder
def svae_2enc(args, train_loader, test_loader, fold_index, save_model=False):
    print('\nTrain double encoder SVAE model...')
    print(args)
    t1 = time.time()
    
    #import the model
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    enc0 = model_enc(args).to(args.device)
    enc1 = model_enc(args).to(args.device)
    dec  = model_dec(args).to(args.device)
    cls  = model_cls(args).to(args.device)

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()

    # optimizers
    optimizer_enc0 = torch.optim.Adam(enc0.parameters(), lr=args.lr)
    optimizer_enc1 = torch.optim.Adam(enc1.parameters(), lr=args.lr)
    optimizer_cls  = torch.optim.Adam(cls.parameters(),  lr=args.lr)
    optimizer_dec  = torch.optim.Adam(dec.parameters(),  lr=args.lr)
    
    enc0_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc0, step_size=args.step_size, gamma=0.5)
    enc1_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc1, step_size=args.step_size, gamma=0.5)
    cls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=args.step_size, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=args.step_size, gamma=0.5)

    # Iterate through train set minibatchs 
    best_acc = 0
    best_auc = 0
    best_metrics = np.zeros(10, dtype=np.float32)
    out_path = os.path.join(args.out_path, str(fold_index))
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(10, dtype=np.float32)
        t0 = time.time()
        model_saved = False
        enc0.train()
        enc1.train()
        dec.train()
        cls.train()
        for instances, labels in train_loader:
            x, y = instances.to(args.device), labels.long().to(args.device)

            # Forward pass
            z_mu1, z_var1, z1 = enc0(x)
            x_ = dec(z1)
            
            # Zero out the gradients
            optimizer_enc0.zero_grad()
            optimizer_enc1.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_dec.zero_grad()
            
            ## Update on first encoder
            rec_loss = criterion_rec(x_, x)
            kl_loss = kl_divergence(z_mu1, z_var1)
            loss1 = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
            # Backward pass
            #loss1.backward(retain_graph=True)
            loss1.backward()
            # Gradient Steps
            optimizer_enc0.step()
            optimizer_dec.step()

            # Forward pass
            z_mu2, z_var2, z2 = enc1(x)
            y_ = cls(z2)
            
            # Zero out the gradients
            optimizer_enc0.zero_grad()
            optimizer_enc1.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_dec.zero_grad()
            
            ## Loss and updates on second encoder
            cls_loss = criterion_cls(y_,y)
            #sim_loss = kl_distance(z_mu1, z_var1, z_mu2, z_var2)#*args.out_features ##sim_loss: torch.Size([128, 100])
            sim_loss = kl_distance(z_mu1.detach(), z_var1.detach(), z_mu2, z_var2)#*args.out_features ##sim_loss: torch.Size([128, 100])
            loss = cls_loss + args.sim_loss_scale*sim_loss #need tune the weight args.rec_kl_scale for sim_loss
            loss.backward()
            
            # Gradient Steps
            optimizer_enc1.step()
            optimizer_cls.step()

            perf_mat[0] += rec_loss.item()
            perf_mat[1] += cls_loss.item()
            perf_mat[2] += kl_loss.item()
            perf_mat[3] += sim_loss.item()
            perf_mat[4] += compute_acc(y_, y)
            perf_mat[5] += compute_auc(y_, y)      

        enc0_scheduler.step()
        enc1_scheduler.step()
        cls_scheduler.step()
        dec_scheduler.step()        
        
        ## Testing
        enc0.eval()
        enc1.eval()
        dec.eval()
        cls.eval()
        with torch.no_grad():
            # Iterate through test set minibatchs 
            for x, y in test_loader:
                x, y = x.to(args.device), y.long().to(args.device)
                
                _, _, z1 = enc0(x)
                x_ = dec(z1)
                _, _, z2 = enc1(x)
                y_ = cls(z2)    

                perf_mat[6] += criterion_rec(x_, x).item()
                perf_mat[7] += criterion_cls(y_,y).item()
                perf_mat[8] += compute_acc(y_, y)   #prediction acc
                perf_mat[9] += compute_auc(y_, y)   #prediction auc
        
        #to save space, only save models of the end half epochs
        start_save_epo = args.total_epochs * 0.5
        if(epoch > start_save_epo and best_acc<perf_mat[8] and not np.isnan(perf_mat[5])):
            best_acc = perf_mat[8]
            best_metrics = perf_mat
            
            if save_model:
                #save the model
                save_four_models(enc0, enc1, dec, cls, out_path, 'svae_2enc_best_acc_'+str(epoch)+'_model_')
                model_saved = True
        if(epoch > start_save_epo and best_auc<perf_mat[9] and not np.isnan(perf_mat[5])): #training AUC is not NAN
            best_auc = perf_mat[9]
            
            if save_model and not model_saved:
                #save the model
                save_four_models(enc0, enc1, dec, cls, out_path, 'svae_2enc_best_auc_'+str(epoch)+'_model_')
        
        perf_mat[0:6] /= len(train_loader)
        perf_mat[6:10] /= len(test_loader)

        print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)   
        
    print('Best ACC|AUC: {:.4f} {:.4f}'.format(best_acc/len(test_loader), best_auc/len(test_loader)))
    
    #print losses and acc of this fold (selected from an approparate epoch)
    print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics
    

# Train a meta-learning based VAE with double encoder
def meta_2enc_vae(args, train_loader, test_loader, fold_index, save_model=False):
    print('\nTrain double encoder META model...')
    print(args)
    t1 = time.time()
    
    #import the model    
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    enc0 = model_enc(args).to(args.device)
    enc1 = model_enc(args).to(args.device)
    dec  = model_dec(args).to(args.device)
    cls  = model_cls(args).to(args.device)
    
    # print(enc0)
    # print(enc1)
    # print(dec)
    # print(cls)

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    #criterion_cls = nn.BCEWithLogitsLoss()
    criterion_rec = nn.MSELoss()

    # optimizers
    optimizer_enc0 = torch.optim.Adam(enc0.parameters(), lr=args.lr)
    optimizer_enc1 = torch.optim.Adam(enc1.parameters(), lr=args.lr)
    optimizer_cls = torch.optim.Adam(cls.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr=args.lr)

    optimizer_enc0_alt = torch.optim.Adam(enc0.parameters(), lr=args.lr)
    optimizer_dec_alt  = torch.optim.Adam(dec.parameters(), lr=args.lr)
    
    enc0_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc0, step_size=args.step_size, gamma=0.5)
    enc1_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc1, step_size=args.step_size, gamma=0.5)
    cls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=args.step_size, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=args.step_size, gamma=0.5)
    enc0_alt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc0_alt, step_size=args.step_size, gamma=0.5)
    dec_alt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec_alt, step_size=args.step_size, gamma=0.5)
    

    lr_2 = args.lr # define learning rate for second-derivative step

    # Iterate through train set minibatchs 
    best_acc = 0
    best_auc = 0
    best_metrics = np.zeros(10, dtype=np.float32)
    out_path = os.path.join(args.out_path, str(fold_index))
    for epoch in range(args.total_epochs):
        t0 = time.time()         
        perf_mat = np.zeros(10, dtype=np.float32)
        model_saved = False
        enc0.train()
        enc1.train()
        dec.train()
        cls.train()
        for instances, labels in train_loader:
            x, y = instances.to(args.device), labels.long().to(args.device)

            # Forward pass            
            z_mu1, z_var1, z1 = enc0(x)
            x_ = dec(z1)
            
            # Zero out the gradients
            optimizer_enc0.zero_grad()
            optimizer_enc1.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_dec.zero_grad()
            
            ## Update on first encoder
            rec_loss = criterion_rec(x_, x)
            kl_loss = kl_divergence(z_mu1, z_var1)
            loss0 = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
            # Backward pass
            #loss0.backward(retain_graph=True)
            loss0.backward()
            optimizer_enc0.step()
            optimizer_dec.step()

            # Forward pass
            z_mu1, z_var1, z1 = enc0(x)
            x_ = dec(z1)
            
            # Zero out the gradients
            optimizer_enc0.zero_grad()
            optimizer_enc1.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_dec.zero_grad()
            optimizer_enc0_alt.zero_grad()

            ##
            fast_weights = OrderedDict((name, param) for (name, param) in enc0.named_parameters())
            rec_loss = criterion_rec(x_, x)
            kl_loss = kl_divergence(z_mu1, z_var1)
            loss0 = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
            grads = grad(loss0, enc0.parameters(),create_graph=True)
            data = [p.data for p in list(enc0.parameters())]              

            # update encoder's weights
            fast_weights = OrderedDict((name, param - lr_2 * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))
            # compute s with updated weights
            z_mu1_alt, z_var1_alt, z1_alt = enc0.forward(x, fast_weights)

            ########################################Why this step????????????????????????????????????
            y_ = cls(z1_alt)
            cls_loss = criterion_cls(y_, y)
            cls_loss.backward()
            ################################################??????????????###########################
            
            # Gradient Steps
            optimizer_enc0_alt.step()

            # Forward pass
            z_mu1, z_var1, z1 = enc0(x)
            z_mu2, z_var2, z2 = enc1(x)
            y_ = cls(z2)
            
            # Zero out the gradients
            optimizer_enc0.zero_grad()
            optimizer_enc1.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_dec.zero_grad()
            optimizer_dec_alt.zero_grad()
            
            ## Update on second encoder
            cls_loss = criterion_cls(y_,y)
            sim_loss = kl_distance(z_mu1, z_var1, z_mu2, z_var2)
            loss = cls_loss + args.sim_loss_scale*sim_loss
            loss.backward()
            
            # Gradient Steps
            optimizer_enc1.step()
            optimizer_cls.step()
            optimizer_dec_alt.step()
            
            perf_mat[0] += rec_loss.item()
            perf_mat[1] += cls_loss.item()
            perf_mat[2] += kl_loss.item()
            perf_mat[3] += sim_loss.item()
            perf_mat[4] += compute_acc(y_, y)
            perf_mat[5] += compute_auc(y_, y) 
        
        if (epoch + 1) % args.step_size == 0:
            lr_2 = lr_2 * 0.5
        enc0_scheduler.step()
        enc1_scheduler.step()
        cls_scheduler.step()
        dec_scheduler.step()    
        enc0_alt_scheduler.step()
        dec_alt_scheduler.step() 
        
        ## Testing
        enc0.eval()
        enc1.eval()
        dec.eval()
        cls.eval()
        with torch.no_grad():
            # Iterate through test set minibatchs 
            for x, y in test_loader:
                x, y = x.to(args.device), y.long().to(args.device)

                _, _, z1 = enc0(x)
                x_ = dec(z1)
                _, _, z2 = enc1(x)
                y_ = cls(z2)   

                perf_mat[6] += criterion_rec(x_, x).item()
                perf_mat[7] += criterion_cls(y_,y).item()
                perf_mat[8] += compute_acc(y_, y)   #prediction acc
                perf_mat[9] += compute_auc(y_, y)   #prediction auc
        
        #to save space, only save models of the end half epochs
        start_save_epo = args.total_epochs * 0.5
        if(epoch > start_save_epo and best_acc<perf_mat[8] and not np.isnan(perf_mat[5])):
            best_acc = perf_mat[8]
            best_metrics = perf_mat
            
            if save_model:
                #save the model
                save_four_models(enc0, enc1, dec, cls, out_path, 'meta_2enc_svae_best_acc_'+str(epoch)+'_model_')
                model_saved = True
        if(epoch > start_save_epo and best_auc<perf_mat[9] and not np.isnan(perf_mat[5])): #training AUC is not NAN
            best_auc = perf_mat[9]
            
            if save_model and not model_saved: #if the model already saved with best acc, don't need to save it again
                #save the model
                save_four_models(enc0, enc1, dec, cls, out_path, 'meta_2enc_svae_best_auc_'+str(epoch)+'_model_')
        
        perf_mat[0:6] /= len(train_loader)
        perf_mat[6:10] /= len(test_loader)

        print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)
        
    print('Best ACC|AUC: {:.4f} {:.4f}'.format(best_acc/len(test_loader), best_auc/len(test_loader)))
     
    #print losses and acc of this fold (selected from an approparate epoch)
    print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics

    
# Train a meta-learning based VAE with signle encoder
# new name: SOS-VAE
def meta_1enc_vae(args, train_loader, test_loader, fold_index, save_model=False):
    print('\nTrain SOS-VAE model...')
    print(args)
    t1 = time.time()
    
    #import the model    
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    enc = model_enc(args).to(args.device)
    dec = model_dec(args).to(args.device)
    cls = model_cls(args).to(args.device)

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()

    # optimizers
    optimizer_enc = torch.optim.Adam(enc.parameters(), lr=args.lr)
    optimizer_cls = torch.optim.Adam(cls.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr=args.lr)
    optimizer_dec_alt  = torch.optim.Adam(dec.parameters(), lr=args.lr)
    
    enc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=args.step_size, gamma=0.5)
    cls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=args.step_size, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=args.step_size, gamma=0.5)
    dec_alt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec_alt, step_size=args.step_size, gamma=0.5)

    lr_2 = args.lr # define learning rate for second-derivative step

    # Iterate through train set minibatchs 
    best_acc = 0
    best_auc = 0
    best_metrics = np.zeros(10, dtype=np.float32)
    out_path = os.path.join(args.out_path, str(fold_index))
    for epoch in range(args.total_epochs):
        t0 = time.time()         
        perf_mat = np.zeros(10, dtype=np.float32)
        model_saved = False
        enc.train()
        dec.train()
        cls.train()
        for instances, labels in train_loader:
            x, y = instances.to(args.device), labels.long().to(args.device)
            
            # Forward pass            
            z_mu, z_var, z = enc(x)
            x_ = dec(z)
            y_ = cls(z)
            
            # Zero out the gradients
            optimizer_enc.zero_grad() 
            optimizer_dec.zero_grad()
                            
            # Update on reconstruction loss
            rec_loss = criterion_rec(x_, x)
            kl_loss = kl_divergence(z_mu.detach(), z_var.detach())
            loss0 = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
            
            loss0.backward()
            
            optimizer_enc.step()
            optimizer_dec.step()
         
            ## Linking update on s
            optimizer_cls.zero_grad()
            optimizer_dec_alt.zero_grad()
            z_mu, z_var, z = enc(x)
            x_ = dec(z)
            
            #retrain the graph
            rec_loss = criterion_rec(x_, x)
            kl_loss = kl_divergence(z_mu, z_var)
            loss = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
            fast_weights = OrderedDict((name, param) for (name, param) in enc.named_parameters())
            
            grads = grad(loss, enc.parameters(),create_graph=True)
            data = [p.data for p in list(enc.parameters())]              

            # update encoder's weights by applying sgd on classification loss
            fast_weights = OrderedDict((name, param - lr_2 * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))
            # compute latent parameters with updated weights
            z_mu_alt, z_var_alt, z_alt = enc.forward(x, fast_weights)
            
            y_  = cls(z_alt)
            
            #Update on classification loss            
            cls_loss = criterion_cls(y_, y)
            cls_loss.backward()
            optimizer_cls.step()
            optimizer_dec_alt.step()
            
            perf_mat[0] += rec_loss.item()
            perf_mat[1] += cls_loss.item()
            perf_mat[2] += kl_loss.item()
            perf_mat[3] += 0 #placeholder, no sim_loss in this model.
            perf_mat[4] += compute_acc(y_, y)
            perf_mat[5] += compute_auc(y_, y)
            
        
        if (epoch + 1) % args.step_size == 0:
            lr_2 = lr_2 * 0.5
        enc_scheduler.step()
        cls_scheduler.step()
        dec_scheduler.step()    
        dec_alt_scheduler.step() 
        
        ## Testing
        enc.eval()
        dec.eval()
        cls.eval()
        with torch.no_grad():
            # Iterate through test set minibatchs 
            for x, y in test_loader:
                x, y = x.to(args.device), y.long().to(args.device)

                _, _, z = enc(x)
                x_ = dec(z)
                y_ = cls(z)

                perf_mat[6] += criterion_rec(x_, x).item()
                perf_mat[7] += criterion_cls(y_,y).item()
                perf_mat[8] += compute_acc(y_, y)   #prediction acc
                perf_mat[9] += compute_auc(y_, y)   #prediction auc
        
        #to save space, only save models of the end half epochs
        start_epo = args.total_epochs * 0.5
        if(epoch > start_epo and best_acc<perf_mat[8] and not np.isnan(perf_mat[5])):
            best_acc = perf_mat[8]
            best_metrics = perf_mat
            
            if save_model:
                #save the model
                save_three_models(enc, dec, cls, out_path, 'meta_svae_best_acc_'+str(epoch)+'_model_')
                model_saved = True
        if(epoch > start_epo and best_auc<perf_mat[9] and not np.isnan(perf_mat[5])): #training AUC is not NAN
            best_auc = perf_mat[9]
            
            if save_model and not model_saved: #if the model already saved with best acc, don't need to save it again
                #save the model
                save_three_models(enc, dec, cls, out_path, 'meta_svae_best_auc_'+str(epoch)+'_model_')
        
        perf_mat[0:6] /= len(train_loader)
        perf_mat[6:10] /= len(test_loader)

        print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)
        
    print('Best ACC|AUC: {:.4f} {:.4f}'.format(best_acc/len(test_loader), best_auc/len(test_loader)))
     
    #print losses and acc of this fold (selected from an approparate epoch)
    print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics


############################variational autoencoder##########################
# Train a standard VAE using L(x,x') + KL(q1||p) => distribution q1. p is the true distribution of the oberservations (input).
def vae(args, train_loader, test_loader, fold_index):
    print('\nTrain a standard VAE model...')
    print(args)
    
    t1 = time.time()
    
    #import the model
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    enc = model_enc(args).to(args.device)
    dec = model_dec(args).to(args.device)

    # Loss and Optimizer
    criterion_rec = nn.MSELoss()

    # optimizers
    optimizer_enc = torch.optim.Adam(enc.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr=args.lr)
    
    enc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=args.step_size, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=args.step_size, gamma=0.5)

    # Iterate through train set minibatchs
    best_metrics = np.zeros(10, dtype=np.float32)
    out_path = os.path.join(args.out_path, str(fold_index))
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(10, dtype=np.float32)
        t0 = time.time()
        model_saved = False
        enc.train()
        dec.train()
        for instances, labels in train_loader:
            x, y = instances.to(args.device), labels.long().to(args.device)

            # Forward pass
            z_mu, z_var, x_sample = enc(x)
            x_ = dec(x_sample)
            
            # Zero out the gradients
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            
            ## Loss and updates
            rec_loss = criterion_rec(x_, x)
            kl_loss = kl_divergence(z_mu, z_var)
            
            loss = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
            loss.backward()

            # Gradient Steps
            optimizer_enc.step()
            optimizer_dec.step()   

            perf_mat[0] += rec_loss.item()
            #perf_mat[1] += 0                    #place holder of kl_dist, just for uniform print
            perf_mat[2] += kl_loss.item()
            # perf_mat[3] += 0                    #place holder of kl_dist, just for uniform print
            # perf_mat[4] += 0                    #place holder of kl_dist, just for uniform print
            # perf_mat[5] += 0                    #place holder of kl_dist, just for uniform print
        
        enc_scheduler.step()
        dec_scheduler.step()
        
        ## Testing
        enc.eval()
        dec.eval()
        with torch.no_grad():
            # Iterate through test set minibatchs 
            for x, y in test_loader:
                x, y = x.to(args.device), y.long().to(args.device)

                z_mu, z_var, x_sample = enc(x)
                x_ = dec(x_sample)            

                perf_mat[6] += criterion_rec(x_, x).item()
                # perf_mat[7] += 0
                # perf_mat[8] += 0
                # perf_mat[9] += 0
        
        #save the model when all epochs finish.
        if(epoch == args.total_epochs -1):
            best_metrics = perf_mat
            #save the model            
            save_two_models(enc, dec, out_path, 'vae_model_') #if change this file name, should also update the function load_pretrained_models.
                
        
        perf_mat[0:6] /= len(train_loader)
        perf_mat[6:10] /= len(test_loader)

        print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)
    
    #print losses and acc of this fold (selected from an approparate epoch)
    print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics

#Fix the encoder and decoder (obtained in vae), and train a classifier using only the classification loss.
def vae_refit(args, train_loader, test_loader, fold_index, save_model=False):
    print('\nTrain a VAE_refit model...')
    print(args)
    
    t1 = time.time()
    
    #import the model
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    #define models
    enc = model_enc(args).to(args.device)  #this will be the pretrained encoder obtained from SVAE.    <=> encoder1
    dec = model_dec(args).to(args.device)
    cls = model_cls(args).to(args.device)
    
    out_path = os.path.join(args.out_path, str(fold_index))
    
    #load the pretrained/fixed enc and cls (that were obtained from the standard VAE)
    best_model_fn = os.path.join(out_path, 'vae_model_') #this name was defined in vae
    pretrained_dict_enc = load_pretrained_model_v2(best_model_fn, 'enc')
    pretrained_dict_dec = load_pretrained_model_v2(best_model_fn, 'dec')
    
    #set model paras to be the pretrained parameters in the dictionary
    enc.load_state_dict(pretrained_dict_enc)
    dec.load_state_dict(pretrained_dict_dec)            

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()

    # optimizers
    optimizer_cls = torch.optim.Adam(cls.parameters(), lr=args.lr)
    
    cls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=args.step_size, gamma=0.5)

    # Iterate through train set minibatchs
    best_acc = 0
    best_auc = 0
    best_metrics = np.zeros(10, dtype=np.float32)
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(10, dtype=np.float32)
        t0 = time.time()
        model_saved = False
        enc.eval() #the pretrained/fixed encoder
        dec.eval() #the pretrained/fixed decoder
        cls.train()
        for instances, labels in train_loader:
            x, y = instances.to(args.device), labels.long().to(args.device)
            
            #the second encoder to be trained here
            _, _, z = enc(x)
            x_ = dec(z)
            
            #compute reconstruction loss, just for report. The encoder and decoder are unoptimized in this model.
            rec_loss = criterion_rec(x_, x)
            
            #classification using the trained encoder and the pretrained/fixed classifier  
            y1_ = cls(z)
            
            optimizer_cls.zero_grad()
            
            cls_loss = criterion_cls(y1_,y)
            cls_loss.backward()
            optimizer_cls.step()

            perf_mat[0] += rec_loss.item()
            perf_mat[1] += cls_loss.item()
            #perf_mat[2] += 0 #it is 0 by default
            #perf_mat[3] += 0
            perf_mat[4] += compute_acc(y1_, y)
            perf_mat[5] += compute_auc(y1_, y)                 
                       
                    
        cls_scheduler.step()

        
        ## Testing
        cls.eval()
        with torch.no_grad():
            # Iterate through test set minibatchs 
            for x, y in test_loader:
                x, y = x.to(args.device), y.long().to(args.device)

                _, _, z = enc(x)
                x_  = dec(z)
                y1_ = cls(z)              

                perf_mat[6] += criterion_rec(x_, x).item()
                perf_mat[7] += criterion_cls(y1_,y).item()
                perf_mat[8] += compute_acc(y1_, y)   #prediction acc
                perf_mat[9] += compute_auc(y1_, y)   #prediction auc    
        
        #to save space, only save models of the end half epochs
        start_save_epo = args.total_epochs * 0.5
        if(epoch > start_save_epo and best_acc<perf_mat[8] and not np.isnan(perf_mat[5])):
            best_acc = perf_mat[8]
            best_metrics = perf_mat
            
            if save_model:
                #save the model
                save_single_model(cls, out_path, 'vae_refit_best_acc_'+str(epoch)+'_model_cls.pt')
                model_saved = True
        if(epoch > start_save_epo and best_auc<perf_mat[9] and not np.isnan(perf_mat[5])):
            best_auc = perf_mat[9]  

            if save_model and not model_saved:
                #save the model
                save_single_model(cls, out_path, 'vae_refit_best_auc_'+str(epoch)+'_model_cls.pt')
        
        perf_mat[0:6] /= len(train_loader)
        perf_mat[6:10] /= len(test_loader)

        print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)

    print('Best test ACC|AUC: {:.4f} {:.4f}'.format(best_acc/len(test_loader), best_auc/len(test_loader)))

    #print losses and acc of this fold (selected from an approparate epoch)
    print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics
    
        
############################variational autoencoder##########################
# Train a SVAE using sum(L(x,x') + L(y,y') + KL(q1||p)) => distribution q1. p is the true distribution of the oberservations (input).
# each mouse has a encoder, all mouse share the same encoder and classifier.
def svae_unique_encoder(args, train_loader, test_loader, fold_index):
    print('\nTrain a standard SVAE model with unique encoder for each mouse...')
    print(args)
    
    t1 = time.time()
    
    #import the model
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    #each mouse has its own encoder    
    num_mouse = len(train_loader)
                   
    enc = nn.ModuleList([model_enc(args).to(args.device) for i in range(num_mouse+1)])
    #print('The encoder:', enc)
    dec = model_dec(args).to(args.device)
    cls = model_cls(args).to(args.device)

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()

    # optimizers
    # collect the parameters of all encoders
    enc_params = []
    for i in range(num_mouse):
        enc_params += list(enc[i].parameters())       
    
    optimizer_enc = torch.optim.Adam(enc_params, lr=args.lr)
    optimizer_enc_test = torch.optim.Adam(enc[-1].parameters(), lr=args.lr) #this encoder is for the test samples
    optimizer_cls = torch.optim.Adam(cls.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr=args.lr)
    
    enc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=args.step_size, gamma=0.5)
    enc_scheduler_test = torch.optim.lr_scheduler.StepLR(optimizer_enc_test, step_size=args.step_size, gamma=0.5)
    cls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=args.step_size, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=args.step_size, gamma=0.5)

    # Iterate through train set minibatchs
    best_acc = 0
    best_auc = 0
    best_metrics = np.zeros(10, dtype=np.float32)
    out_path = os.path.join(args.out_path, str(fold_index))
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(10, dtype=np.float32)
        t0 = time.time()
        model_saved = False
        # for i in range(args.num_domains):
        #enc.train() ###to test....      enc[:].train() 
        enc[0:num_mouse].train()        
        dec.train()
        cls.train()
        
        for it in range(100):
            # loop each mouse
            X_domain = []   # each mouse will be saved into one entry in this list
            y_domain = []
            
            # loop each training mouse
            loss = 0
            cls_loss = 0
            rec_loss = 0
            kl_loss = 0
            
            # Zero out the gradients
            optimizer_enc.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_dec.zero_grad()  
            
            # loop each mouse
            for i in range(len(train_loader)):
                # get data of a mouse
                try:
                    x, y = next(iter(train_loader[i]))
                except StopIteration:
                    print('in exeption')
                    train_loader[i] = enumerate(train_loader[i])
                    x, y = next(iter(train_loader[i]))
                
                #print(x.dtype) #torch.float64
                x, y = x.float().to(args.device), y.long().to(args.device)
                
                # encode
                z_mu, z_var, x_sample = enc[i](x)
                x_ = dec(x_sample)
                y_ = cls(x_sample)
        
                # loss
                cls_loss = criterion_cls(y_,y)
                rec_loss = criterion_rec(x_, x)
                kl_loss = kl_divergence(z_mu, z_var)

                loss += cls_loss + args.invmu*rec_loss + args.rec_kl_scale*kl_loss 
            
                #print('len(train_loader[i]): ', len(train_loader[i])) #18, 19
                perf_mat[0] += rec_loss.item() /len(train_loader)
                perf_mat[1] += cls_loss.item() /len(train_loader)
                perf_mat[2] += kl_loss.item()  /len(train_loader)
    #            perf_mat[3] += 0                    #place holder of kl_dist, just for uniform print
                perf_mat[4] += compute_acc(y_, y) /len(train_loader)
                perf_mat[5] += compute_auc(y_, y) /len(train_loader)          
            
            loss.backward()

            # Gradient Steps
            optimizer_enc.step()
            optimizer_cls.step()
            optimizer_dec.step()            
        
        print('{}: {:03d} {:.2f}s || TRAIN LOSS: [REC|CLS|KL] {:.4f} {:.4f} {:.4f} ACC {:.4f} AUC {:.4f}'.format('Epoch', epoch, time.time() - t0, 
        perf_mat[0], perf_mat[1], perf_mat[2], perf_mat[4],perf_mat[5]))
        
        enc_scheduler.step()
        cls_scheduler.step()
        dec_scheduler.step()
        
        
        # Testing, after the model get optimized
        if epoch > args.test_epoch:
            # fix the decoder and classifer, then train a encoder for the testing sampels
            enc[-1].train()
            dec.eval()
            cls.eval()            

            # Iterate through test set minibatchs 
            for x, y in test_loader:
                x, y = x.float().to(args.device), y.long().to(args.device)
                # print('x: ', x.shape) # torch.Size([128, 1980])
                # print('y: ', y.shape) # torch.Size([128])

                z_mu, z_var, x_sample = enc[-1](x)
                x_ = dec(x_sample)
                y_ = cls(x_sample) 
                
                optimizer_enc_test.zero_grad()
                
                # loss
                #cls_loss = criterion_cls(y_,y)
                rec_loss = criterion_rec(x_, x)
                kl_loss  = kl_divergence(z_mu, z_var)
            
                #loss = cls_loss + args.invmu*rec_loss + args.rec_kl_scale*kl_loss
                loss = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
                loss.backward()
                
                optimizer_enc_test.step()

                perf_mat[6] += rec_loss.item() / len(test_loader)
                perf_mat[7] += 0 #cls_loss.item() / len(test_loader)
                perf_mat[8] += compute_acc(y_, y) / len(test_loader)   #prediction acc
                perf_mat[9] += compute_auc(y_, y) / len(test_loader)  #prediction auc                
                
            enc_scheduler_test.step()        
        
            if(best_acc<perf_mat[8]):
                best_acc = perf_mat[8]
                best_metrics = perf_mat
            
            if(best_auc<perf_mat[9]):
                best_auc = perf_mat[9]

            print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)

            print('Best test ACC|AUC: {:.4f} {:.4f}'.format(best_acc, best_auc))
    
            #print losses and acc of this fold (selected from an approparate epoch)
            print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics



############################variational autoencoder##########################
# Train a SOS-VAE
# each mouse has a encoder, all mouse share the same encoder and classifier.
def sos_vae_unique_encoder(args, train_loader, test_loader, fold_index):
    print('\nTrain a SOS-VAE model with unique encoder for each mouse...')
    print(args)
    
    t1 = time.time()
    
    #import the model
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    #each mouse has its own encoder    
    num_mouse = len(train_loader)
                   
    enc = nn.ModuleList([model_enc(args).to(args.device) for i in range(num_mouse+1)])
    #print('The encoder:', enc)
    dec = model_dec(args).to(args.device)
    cls = model_cls(args).to(args.device)

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()

    # optimizers
    # collect the parameters of all encoders for training mouse
    enc_params = []
    for i in range(num_mouse):
        enc_params += list(enc[i].parameters())       
    
    optimizer_enc = torch.optim.Adam(enc_params, lr=args.lr)
    optimizer_enc_test = torch.optim.Adam(enc[-1].parameters(), lr=args.lr) #this encoder is for the test samples
    optimizer_cls = torch.optim.Adam(cls.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr=args.lr)
    optimizer_dec_alt  = torch.optim.Adam(dec.parameters(), lr=args.lr)
    
    enc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=args.step_size, gamma=0.5)
    enc_scheduler_test = torch.optim.lr_scheduler.StepLR(optimizer_enc_test, step_size=args.step_size, gamma=0.5)
    cls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=args.step_size, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=args.step_size, gamma=0.5)
    dec_alt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec_alt, step_size=args.step_size, gamma=0.5)

    lr_2 = args.lr # define learning rate for second-derivative step

    # Iterate through train set minibatchs
    best_acc = 0
    best_auc = 0
    best_metrics = np.zeros(10, dtype=np.float32)
    out_path = os.path.join(args.out_path, str(fold_index))
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(10, dtype=np.float32)
        t0 = time.time()
        model_saved = False
        # for i in range(args.num_domains):
        enc[0:num_mouse].train()
        dec.train()
        cls.train()
        
        for it in range(100):
            
            # loop each training mouse
            loss0 = 0
            cls_loss = 0
            rec_loss = 0
            kl_loss = 0
            
            # Zero out the gradients
            optimizer_enc.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_dec.zero_grad() 
            
            # loop each mouse
            for i in range(len(train_loader)):
                # get data of a mouse
                try:
                    x, y = next(iter(train_loader[i]))
                except StopIteration:
                    print('in exeption')
                    train_loader[i] = enumerate(train_loader[i])
                    x, y = next(iter(train_loader[i]))
                
                #print(x.dtype) #torch.float64
                x, y = x.float().to(args.device), y.long().to(args.device)
                
                # encode
                z_mu, z_var, x_sample = enc[i](x)
                x_ = dec(x_sample)
                y_ = cls(x_sample)
        
                # loss
                rec_loss = criterion_rec(x_, x)
                kl_loss = kl_divergence(z_mu, z_var)

                loss0 += args.invmu*rec_loss + args.rec_kl_scale*kl_loss
                
            loss0.backward()
            
            optimizer_enc.step()
            optimizer_dec.step() 
             
            #torch.autograd.set_detect_anomaly(True)
            optimizer_cls.zero_grad()
            optimizer_dec_alt.zero_grad()
            # loop each mouse
            for i in range(len(train_loader)):
                # get data of a mouse
                try:
                    x, y = next(iter(train_loader[i]))
                except StopIteration:
                    print('in exeption')
                    train_loader[i] = enumerate(train_loader[i])
                    x, y = next(iter(train_loader[i]))
                
                #print(x.dtype) #torch.float64
                x, y = x.float().to(args.device), y.long().to(args.device)
                
                ## Linking update on s
                z_mu, z_var, z = enc[i](x)
                x_ = dec(z)
                
                #retrain the graph
                rec_loss = criterion_rec(x_, x)
                kl_loss = kl_divergence(z_mu, z_var)
                loss = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
                fast_weights = OrderedDict((name, param) for (name, param) in enc[i].named_parameters())
                
                grads = grad(loss, enc[i].parameters(),create_graph=True)
                data = [p.data for p in list(enc[i].parameters())]              

                # update encoder's weights by applying sgd on classification loss
                fast_weights = OrderedDict((name, param - lr_2 * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))
                # compute latent parameters with updated weights
                z_mu_alt, z_var_alt, z_alt = enc[i].forward(x, fast_weights)
                
                y_  = cls(z_alt)
                
                #Update on classification loss            
                cls_loss += criterion_cls(y_, y)
                
                #print('len(train_loader[i]): ', len(train_loader[i])) #18, 19
                perf_mat[0] += rec_loss.item()
                perf_mat[1] = cls_loss.item()
                perf_mat[2] += kl_loss.item()
    #            perf_mat[3] += 0                    #place holder of kl_dist, just for uniform print
                perf_mat[4] += compute_acc(y_, y)
                perf_mat[5] += compute_auc(y_, y)
            
            perf_mat[0:6] /= len(train_loader)
            
            cls_loss.backward(retain_graph=True)
                
            optimizer_cls.step()
            optimizer_dec_alt.step()
            
        
            
      
        print('{}: {:03d} {:.2f}s || TRAIN LOSS: [REC|CLS|KL] {:.4f} {:.4f} {:.4f} ACC {:.4f} AUC {:.4f}'.format('Epoch', epoch, time.time() - t0, 
        perf_mat[0], perf_mat[1], perf_mat[2], perf_mat[4],perf_mat[5]))
        
        enc_scheduler.step()
        cls_scheduler.step()
        dec_scheduler.step()
        dec_alt_scheduler.step() 
        
        # Testing, after the model get optimized
        if epoch > args.test_epoch:
            # fix the decoder and classifer, then train a encoder for the testing sampels
            enc[-1].train()
            dec.eval()
            cls.eval()
            
            for it in range(100):
                # loop each mouse, to train a encoder for the test mice
                for i in range(len(test_loader)):
                    # get data of a mouse
                    try:
                        x, y = next(iter(test_loader))
                    except StopIteration:
                        print('in exeption')
                        test_loader = enumerate(test_loader)
                        x, y = next(iter(test_loader))
                    x, y = x.float().to(args.device), y.long().to(args.device)    
                    
                    # print('x: ', x.shape) # torch.Size([128, 1980])
                    # print('y: ', y.shape) # torch.Size([128])

                    z_mu, z_var, x_sample = enc[-1](x)
                    x_ = dec(x_sample)
                    y_ = cls(x_sample) 
                    
                    optimizer_enc_test.zero_grad()
                    
                    # loss
                    cls_loss = criterion_cls(y_,y)
                    rec_loss = criterion_rec(x_,x)
                    kl_loss  = kl_divergence(z_mu, z_var)
                
                    loss = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
                    loss.backward()
                    
                    optimizer_enc_test.step()
                    
            # test the trained encoder
            # Iterate through test set minibatchs 
            enc[-1].eval()
            with torch.no_grad():
                for x, y in test_loader:
                    z_mu, z_var, x_sample = enc[-1](x)
                    x_ = dec(x_sample)
                    y_ = cls(x_sample)
                    
                    # loss
                    cls_loss = criterion_cls(y_,y)
                    rec_loss = criterion_rec(x_, x)

                    perf_mat[6] += rec_loss.item() / len(test_loader)
                    perf_mat[7] += cls_loss.item() / len(test_loader)
                    perf_mat[8] += compute_acc(y_, y) / len(test_loader)   #prediction acc
                    perf_mat[9] += compute_auc(y_, y) / len(test_loader)  #prediction auc                
                
            enc_scheduler_test.step()        
        
            if(best_acc<perf_mat[8]):
                best_acc = perf_mat[8]
                best_metrics = perf_mat
            
            if(best_auc<perf_mat[9]):
                best_auc = perf_mat[9]

            print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)

            print('Best test ACC|AUC: {:.4f} {:.4f}'.format(best_acc, best_auc))
    
            #print losses and acc of this fold (selected from an approparate epoch)
            print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics    

############################variational autoencoder##########################
# Train a SOS-VAE
# each mouse has a encoder, all mouse share the same encoder and classifier.
def sos_vae_unique_encoder(args, train_loader, test_loader, fold_index):
    print('\nTrain a SOS-VAE model with unique encoder for each mouse...')
    print(args)
    
    t1 = time.time()
    
    #import the model
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    #each mouse has its own encoder    
    num_mouse = len(train_loader)
                   
    enc = nn.ModuleList([model_enc(args).to(args.device) for i in range(num_mouse+1)])
    #print('The encoder:', enc)
    dec = model_dec(args).to(args.device)
    cls = model_cls(args).to(args.device)

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()

    # optimizers
    # collect the parameters of all encoders for training mouse
    enc_params = []
    for i in range(num_mouse):
        enc_params += list(enc[i].parameters())       
    
    optimizer_enc = torch.optim.Adam(enc_params, lr=args.lr)
    optimizer_enc_test = torch.optim.Adam(enc[-1].parameters(), lr=args.lr) #this encoder is for the test samples
    optimizer_cls = torch.optim.Adam(cls.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr=args.lr)
    optimizer_dec_alt  = torch.optim.Adam(dec.parameters(), lr=args.lr)
    
    enc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=args.step_size, gamma=0.5)
    enc_scheduler_test = torch.optim.lr_scheduler.StepLR(optimizer_enc_test, step_size=args.step_size, gamma=0.5)
    cls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=args.step_size, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=args.step_size, gamma=0.5)
    dec_alt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dec_alt, step_size=args.step_size, gamma=0.5)

    lr_2 = args.lr # define learning rate for second-derivative step

    # Iterate through train set minibatchs
    best_acc = 0
    best_auc = 0
    best_metrics = np.zeros(10, dtype=np.float32)
    out_path = os.path.join(args.out_path, str(fold_index))
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(10, dtype=np.float32)
        t0 = time.time()
        model_saved = False
        # for i in range(args.num_domains):
        enc[0:num_mouse].train()
        dec.train()
        cls.train()
        
        for it in range(100):
            
            # loop each training mouse
            loss0 = 0
            cls_loss = 0
            rec_loss = 0
            kl_loss = 0
            
            # Zero out the gradients
            optimizer_enc.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_dec.zero_grad() 
            optimizer_enc_test.zero_grad() 
            optimizer_dec_alt.zero_grad() 
            # loop each mouse
            for i in range(len(train_loader)):
                # get data of a mouse
                try:
                    x, y = next(iter(train_loader[i]))
                except StopIteration:
                    print('in exeption')
                    train_loader[i] = enumerate(train_loader[i])
                    x, y = next(iter(train_loader[i]))
                
                #print(x.dtype) #torch.float64
                x, y = x.float().to(args.device), y.long().to(args.device)
                
                # encode
                z_mu, z_var, x_sample = enc[i](x)
                x_ = dec(x_sample)
                y_ = cls(x_sample)
        
                # loss
                rec_loss = criterion_rec(x_, x)
                kl_loss = kl_divergence(z_mu, z_var)

                loss0 += args.invmu*rec_loss + args.rec_kl_scale*kl_loss
                
            loss0.backward()
            
            optimizer_enc.step()
            optimizer_dec.step() 
             
            #torch.autograd.set_detect_anomaly(True)
            optimizer_cls.zero_grad()
            #optimizer_dec_alt.zero_grad()
            # loop each mouse
            for i in range(len(train_loader)):
                # get data of a mouse
                try:
                    x, y = next(iter(train_loader[i]))
                except StopIteration:
                    print('in exeption')
                    train_loader[i] = enumerate(train_loader[i])
                    x, y = next(iter(train_loader[i]))
                
                #print(x.dtype) #torch.float64
                x, y = x.float().to(args.device), y.long().to(args.device)
                
                ## Linking update on s
                z_mu, z_var, z = enc[i](x)
                # x_ = dec(z)
                
                # #retrain the graph
                # rec_loss = criterion_rec(x_, x)
                # kl_loss = kl_divergence(z_mu, z_var)
                # loss = args.invmu*rec_loss + args.rec_kl_scale*kl_loss
                # fast_weights = OrderedDict((name, param) for (name, param) in enc[i].named_parameters())
                
                # grads = grad(loss, enc[i].parameters(),create_graph=True)
                # data = [p.data for p in list(enc[i].parameters())]              

                # # update encoder's weights by applying sgd on classification loss
                # fast_weights = OrderedDict((name, param - lr_2 * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))
                # # compute latent parameters with updated weights
                # z_mu_alt, z_var_alt, z_alt = enc[i].forward(x, fast_weights)
                
                # y_  = cls(z_alt)
                y_  = cls(z)
                
                #Update on classification loss            
                cls_loss += criterion_cls(y_, y)
                
                #print('len(train_loader[i]): ', len(train_loader[i])) #18, 19
                perf_mat[0] += rec_loss.item()
                perf_mat[1] = cls_loss.item()
                perf_mat[2] += kl_loss.item()
    #            perf_mat[3] += 0                    #place holder of kl_dist, just for uniform print
                perf_mat[4] += compute_acc(y_, y)
                perf_mat[5] += compute_auc(y_, y)
            
            perf_mat[0:6] /= len(train_loader)
            
            cls_loss.backward(retain_graph=True)
                
            optimizer_cls.step()
            #optimizer_dec_alt.step()
            
        
            
      
        print('{}: {:03d} {:.2f}s || TRAIN LOSS: [REC|CLS|KL] {:.4f} {:.4f} {:.4f} ACC {:.4f} AUC {:.4f}'.format('Epoch', epoch, time.time() - t0, 
        perf_mat[0], perf_mat[1], perf_mat[2], perf_mat[4],perf_mat[5]))
        
        enc_scheduler.step()
        cls_scheduler.step()
        dec_scheduler.step()
        dec_alt_scheduler.step() 
        
        # Testing, after the model get optimized
        if epoch > args.test_epoch:
            # fix the decoder and classifer, then train a encoder for the testing sampels
            enc[-1].train()
            dec.eval()
            cls.eval()
            loss = 0
            for it in range(100):
                optimizer_enc_test.zero_grad()
                
                # loop each mouse, to train a encoder for the test mice
                for i in range(len(test_loader)):
                    # get data of a mouse
                    try:
                        x, y = next(iter(test_loader))
                    except StopIteration:
                        print('in exeption')
                        test_loader = enumerate(test_loader)
                        x, y = next(iter(test_loader))
                    x, y = x.float().to(args.device), y.long().to(args.device)    
                    
                    # print('x: ', x.shape) # torch.Size([128, 1980])
                    # print('y: ', y.shape) # torch.Size([128])

                    z_mu, z_var, x_sample = enc[-1](x)
                    x_ = dec(x_sample)
                    #y_ = cls(x_sample) 
                    
                    
                    
                    # loss
                    #cls_loss = criterion_cls(y_,y)
                    rec_loss = criterion_rec(x_,x)
                    kl_loss  = kl_divergence(z_mu, z_var)
                
                    loss += args.invmu*rec_loss + args.rec_kl_scale*kl_loss
                loss.backward()
                
                optimizer_enc_test.step()
                    
            # test the trained encoder
            # Iterate through test set minibatchs 
            enc[-1].eval()
            with torch.no_grad():
                for x, y in test_loader:
                    z_mu, z_var, x_sample = enc[-1](x)
                    x_ = dec(x_sample)
                    y_ = cls(x_sample)
                    
                    # loss
                    cls_loss = criterion_cls(y_,y)
                    rec_loss = criterion_rec(x_, x)

                    perf_mat[6] += rec_loss.item() / len(test_loader)
                    perf_mat[7] += cls_loss.item() / len(test_loader)
                    perf_mat[8] += compute_acc(y_, y) / len(test_loader)   #prediction acc
                    perf_mat[9] += compute_auc(y_, y) / len(test_loader)  #prediction auc                
                
            enc_scheduler_test.step()        
        
            if(best_acc<perf_mat[8]):
                best_acc = perf_mat[8]
                best_metrics = perf_mat
            
            if(best_auc<perf_mat[9]):
                best_auc = perf_mat[9]

            print_metrics(perf_mat, 'EPOCH', epoch, time.time() - t0)

            print('Best test ACC|AUC: {:.4f} {:.4f}'.format(best_acc, best_auc))
    
            #print losses and acc of this fold (selected from an approparate epoch)
            print_metrics(best_metrics, 'FOLD', fold_index, time.time() - t1)
    
    return best_metrics 


#Use a static fitted decoder from SVAE, to train an encoder of VAE using VAE loss
def vae_refit_enc(args, train_loader, test_loader, fold_index, dec_model_pathname, save_model=False):
    print('\nTrain a vae_refit_enc model...')
    print(args)
    
    t1 = time.time()
    
    #import the model
    model_enc, model_dec, model_cls = import_model(args.decoder_fn)

    #define models
    enc = model_enc(args).to(args.device)  #this will be the pretrained encoder obtained from SVAE.    <=> encoder1
    dec = model_dec(args).to(args.device)
    
    out_path = os.path.join(args.out_path, str(fold_index))
    
    #load the pretrained/fixed enc (that were obtained from the standard SVAE)    
    pretrained_dict_dec = torch.load(dec_model_pathname)
    
    #set model paras to be the pretrained parameters in the dictionary
    dec.load_state_dict(pretrained_dict_dec)            

    # Loss and Optimizer
    criterion_rec = nn.MSELoss()

    # optimizers
    optimizer_enc = torch.optim.Adam(enc.parameters(), lr=args.lr)
    
    enc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=args.step_size, gamma=0.5)

    # Iterate through train set minibatchs
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(2, dtype=np.float32)
        t0 = time.time()
        model_saved = False
        enc.train() 
        dec.eval() #the pretrained/fixed decoder
     
        for instances, labels in train_loader:
            x, y = instances.to(args.device), labels.long().to(args.device)
            
            #the second encoder to be trained here
            _, _, z = enc(x)
            x_ = dec(z)
            
            #compute reconstruction loss, just for report. The encoder and decoder are unoptimized in this model.
            rec_loss = criterion_rec(x_, x)
            
            optimizer_enc.zero_grad()
            
            optimizer_enc.step()

            perf_mat[0] += rec_loss.item()                          
                       
                    
        enc_scheduler.step()

        
        ## Testing
        enc.eval()
        with torch.no_grad():
            # Iterate through test set minibatchs 
            for x, y in test_loader:
                x, y = x.to(args.device), y.long().to(args.device)

                _, _, z = enc(x)
                x_  = dec(z)         

                perf_mat[1] += criterion_rec(x_, x).item()  
        
        if(epoch == args.total_epochs-1):
            if save_model:
                #save the model
                save_single_model(enc, out_path, 'vae_refit_enc_'+str(epoch)+'_model_enc.pt')
                model_saved = True
        
        
        perf_mat[0] /= len(train_loader)
        perf_mat[1] /= len(test_loader)

        print('{:03d} {:.2f}s || LOSS [TRAIN|TEST]: {:.4f} {:.4f}'.format(epoch, time.time() - t0, perf_mat[0], perf_mat[1]))
    
def compute_acc(y_pred, y_true):
    predictions = torch.argmax(y_pred, dim=1)
    #return torch.sum((predictions == y_true).float())
    return torch.mean((predictions == y_true).float())
                
def compute_auc(y_pred, y_true):
        
    s = 0
    
    #y_pred is torch.Size([100, 2]), convert it the same as labels: torch.Size([100])
    predictions = torch.argmax(y_pred, dim=1)
    # print('predictions:', predictions)
        
    #for binary classes
    if (y_pred.shape[1] == 2):
               
        ##one way. (while labels only contains one class (the data very imbalanced), say error: "ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.")
        y_true = y_true.cpu().detach().numpy()
        y_score = predictions.cpu().detach().numpy()
        s = roc_auc_score(y_true, y_score)
        
        # #a second way
        # lt = y_true.cpu()
        # yt = predictions.cpu()
        # fpr,tpr,_ = roc_curve(lt.detach().numpy(),yt.detach().numpy())
        # s = auc(fpr,tpr)
    
    #for multiclass
    else:
        y_true = y_true.cpu().detach().numpy()
        y_score = predictions.cpu().detach().numpy()
        # print(y_true.shape)     #N*num_label
        # print(y_score.shape)    #N*num_label

        lb = LabelBinarizer()
        lb.fit(y_true)
        y_true = lb.transform(y_true)
        y_score = lb.transform(y_score)
        # print(y_true.shape)     #N*num_label
        # print(y_score.shape)    #N*num_label
        try:
          s = roc_auc_score(y_true, y_score, average='macro')
        except:
          print("An exception occurred at roc_auc_score")
          print('----Due to all positive/negtive sample in the batch-----')
          print(y_true.shape)     #N*num_label
          print(y_score.shape)    #N*num_label
          s=0        
    
    return s


    
