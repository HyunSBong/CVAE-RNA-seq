import os
import numpy as np
import argparse
import pandas as pd
import pickle
import random
from utils import *
from model import CVAE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data_npy(multivariate, geo):
    # multivariate 1 : use gaussian decoder
    # multivariate 0 : use bernoulli decoder
    
    if geo == 1 and multivariate == 1:
        gene_names = np.load('npy_data/rna_geo_gene_names.npy', allow_pickle=True)
        X_train = np.load('npy_data/rna_geo_x_train_standardScaled.npy', allow_pickle=True)
        X_test = np.load('npy_data/rna_geo_x_test_standardScaled.npy', allow_pickle=True)
        Y_train = np.load('npy_data/rna_geo_y_train.npy', allow_pickle=True)
        Y_test = np.load('npy_data/rna_geo_y_test.npy', allow_pickle=True)
        X = np.load('npy_data/rna_geo_x_log_trainsform.npy', allow_pickle=True)
        Y = np.load('npy_data/rna_geo_y.npy', allow_pickle=True)
        

    rna_dataset = {'train_set': (X_train, Y_train),
                            'test_set': (X_test, Y_test),
                            'X': X,
                            'Y': Y,
                            'gene_names': gene_names}
    return rna_dataset

def main(args, rna_dataset):
    
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # GPU 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    # use bernoulli decoder
    multivariate = False
    if args.multivariate == 1:
        # use gaussian decoder
        multivariate = True
    
    # Train/Test dataset & tissues
    X = rna_dataset['X'] # [COL4A1, IFT27,,,].values
    Y = rna_dataset['Y'] # [["TISSUE_GTEX", "DATASET"]].values
    (X_train, Y_train) = rna_dataset['train_set']
    (X_test, Y_test) = rna_dataset['test_set']
    Y_tissues = Y[:,0]
    Y_datasets = Y[:,1]
    Y_tissue_datasets = Y[:,2]
    Y_train_tissues = Y_train[:,0]
    Y_train_datasets = Y_train[:,1]
    Y_train_tissue_datasets = Y_train[:,2]
    Y_test_tissues = Y_test[:,0]
    Y_test_datasets = Y_test[:,1]
    Y_test_tissue_datasets = Y_test[:,2]
    gene_names = rna_dataset['gene_names'] # ex) COL4A1, IFT27,,,
    
    num_tissue = len(set(Y_tissue_datasets)) # 15 (breast, lung, liver 등 15개 tissue)
    view_size = X.shape[1]
    
    # Validation dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train_tissue_datasets, test_size=0.3, random_state=0)
    Y_train_tissue_datasets = y_train
    Y_val_tissue_datasets = y_val
    
    # one-hot encoding
    le = LabelEncoder()
    le.fit(Y_train_tissue_datasets)
    train_labels = le.transform(Y_train_tissue_datasets) # bladder,uterus,,, -> 0,14,,,
    test_labels = le.transform(Y_test_tissue_datasets)
    val_labels = le.transform(Y_val_tissue_datasets)
    
    # DataLoader
    train_label = torch.as_tensor(train_labels)
    train = torch.tensor(X_train.astype(np.float32))
    train_tensor = TensorDataset(train, train_label)
    train_loader = DataLoader(dataset=train_tensor, batch_size=args.batch_size, shuffle=True)

    test_label = torch.as_tensor(test_labels)
    test = torch.tensor(X_test.astype(np.float32))
    test_tensor = TensorDataset(test, test_label)
    test_loader = DataLoader(dataset=test_tensor, batch_size=args.batch_size, shuffle=True)
    
    val_label = torch.as_tensor(val_labels)
    val = torch.tensor(X_val.astype(np.float32))
    val_tensor = TensorDataset(val, val_label)
    val_loader = DataLoader(dataset=val_tensor, batch_size=args.batch_size, shuffle=True)
    
    # loss function
    mse_criterion = nn.MSELoss(size_average=False, reduction="sum")
    def loss_fn_gaussian(x, mean, log_var, z_mean, z_sigma):
        # reconstruction error
        reconstr_loss = mse_criterion(
            z_mean, x)
            
        # Kullback-Leibler divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        elbo = (reconstr_loss + kl_loss) / x.size(0)
        
        return {'elbo': elbo, 'reconstr_loss': reconstr_loss, 'kl_loss': kl_loss}
    
    def loss_fn_bernoulli(recon_x, x, mean, log_var):
        # reconstruction error
        reconstr_loss = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, view_size), x.view(-1, view_size), reduction='sum')
        # Kullback-Leibler divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())
        elbo = (reconstr_loss + kl_loss) / x.size(0)
        
        return {'elbo': elbo, 'reconstr_loss': reconstr_loss, 'kl_loss': kl_loss}
    
    compress_dims = [1000, 512, 256]
    decompress_dims = [256, 512, 1000]
    
    vae = CVAE(
        data_dim=X_train.shape[1],
        compress_dims=compress_dims,
        latent_size=args.latent_size,
        decompress_dims=decompress_dims,
        conditional=args.conditional,
        view_size = view_size,
        multivariate = multivariate,
        num_labels=num_tissue if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    
    # Train
    stop_point = 10
    best_score = 0.0000000000001
    initial_stop_point = stop_point
    stop_point_done = False
    losses = []
    
    score = 0

    for epoch in range(args.epochs):
        train_loss = 0
        sum_elbo = 0
        sum_kl_loss = 0
        sum_reconstr_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if x.is_cuda != True:
                x = x.cuda()

            if args.conditional and multivariate:
                mean, log_var, z_mean, z_sigma = vae(x, y)
                losses = loss_fn_gaussian(x, mean, log_var, z_mean, z_sigma)
            elif args.conditional and multivariate==False:
                recon_x, mean, log_var, z = vae(x, y)
                losses = loss_fn_bernoulli(recon_x, x, mean, log_var)
            else:
                recon_x, mean, log_var, z = vae(x)
                losses = loss_fn_bernoulli(recon_x, x, mean, log_var)
            
            loss = losses['elbo'].clone() #  KL-Divergence + reconstruction error / x.size(0)
            train_loss += loss
            
            sum_elbo += losses['elbo']
            sum_kl_loss += losses['kl_loss']
            sum_reconstr_loss += losses['reconstr_loss']
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
        print(f'stop point : {stop_point}')
        c = torch.from_numpy(test_labels) # le.fit_transform(Y_train_tissues)
        x_syn = vae.inference(n=c.size(0), c=c)
        score = score_fn(X_test, x_syn.detach().cpu().numpy())
        
        if score > best_score or epoch % 50 == 0:
            best_score = score
            stop_point = initial_stop_point
            x_syn = save_synthetic(vae, x_syn, Y_test, epoch, args.batch_size, args.learning_rate, X.shape[1])
        else:
            stop_point -= 1
        
        avg_loss = sum_elbo / len(train_loader)
        avg_kl_loss = sum_kl_loss / len(train_loader)
        avg_reconstr_loss = sum_reconstr_loss / len(train_loader)
        
        print("Epoch {:02d}/{:02d} Loss {:9.4f}, KL {:9.4f}, Reconstruction {:9.4f}".format(
            epoch, args.epochs, avg_loss, avg_kl_loss, avg_reconstr_loss))
        print(f'==>Gamma Score : {score}')
        
        ## early stopping
        # if stop_point == 0:
        #     x_syn = save_synthetic(vae, x, epoch, args.batch_size, args.learning_rate, X.shape[1])
        #     x_syn = generate_synthetic_n_save(vae, le, X, gene_names, Y_test_tissues, train_epoch, 'trial_005', X.shape[1])
        #     break
        
    with torch.no_grad():
        for epoch in range(args.epochs):
            test_loss = 0
            sum_elbo = 0
            sum_kl_loss = 0
            sum_reconstr_loss = 0

            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                if x.is_cuda != True:
                    x = x.cuda()

                if args.conditional and multivariate:
                    mean, log_var, z_mean, z_sigma = vae(x, y)
                    losses = loss_fn_gaussian(x, mean, log_var, z_mean, z_sigma)
                elif args.conditional and multivariate==False:
                    recon_x, mean, log_var, z = vae(x, y)
                    losses = loss_fn_bernoulli(recon_x, x, mean, log_var)
                else:
                    recon_x, mean, log_var, z = vae(x)
                    losses = loss_fn_bernoulli(recon_x, x, mean, log_var)

                loss = losses['elbo'].clone() #  KL-Divergence + reconstruction error / x.size(0)
                test_loss += loss

                sum_elbo += losses['elbo']
                sum_kl_loss += losses['kl_loss']
                sum_reconstr_loss += losses['reconstr_loss']

            avg_val_loss = sum_elbo / len(val_loader)
            avg_kl_loss = sum_kl_loss / len(val_loader)
            avg_reconstr_loss = sum_reconstr_loss / len(val_loader)

            print("Epoch {:02d}/{:02d} Test Loss {:9.4f}, KL {:9.4f}, Reconstruction {:9.4f}".format(
                epoch, args.epochs, avg_val_loss, avg_kl_loss, avg_reconstr_loss))
            

    x_syn = save_synthetic(vae, x, Y_test, args.epochs, args.batch_size, args.learning_rate, X.shape[1])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-04) # bernoulli 0.001
    parser.add_argument("--l2scale",type=float, default=0.00001)
    parser.add_argument("--compress_dims", type=list, default=[1000, 512, 256])
    parser.add_argument("--decompress_dims", type=list, default=[256, 512, 1000])
    parser.add_argument("--latent_size", type=int, default=50)
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--multivariate", type=int, default=1)
    parser.add_argument("--geo", type=int, default=1)
    parser.add_argument("--hidden_dims", type=int, default=3) # 0.144

    args = parser.parse_args()
    
    rna_dataset = load_data_npy(args.multivariate, args.geo)
    main(args, rna_dataset)
