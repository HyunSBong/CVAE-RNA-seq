import os
import numpy as np
import time
import argparse
import pandas as pd
import pickle
import random
import datetime
from datetime import datetime
from utils import *
from model import CVAE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import wandb

def load_data(num_genes, multivariate):
    train = pd.read_csv(f'../data/RNAseqDB/x_train_all_tissue.csv')
    test = pd.read_csv(f'../data/RNAseqDB/x_test_all_tissue.csv')
    df_real = pd.concat([train, test])
    
    df_real_gene_columns = df_real.iloc[:,1:-2].columns
    train_genes = list(df_real_gene_columns)[:num_genes]

    X = df_real[train_genes].values
    # Log-transform data
    X = np.log(1 + X)
    X = np.float32(X)

    df_real_y_copy = df_real[["TISSUE_GTEX", "DATASET"]].copy()
    df_real_y_copy['TISSUE_DATASET'] = df_real_y_copy['TISSUE_GTEX'] + "-" + df_real_y_copy['DATASET']
    Y = df_real_y_copy[["TISSUE_GTEX", "DATASET", "TISSUE_DATASET"]].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    gene_names = train_genes

    # Standardization & Normalization data
    std_scaler = StandardScaler().fit(X_train)
    X_train = std_scaler.transform(X_train)
    X_test = std_scaler.transform(X_test)
    if multivariate == 0:
        minmax_scaler = MinMaxScaler().fit(X_train)
        X_train = minmax_scaler.transform(X_train)
        X_test = minmax_scaler.transform(X_test)

    dict_train_test_splited = {'train_set': (X_train, Y_train),
                            'test_set': (X_test, Y_test),
                            'X': X,
                            'Y': Y,
                            'gene_names': gene_names}
    return dict_train_test_splited

def load_data_npy(num_genes, multivariate, geo):
    # multivariate 1 : use gaussian decoder
    # multivariate 0 : use bernoulli decoder
    if geo == 0 and multivariate == 0:
        gene_names = np.load('npy_data/rna_gene_names.npy', allow_pickle=True)
        X_train = np.load('npy_data/rna_x_train_minmaxScaled.npy', allow_pickle=True)
        X_test = np.load('npy_data/rna_x_test_minmaxScaled.npy', allow_pickle=True)
        Y_train = np.load('npy_data/rna_y_train.npy', allow_pickle=True)
        Y_test = np.load('npy_data/rna_y_test.npy', allow_pickle=True)
        X = np.load('npy_data/rna_x_log_trainsform.npy', allow_pickle=True)
        Y = np.load('npy_data/rna_y.npy', allow_pickle=True)
        
    elif geo == 0 and multivariate == 1:
        gene_names = np.load('npy_data/rna_gene_names.npy', allow_pickle=True)
        X_train = np.load('npy_data/rna_x_train_standardScaled.npy', allow_pickle=True)
        X_test = np.load('npy_data/rna_x_test_standardScaled.npy', allow_pickle=True)
        Y_train = np.load('npy_data/rna_y_train.npy', allow_pickle=True)
        Y_test = np.load('npy_data/rna_y_test.npy', allow_pickle=True)
        X = np.load('npy_data/rna_x_log_trainsform.npy', allow_pickle=True)
        Y = np.load('npy_data/rna_y.npy', allow_pickle=True)
        
    elif geo == 1 and multivariate == 0:
        gene_names = np.load('npy_data/rna_geo_gene_names.npy', allow_pickle=True)
        X_train = np.load('npy_data/rna_geo_x_train_minmaxScaled.npy', allow_pickle=True)
        X_test = np.load('npy_data/rna_geo_x_test_minmaxScaled.npy', allow_pickle=True)
        Y_train = np.load('npy_data/rna_geo_y_train.npy', allow_pickle=True)
        Y_test = np.load('npy_data/rna_geo_y_test.npy', allow_pickle=True)
        X = np.load('npy_data/rna_geo_x_log_trainsform.npy', allow_pickle=True)
        Y = np.load('npy_data/rna_geo_y.npy', allow_pickle=True)
    
    elif geo == 1 and multivariate == 1:
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
    # wandb
    wandb.init(project="Multivariate Gaussian CVAE loss test", name="newModel2 gaussian mse covariance d1000 b50 lr1e-04", reinit=True)
    wandb.config.update(args)
    
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
    
    # one-hot encoding
    le = LabelEncoder()
    le.fit(Y_train_tissue_datasets)
    train_labels = le.transform(Y_train_tissue_datasets) # bladder,uterus,,, -> 0,14,,,
    test_labels = le.transform(Y_test_tissue_datasets)
    
    # DataLoader
    train_label = torch.as_tensor(train_labels)
    train = torch.tensor(X_train.astype(np.float32))
    train_tensor = TensorDataset(train, train_label)
    train_loader = DataLoader(dataset=train_tensor, batch_size=args.batch_size, shuffle=True)

    test_label = torch.as_tensor(test_labels)
    test = torch.tensor(X_test.astype(np.float32))
    test_tensor = TensorDataset(test, test_label)
    test_loader = DataLoader(dataset=test_tensor, batch_size=args.batch_size, shuffle=True)
    
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

    vae = CVAE(
        data_dim=X_train.shape[1],
        compress_dims=args.compress_dims,
        latent_size=args.latent_size,
        decompress_dims=args.decompress_dims,
        conditional=args.conditional,
        view_size = view_size,
        multivariate = multivariate,
        num_labels=num_tissue if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    wandb.watch(vae)
    
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
        wandb.log({
            "ELBO": avg_loss,
            "Reconstruction Error": avg_reconstr_loss,
            "KL-Divergence": avg_kl_loss,
            "Gamma Score": score
        })
        if stop_point == 0:
            x_syn = save_synthetic(vae, x, epoch, args.batch_size, args.learning_rate, X.shape[1])
            x_syn = generate_synthetic_n_save(vae, le, X, gene_names, Y_test_tissues, train_epoch, 'trial_005', X.shape[1])
            break

    with torch.no_grad():
        for epoch in range(args.epochs):
            test_loss = 0
            for batch_idx, (x, y) in enumerate(test_loader):
                if x.is_cuda != True:
                    x = x.cuda()
                if multivariate:
                    losses = loss_fn_gaussian(x, mean, log_var, z_mean, z_log_var)
                elif multivariate == False:
                    losses = loss_fn_bernoulli(recon_x, x, mean, log_var)
                test_loss = losses['elbo'].clone()

                if batch_idx == len(test_loader) - 1:
                    print('====> Test set loss: {:.4f}'.format(test_loss.item()))

    x_syn = save_synthetic(vae, x, Y_test, args.epochs, args.batch_size, args.learning_rate, X.shape[1])
    draw_umap(X_test, x_syn, Y_test_tissues, Y_test_datasets)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-03) # bernoulli 0.001
    parser.add_argument("--l2scale",type=float, default=0.00001)
    parser.add_argument("--compress_dims", type=list, default=[1000, 512, 256])
    parser.add_argument("--decompress_dims", type=list, default=[256, 512, 1000])
    parser.add_argument("--latent_size", type=int, default=50)
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--num_genes", type=int, default=18154)
    parser.add_argument("--multivariate", type=int, default=1)
    parser.add_argument("--geo", type=int, default=1)

    args = parser.parse_args()
    
    # rna_dataset = load_data(args.num_genes, args.multivariate)
    rna_dataset = load_data_npy(args.num_genes, args.multivariate, args.geo)
    main(args, rna_dataset)
