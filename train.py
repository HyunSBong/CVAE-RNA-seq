import os
import numpy as np
import time
import argparse
import pandas as pd
import pickle
import datetime
from datetime import datetime
from collections import defaultdict
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
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def draw_umap(X_test, x_syn, Y_test_tissues, Y_test_datasets, normalize=False):
    if normalize:
        x_log = np.log1p(x_syn)
        x_log = np.float32(x_log)
        x_mean = np.mean(x_log, axis=0)
        x_std = np.std(x_log, axis=0)
        x_syn = standardize(x_log, mean=x_mean, std=x_std)

    x = np.concatenate((X_test, x_syn), axis=0)
    model = umap.UMAP(n_neighbors=300,
                    min_dist=0.7,
                    n_components=2,
                    random_state=1111)
    model.fit(x)
    emb_2d = model.transform(x)
    plot_umap(emb_2d, X_test, x_syn, Y_test_tissues, Y_test_datasets)

def load_data(num_genes):
    train = pd.read_csv(f'../data/RNAseqDB/x_train_all_tissue.csv')
    test = pd.read_csv(f'../data/RNAseqDB/x_test_all_tissue.csv')
    df_real = pd.concat([train, test])
    
    df_real_gene_columns = df_real.iloc[:,1:-2].columns
    # 학습할 genes
    train_genes = list(df_real_gene_columns)[:num_genes]

    X = df_real[train_genes].values
    # Log-transform data
    X = np.log(1 + X)
    X = np.float32(X)

    Y = df_real[["TISSUE_GTEX", "DATASET"]].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    gene_names = train_genes

    # Standardization & Normalization data
    std_scaler = StandardScaler().fit(X_train)
    X_train = std_scaler.transform(X_train)
    X_test = std_scaler.transform(X_test)
    minmax_scaler = MinMaxScaler().fit(X_train)
    X_train = minmax_scaler.transform(X_train)
    X_test = minmax_scaler.transform(X_test)

    dict_train_test_splited = {'train_set': (X_train, Y_train),
                            'test_set': (X_test, Y_test),
                            'X': X,
                            'Y': Y,
                            'gene_names': gene_names}
    return dict_train_test_splited

def generate_synthetic_n_save(vae_model, le, X, gene_names, Y_test_tissues, epoch, trial_name, dim_size):
    genes_to_validate = 40
    original_means = np.mean(X, axis=0)
    original_vars = np.var(X, axis=0)
    model_dir = '../checkpoints/models/cvae/'

    with torch.no_grad():
        x_synthetic = []
        y_synthetic = []
        
        all_samples = Y_test_tissues
        le = LabelEncoder()
        onehot_c = le.fit_transform(all_samples)
        
        c = all_samples # ['bladder' 'bladder' 'bladder' ... 'uterus' 'uterus' 'uterus']
        c = torch.from_numpy(onehot_c) # [0 0 0 ... 14 14 14]
        x = vae_model.inference(n=len(all_samples), c=c)
        
        x_syn = x.detach().cpu().numpy() # (7500,1000)

        x_synthetic += list(x.detach().cpu().numpy())
        y_synthetic += list(np.ravel(le.transform(all_samples)))
        
        print(f'x_syn.shape : {x_syn.shape}')
        date_val = datetime.today().strftime("%Y%m%d%H%M")

        file = f'../checkpoints/models/cvae/gen_rnaseqdb_cvae_{date_val}_{trial_name}_epoch{epoch}_dim{dim_size}_.pkl'
        data = {'model': vae_model,
                'x_syn': x_syn
                }
        with open(file, 'wb') as files:
            pickle.dump(data, files)
            
    return x_syn

def main(args, rna_dataset):
    torch.manual_seed(args.seed)
    latest_loss = torch.tensor(1)
    
    # GPU 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    multivariate = False
    if args.multivariate == 1:
        multivariate = True
    
    # Train/Test dataset & tissues
    X = rna_dataset['X'] # [COL4A1, IFT27,,,].values
    Y = rna_dataset['Y'] # [["TISSUE_GTEX", "DATASET"]].values
    (X_train, Y_train) = rna_dataset['train_set']
    (X_test, Y_test) = rna_dataset['test_set']
    Y_tissues = Y[:,0]
    Y_datasets = Y[:,1]
    Y_train_tissues = Y_train[:,0]
    Y_train_datasets = Y_train[:,1]
    Y_test_tissues = Y_test[:,0]
    Y_test_datasets = Y_test[:,1]
    gene_names = rna_dataset['gene_names'] # ex) COL4A1, IFT27,,,
    
    num_tissue = len(set(Y_tissues)) # 15 (breast, lung, liver 등 15개 tissue)
    view_size = X.shape[1]
    log2pi = torch.log2(torch.Tensor([np.pi]))
    
    # one-hot encoding
    le = LabelEncoder()
    le.fit(Y_train_tissues)
    train_labels = le.transform(Y_train_tissues) # bladder,uterus,,, -> 0,14,,,
    test_labels = le.transform(Y_test_tissues)
    
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
    mse_criterion = nn.MSELoss(size_average=False)
    def loss_fn(mean, log_var, recon_x=None, z=None, z_mean=None, z_log_var=None, multivariate=False):
        if multivariate:
            # reconstruction error
            # reconstr_loss = mse_criterion(
                # z_mean.view(-1, view_size), x.view(-1, view_size), reduction='sum')
            reconstr_loss = log2pi + z_log_var + (x - z_mean) ** 2 / (2 * torch.exp(z_log_var))
            
            # Kullback-Leibler divergence
            latent_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            MSE = torch.sum(x - 1)**2 # ignore
            
            return {'reconstr_latent_loss': (reconstr_loss + latent_loss) / x.size(0), 'MSE': MSE, 'reconstr_loss': reconstr_loss, 'latent_loss': latent_loss}
        else:
            # reconstruction error
            reconstr_loss = torch.nn.functional.binary_cross_entropy(
                recon_x.view(-1, view_size), x.view(-1, view_size), reduction='sum')
            # Kullback-Leibler divergence
            latent_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            MSE = torch.sum((x - recon_x)**2)
            
            return {'reconstr_latent_loss': (reconstr_loss + latent_loss) / x.size(0), 'MSE': MSE, 'reconstr_loss': reconstr_loss, 'latent_loss': latent_loss}

    vae = CVAE(
        data_dim=X_train.shape[1],
        compress_dims=args.compress_dims,
        latent_size=args.latent_size,
        decompress_dims=args.decompress_dims,
        conditional=args.conditional,
        view_size = view_size,
        multivariate = multivariate,
        num_labels=num_tissue if args.conditional else 0).to(device)

    dataiter = iter(train_loader)
    genes, labels = dataiter.next()

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    
    # Train
    stop_point = 10
    best_score = -np.inf
    initial_stop_point = stop_point
    stop_point_done = False
    losses = []

    for epoch in range(args.epochs):
        train_epoch = epoch
        if stop_point_done:
            train_epoch = epoch - 1
            break
        train_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional and multivariate:
                if x.is_cuda != True:
                    x = x.cuda()
                mean, log_var, z_mean, z_log_var = vae(x, y)
                losses = loss_fn(
                    mean = mean, 
                    log_var = log_var, 
                    z_mean = z_mean, 
                    z_log_var = z_log_var, 
                    multivariate = multivariate)
                print(f'z_mean : {z_mean.shape}')
                print(f'x : {x.shape}')
            elif args.conditional and multivariate==False:
                if x.is_cuda != True:
                    x = x.cuda()
                recon_x, mean, log_var, z = vae(x, y)
                losses = loss_fn(
                    mean = mean, 
                    log_var = log_var, 
                    recon_x = recon_x, 
                    z = z, 
                    multivariate = multivariate)
            else:
                recon_x, mean, log_var, z = vae(x)
                losses = loss_fn(
                    mean = mean, 
                    log_var = log_var, 
                    recon_x = recon_x, 
                    z = z, 
                    multivariate = multivariate)
            # losses = loss_fn(mean, log_var, recon_x, z, z_mean, z_log_var, multivariate)
            # losses = loss_fn(x, mean, log_var, z_mean, z_log_var, multivariate)
            loss = losses['reconstr_latent_loss'].clone() #  KL-Divergence + reconstruction error / x.size(0)
            train_loss += loss

            # latest_loss = losses['MSE'].detach()

            optimizer.zero_grad()
            
            # writer.add_scalar("Train/Reconstruction Error",losses['BCE'].detach().item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
            # writer.add_scalar("Train/KL-Divergence", losses['KLD'].detach().item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
            # writer.add_scalar("Train/Total Loss" , loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
            
            loss.backward(retain_graph=True)
            optimizer.step()

            # logs['loss'].append(loss.item())
            
            if batch_idx % 100 == 0 or batch_idx == len(train_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, batch_idx, len(train_loader)-1, loss.item()))

                if args.conditional:
                    if epoch % 5 == 0:
                        # generate gene
                        c = torch.from_numpy(test_labels) # le.fit_transform(Y_train_tissues)
                        x = vae.inference(n=c.size(0), c=c)
                        if multivariate:
                            mean, log_var, z_mean, z_log_var = vae(x, y)
                        elif multivariate == False:
                            x = vae.inference(n=c.size(0), c=c)
                        score = score_fn(X_test, x.detach().cpu().numpy())
                        if score > best_score:
                            best_score = score
                            stop_point = initial_stop_point
                            x_syn = save_synthetic(vae, x, train_epoch, 'trial_005', X.shape[1])
                            # x_syn = generate_synthetic_n_save(vae, le, X, gene_names, Y_test_tissues, train_epoch, 'trial_004', X.shape[1])
                        else:
                            stop_point -= 1
                        print(f'==>Gamma Score : {score}')
                        print(f'==>stop_point : {stop_point}')
                else:
                    x = vae.inference()
                    
            if stop_point == 0:
                train_epoch = epoch
                x_syn = save_synthetic(vae, x, train_epoch, 'trial_005', X.shape[1])
                # x_syn = generate_synthetic_n_save(vae, le, X, gene_names, Y_test_tissues, train_epoch, 'trial_005', X.shape[1])
                stop_point_done = True
        
    with torch.no_grad():
        for epoch in range(args.epochs):
            test_loss = 0
            for batch_idx, (x, y) in enumerate(test_loader):
                recon_x, mean, log_var, z = vae(x, y)
                if x.is_cuda != True:
                    x = x.cuda()
                    
                losses = loss_fn(recon_x, x, mean, log_var)
                # writer.add_scalar("Test/Reconstruction Error",losses['BCE'].detach().item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
                # writer.add_scalar("Test/KL-Divergence", losses['KLD'].detach().item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
                # writer.add_scalar("Test/Total Loss" , loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
                test_loss = losses['GL'].clone()

                if batch_idx == len(test_loader) - 1:
                    print('====> Test set loss: {:.4f}'.format(test_loss.item()))

    # writer.close()
    draw_umap(X_test, x_syn, Y_test_tissues, Y_test_datasets)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--l2scale",type=float, default=0.00001)
    parser.add_argument("--compress_dims", type=list, default=[1000, 512, 256])
    parser.add_argument("--decompress_dims", type=list, default=[256, 512, 1000])
    parser.add_argument("--latent_size", type=int, default=50)
    parser.add_argument("--num_genes", type=int, default=1000)
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--multivariate", type=int, default=0)

    args = parser.parse_args()
    
    rna_dataset = load_data(args.num_genes)
    main(args, rna_dataset)
