import os
import numpy as np
from copy import copy
import time
import math
import argparse
import pandas as pd
import warnings
import random
import sys
import pickle
import argparse
import wandb

import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from model import VAE

from pyensembl import EnsemblRelease
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def load_data():
    train = pd.read_csv(f'../data/RNAseqDB/x_train_all_tissue.csv')
    test = pd.read_csv(f'../data/RNAseqDB/x_test_all_tissue.csv')
    df_real = pd.concat([train, test])
    
    df_real_gene_columns = df_real.iloc[:,1:-2].columns
    # 처음 1000개
    first_1000_genes = list(df_real_gene_columns)[:1000]

    # normalize expression data for nn
    steps = [('standardization', StandardScaler()), ('normalization', MinMaxScaler())]
    pre_processing_pipeline = Pipeline(steps)
    transformed_data = pre_processing_pipeline.fit_transform(df_real[first_1000_genes])

    scaled_df = pd.DataFrame(transformed_data, columns=first_1000_genes)
    X = scaled_df.values
    Y = df_real["TISSUE_GTEX"].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    gene_names = df_real_gene_columns.tolist()

    dict_train_test_splited = {'train_set': (X_train, Y_train),
                               'test_set': (X_test, Y_test),
                               'X_df': scaled_df.values,
                               'Y': Y,
                               'gene_names': gene_names}
    return dict_train_test_splited

def main(args, rna_dataset):
    torch.manual_seed(args.seed)
    latest_loss = torch.tensor(1)
    
    # GPU 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)  
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # time
    ts = time.time()
    
    # Train/Test dataset
    (X_train, Y_train) = rna_dataset['train_set']
    (X_test, Y_test) = rna_dataset['test_set']
    scaled_df_values = rna_dataset['X_df']
    gene_names = rna_dataset['gene_names'] # ex) COL4A1, IFT27,,,
    Y = rna_dataset['Y'] # ["TISSUE_GTEX"].values
    LABELS_NUM = len(set(Y)) # 15 (breast, lung, liver 등 15개 tissue)
    
    # one-hot encoding
    le = LabelEncoder()
    le.fit(Y_train)
    train_targets = le.transform(Y_train)
    test_targets = le.transform(Y_test)

    train_target = torch.as_tensor(train_targets)
    train = torch.tensor(X_train.astype(np.float32))
    train_tensor = TensorDataset(train, train_target)
    data_loader = DataLoader(dataset=train_tensor, batch_size=args.batch_size, shuffle=True)

    test_target = torch.as_tensor(test_targets)
    test = torch.tensor(X_test.astype(np.float32))
    test_tensor = TensorDataset(test, test_target)
    test_loader = DataLoader(dataset=test_tensor, batch_size=args.batch_size, shuffle=True)
    
    # loss function
    def loss_fn(recon_x, x, mean, log_var):
        view_size = 1000
        ENTROPY = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, view_size), x.view(-1, view_size), reduction='sum')
        HALF_LOG_TWO_PI = 0.91893
        MSE = torch.sum((x - recon_x)**2)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        gamma_square = 0
        if torch.eq(latest_loss, torch.tensor(1)):
            gamma_square = MSE
        else:
            gamma_square = min(MSE, latest_loss.clone())
        beta = 0.9
        return {'GL': (ENTROPY + KLD) / x.size(0), 'MSE': MSE}

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=LABELS_NUM if args.conditional else 0).to(device)

    dataiter = iter(data_loader)
    genes, labels = dataiter.next()

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):
        train_loss = 0
        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            multiple_losses = loss_fn(recon_x, x, mean, log_var)
            loss = multiple_losses['GL'].clone()
            train_loss += loss

            latest_loss = multiple_losses['MSE'].detach()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if args.conditional:
                    c = torch.arange(0, LABELS_NUM).long().unsqueeze(1)
                    x = vae.inference(n=c.size(0), c=c)
                else:
                    x = vae.inference()

    with torch.no_grad():
        for epoch in range(args.epochs):
            test_loss = 0
            for iteration, (x, y) in enumerate(test_loader):
                recon_x, mean, log_var, z = vae(x, y)
                test_loss = loss_fn(recon_x, x, mean, log_var)['GL']

                if iteration == len(test_loader) - 1:
                    print('====> Test set loss: {:.4f}'.format(test_loss.item()))
    # check quality of reconstruction
    check_reconstruction_and_sampling_fidelity(vae, le,  scaled_df_values, Y, gene_names)


def check_reconstruction_and_sampling_fidelity(vae_model, le, scaled_df_values, Y, gene_names):
    # get means of original columns based on 100 first rows
    genes_to_validate = 40
    original_means = np.mean(scaled_df_values, axis=0)
    original_vars = np.var(scaled_df_values, axis=0)

    #mean, logvar = vae_model.encode(scaled_df_values, Y)
    #z = vae_model.reparameterize(mean, logvar)

    #plot_dataset_in_3d_space(z, y_values)

    #x_decoded = vae_model.decode(z, Y)

    #decoded_means = np.mean(x_decoded, axis=0)
    #decoded_vars = np.var(x_decoded, axis=0)

    with torch.no_grad():
        number_of_samples = 500
        labels_to_generate = []
        for label_value in le.classes_:
            label_to_generate = [label_value for i in range(number_of_samples)]
            labels_to_generate += label_to_generate
        all_samples = np.array(labels_to_generate)
        #c = torch.from_numpy(all_samples)
        c = all_samples
        #print(c)
        x = vae_model.inference(n=len(all_samples), c=c)
        print(x)

    sampled_means = np.mean(x.detach().numpy(), axis=0)
    sampled_vars = np.var(x.detach().numpy(), axis=0)

    plot_reconstruction_fidelity(original_means[:genes_to_validate], sampled_means[:genes_to_validate], metric_name='Mean', df_columns=gene_names)
    plot_reconstruction_fidelity(original_vars[:genes_to_validate], sampled_vars[:genes_to_validate], metric_name='Variance', df_columns=gene_names)

def plot_reconstruction_fidelity(original_values, sampled_values=[], metric_name='', df_columns=[]):
    n_groups = len(original_values)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    if len(sampled_values) > 0:
        plt.bar(index, original_values, bar_width, alpha=opacity, color='b', label='Original')
        plt.bar(index + bar_width, sampled_values, bar_width, alpha=opacity, color='g', label='Reconstructed')
        plt.title('Original VS Reconstructed ' + metric_name)
        plt.xticks(index + bar_width, list(df_columns)[:n_groups], rotation=90)
        plt.ylabel(metric_name + ' Expression Level (Scaled)')
        plt.legend()
    else:
        plt.bar(index, original_values, bar_width, alpha=opacity, color='b')
        plt.title(metric_name)
        plt.xticks(index, list(df_columns)[:n_groups], rotation=90)
        plt.ylabel('Expression Level (Scaled)')
        plt.legend()

    plt.xlabel('Gene Name')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1000, 512, 256]) #[1000, 512, 256]
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 512, 1000]) # [256, 512, 1000]
    parser.add_argument("--latent_size", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    
    rna_dataset = load_data()
    main(args, rna_dataset)
