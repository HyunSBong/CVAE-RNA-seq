import os
import numpy as np
import time
import argparse
import pandas as pd
import argparse
from collections import defaultdict
from utils import *
from model import VAE

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import wandb
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
    train_targets = le.transform(Y_train) # bladder,uterus,,, -> 0,14,,,
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
        print(gamma_square)
        beta = 0.9

        return {'GL': (ENTROPY + KLD) / x.size(0), 'MSE': MSE}
    
    print(f'device : {device}')
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=LABELS_NUM if args.conditional else 0).to(device)

    dataiter = iter(data_loader)
    genes, labels = dataiter.next()
    # writer.add_graph(vae, genes)
    # writer.close()

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):
        train_loss = 0
        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                # print(f'in dataLoader : x is cuda? {x.is_cuda}, y is cuda? {y.is_cuda}')
                if x.is_cuda != True:
                    x = x.cuda()
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
                # print(f'in testLoader : recon_x is cuda? {recon_x.is_cuda}, mean is cuda? {mean.is_cuda}')
                # print(f'in testLoader : log_var is cuda? {log_var.is_cuda}, x is cuda? {x.is_cuda}')
                if x.is_cuda != True:
                    x = x.cuda()
                test_loss = loss_fn(recon_x, x, mean, log_var)['GL']

                if iteration == len(test_loader) - 1:
                    print('====> Test set loss: {:.4f}'.format(test_loss.item()))

    # check quality of reconstruction and draw umap plot
    check_reconstruction_and_sampling_fidelity(vae, le,  scaled_df_values, Y, gene_names)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
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
