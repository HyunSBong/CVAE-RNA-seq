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

def draw_umap(X_test, x_syn, Y_test_tissues, Y_test_datasets):
    x_log = np.log1p(x_syn)
    x_log = np.float32(x_log)
    x_mean = np.mean(x_log, axis=0)
    x_std = np.std(x_log, axis=0)
    x_log = standardize(x_log, mean=x_mean, std=x_std)

    x = np.concatenate((X_test, x_syn), axis=0)
    model = umap.UMAP(n_neighbors=300,
                    min_dist=0.7,
                    n_components=2,
                    random_state=1111)
    model.fit(x)
    emb_2d = model.transform(x)
    plot_umap(emb_2d, X_test, x_syn, Y_test_tissues, Y_test_datasets)

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
    # transformed_data = pre_processing_pipeline.fit_transform(df_real[df_real_gene_columns])

    scaled_df = pd.DataFrame(transformed_data, columns=first_1000_genes)
    # scaled_df = pd.DataFrame(transformed_data, columns=df_real_gene_columns)
    X = scaled_df.values
    Y = df_real[["TISSUE_GTEX", "DATASET"]].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    gene_names = df_real_gene_columns.tolist()

    dict_train_test_splited = {'train_set': (X_train, Y_train),
                            'test_set': (X_test, Y_test),
                            'X': X,
                            'Y': Y,
                            'gene_names': gene_names}
    return dict_train_test_splited

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
    
    # Train/Test dataset & tissues
    (X_train, Y_train) = rna_dataset['train_set']
    (X_test, Y_test) = rna_dataset['test_set']
    Y_train_tissues = Y_train[:,0]
    Y_train_datasets = Y_train[:,1]
    Y_test_tissues = Y_test[:,0]
    Y_test_datasets = Y_test[:,1]
    gene_names = rna_dataset['gene_names'] # ex) COL4A1, IFT27,,,
    X = rna_dataset['X']
    Y = rna_dataset['Y'] # [["TISSUE_GTEX", "DATASET"]].values
    Y_tissues = Y[:,0]
    Y_datasets = Y[:,1]
    num_tissue = len(set(Y_tissues)) # 15 (breast, lung, liver 등 15개 tissue)
    
    # one-hot encoding
    le = LabelEncoder()
    le.fit(Y_train_tissues)
    train_targets = le.transform(Y_train_tissues) # bladder,uterus,,, -> 0,14,,,
    test_targets = le.transform(Y_test_tissues)

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
        # print(gamma_square)
        beta = 0.9

        return {'GL': (ENTROPY + KLD) / x.size(0), 'MSE': MSE}
    
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=num_tissue if args.conditional else 0).to(device)

    dataiter = iter(data_loader)
    genes, labels = dataiter.next()
    # writer.add_graph(vae, genes)
    # writer.close()

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)
    
    stop_point = 10
    best_score = -np.inf
    initial_stop_point = stop_point

    for epoch in range(args.epochs):
        train_loss = 0
        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for enumer_idx, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
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
            
            if args.conditional:
                c = torch.arange(0, num_tissue).long().unsqueeze(1)
                c = torch.from_numpy(test_targets)
                x = vae.inference(n=c.size(0), c=c)
            else:
                x = vae.inference()
            
            if enumer_idx % args.verbose == 0 or enumer_idx == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, enumer_idx, len(data_loader)-1, loss.item()))

                if args.conditional:
                    c = torch.arange(0, num_tissue).long().unsqueeze(1)
                    c = torch.from_numpy(test_targets) # le.fit_transform(Y_train_tissues)
                    x = vae.inference(n=c.size(0), c=c)
                    
                    if epoch % 5 == 0:
                        score = score_fn(X_test, x.detach().cpu().numpy())
                        if score > best_score:
                            best_score = score
                            stop_point = initial_stop_point
                        else:
                            stop_point -= 1
                        print(f'==>Gamma Score : {score}')
                else:
                    x = vae.inference()
                    
            if stop_point == 0:
                break   

    with torch.no_grad():
        for epoch in range(args.epochs):
            test_loss = 0
            for enumer_idx, (x, y) in enumerate(test_loader):
                recon_x, mean, log_var, z = vae(x, y)
                if x.is_cuda != True:
                    x = x.cuda()
                test_loss = loss_fn(recon_x, x, mean, log_var)['GL']

                if enumer_idx == len(test_loader) - 1:
                    print('====> Test set loss: {:.4f}'.format(test_loss.item()))

    x_syn = generate_synthetic(vae, le, X, gene_names, Y_test_tissues)
    draw_umap(X_test, x_syn, Y_test_tissues, Y_test_datasets)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=list, default=[3000, 512, 256]) 
    parser.add_argument("--decompress_dims", type=list, default=[256, 512, 3000]) 
    parser.add_argument("--latent_size", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    
    rna_dataset = load_data()
    main(args, rna_dataset)
