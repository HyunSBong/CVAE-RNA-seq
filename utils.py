import torch
import numpy as np

import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

def tsne_2d(data, **kwargs):
    """
    Transform data to 2d tSNE representation
    :param data: expression data. Shape=(dim1, dim2)
    :param kwargs: tSNE kwargs
    :return:
    """
    print('... performing tSNE')
    tsne = TSNE(n_components=2, **kwargs)
    return tsne.fit_transform(data)


def plot_tsne_2d(data, labels, **kwargs):
    """
    Plots tSNE for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    dim1, dim2 = data.shape

    # Prepare label dict and color map
    label_set = set(labels)
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Perform tSNE
    if dim2 == 2:
        # print('plot_tsne_2d: Not performing tSNE. Shape of second dimension is 2')
        data_2d = data
    elif dim2 > 2:
        data_2d = tsne_2d(data, **kwargs)
    else:
        raise ValueError('Shape of second dimension is <2: {}'.format(dim2))

    # Plot scatterplot
    for k, v in label_dict.items():
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v)
    plt.legend()
    return plt.gca()

def scatter_2d(data_2d, labels, colors=None, **kwargs):
    """
    Scatterplot for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    # Prepare label dict and color map
    label_set = list(set(labels))[::-1]
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Plot scatterplot
    i = 0
    for k, v in label_dict.items():
        c = None
        if colors is not None:
            c = colors[i]
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v, color=c, **kwargs)
        i += 1
    lgnd = plt.legend(markerscale=3)
    return plt.gca()

def scatter_2d_cancer(data_2d, labels, cancer, colors=None, **kwargs):
    # Prepare label dict and color map
    label_set = list(set(labels))[::-1]
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Plot scatterplot
    i = 0
    for k, v in label_dict.items():
        c = None
        if colors is not None:
            c = colors[i]

        idxs = np.logical_and(labels == v, cancer == 'normal')
        plt.scatter(data_2d[idxs, 0], data_2d[idxs, 1],
                    label=v, color=c, marker='o', s=7, **kwargs)
        idxs = np.logical_and(labels == v, cancer == 'cancer')
        plt.scatter(data_2d[idxs, 0], data_2d[idxs, 1], color=c, marker='+', **kwargs)
        i += 1
    lgnd = plt.legend(markerscale=3)
    return plt.gca()

def check_reconstruction_and_sampling_fidelity(vae_model, le, scaled_df_values, Y, gene_names):
    genes_to_validate = 40
    original_means = np.mean(scaled_df_values, axis=0)
    original_vars = np.var(scaled_df_values, axis=0)

    with torch.no_grad():
        number_of_samples = 500
        labels_to_generate = []
        for label_value in le.classes_:
            label_to_generate = [label_value for i in range(number_of_samples)]
            labels_to_generate += label_to_generate
        all_samples = np.array(labels_to_generate)
        
        le = LabelEncoder()
        onehot_c = le.fit_transform(all_samples)
        
        c = all_samples # ['bladder' 'bladder' 'bladder' ... 'uterus' 'uterus' 'uterus']
        c = torch.from_numpy(onehot_c) # [0 0 0 ... 14 14 14]
        x = vae_model.inference(n=len(all_samples), c=c)
        print(f'generated x => {x}')

    sampled_means = np.mean(x.detach().cpu().numpy(), axis=0)
    sampled_vars = np.var(x.detach().cpu().numpy(), axis=0)

    plot_reconstruction_fidelity(original_means[:genes_to_validate], sampled_means[:genes_to_validate], metric_name='Mean', df_columns=gene_names)
    plot_reconstruction_fidelity(original_vars[:genes_to_validate], sampled_vars[:genes_to_validate], metric_name='Variance', df_columns=gene_names)
    
    x_syn = x.detach().cpu().numpy()
    plot_umap(x_syn, all_samples)
    
def plot_umap(x, tissue):
    model = umap.UMAP(n_neighbors=800,
                  min_dist=0.0,
                  n_components=2,
                  random_state=1111)
    model.fit(x)
    emb_2d = model.transform(x)

    plt.figure(figsize=(18, 6))
    ax = plt.gca()

    plt.subplot(1, 2, 1)
    colors =  plt.get_cmap('tab20').colors
    ax = scatter_2d(emb_2d, tissue, s=10, marker='.')
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=3, markerscale=5)
    plt.axis('off')

    # plt.subplot(1, 2, 2)
    # c = np.array(['normal' if dataset_dict_inv[q] != 'tcga-t' else 'cancer' for q in cat_covs[:, 1]])
    # # colors = ['brown', 'lightgray']
    # colors = ['lightgray', 'brown']
    # ax = scatter_2d(emb_2d, c, colors=colors, s=10, marker='.')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #               fancybox=True, shadow=True, ncol=3, markerscale=5)
    # plt.axis('off')
    
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