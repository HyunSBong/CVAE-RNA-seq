# CVAE-RNA-seq
github for "Conditional Variational Autoencoder-based Generative Model for Gene Expression Data Augmentation"
https://arxiv.org/abs/180xxxx [late,,]

Conditional variational generation of gene expression data

To secure sufficient gene expression data, a study that develops and proposes a Contidional Variational Auto-Encoder 

Dataset
----------
In this study, samples of 15 common tissues (lung, breast, kidney, thyroid, colon, stomach, prostate, saliva, liver, esophageal myopathy, esophageal mucosa, esophageal gastrointestinal tract, bladder, uterus, and cervix) of GTEx and TCGA were used. We followed the [pipeline](https://github.com/mskcc/RNAseqDB) described by Wang et al. (2018) to integrate data and modify the deployment effect. Thus, the entire dataset consists of 9147 samples and 18154 genes.
- GTEx(Genotype-Tissue Expression) Dataset
- TCGA(Cancer Genome Atlas) Dataset
- RNA-seq(human transcriptomics) Dataset (9147 samples and 18154 genes )

UMAP
----------
- Test 2745 samples, 18154 genes.
- Gamma score 0.98
<img width="712" alt="스크린샷 2023-02-24 오전 2 25 11" src="https://user-images.githubusercontent.com/69189272/220983236-25ae773f-0264-47aa-830d-bee86627d5ef.png">
- Compare with datasets such as [Ramon Viñas, Helena Andrés-Terré, Pietro Liò, Kevin Bryson, Adversarial generation of gene expression data, Bioinformatics, Volume 38, Issue 3, February 2022, Pages 730–737]
- Gamma score 0.96
<img width="1417" alt="bioinfo_sample" src="https://user-images.githubusercontent.com/69189272/217756386-82593ab1-4e83-4e23-9089-27ad56917c97.png">


Quick start
----------

1. First, like  [RNAseqDB](https://github.com/mskcc/RNAseqDB) , create human transcriptomics data for 15 common tissues.

2. You can train with the following command (num_genes: train genes, multivatiable: decoder 0;bernoulli, 1;gaussian)
```
python train.py --gpu_id=0 --epochs=300 --num_genes=1000 --multivatiable=0
```
3. You can see an umap plot is automatically drawn after the train.
