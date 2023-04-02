# CVAE-RNA-seq
<img width="749" alt="스크린샷 2023-04-02 오후 11 47 58" src="https://user-images.githubusercontent.com/69189272/229360369-fd217d1c-6749-462f-b617-30adc314c4f1.png">
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
- Test 2745 samples, 978 L1000 landmark genes.

- Gamma score 0.98
<img width="766" alt="스크린샷 2023-04-02 오후 11 48 23" src="https://user-images.githubusercontent.com/69189272/229360395-d363555e-2e55-4405-bd3c-226868499f6d.png">

- Compare with datasets such as [Ramon Viñas, Helena Andrés-Terré, Pietro Liò,
Kevin Bryson, Adversarial generation of gene expression data, Bioinformatics, Volume 38, Issue 3, February 2022, Pages 730–737]
- Gamma score 0.96
<img width="688" alt="스크린샷 2023-04-02 오후 11 48 56" src="https://user-images.githubusercontent.com/69189272/229360428-698ee774-7aac-450d-9a6e-5c232814d65f.png">


Quick start
----------

1. First, like  [RNAseqDB](https://github.com/mskcc/RNAseqDB) , create human transcriptomics data for 15 common tissues.

2. You can train with the following command (num_genes: train genes, multivatiable: decoder 0;bernoulli, 1;gaussian)
```
python train.py --gpu_id=0 --epochs=300 --num_genes=1000 --multivatiable=0
```
3. You can see an umap plot is automatically drawn after the train.
