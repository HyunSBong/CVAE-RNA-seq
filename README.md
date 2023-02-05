# CVAE-RNA-seq
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
Test 2745 samples, 15000 genes
<img width="1497" alt="sample" src="https://user-images.githubusercontent.com/69189272/216815980-f708dd30-0adf-455d-9cdc-5feb1f9cccfa.png">

Quick start
----------

1. First, like  [RNAseqDB](https://github.com/mskcc/RNAseqDB) , we create human transcriptomics data for 15 common tissues.

2. You can train with the following command (num_genes: train genes)
```
python train.py --gpu_id=0 --epochs=300 --num_genes=1000
```
3. You can see an umap plot is automatically drawn after the train.
