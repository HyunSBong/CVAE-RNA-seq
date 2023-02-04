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
2745 samples, 3000 genes
<img width="1433" alt="스크린샷 2023-02-04 오후 10 38 24" src="https://user-images.githubusercontent.com/69189272/216770798-acfb75a2-5e86-4be8-9930-a032d2bafd3f.png">

1. First, like  [RNAseqDB](https://github.com/mskcc/RNAseqDB) below, we create human transcriptomics data for 15 common tissues.

2. You can train with the following command
```
python train.py --gpu_id=0 --epochs=300 
```
3. You can see an umap plot is automatically drawn after the train.
