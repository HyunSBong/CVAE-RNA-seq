# CVAE-RNA-seq
<img width="749" alt="스크린샷 2023-04-02 오후 11 47 58" src="https://user-images.githubusercontent.com/69189272/229360369-fd217d1c-6749-462f-b617-30adc314c4f1.png">
github for "Conditional Variational Autoencoder-based Generative Model for Gene Expression Data Augmentation"
RNA-seq generation model | [Paper](https://doi.org/10.5909/JBE.2023.28.3.275) | [Code](https://github.com/HyunSBong/CVAE-RNA-seq)


Overview
----------
Gene expression data can be utilized in various studies, including the prediction of disease prognosis. However, there are challenges associated with collecting enough data due to cost constraints. In this paper, we propose a gene expression data generation model based on Conditional Variational Autoencoder. Our results demonstrate that the proposed model generates synthetic data with superior quality compared to two other state-of-the-art models for gene expression data generation, namely the Wasserstein Generative Adversarial Network with Gradient Penalty based model and the structured data generation models CTGAN and TVAE.

Simple Result
----------
- Test 2745 samples, 969 L1000 landmark genes.

- Gamma score 0.98
<img width="766" alt="스크린샷 2023-04-02 오후 11 48 23" src="https://user-images.githubusercontent.com/69189272/229360395-d363555e-2e55-4405-bd3c-226868499f6d.png">
- Compare with datasets such as [Ramon Viñas, Helena Andrés-Terré, Pietro Liò,
Kevin Bryson, Adversarial generation of gene expression data, Bioinformatics, Volume 38, Issue 3, February 2022, Pages 730–737]
- Gamma score 0.96
<img width="688" alt="스크린샷 2023-04-02 오후 11 48 56" src="https://user-images.githubusercontent.com/69189272/229360428-698ee774-7aac-450d-9a6e-5c232814d65f.png">

Dataset
----------
In this study, samples of 15 common tissues (lung, breast, kidney, thyroid, colon, stomach, prostate, saliva, liver, esophageal myopathy, esophageal mucosa, esophageal gastrointestinal tract, bladder, uterus, and cervix) of GTEx and TCGA were used. We followed the [pipeline](https://github.com/mskcc/RNAseqDB) described by Wang et al. (2018) to integrate data and modify the deployment effect. Since then, 969 common genes with the L1000 landmark gene set were selected to create a dataset consisting of 9,146 samples and 969 genes.
- GTEx(Genotype-Tissue Expression) Dataset
- TCGA(Cancer Genome Atlas) Dataset
- L1000 landmark 
- RNA-seq(human transcriptomics) Dataset (9147 samples and 18154 genes )

Install dependencies
----------
- torch >= 1.12.1
- umap-learn >= 0.5.3
- scikit-learn >= 1.1.1
