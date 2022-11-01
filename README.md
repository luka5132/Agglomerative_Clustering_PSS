# Overview of Repo

## General supervision
This repo hosts the code as well as the paper for my (Lukas Busch) master these at the University of Amsterdam.
I researched the use of agglomerative clustering for page stream segmentation (PSS) on Dutch WOO (WOB) requests. These are long streams of concatenated documents, that we through PSS try to split up into the original documents. An important first step to make these streams more readable and to make the Open Government act funcion better. 

## Acknowledgements
In this researched I have been supervised by [Maarten Marx](https://www.uva.nl/profiel/m/a/m.j.marx/m.j.marx.html).

We used agglomerative clustering instead of the standard state-of-the-art approach, which is first page classification. To compare our results and to obtain finetuned deep learned vector representations we use the CNN models created by [Wiedemann et al.](https://github.com/uhh-lt/pss-lrev)

> Wiedemann, G., Heyer, G. Multi-modal page stream segmentation with convolutional neural networks. Lang Resources & Evaluation 55, 127–150 (2021). https://doi.org/10.1007/s10579-019-09476-2


## Documents

Link to final version of [Thesis](https://github.com/luka5132/Agglomerative_Clustering_PSS/blob/main/DSS_Thesis_LukasBusch.pdf)

This repo contains the following files:
1) metricutils.py: script to calculate metric scores
2) utils.py: script with utils function, to quickly and easily perform clustering
3) main.ipynb: notebook that shows the main resutls obtained for clustering and classification with different modalities
4) EDA.ipynb: analysis of data en trying different methods of improvement
5) prep.ipynb: data preparation and creating deep learned vector representations

Link to data: YET TO BE UPLOADED
