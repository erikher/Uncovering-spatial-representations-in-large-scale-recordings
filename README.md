# Uncovering spatial representation in large scale recordings using topological data analysis and unsupervised ensemble detection

Code for the analysis conducted in "Uncovering 2-D toroidal representations in grid cell ensemble activity during 1-D behavior" - E. Hermansen, D. A. Klindt. & B. A. Dunn (https://www.biorxiv.org/content/10.1101/2022.11.25.517966v1)

This has been tested on a Macbook Pro M2, macOS Ventura (13) using the following versions:

Open-source software: Jupyter notebook 6.4.8, Python version 3.9.12, MySQL 5.7
 
Open-source Python packages:
umap  0.5.5
ripser 0.6.4
numba 0.58.1
scipy 1.11.4
numpy 1.26.2
matplotlib 3.8.2
h5py 3.6.0
gtda 0.6.0
cv2 4.8.1
pandas 1.4.2
datajoint 0.13.5
IPython 8.2.0

Install python/Anaconda and relevant packages using pip or conda. 
Install time ~30min-1hr

Consecutive blocks of the notebooks are to be run in order, with output figures given per cell.
Run time depends on notebook but usually takes 15-30 mins. 
Note that shuffles of barcodes for homological dimensions >= 2 were run using a high-performance computing cluster. 
