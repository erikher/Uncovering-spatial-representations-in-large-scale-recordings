# Uncovering spatial representation in large scale recordings using topological data analysis and unsupervised ensemble detection

Code for the analysis conducted in "Uncovering 2-D toroidal representations in grid cell ensemble activity during 1-D behavior" - E. Hermansen, D. A. Klindt. & B. A. Dunn (https://www.biorxiv.org/content/10.1101/2022.11.25.517966v1)

The data are publicly shared by the respective authors at: 
- Campbell et al. (2021) "Distance-tuned neurons drive specialized path integration calculations in medial entorhinal cortex" https://plus.figshare.com/articles/dataset/VR_Data_Neuropixel_supporting_Distance-tuned_neurons_drive_specialized_path_integration_calculations_in_medial_entorhinal_cortex_/15041316
- Zong et al. (2022) "Large-scale two-photon calcium imaging in freely moving mice" https://archive.sigma2.no/pages/public/datasetDetail.jsf?id=10.11582/2022.00008
- Obenhaus et al. (2022) "Functional network topography of the medial entorhinal cortex" https://archive.sigma2.no/pages/public/datasetDetail.jsf?id=10.11582/2022.00005
- Garnder et al. (2022) "Toroidal topology of population activity in grid cells" https://figshare.com/articles/dataset/Toroidal_topology_of_population_activity_in_grid_cells/16764508
- Peyrache et al. (2015) "Internally organized mechanisms of the head direction sense" https://crcns.org/data-sets/thalamus/th-1
- Waaga et al. (2022) "Grid-cell modules remain coordinated when neural activity is dissociated from external sensory cues" https://zenodo.org/records/6200517

The data needs to be downloaded and the directories referenced appropriately where relevant.

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
Cebra 0.3.1

Install python/Anaconda and relevant packages using pip or conda. 
Install time ~30min-1hr

Download utils_new.py and add to folder of relevant notebook (or referred to in PATH)
Consecutive blocks of the notebooks are to be run in order, with output figures given per cell.
Run time depends on notebook but usually takes 15-30 mins. 
Note that shuffles of barcodes for homological dimensions >= 2 were run using a high-performance computing cluster. 
