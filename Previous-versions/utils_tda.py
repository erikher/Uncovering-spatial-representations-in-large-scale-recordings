import sys
import os 
import cv2 as cv
import numpy as np
import functools
import glob
import numba

from matplotlib import gridspec, transforms, pyplot as plt
from matplotlib.collections import PathCollection

from sklearn import preprocessing, decomposition as dimred

from scipy.stats import wilcoxon, binned_statistic, binned_statistic_2d, pearsonr, multivariate_normal
from scipy.ndimage import affine_transform, shift, gaussian_filter,  gaussian_filter1d, rotate, binary_dilation, binary_closing
import scipy.io as sio
from scipy.interpolate import CubicSpline
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import scipy.optimize as opt
from scipy.special import factorial
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr


from ripser import ripser
from gtda.homology import VietorisRipsPersistence
    
ylims = {}
ylims['corr'] = [-0.05, 1]
ylims['dist'] = [0, 2.6/(2*np.pi)*360]
ylims['info'] = [0,1.1]
ylims['ED_OF'] = [0,0.15]
ylims['ED_WW'] = [0,0.2]
ytics = {}
ytics['corr'] = [0.0, 0.25, 0.5, 0.75, 1]
ytics['dist'] = [0, 50, 100, 150]
ytics['ED_OF'] = [0, 0.05,0.1, 0.15, 0.2]
ytics['ED_WW'] = [0, 0.05,0.1, 0.15, 0.2]
ytics['info'] = [0.0, 0.25, 0.50, 0.75,1.0]

combs_all = {}
combs_all[0] = [[0, 0, 1], 
                [1, 0, 1],
                [0, 0, 3],
                [1, 0, 3],
                [0, 1, 1],
                [1, 1, 1],
                [0, 1, 3],
                [1, 1, 3],                
                [0, 2, 0],
                [0, 3, 0],
                [1, 2, 0],
                [1, 3, 0]]

combs_all[1] = [[0, 0, 2], 
                [1, 0, 4], 
                [0, 1, 2], 
                [1, 1, 4],
                [1, 0, 2],
                [0, 0, 4], 
                [1, 1, 2], 
                [0, 1, 4],                
                [0, 2, 0],
                [0, 3, 0],
                [1, 2, 0],
                [1, 3, 0]]

combs_all[2] = [[0, 2, 2],
                [1, 2, 2], 
                [0, 2, 4], 
                [1, 2, 4], 
                [0, 3, 2], 
                [1, 3, 2], 
                [0, 3, 4], 
                [1, 3, 4],                
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0]]

combs_all[3] = [[0, 2, 1], 
                [1, 2, 3], 
                [0, 3, 1], 
                [1, 3, 3],
                [1, 2, 1], 
                [0, 2, 3], 
                [1, 3, 1], 
                [0, 3, 3],
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0]]

ks = np.array([[0,0], [1,0], [0,1], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])


def plt_sqr(coords, data, numbins = 10, bSqr = True, folder = '', fname = ''):    
    num_neurons = len(data[0,:])
    mcstemp, mtot_all = get_ratemaps_center(coords, data, numbins = numbins,)
    intervals = np.linspace(0, 2*np.pi, 100)
    intervals = intervals[1:] - (intervals[1]-intervals[0])/2
    c1,c2 = np.meshgrid(intervals,intervals)
#    coords_uniform = np.random.rand(len(coords),2)*2*np.pi
    coords_uniform = np.concatenate((c1.flatten()[:,np.newaxis], c2.flatten()[:,np.newaxis]), 1)
    spk_sim_sqr = get_sim(coords_uniform, mcstemp, numbins = numbins, simtype = 'sqr')
        
    if len(folder) >0:
        fig,ax = plt.subplots(1,2)
        print(theta)
        plot_centered_ratemaps(coords, data, mcstemp, numbins, ax[0])
        plot_centered_ratemaps(coords_uniform, spk_sim_sqr, mcstemp, numbins, ax[1])            
        fig.tight_layout()
        fig.savefig(folder + '/sqr_ratemap' + fname, transparent = True)
        plt.close()

def from_ripser_to_giotto(dgm, infmax= np.inf):
    dgm_ret = []
    for dim, i in enumerate(dgm):
        i[np.isinf(i)] = infmax
        for j in i:
            dgm_ret.append([j[0],j[1], dim])
    return dgm_ret

def from_giotto_to_ripser(dgm):
    dgm_ret = []
    for i in range(int(np.max(dgm[:, 2]))+1):
        dgm_ret.append(dgm[dgm[:,2]==i, :2])
    return dgm_ret

def pca(data, dim=2):
    if dim < 1:
        return data, [0]
    m, n = data.shape
    # mean center the data
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eig(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dim]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    tot = np.sum(evals)
    var_exp = [(i / tot) * 100 for i in sorted(evals[:dim], reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    components = np.dot(evecs.T, data.T).T
    return components, evecs, evals[:dim], var_exp

def get_coords_all(sspikes2, coords1, times_cube, indstemp, dim = 7, spk2 = [], bPred = False, bPCA = False):
    num_circ = len(coords1)
    spkmean = np.mean(sspikes2[times_cube,:], axis = 0)
    spkstd = np.std(sspikes2[times_cube,:], axis = 0)
    spkscale = (sspikes2-spkmean)/spkstd
    dspk1 = spkscale.copy()
    if bPCA:
        __, e1, e2,__ = pca(spkscale[times_cube,:], dim = dim)
        dspk1 = np.dot(e1.T, spkscale.T).T    
        dspk1 /= np.sqrt(e2)    
        
    dspk = dspk1[indstemp,:]
    if len(spk2)>0:
        dspk1 = (spk2-spkmean)/spkstd
        if bPCA:
            dspk1 = np.dot(e1.T, dspk1.T).T    
            dspk1 /= np.sqrt(e2)    


    if bPred:
        coords_mod1 = np.zeros((len(sspikes2), num_circ))
        coords_mod1[:,0] = predict_color(coords1[0,:], dspk1, dspk, 
                                     dist_measure='cosine',  k = 30)
        coords_mod1[:,1] = predict_color(coords1[1,:],  dspk1, dspk,
                                         dist_measure='cosine',  k = 30)
    else:
        num_neurons = len(dspk[0,:])
        centcosall = np.zeros((num_neurons, num_circ, len(indstemp)))
        centsinall = np.zeros((num_neurons, num_circ, len(indstemp)))    
        for neurid in range(num_neurons):
            spktemp = dspk[:, neurid].copy()
            centcosall[neurid,:,:] = np.multiply(np.cos(coords1[:, :]*2*np.pi),spktemp)
            centsinall[neurid,:,:] = np.multiply(np.sin(coords1[:, :]*2*np.pi),spktemp)

        a = np.zeros((len(dspk1), num_circ, num_neurons))
        for n in range(num_neurons):
            a[:,:,n] = np.multiply(dspk1[:,n:n+1],np.sum(centcosall[n,:,:],1))

        c = np.zeros((len(dspk1), num_circ, num_neurons))
        for n in range(num_neurons):
            c[:,:,n] = np.multiply(dspk1[:,n:n+1],np.sum(centsinall[n,:,:],1))

        mtot2 = np.sum(c,2)
        mtot1 = np.sum(a,2)
        coords_mod1 = np.arctan2(mtot2,mtot1)%(2*np.pi)
    return coords_mod1

def align_coords(coords_mod1, coords_mod2, theta, theta1, times = [], times_cube = [], 
                 indstemp = [], data_ensemble1 = [], data_ensemble2 = []):
    ks = np.array([[0,0], [1,0], [0,1], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])
    sig_smooth = 20
    
    combs = get_combs(theta, theta1)
    
    if len(coords_mod1) != len(coords_mod2):
        coords_smooth_1 = get_coords_all(data_ensemble1, coords_mod1[indstemp,:2].T/(2*np.pi), times_cube, indstemp, 
                                         spk2 = data_ensemble2,  dim = 7, bPred = False, bPCA = True)
    else:        
        coords_smooth_1 = coords_mod1.copy()
        
    coords_smooth_1 = np.arctan2(gaussian_filter1d(np.sin(coords_smooth_1), sigma = sig_smooth, axis = 0),
        gaussian_filter1d(np.cos(coords_smooth_1), sigma = sig_smooth, axis = 0))%(2*np.pi)
    coords_smooth_2 = np.arctan2(gaussian_filter1d(np.sin(coords_mod2), sigma = sig_smooth, axis = 0),
                    gaussian_filter1d(np.cos(coords_mod2), sigma = sig_smooth, axis = 0))%(2*np.pi)
    if len(times) == 0:
        times = np.arange(0,len(coords_smooth_1), 10)
    
    res2 = fit_derivative(coords_smooth_1[times,:],
                          coords_smooth_2[times,:],
                          combs, 
                          times, 
                          thresh = 0.05
                         )   
    comb = combs[np.argmin(res2)]
    pshift = get_pshift(coords_smooth_1,
                       comb, coords_smooth_2,)
    coords_ret = align(coords_mod1, comb, pshift)
    return coords_ret


def get_data(files, sigma = 6000):
    ################### Get good cells ####################    
    good_cells = []
    for fi in files:
        data = loadmat(fi)
        anatomy = data['anatomy']
        if 'parent_shifted' in anatomy:
            group = anatomy['parent_shifted']
        else:
            group = anatomy['cluster_parent']
        regions = ('MEC',)#'VISp','RS')

        idx = [str(ss).startswith(regions) for ss in group]
        idxagl = [str(ss).startswith('RSPagl') for ss in group]
        region_idx = np.array(idx) & np.logical_not(np.array(idxagl))
        _,sn = os.path.split(fi)
        good_cells.extend(data['sp']['cids'][(data['sp']['cgs']==2) & region_idx])
    good_cells = np.where(np.bincount(good_cells)==len(files))[0]

    data_trials = {}
    data_pos = {}
    sspk1 = {}
    spk1 = {}
    speed = {}
    posx = {}
    post = {}
    posxx = {}
    postt = {}
    postrial = {}
    gain = {}
    contrast = {}
    lickt = {}
    lickx = {}
    res = 100000
    dt = 1000

    thresh = sigma*5
    num_thresh = int(thresh/dt)
    num2_thresh = int(2*num_thresh)
    sig2 = 1/(2*(sigma/res)**2)
    ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
    kerwhere = np.arange(-num_thresh,num_thresh)*dt

    for fi in files:
        data = loadmat(fi)
        spikes = {}
        for cell_idx in range(len(good_cells)):   
            spikes[cell_idx] = data['sp']['st'][data['sp']['clu']==good_cells[cell_idx]]        
        sspikes = np.zeros((1,len(spikes)))
        min_time = 0
        max_time = data['sp']['st'].max()*res+dt
        tt = np.arange(np.floor(min_time), np.ceil(max_time), dt)

        spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes)))
        for n, spk in enumerate(spikes):
            spk = spikes[spk]
            spikes1 = np.array(spk*res-min_time, dtype = int)
            spikes1 = spikes1[(spikes1 < (max_time-min_time)) & (spikes1 > 0)]
            spikes_mod = dt-spikes1%dt
            spikes1 = np.array(spikes1/dt, int)
            for m, j in enumerate(spikes1):
                spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
        spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
        sspikes = np.concatenate((sspikes, spikes_temp),0)
        sspikes = sspikes[1:,:]
        sspikes *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
        sspk1[fi] = sspikes.copy()

        print(sspikes.shape)
        spikes_bin = np.zeros((len(tt), len(spikes)), dtype = int)    
        for n in spikes:
            spike_times = np.array(spikes[n]*res-min_time, dtype = int)
            spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
            spikes_mod = dt-spike_times%dt
            spike_times= np.array(spike_times/dt, int)
            for m, j in enumerate(spike_times):
                spikes_bin[j, n] += 1

        spk1[fi] = spikes_bin.copy()

        tt /= res

        xtmp = data['posx'].copy()
        xtmp += (data['trial']-1)*400
        xtmp = gaussian_filter1d(xtmp, sigma = 5)        
        xx_spline = CubicSpline(data['post'], xtmp)
        speed0 = xx_spline(tt,1)
        
        speed[fi] = speed0.copy()

        xx_spline = CubicSpline(data['post'], data['posx'])

        posxx[fi] = xx_spline(tt).copy()%400
        postt[fi] = tt.copy()

        posx[fi] = data['posx']
        post[fi] = data['post']
        postrial[fi] = data['trial']
        gain[fi] = data['trial_gain']
        contrast[fi] = data['trial_contrast']
        lickt[fi] = data['lickt']        
        lickx[fi] = data['lickx']
        tt_dig = np.digitize(tt, data['post'])-1    
        postrial[fi] = data['trial'][tt_dig]
        data_trials[fi] = np.unique(data['trial'])

    ################### Filter firing rate ####################    
    num_neurons = len(good_cells)
    indsnull = np.ones(num_neurons,dtype =bool)
    for fi in spk1:
        spksum = np.sum(spk1[fi], 0)/len(spk1[fi])*100
        indsnull[(spksum<0.05) | (spksum>10)] = False
    ################### concatenate spikes ####################
    sspikes1 = np.zeros((1,sum(indsnull)))
    for fi in files:
        sspikes1 = np.concatenate((sspikes1, sspk1[fi][:,indsnull]),0)            
    sspikes1 = sspikes1[1:,:]
    speed1 = []
    for fi in files:
        speed1.extend(speed[fi])
    speed1 = np.array(speed1)
    return sspikes1, speed1, sspk1, spk1, good_cells, indsnull, speed, data_trials, data_pos, posx, post, posxx, postt, postrial, gain, contrast, lickt, lickx


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

@numba.njit(parallel=True, fastmath=True)
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                #val = ((knn_dists[i, j] - rhos[i]) / (sigmas[i]))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals
@numba.njit(
    fastmath=True
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=0.0, bandwidth=1.0):
    target = np.log2(k) * bandwidth
#    target = np.log(k) * bandwidth
#    target = k
    
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = np.inf
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > 1e-5:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
#                    psum += d / mid
 
                else:
                    psum += 1.0
#                    psum += 0

            if np.fabs(psum - target) < 1e-5:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < 1e-3 * mean_ith_distances:
                result[i] = 1e-3 * mean_ith_distances
        else:
            if result[i] < 1e-3 * mean_distances:
                result[i] = 1e-3 * mean_distances

    return result, rho
def sample_denoising(data,  k = 10, num_sample = 500, omega = 1, metric = 'euclidean'):    
    n = data.shape[0]
    leftinds = np.arange(n)
    F_D = np.zeros(n)

    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)

    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    X = result.toarray()
    F = np.sum(X,1)
    print(np.mean(F),np.median(F))
    Fs = np.zeros(num_sample)
    Fs[0] = np.max(F)
    i = np.argmax(F)
    inds_all = np.arange(n)
    inds_left = inds_all>-1
    inds_left[i] = False
    inds = []
    j = 0
    for j in np.arange(1,num_sample):
        F -= omega*X[i,:]
        Fmax = np.argmax(F[inds_left])
        Fs[j] = F[inds_left][Fmax]
        i = inds_all[inds_left][Fmax]
        
        inds_left[i] = False   
        inds.extend([i])
    return inds, Fs

def get_dgms(sspikes2, maxdim = 1, omega = 1, k  = 1000, 
    n_points = 800, dim = 7, nbs = 800, eps = 1, metric = 'cosine', indstemp = []):    
    dgms_all = {}
    dim_red_spikes_move_scaled = preprocessing.scale(sspikes2, axis = 0)
    dim_red_spikes_move_scaled, e1, e2, var_exp = pca(dim_red_spikes_move_scaled, dim = dim)
    dim_red_spikes_move_scaled /= np.sqrt(e2[:dim])
    if len(indstemp)==0:
        startindex = np.argmax(np.sum(np.abs(dim_red_spikes_move_scaled),1))
        movetimes1 = radial_downsampling(dim_red_spikes_move_scaled, metric = 'euclidean', epsilon = eps, 
            startindex = startindex)
        indstemp,__  = sample_denoising(dim_red_spikes_move_scaled[movetimes1,:],  k, 
                                           n_points, omega, metric)
        indstemp = movetimes1[indstemp]
    else:
        movetimes1 = []
    dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
    X = squareform(pdist(dim_red_spikes_move_scaled[:,:], metric))
    knn_indices = np.argsort(X)[:, :nbs]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
    sigmas, rhos = smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    d = result.toarray()
    d = -np.log(d)
    np.fill_diagonal(d,0)
    thresh = np.max(d[~np.isinf(d)])
    if maxdim == 2:
        hom_dims = list(range(maxdim+1))
        VR = VietorisRipsPersistence(
        homology_dimensions=hom_dims,
        metric='precomputed',
        coeff=47,
        max_edge_length= thresh,
        collapse_edges=False,  # True faster?
        n_jobs=None  # -1 faster?
        )
        diagrams = VR.fit_transform([d])
        dgms_all[0] = from_giotto_to_ripser(diagrams[0])
        persistence = []
    else:
        persistence = ripser(d, maxdim=1, coeff=47, do_cocycles= True, distance_matrix = True, thresh = thresh)    
        dgms_all[0] = persistence['dgms'] 
    return dgms_all, persistence, indstemp, movetimes1, var_exp


def get_coords_ds(rips_real, len_indstemp, ph_classes = [0,1], dec_thresh = 0.99, coeff = 47):
    num_circ = len(ph_classes)    
    ################### Decode coordinates ####################
    diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
    cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
    dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
    births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
    deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
    deaths1[np.isinf(deaths1)] = 0
    lives1 = deaths1-births1 # the lifetime for the 1-dim classes
    iMax = np.argsort(lives1)
    coords1 = np.zeros((num_circ, len_indstemp))
    for j,c in enumerate(ph_classes):
        cocycle = cocycles[iMax[-(c+1)]]
        threshold = births1[iMax[-(c+1)]] + (deaths1[iMax[-(c+1)]] - births1[iMax[-(c+1)]])*dec_thresh
        coordstemp,inds = get_coords(cocycle, threshold, len_indstemp, dists_land, coeff)
        coords1[j,inds] = coordstemp
    return coords1


    
def radial_downsampling(data_in, epsilon = 0.1, metric = 'euclidean', startindex = -1):    
    n = data_in.shape[0]
    np.random.seed(0) 
    if epsilon > 0:
        n = data_in.shape[0]
        if startindex == -1:
            startindex = np.randint(n)
        i = startindex
        j = 1
        inds = np.zeros((n, ), dtype=int)
        inds1 = np.arange(n, dtype=int)
        dists = np.zeros((n, ))
        while j < n+1:
            disttemp = (cdist(data_in[i, :].reshape(1, -1), data_in[inds1, :], metric=metric) - epsilon)[0]                        
            dists[inds1] = np.max(np.concatenate((dists[inds1][:,np.newaxis], disttemp[:,np.newaxis]),1),1)
            inds[i] = j
            inds1 = inds1[disttemp>0]
            j = j+1
            if len(inds1)>0:
                i = inds1[np.argmin(dists[inds1])]
#                i = inds1[np.argmax(dists[inds1])]
#                i = inds1[np.argmax(disttemp[disttemp>0])]

            else:
                break
    else:
        inds = np.ones(range(np.shape(data_in)[0]))
    inds = np.where(inds)[0]
    return inds


def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    colormap1 = "default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    lifetime=False,
    rel_life= False,
    legend=True,
    show=False,
    ax=None,
    torus_colors = [],
    lw = 2.5,
    cs = ['#1f77b4','#ff7f0e', '#2ca02c', '#d62728']

):


    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = [
            "$H_0$",
            "$H_1$",
            "$H_2$",
            "$H_3$",
            "$H_4$",
            "$H_5$",
            "$H_6$",
            "$H_7$",
            "$H_8$",
        ]

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if len(plot_only)>0:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
    
    if lifetime:
        ylabel = "Lifetime"

        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]
    elif rel_life:
        ylabel = "Relative Lifetime"

        for dgm in diagrams:
            dgm[dgm[:,0]>0, 1] /= dgm[dgm[:,0]>0, 0]
    aspect = 'equal'
    # find min and max of all visible diagrams
    concat_dgms_b = np.concatenate(diagrams)[:,0]#
    finite_dgms_b = concat_dgms_b[np.isfinite(concat_dgms_b)]
    concat_dgms_d = np.concatenate(diagrams)[:,1]#
    finite_dgms_d = concat_dgms_d[np.isfinite(concat_dgms_d)]
    has_inf = np.any(np.isinf(concat_dgms_d))
    
    if not xy_range:
        ax_min, ax_max = np.min(finite_dgms_b), np.max(finite_dgms_b)
        x_r = ax_max - ax_min
        buffer = 1 if xy_range == 0 else x_r / 5
        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        ax_min, ax_max = np.min(finite_dgms_d), np.max(finite_dgms_d)
        y_r = ax_max - ax_min
        buffer = 1 if xy_range == 0 else x_r / 5
        y_down = ax_min - buffer / 2
        y_up = ax_max + buffer
        
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down
    if lifetime | rel_life:
        y_down = -yr * 0.05
        y_up = yr - y_down
        

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    i = 0
    for dgm, label in zip(diagrams, labels):
        c = cs[plot_only[i]]
        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none", c = c)
        i += 1
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    if len(torus_colors)>0:
        births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
        deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
        deaths1[np.isinf(deaths1)] = 0
        inds1 = np.argsort(deaths1)
        ax.scatter(diagrams[1][inds1[-1], 0], diagrams[1][inds1[-1], 1], 
                   10*size, linewidth =lw, edgecolor=torus_colors[0], facecolor = "none")
        ax.scatter(diagrams[1][inds1[-2], 0], diagrams[1][inds1[-2], 1], 
                   10*size, linewidth =lw, edgecolor=torus_colors[1], facecolor = "none")
        
        
        births2 = diagrams[2][:, ] #the time of birth for the 1-dim classes
        deaths2 = diagrams[2][:, 1] #the time of death for the 1-dim classes
        deaths2[np.isinf(deaths2)] = 0
        inds2 = np.argsort(deaths2)
        ax.scatter(diagrams[2][inds2[-1], 0], diagrams[2][inds2[-1], 1], 
                   10*size, linewidth =lw, edgecolor=torus_colors[2], facecolor = "none")
        
        
    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect(1/ax.get_data_ratio())

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="upper right")

    if show is True:
        plt.show()
    return

def pearson_correlate2d(in1, in2, mode='same', fft=True, nan_to_zero=True):
    """
    Pearson cross-correlation of two 2-dimensional arrays.

    Cross correlate `in1` and `in2` with output size determined by `mode`.
    NB: `in1` is kept still and `in2` is moved.

    Array in2 is shifted with respect to in1 and for each possible shift
    the Pearson correlation coefficient for the overlapping part of
    the two arrays is determined and written in the output rate.
    For in1 = in2 this results in the  Pearson auto-correlogram.
    Note that erratic values (for example seeking the correlation in
    array of only zeros) result in np.nan values which are by default set to 0.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
        If operating in 'valid' mode, either `in1` or `in2` must be
        at least as large as the other in every dimension.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    Returns
    -------
    pearson_corr : ndarray
        A 2-dimensional array containing a subset of the discrete pearson
        cross-correlation of `in1` with `in2`.
    """
    kwargs = dict(mode=mode, fft=fft, normalize=True,
                  set_small_values_zero=1e-10)
    corr = functools.partial(correlate2d, **kwargs)
    ones = np.ones_like(in1)
    pearson_corr = (
        (corr(in1, in2) - corr(ones, in2) * corr(in1, ones))
        / (
            np.sqrt(corr(in1 ** 2, ones) - corr(in1, ones) ** 2)
            * np.sqrt(corr(ones, in2 ** 2) - corr(ones, in2) ** 2)
        )
    )
    if nan_to_zero:
        pearson_corr[np.isnan(pearson_corr)] = 0.
    return pearson_corr


def correlate2d(in1, in2, mode, fft, normalize=True,
                set_small_values_zero=None):
    """
    Correlate two 2-dimensional arrays using FFT and possibly normalize

    NB: `in1` is kept still and `in2` is moved.

    Convenience function. See signal.convolve2d or signal.correlate2d
    for documenation.
    Parameters
    ----------
    normalize : Bool
        Decide wether or not to normalize each element by the
        number of overlapping elements for the associated displacement
    set_small_values_zero : float, optional
        Sometimes very small number occur. In particular FFT can lead to
        very small negative numbers.
        If specified, all entries with absolute value smalle than
        `set_small_values_zero` will be set to 0.
    Returns
    -------

    """
    if normalize:
        ones = np.ones_like(in1)
        n = signal.fftconvolve(ones, ones, mode=mode)
    else:
        n = 1
    if fft:
        # Turn the second array to make it a correlation
        ret = signal.fftconvolve(in1, in2[::-1, ::-1], mode=mode) / n
        if set_small_values_zero:
            condition = (np.abs(ret) < set_small_values_zero)
            if condition.any():
                ret[condition] = 0.
        return ret
    else:
        return signal.correlate2d(in1, in2, mode=mode) / n
    


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict







def get_corr_dist(masscenters_1,masscenters_2, mtot_1, mtot_2, sig = 2.75, num_shuffle = 1000):
    numangsint = len(mtot_1[0,:,0])+1
    num_neurons = len(masscenters_1[:,0])
    cells_all = np.arange(num_neurons)
    corr = np.zeros(num_neurons)
    corr1 = np.zeros(num_neurons)
    corr2 = np.zeros(num_neurons)
    mtot_1_shuf = np.zeros_like(mtot_1)
    mtot_2_shuf = np.zeros_like(mtot_2)
    for n in cells_all:
        m1 = mtot_1[n,:,:].copy()
        m1[np.isnan(m1)] = np.min(m1[~np.isnan(m1)])
        m2 = mtot_2[n,:,:].copy()
        m2[np.isnan(m2)] = np.min(m2[~np.isnan(m2)])
        m1 = smooth_tuning_map(m1, numangsint, sig, bClose = False)
        m2 = smooth_tuning_map(m2, numangsint, sig, bClose = False)
        corr[n] = pearsonr(m1.flatten(), m2.flatten())[0]
        mtot_1_shuf[n,:,:]= m1
        mtot_2_shuf[n,:,:]= m2

    dist =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2),
                                  np.cos(masscenters_1 - masscenters_2))),1))
    dist_shuffle = np.zeros((num_shuffle, num_neurons))
    corr_shuffle = np.zeros((num_shuffle, num_neurons))
    np.random.seed(47)
    for i in range(num_shuffle):
        inds = np.arange(num_neurons)
        np.random.shuffle(inds)
        for n in cells_all:
            corr_shuffle[i,n] = pearsonr(mtot_1_shuf[n,:,:].flatten(), mtot_2_shuf[inds[n],:,:].flatten())[0]
        dist_shuffle[i,:] =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2[inds,:]),
                np.cos(masscenters_1 - masscenters_2[inds,:]))),1))
    return corr, dist, corr_shuffle, dist_shuffle

def plot_cumulative_stat(stat, stat_shuffle, stat_range, stat_scale, xs, ys, xlim, ylim):
    num_neurons = len(stat)
    num_shuffle = len(stat_shuffle)
    fig = plt.figure()
    ax = plt.axes()
    numbins = 30
    meantemp = np.zeros(numbins)
    stat_all = np.array([])
    mean_stat_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        stat_all = np.concatenate((stat_all, stat_shuffle[i]))
        mean_stat_all[i] = np.mean(stat_shuffle[i])

    meantemp1 = np.histogram(stat_all, range = stat_range, bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(stat, range = stat_range, bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= stat_scale
    ax.plot(x, meantemp, c = 'g', alpha = 0.8, lw = 5)
    ax.plot(x,y, c = 'r', alpha = 0.8, lw = 5)

    ax.set_xticks(xs)
    ax.set_xticklabels(np.zeros(len(xs),dtype=str))
    ax.xaxis.set_tick_params(width=1, length =5)
    ax.set_yticks(ys)
    ax.set_yticklabels(np.zeros(len(ys),dtype=str))
    ax.yaxis.set_tick_params(width=1, length =5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()

    ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*1.4))

    plt.gca().axes.spines['top'].set_visible(False)
    plt.gca().axes.spines['right'].set_visible(False)

def get_coords(cocycle, threshold, num_sampled, dists, coeff):
    zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
    cocycle[zint, 2] = cocycle[zint, 2] - coeff
    d = np.zeros((num_sampled, num_sampled))
    d[np.tril_indices(num_sampled)] = np.NaN
    d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
    d[dists > threshold] = np.NaN
#    d[dists == 0] = np.NaN
    edges = np.where(~np.isnan(d))
    verts = np.array(np.unique(edges))
    num_edges = np.shape(edges)[1]
    num_verts = np.size(verts)

    values = d[edges]
    A = np.zeros((num_edges, num_verts), dtype=int)
    v1 = np.zeros((num_edges, 2), dtype=int)
    v2 = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        v1[i, :] = [i, np.where(verts == edges[0][i])[0]]
        v2[i, :] = [i, np.where(verts == edges[1][i])[0]]

    A[v1[:, 0], v1[:, 1]] = -1
    A[v2[:, 0], v2[:, 1]] = 1

    L = np.ones((num_edges,))
    Aw = A * np.sqrt(L[:, np.newaxis])
    Bw = values * np.sqrt(L)

    f = lsmr(Aw, Bw)[0]%1
    return f, verts




def get_coords1(cocycles, threshold, dists, coeff, smooth_circle = '', weights = []):
    num_sampled = len(dists)
    values = []
    for it, cocycle in enumerate(cocycles):
        d = np.zeros((num_sampled, num_sampled))
        zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
        cocycle[zint, 2] = cocycle[zint, 2] - coeff


        d[np.tril_indices(num_sampled)] = np.NaN
        d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
        d[dists > threshold] = np.NaN
#        d[dists == 0] = np.NaN

        values.append(d[np.where(~np.isnan(d))])
    values = np.array(values).T 
    edges = np.where(~np.isnan(d))
    verts = np.array(np.unique(edges))
    num_edges = np.shape(edges)[1]
    num_verts = np.size(verts)
    A = np.zeros((num_edges, num_verts), dtype=int)
    v1 = np.zeros((num_edges, 2), dtype=int)
    v2 = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        v1[i, :] = [i, np.where(verts == edges[0][i])[0]]
        v2[i, :] = [i, np.where(verts == edges[1][i])[0]]
    A[v1[:, 0], v1[:, 1]] = -1
    A[v2[:, 0], v2[:, 1]] = 1

    f = np.zeros((num_sampled, len(cocycles)))
    for i in range(len(cocycles)):
        if smooth_circle=='graph':
            L = get_weights_graph(A, values[:, i:i+1], edges, verts, dists, num_edges, num_verts)
        elif smooth_circle =='perea':
            L = get_weights_perea(edges, dists, threshold, num_edges)
        elif smooth_circle =='manual':
            L = np.power(weights[v1[:, 1],v2[:, 1]]/dists[v1[:, 1],v2[:, 1]],2)
        else:
            L = np.ones(num_edges)

        Aw = A * np.sqrt(L[:, np.newaxis])
        Bw = np.multiply(values[:,i], np.sqrt(L))

        f[verts,i] = lsmr(Aw, Bw)[0]%1
    return f

def get_weights_graph(A, vals, edges, verts, dists, num_edges, num_verts):


    row = np.zeros((num_edges,), dtype=int)
    col = np.zeros((num_edges,), dtype=int)
    G = np.zeros((num_verts, num_verts))
    edgewhere = np.zeros((num_verts, num_verts), dtype=int)
    nextv = np.zeros((num_verts, num_verts), dtype=int)
    nextv[:] = -1
    L = np.zeros((num_edges,))
    distpo = np.zeros((num_verts, num_verts))
    for i in range(len(vals[0,:])):
        values = vals[:,i]
        f0 = lsmr(-1 * A, values)[0]
        B = values + np.dot(A, f0)
        for e in range(num_edges):
            if B[e] >= 0:
                e0 = np.where(verts == edges[0][e])[0]
                e1 = np.where(verts == edges[1][e])[0]
            else:
                e0 = np.where(verts == edges[1][e])[0]
                e1 = np.where(verts == edges[0][e])[0]
            row[e] = e0
            col[e] = e1
            G[e0, e1] = dists[verts[e0], verts[e1]]
            distpo[e0, e1] = 1# / np.power(dists[verts[e0], verts[e1]], 2)
    #        distpo[e0, e1] = max([0, threshold - dists[verts[e0], verts[e1]]])#
            edgewhere[e0, e1] = e
            nextv[e0, e1] = e1

        G[G == 0] = np.inf
        it = 0
        for k in range(num_verts):
            for i in range(num_verts):
                for j in range(num_verts):
                    if G[i, j] > (G[i, k] + G[k, j]):
                        it = it + 1
                        G[i, j] = (G[i, k] + G[k, j])
                        nextv[i, j] = nextv[i, k]

        for e in range(num_edges):
            it = 0
            i = nextv[col[e], row[e]]
            j = col[e]
            while i != row[e] and i != -1 and it <= 10000:
                it = it + 1
                L[edgewhere[j, i]] += distpo[j, i]
                j = i
                i = nextv[i, row[e]]
            L[edgewhere[j, i]] += distpo[j, i]
            L[e] += distpo[row[e], col[e]]
    return L


def get_weights_perea(edges, dists, threshold, num_edges):
    L = np.zeros((num_edges,))
    for e in range(num_edges):
        L[e] = max([0, threshold - dists[edges[0][e], edges[1][e]]])
    return L


def predict_color(circ_coord_sampled, data, sampled_data, dist_measure='euclidean', num_batch =20000, k = 10):
    num_tot = len(data)
#    zero_spikes = np.where(np.sum(data,1) == 0)[0]
#    if len(zero_spikes):
#       data[zero_spikes,:] += 1e-10 
    circ_coord_tot = np.zeros(num_tot)
    circ_coord_dist = np.zeros(num_tot)
    circ_coord_tmp = circ_coord_sampled*2*np.pi
    j = -1
    for j in range(int(num_tot/num_batch)):
        dist_landmarks = cdist(data[j*num_batch:(j+1)*num_batch, :], sampled_data, metric = dist_measure)
        closest_landmark = np.argsort(dist_landmarks, 1)[:,:k]
        weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,-1]] for i in range(num_batch)])
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1
        weights /= np.sum(weights, 1)[:,np.newaxis] 

        sincirc = [np.dot(np.sin(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(num_batch)]
        coscirc = [np.dot(np.cos(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(num_batch)]
        circ_coord_tot[j*num_batch:(j+1)*num_batch] = np.arctan2(sincirc, coscirc)%(2*np.pi)

    dist_landmarks = cdist(data[(j+1)*num_batch:, :], sampled_data, metric = dist_measure)
    closest_landmark = np.argsort(dist_landmarks, 1)[:,:k]
    lenrest = len(closest_landmark[:,0])
    weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,k-1:k]] for i in range(lenrest)])
    if np.shape(weights)[0] == 0:
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1 
        weights /= np.sum(weights)
    else:
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1
        weights /= np.sum(weights, 1)[:,np.newaxis] 
    sincirc = [np.dot(np.sin(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(lenrest)]
    coscirc = [np.dot(np.cos(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(lenrest)]
    circ_coord_tot[(j+1)*num_batch:] = np.arctan2(sincirc, coscirc)%(2*np.pi)
    return circ_coord_tot



def get_sspikes(files, res = 100000, sigma = 5000, dt = 1000, 
                indsnull = [], good_cells = [], inds = []):
    ################### Get good cells ####################    
    if len(good_cells) == 0:
        for fi in files:
            data = loadmat(fi)
            anatomy = data['anatomy']
            if 'parent_shifted' in anatomy:
                group = anatomy['parent_shifted']
            else:
                group = anatomy['cluster_parent']
            regions = ('MEC',)#'VISp','RS')

            idx = [str(ss).startswith(regions) for ss in group]
            idxagl = [str(ss).startswith('RSPagl') for ss in group]
            region_idx = np.array(idx) & np.logical_not(np.array(idxagl))
            _,sn = os.path.split(fi)
            good_cells.extend(data['sp']['cids'][(data['sp']['cgs']==2) & region_idx])
        good_cells = np.where(np.bincount(good_cells)==len(files))[0]
    if len(indsnull) == 0:
        ## TODO
        indsnull = np.ones(len(goodcells), dtype = bool)
    if len(inds) == 0:
        ## TODO
        indsnull = np.ones(len(indsnull), dtype = bool)
    sspk1 = {}
    thresh = sigma*5
    num_thresh = int(thresh/dt)
    num2_thresh = int(2*num_thresh)
    sig2 = 1/(2*(sigma/res)**2)
    ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
    kerwhere = np.arange(-num_thresh,num_thresh)*dt

    for fi in files:
        data = loadmat(fi)
        spikes = {}
        for cell_idx in range(len(good_cells)):   
            if cell_idx in np.where(indsnull)[0][inds]:
                spikes[cell_idx] = data['sp']['st'][data['sp']['clu']==good_cells[cell_idx]]        

        sspikes = np.zeros((1,len(spikes)))
        min_time = 0
        max_time = data['sp']['st'].max()*res+dt
        tt = np.arange(np.floor(min_time), np.ceil(max_time), dt)

        spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes)))
        for n, spk in enumerate(spikes):
            spk = spikes[spk]
            spikes1 = np.array(spk*res-min_time, dtype = int)
            spikes1 = spikes1[(spikes1 < (max_time-min_time)) & (spikes1 > 0)]
            spikes_mod = dt-spikes1%dt
            spikes1 = np.array(spikes1/dt, int)
            for m, j in enumerate(spikes1):
                spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
        spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
        sspikes = np.concatenate((sspikes, spikes_temp),0)
        sspikes = sspikes[1:,:]
        sspikes *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
        sspk1[fi] = sspikes.copy()


        sspikes1 = np.zeros((1,len(sspk1[fi][0,:])))
    for fi in files:
        sspikes1 = np.concatenate((sspikes1, sspk1[fi]),0)            
    sspikes1 = sspikes1[1:,:]
    return sspikes1

def get_speed1(files):
    res = 100000
    dt = 1000

    speed = {}

    for fi in files:
        data = loadmat(fi)
        min_time = 0
        max_time = data['sp']['st'].max()*res+dt
        tt = np.arange(np.floor(min_time), np.ceil(max_time), dt)

        tt /= res

        xtmp = data['posx'].copy()
        xtmp += (data['trial']-1)*400
        speed_spline = CubicSpline(data['post'], xtmp)
        dxx = speed_spline(tt,1)
        idx_v = np.flatnonzero(np.logical_not(np.isnan(dxx)))
        idx_n = np.flatnonzero(np.isnan(dxx))
        dxx[idx_n]=np.interp(idx_n,idx_v,dxx[~np.isnan(dxx)])
        dxx = spi.gaussian_filter1d(dxx, 50)
        speed[fi] = dxx.copy()
    return speed

def get_speed(files):
    res = 100000
    dt = 1000

    speed = {}

    for fi in files:
        data = loadmat(fi)
        min_time = 0
        max_time = data['sp']['st'].max()*res+dt
        tt = np.arange(np.floor(min_time), np.ceil(max_time), dt)/res
        xtmp = data['posx'].copy()
        xtmp += (data['trial']-1)*400
        xtmp = gaussian_filter1d(xtmp, sigma = 5)        
        xx_spline = CubicSpline(data['post'], xtmp)
        speed0 = xx_spline(tt,1)
        print(len(speed0))
        speed[fi] = speed0.copy()
    speed1 = []
    for fi in files:
        speed1.extend(speed[fi])
    speed1 = np.array(speed1)
    return speed, speed1



def compca(sspikes2, speed1,lencube = 100000, maxdim = 1,     
             omega = 1, k  = 1000, n_points = 800, sp = 10,
             dim = 7, nbs = 800, eps = 1, metric = 'cosine',
             indstemp = [], speed_times = []):
    
    indstemp = []
    speed_times = []
    dgms_all = {}
    num_times = int(np.round(len(sspikes2)/lencube))
    ii = 0
    speed_times = np.arange(ii,len(sspikes2),num_times)
    speed_times = speed_times[np.where(speed1[speed_times]>sp)[0]]
    dim_red_spikes_move_scaled = preprocessing.scale(sspikes2[speed_times,:], axis = 0)
    dim_red_spikes_move_scaled, e1, e2,__ = pca(dim_red_spikes_move_scaled, dim = dim)
    plt.plot(e2)
    print(e1)
    print(np.diff(e1))
    plt.show()
    return e1



def smooth_tuning_map(mtot, numangsint, sig, bClose = True):
    numangsint_1 = numangsint-1
    mid = int((numangsint_1)/2)
    indstemp1 = np.zeros((numangsint_1,numangsint_1), dtype=int)
    indstemp1[indstemp1==0] = np.arange((numangsint_1)**2)
    indstemp1temp = indstemp1.copy()
    mid = int((numangsint_1)/2)
    mtemp1_3 = mtot.copy()
    for i in range(numangsint_1):
        mtemp1_3[i,:] = np.roll(mtemp1_3[i,:],int(i/2))
    mtot_out = np.zeros_like(mtot)
    mtemp1_4 = np.concatenate((mtemp1_3, mtemp1_3, mtemp1_3),1)
    mtemp1_5 = np.zeros_like(mtemp1_4)
    mtemp1_5[:, :mid] = mtemp1_4[:, (numangsint_1)*3-mid:]  
    mtemp1_5[:, mid:] = mtemp1_4[:,:(numangsint_1)*3-mid]      
    if bClose:
        mtemp1_6 = np.concatenate((mtemp1_5,mtemp1_4,mtemp1_5)) 
        nans = np.isnan(mtemp1_6)
        mtemp1_6[nans] = np.mean(mtemp1_6[~nans])
        mtemp1_6 = gaussian_filter(mtemp1_6,sigma = sig)
        mtemp1_6[nans] = np.nan
        radius = 1
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        kernel = np.array((X ** 2 + Y ** 2) <= radius ** 2).astype(np.uint8)
        
        mtemp1_6 = cv.morphologyEx(mtemp1_6, cv.MORPH_CLOSE, kernel, iterations = 1)
    else:
        mtemp1_6 = gaussian_filter(np.concatenate((mtemp1_5,mtemp1_4,mtemp1_5)) ,sigma = sig)
    for i in range(numangsint_1):
        mtot_out[i, :] = mtemp1_6[(numangsint_1)+i, 
                                          (numangsint_1) + (int(i/2) +1):(numangsint_1)*2 + (int(i/2) +1)] 
    return mtot_out


def plot_trajectories(posxx, coords1, postrial, gain, theta, folder, files, num_rhomb = 4,   sig_smooth = 50 ): 

    bSaveFigs = True
    t0 = 0
    coords = coords1.copy()
    if not theta:
        coords[:,0] = 2*np.pi-coords[:,0]

    dcoords = np.diff(coords, axis = 0)
    if np.sum(dcoords<0)>np.sum(dcoords>0):
        coords = 2*np.pi-coords

    for itfi, fi in enumerate(files):
        times = np.arange(t0,t0+len(posxx[fi]))
        t0+len(posxx[fi])
        coords_smooth_sess = np.arctan2(gaussian_filter1d(np.sin(coords[times,:2]),sigma = 50,axis = 0),
                        gaussian_filter1d(np.cos(coords[times,:2]),sigma = 50,axis = 0))%(2*np.pi)
        postrial_curr = postrial[fi]
        trial_range = np.unique(postrial_curr)[1:-1]
        if np.sum((gain[fi]*100).astype(int) - gain[fi]*100) != 0:
            gain_curr = np.ones(len(trial_range))
            print('Weird gain values ', fi)
        else:
            gain_curr = gain[fi][trial_range-1]

        lbls = np.digitize(gain_curr, np.unique(gain_curr))
        plt.viridis()
        plot_cluster_trials(coords_smooth_sess,
                            trial_range,
                            postrial_curr,
                            lbls,
                            posxx[fi][times],
                           folder = folder,
                           fname = fi.replace('giocomo_data\\', '').replace('.mat', '') + '_trajs', 
                           num_rhomb  = num_rhomb, 
                           bSaveFigs = True)
        
def plot_2dcoords(mouse_sess, cmod, speed, coords1, posxx, postt, gain, contrast, postrial, files, folder_curr, ):
    sp = -np.inf
    cs =   {}
    t0 = 0
    for fi in files:
        times = np.arange(t0,t0+sum(speed[fi]>sp))
        cs[fi] = coords1[times,:]
        t0+=sum(speed[fi]>sp)

    im = plt.imread('image044.png')
    im = np.array(im)
    a1 = np.rot90(im, 0)
    sp = -np.inf
    sig = 50
    for fi in files:    
        coordscurr = cs[fi].copy()#[speed[fi] > sp]
        len_plot = postt[fi][-1]
        times = np.arange(0,np.where(postt[fi][speed[fi] > sp]>=len_plot)[0][0])
        dx = np.abs(posxx[fi][1:]-posxx[fi][:-1])[speed[fi][:-1]>sp][times]
        times = times[dx<10]
        cc = np.arctan2(gaussian_filter1d(np.sin(coordscurr[times,:]),sigma = sig,axis = 0),
                       gaussian_filter1d(np.cos(coordscurr[times,:]),sigma = sig,axis = 0))%(2*np.pi)
        bCos = True
        if bCos:
            eps = 0.0001 
            digitized = np.concatenate((np.digitize(np.cos(cc[:, 0]), np.linspace(-1-eps,1+eps, len(a1)+1))[:,np.newaxis], 
                                np.digitize(np.cos(cc[:, 1]), np.linspace(-1-eps,1+eps, len(a1)+1))[:,np.newaxis]),1)
        else:
            digitized = np.concatenate((np.digitize(cc[:, 0], np.linspace(0,2*np.pi, len(a1)+1))[:,np.newaxis], 
                                       np.digitize(cc[:, 1], np.linspace(0,2*np.pi, len(a1)+1))[:,np.newaxis]),1)
        cc1 = []
        for i in range(len(times)):
            cc1.append(a1[digitized[i,1]-1, digitized[i,0]-1]) 
        cs11 = CubicSpline(times, np.sin(cc))
        cc11 = CubicSpline(times, np.cos(cc))
        angular_rate1 = np.sum(np.sqrt(np.square(cs11(times,1)) + np.square(cc11(times,1))),1)

        fig = plt.figure(figsize = (10,5), dpi = 200)
        plt.axis('off')
        plt.hsv()
        ax1 = fig.add_subplot(111)
        ax1.axis('off')
        plt.xlim([0,len_plot])
        im = ax1.scatter(postt[fi][speed[fi] > sp][times], 
                        posxx[fi][speed[fi] > sp][times], s = 10, c = cc1)
        plt.savefig(folder_curr + '/' + fi.replace('giocomo_data\\', '').replace('.mat', '') +'_ind' + str(cmod) + '_2dcoords', transparent = True)
        plt.close()

        fig = plt.figure(figsize = (10,1), dpi = 100)
        ax1 = fig.add_subplot(111)
        ax1.axis('off')
        ax1.plot([0,len_plot], [0,0], c = 'k')
        tspace = np.linspace(0, len_plot, 5)
        dt = tspace[1]-tspace[0]
        for i in tspace:
            ax1.scatter([i], [0], marker = 'X', s = 10, c = 'k')
            ax1.text(i-dt*0.15, -0.1, str(int(i)))        
        ax1.set_xlim([0,len_plot])    
        ax1.set_ylim([-0.2,0.2])
        plt.savefig(folder_curr + '/' + fi.replace('giocomo_data\\', '').replace('.mat', '') +'_ind' + str(cmod) + '_2dcoords', transparent = True)
        

        fig = plt.figure(figsize = (10,1), dpi = 100)
        ax1 = fig.add_subplot(111)
        plt.xlim([0,len_plot])
        plt.plot(postt[fi][speed[fi] > sp][times], 
                preprocessing.minmax_scale(gaussian_filter1d(angular_rate1, sigma = 50)), alpha = 1)
        plt.plot(postt[fi][speed[fi] > sp][times], 
                preprocessing.minmax_scale(gaussian_filter1d(speed[fi][speed[fi] > sp][times], sigma = 50)), 
                ls = '--', alpha = 1, lw = 0.9)
        plt.savefig(folder_curr + '/' + fi.replace('giocomo_data\\', '').replace('.mat', '') +'_ind' + str(cmod) + '_time', transparent = True)
        plt.close()
        if (fi.find('gain')>-1) or (fi.find('contrast')>-1):
            fig = plt.figure(figsize = (10,1), dpi = 100)
            ax1 = fig.add_subplot(111)
            ax1.set_xlim([0,len_plot])    
            ax1.set_ylim([-0.1, 1.1])    
            ax1.plot(postt[fi][speed[fi] > sp][times], contrast[fi][postrial[fi]-1][speed[fi] > sp][times]/100, c = 'r')
            ax1.plot(postt[fi][speed[fi] > sp][times], gain[fi][postrial[fi]-1][speed[fi] > sp][times], c = 'g')
            ax1.set_xticklabels('')
            ax1.set_yticklabels('')
            plt.savefig(folder_curr + '/' + fi.replace('giocomo_data\\', '').replace('.mat', '') +'_ind' + str(cmod) + '_gaincontrast', transparent = True)
            plt.close()
    plt.viridis()
        
def plot_barcode(mouse_sess, cmod, folder_curr, persistence, file_name = '', shuffle_name = ''):
    if np.sum(np.isinf(persistence[0])) == 0:
        persistence[0] = np.concatenate((persistence[0], np.array([0,np.inf])[np.newaxis, :]),0)

    diagrams_roll = {}
    if len(shuffle_name)>0:
        f = np.load(shuffle_name + '.npz', allow_pickle = True)
        diagrams_roll = f['dgms_shuffles'][()]
        f.close() 

    cs = np.repeat([[0,0.55,0.2]],3).reshape(3,3).T
    alpha=1
    inf_delta=0.1
    legend=True
    colormap=cs
    maxdim = len(persistence)-1
    dims =np.arange(maxdim+1)
    num_rolls = len(diagrams_roll)
    print(num_rolls)
    if num_rolls>0:
        diagrams_all = np.copy(diagrams_roll[0][0])
        for i in np.arange(1, num_rolls):
            for d in dims:
                diagrams_all[d] = np.concatenate((diagrams_all[d], diagrams_roll[i][0][d]),0)
        infs = np.isinf(diagrams_all[0])
        diagrams_all[0][infs] = 0
        diagrams_all[0][infs] = np.max(diagrams_all[0])
        infs = np.isinf(diagrams_all[0])
        diagrams_all[0][infs] = 0
        diagrams_all[0][infs] = np.max(diagrams_all[0])


    min_birth, max_death = 0,0            
    for dim in dims:
        persistence_dim = persistence[dim][~np.isinf(persistence[dim][:,1]),:]
        min_birth = min(min_birth, np.min(persistence_dim))
        max_death = max(max_death, np.max(persistence_dim))
    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta
    axis_start = min_birth - delta            
    plotind = (dims[-1]+1)*100 + 10 +1
    fig = plt.figure(figsize = (5,10), dpi = 160)
    gs = gridspec.GridSpec(len(dims),1)

    indsall =  0
    labels = ["$H_0$", "$H_1$", "$H_2$"]
    for dit, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes.axis('off')
        d = np.copy(persistence[dim])
        d[np.isinf(d[:,1]),1] = infinity
        dlife = (d[:,1] - d[:,0])
        dinds = np.argsort(dlife)[-30:]
        dl1,dl2 = dlife[dinds[-2:]]
        if dim>0:
            dinds = dinds[np.flip(np.argsort(d[dinds,0]))]
        axes.barh(
            0.5+np.arange(len(dinds)),
            dlife[dinds],
            height=0.8,
            left=d[dinds,0],
            alpha=alpha,
            color=colormap[dim],
            linewidth=0,
        )
        indsall = len(dinds)
        if num_rolls>0:
            bins = 50
            cs = np.flip([[0.4,0.4,0.4], [0.6,0.6,0.6], [0.8, 0.8,0.8]])
            cs = np.repeat([[1,0.55,0.1]],3).reshape(3,3).T
            cc = 0
            lives1_all = diagrams_all[dim][:,1] - diagrams_all[dim][:,0]
            x1 = np.linspace(diagrams_all[dim][:,0].min()-1e-5, diagrams_all[dim][:,0].max()+1e-5, bins-2)
            
            dx1 = (x1[1] - x1[0])
            x1 = np.concatenate(([x1[0]-dx1], x1, [x1[-1]+dx1]))
            dx = x1[:-1] + dx1/2
            ytemp = np.zeros((bins-1))
            binned_birth = np.digitize(diagrams_all[dim][:,0], x1)-1
            x1  = d[dinds,0]
            ytemp =x1 + np.max(lives1_all)
            axes.fill_betweenx(0.5+np.arange(len(dinds)), x1, ytemp, color = cs[(dim)], zorder = -2, alpha = 0.3)

        axes.plot([0,0], [0, indsall], c = 'k', linestyle = '-', lw = 1)
        axes.plot([0,indsall],[0,0], c = 'k', linestyle = '-', lw = 1)
        axes.set_xlim([0, infinity])
    
    plt.tight_layout()
    if len(file_name)>0:
        plt.savefig(folder_curr + '/' + mouse_sess + '_' + str(cmod) + file_name + '_barcode' + str(int(np.round(infinity,0))),
                    transparent = True)
        plt.close()
    else:
        plt.show()

def plot_phase_stats(mouse_sess, cmod, coords, speed, data_ensemble, e1, folder_curr, numangsint = 16, files = []):
    numangsint_1 = numangsint-1
    bins = np.linspace(0,2*np.pi, numangsint)
    xv, yv = np.meshgrid(bins[0:-1] + (bins[1:] -bins[:-1])/2, 
                    bins[0:-1] + (bins[1:] -bins[:-1])/2)
    pos  = np.concatenate((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]),1)
    ccos = np.cos(pos)
    csin = np.sin(pos)
    
    num_neurons = len(e1)
    mtots =  {}
    mcs = {}
    t0 = 0
    for fi in files:
        if len(coords) == len(files):
            cc1 = coords[fi].copy()
            currspk = data_ensemble[fi][:,e1].copy()
        else:
            currspk = data_ensemble[times,:].copy()
            times = np.arange(t0,t0+len(speed[fi]))
            t0+=len(speed[fi])
            cc1 = coords[times,:]
        mcs[fi] = np.zeros((num_neurons,2))
        mtots[fi] = np.zeros((num_neurons, numangsint_1, numangsint_1))

        for n in range(num_neurons):
            spk = currspk[:,n]
            mtot_tmp, x_edge, y_edge,c2 = binned_statistic_2d(cc1[:,0],cc1[:,1], 
                                                 spk, statistic='mean', 
                                                 bins=bins, range=None, expand_binnumbers=True)
            mtots[fi][n, :, :] = mtot_tmp.copy()
            nans = np.isnan(mtot_tmp)
            mtot_tmp[nans] = 0
            mtot_tmp = smooth_tuning_map(mtot_tmp, numangsint, sig = 5, bClose = False)
            mtot_tmp[nans] = np.nan
            mtot_tmp1 = mtot_tmp.flatten()
            nans  = ~np.isnan(mtot_tmp1) 
            centcos = np.sum(np.multiply(ccos[nans,:],mtot_tmp1[nans,np.newaxis]),0)
            centsin = np.sum(np.multiply(csin[nans,:],mtot_tmp1[nans,np.newaxis]),0)
            mcs[fi][n,:] = np.arctan2(centsin,centcos)%(2*np.pi)
            
    if len(files) == 1:
        fi = files[0]
        plot_phase_distribution(mcs[fi], [], )
        fii = fi.replace('giocomo_data\\', '').replace('.mat', '')
        plt.tight_layout()
        plt.savefig(folder_curr + '/' + fii + '_ind' + str(cmod) + '_phase_dist', transparent = True)
        plt.close()
    else:
        for i, fi in enumerate(files):
              for j, fi1 in enumerate(files[i+1:]):
                    corr, dist, corr_shuffle, dist_shuffle = get_corr_dist(mcs[fi], mcs[fi1], 
                                                                  mtots[fi], mtots[fi1])
                    plot_phase_distribution(mcs[fi], mcs[fi1], )
                    fii = fi.replace('giocomo_data\\', '').replace('.mat', '')
                    fii1 = fi1.replace('giocomo_data\\', '').replace('.mat', '')
                    fii1 = fii1[9:]
                    #                     plt.title(fii + fii1 +'_ind' + str(cmod) + '_phase_dist')
                    plt.tight_layout()
                    plt.savefig(folder_curr + '/' + fii + fii1 +'_ind' + str(cmod) + '_phase_dist', transparent = True)
                    plt.close()
                    plot_cumulative_stat(corr, corr_shuffle, (-1,1), 1, 
                                [-1, -0.5, 0.0, 0.5, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], [-1,1], [-0.05,1.05])
                    #plt.title(fii + fii1 +'_ind' + str(cmod) + '_corrsum')
                    plt.tight_layout()
                    plt.savefig(folder_curr + '/' + fii + fii1 +'_ind' + str(cmod) + '_corrsum', transparent = True)
                    plt.close()
                    plot_cumulative_stat(dist, dist_shuffle, (0,np.sqrt(2)*np.pi), 180/np.pi, 
                                [0, 62.5, 125, 187.5, 250], [0.0, 0.25, 0.5, 0.75, 1.0], [0,255], [-0.05,1.05])
                    #plt.title(fii + fii1 +'_ind' + str(cmod) + '_distsum')
                    plt.tight_layout()
                    plt.savefig(folder_curr + '/' + fii + fii1 +'_ind' + str(cmod) + '_distsum', transparent = True)
                    plt.close()

                    

def plot_phase_distribution(masscenters_1, masscenters_2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')
    num_neurons = len(masscenters_1[:,0])
    for i in np.arange(num_neurons):
        ax.scatter(masscenters_1[i,0], masscenters_1[i,1], s = 10, c = 'r')
        if len(masscenters_2)>0:
            ax.scatter(masscenters_2[i,0], masscenters_2[i,1], s = 10, c = 'g')
            line = masscenters_1[i,:] - masscenters_2[i,:]
            dline = line[1]/line[0]
            if line[0]< - np.pi and line[1] < -np.pi:
                line = (-2*np.pi + masscenters_2[i,:]) - masscenters_1[i,:]
                dline = line[1]/line[0]
                if (masscenters_1[i,1] + (- masscenters_1[i,0])*dline)>0:
                    ax.plot([masscenters_1[i,0], 0],
                            [masscenters_1[i,1], masscenters_1[i,1] + (- masscenters_1[i,0])*dline],
                            c = 'b', alpha = 0.5)

                    ax.plot([2*np.pi,2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline], 
                            [masscenters_1[i,1] + (- masscenters_1[i,0])*dline, 0],
                            c = 'b', alpha = 0.5)

                    ax.plot([2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline, 
                             masscenters_2[i,0]], 
                            [2*np.pi,masscenters_2[i,1]],
                            c = 'b', alpha = 0.5)
                else:
                    ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (- masscenters_1[i,1])/dline],
                            [masscenters_1[i,1], 0],
                            c = 'b', alpha = 0.5)

                    ax.plot([masscenters_1[i,0] + (- masscenters_1[i,1])/dline, 0],
                            [2*np.pi, 2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline], 
                            c = 'b', alpha = 0.5)

                    ax.plot([2*np.pi, 
                             masscenters_2[i,0]], 
                            [2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline,
                            masscenters_2[i,1]],
                            c = 'b', alpha = 0.5)
            elif line[0]> np.pi and line[1] >np.pi:
                line = (2*np.pi + masscenters_2[i,:]) - masscenters_1[i,:]
                dline = line[1]/line[0]
                if (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline)<2*np.pi:
                    ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline],

                           [masscenters_1[i,1],2*np.pi],
                           c = 'b', alpha = 0.5)
                    ax.plot([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline, 2*np.pi],
                           [0,(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline], 
                           c = 'b', alpha = 0.5)
                    ax.plot([0,masscenters_2[i,0]],
                           [(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline, 
                            masscenters_2[i,1]], 
                           c = 'b', alpha = 0.5)          
                else:
                    ax.plot([masscenters_1[i,0],2*np.pi],
                            [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                            c = 'b', alpha = 0.5)

                    ax.plot([0,(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline], 
                            [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, 2*np.pi],
                            c = 'b', alpha = 0.5)

                    ax.plot([(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline, 
                             masscenters_2[i,0]], 
                            [0,masscenters_2[i,1]],
                            c = 'b', alpha = 0.5)#
            elif line[0]>np.pi and line[1] <-np.pi:  
                line = [2*np.pi + masscenters_2[i,0], -2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
                dline = line[1]/line[0]            
                if (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline)>0:
                    ax.plot([masscenters_1[i,0],2*np.pi],
                            [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                            c = 'b', alpha = 0.5)

                    ax.plot([0,(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline], 
                            [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, 0],
                            c = 'b', alpha = 0.5)

                    ax.plot([(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline, 
                             masscenters_2[i,0]], 
                            [2*np.pi,masscenters_2[i,1]],
                            c = 'b', alpha = 0.5)

                else:
                    line = [2*np.pi + masscenters_2[i,0], -2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
                    dline = line[1]/line[0]
                    ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (- masscenters_1[i,1])/dline],
                            [masscenters_1[i,1], 0],
                            c = 'b', alpha = 0.5)

                    ax.plot([masscenters_1[i,0] + (- masscenters_1[i,1])/dline, 2*np.pi], 
                            [2*np.pi, 2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline],
                            c = 'b', alpha = 0.5)

                    ax.plot([0, masscenters_2[i,0]], 
                            [2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline,masscenters_2[i,1]],
                            c = 'b', alpha = 0.5)
            elif line[0]<-np.pi and line[1] >np.pi:
                line = [-2*np.pi + masscenters_2[i,0], 2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
                dline = line[1]/line[0]
                if ((masscenters_1[i,1] + -(masscenters_1[i,0])*dline)<2*np.pi):

                    ax.plot([masscenters_1[i,0],0],
                            [masscenters_1[i,1], masscenters_1[i,1] + -(masscenters_1[i,0])*dline],
                            c = 'b', alpha = 0.5)

                    ax.plot([2*np.pi, 2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline], 
                            [masscenters_1[i,1] + -(masscenters_1[i,0])*dline, 2*np.pi],
                            c = 'b', alpha = 0.5)

                    ax.plot([2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline, 
                             masscenters_2[i,0]], 
                            [0,masscenters_2[i,1]],
                            c = 'b', alpha = 0.5)
                else:
                    ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline],
                            [masscenters_1[i,1], 2*np.pi],
                            c = 'b', alpha = 0.5)

                    ax.plot([masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline, 0], 
                            [0, 0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline],
                            c = 'b', alpha = 0.5)

                    ax.plot([2*np.pi, masscenters_2[i,0]], 
                            [0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline,masscenters_2[i,1]],
                            c = 'b', alpha = 0.5)

            elif line[0]< -np.pi:
                line = [(2*np.pi + masscenters_1[i,0]), masscenters_1[i,1]] - masscenters_2[i,:]
                dline = line[1]/line[0]
                ax.plot([masscenters_2[i,0],2*np.pi],
                        [masscenters_2[i,1], masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline], 
                        alpha = 0.5, c = 'b')            
                ax.plot([0,masscenters_1[i,0]],
                        [masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline, masscenters_1[i,1]], 
                        alpha = 0.5, c = 'b')
            elif line[0]> np.pi:
                line = [ masscenters_2[i,0]+ 2*np.pi, masscenters_2[i,1]] - masscenters_1[i,:]
                dline = line[1]/line[0]


                ax.plot([masscenters_1[i,0],2*np.pi],
                        [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                        c = 'b', alpha = 0.5)
                ax.plot([0,masscenters_2[i,0]],
                        [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, masscenters_2[i,1]], 
                        alpha = 0.5, c = 'b')
            elif line[1]< -np.pi:
                line = [ masscenters_1[i,0], (2*np.pi + masscenters_1[i,1])] - masscenters_2[i,:]
                dline = line[1]/line[0]

                ax.plot([masscenters_2[i,0], masscenters_2[i,0] + (2*np.pi - masscenters_2[i,1])/dline], 
                        [masscenters_2[i,1],2*np.pi], alpha = 0.5, c = 'b'),
                ax.plot([masscenters_1[i,0] - masscenters_1[i,1]/dline,masscenters_1[i,0]],
                        [0, masscenters_1[i,1]], 
                        alpha = 0.5, c = 'b')
            elif line[1]> np.pi:
                line = [ masscenters_2[i,0], masscenters_2[i,1]+ 2*np.pi] - masscenters_1[i,:]
                dline = line[1]/line[0]

                ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline], 
                        [masscenters_1[i,1], 2*np.pi], alpha = 0.5, c = 'b'),

                ax.plot([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline,masscenters_2[i,0]],
                        [0, masscenters_2[i,1]], 
                        alpha = 0.5, c = 'b')
            else:
                ax.plot([masscenters_1[i,0],masscenters_2[i,0]],
                        [masscenters_1[i,1],masscenters_2[i,1]], 
                        alpha = 0.5, c = 'b')

    ax.plot([0,0], [0,2*np.pi], c = 'k')
    ax.plot([0,2*np.pi], [0,0], c = 'k')
    ax.plot([2*np.pi,2*np.pi], [0,2*np.pi], c = 'k')
    ax.plot([0,2*np.pi], [2*np.pi,2*np.pi], c = 'k')

    r_box = transforms.Affine2D().skew_deg(15,15)
    for x in ax.images + ax.lines + ax.collections + ax.get_xticklabels() + ax.get_yticklabels():
        trans = x.get_transform()
        x.set_transform(r_box+trans) 
        if isinstance(x, PathCollection):
            transoff = x.get_offset_transform()
            x._transOffset = r_box+transoff 
    ax.set_xlim([0,2*np.pi + 3/5*np.pi])
    ax.set_ylim([0,2*np.pi + 3/5*np.pi])
    ax.set_aspect('equal', 'box')
    return ax


        
def plot_toroidal_ratemaps(mouse_sess, data_ensemble, files, e1, coords1, speed, 
                           pos_trial, posxx, theta, folder_curr, sp = 10):    
    num_neurons = len(e1)
    r_box = transforms.Affine2D().skew_deg(15,15)
    numangsint = 51
    sig = 2.75
    numbins = 50
    bins = np.linspace(0,2*np.pi, numbins+1)
    plt.viridis()
    numfigs = len(files)
    numw = 4
    numh = int(np.ceil(num_neurons/numw))
    outer1 = gridspec.GridSpec(1, numw)
    fig = plt.figure(figsize=(np.ceil((numw*numfigs+numw-1)*1.05), np.ceil(numh*1.1)))
    len_acorr = 500 #len(acorr_sess[fi][0,:])
    nw = 0
    mtots =  {}
    files1 = glob.glob('giocomo_data/' + mouse_sess + '*.mat')
    for fi in files1:
        if fi.find('dark')>-1:
            acorr_real = get_acorrs(data_ensemble[fi][:,e1][speed[fi]>sp,:], pos_trial[fi][speed[fi]>sp], posxx[fi][speed[fi]>sp])

    
    for fi in files:
        mtots[fi] = np.zeros((num_neurons, numbins, numbins))
    for nn, n in enumerate(range(num_neurons)):
        nnn = nn%numh
        if nnn == 0:
            outer2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer1[nw], wspace = .1)
            gs2 = gridspec.GridSpecFromSubplotSpec(numh, len(files)+1, subplot_spec = outer2[0], wspace = .1)
            nw += 1
        posnum = 0 
        xnum = 0
        t0 = 0
        for fi in files[:]:
            if len(coords1)== len(files):
                cc1 = coords1[fi]
                spk = data_ensemble[fi][:,e1[n]]
            else:
                times = np.arange(t0,t0+len(speed[fi]))
                t0+=len(speed[fi])
                spk = data_ensemble[:,n][times]
                cc1 = coords1[times,:]
            if xnum == 0:
                ax = plt.subplot(gs2[nnn,xnum]) 

                ax.bar(np.arange(300),acorr_real[n,:], width = 1, color = 'k')
                ax.set_xlim([0,300])
                ax.set_ylim([0, 0.4])

                ax.set_xticks([0, 100, 200, 300])
                ax.set_yticks([0.0, 0.2, 0.4])
                ax.set_yticklabels('')
                ax.set_xticklabels('')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_aspect(1/ax.get_data_ratio())

                xnum += 1
            ax = plt.subplot(gs2[nnn,xnum])
            mtot_tmp, x_edge, y_edge,c2 = binned_statistic_2d(cc1[:,0],cc1[:,1], 
                                                 spk, statistic='mean', 
                                                 bins=bins, range=None, expand_binnumbers=True)
            mtots[fi][nn, :,:] = mtot_tmp.copy()
            nans = np.isnan(mtot_tmp)
            mtot_tmp[np.isnan(mtot_tmp)] = np.mean(mtot_tmp[~np.isnan(mtot_tmp)])
            if theta:
                mtot_tmp = np.rot90(mtot_tmp,1)
          
            mtot_tmp = smooth_tuning_map(mtot_tmp, numangsint, sig, bClose = True) 
            mtot_tmp[nans] = -np.inf
            ax.imshow(mtot_tmp, origin = 'lower', extent = [0,2*np.pi,0, 2*np.pi], vmin = 0, vmax = np.max(mtot_tmp) *0.975)
            ax.set_xticks([])
            ax.set_yticks([])
            for x in ax.images + ax.lines + ax.collections:
                trans = x.get_transform()
                x.set_transform(r_box+trans) 
                if isinstance(x, PathCollection):
                    transoff = x.get_offset_transform()
                    x._transOffset = r_box+transoff     
            ax.set_xlim(0, 2*np.pi + 3*np.pi/5)
            ax.set_ylim(0, 2*np.pi + 3*np.pi/5)
            ax.set_aspect('equal', 'box') 
            ax.axis('off')
            xnum += 1            
    fig.savefig(folder_curr + '/ratemaps', transparent = True)
    plt.close()



def load_data(mouse_sess, cmod, data_dir, bPCA=True, bPred=False):
    f = np.load(data_dir + '/' + mouse_sess + '_mods.npz',allow_pickle = True)
    ind = f['ind']
    f.close()
    e1 = ind == cmod
    print('')
    print(mouse_sess, 'ind ' + str(cmod), sum(e1))
    
    files = glob.glob('giocomo_data/' + mouse_sess + '_*.mat')
    files.sort()
    ff = glob.glob(data_dir + '/' + mouse_sess + '_data.npz')

    if len(ff) == 0:
        (sspikes1, speed1, sspk1, spk1, good_cells, indsnull, speed, 
           pos_trial,data_pos, posx, post, posxx, postt, 
           postrial, gain, contrast, lickt, lickx) =  get_data(files)
        np.savez(data_dir + '/' + mouse_sess + '_data', sspikes1 = sspikes1, speed1 = speed1, spk1 = spk1, good_cells = good_cells, indsnull = indsnull, 
                 speed = speed, pos_trial = pos_trial, data_pos = data_pos, posx = posx, post = post, posxx = posxx, postt = postt, postrial = postrial, gain = gain, contrast = contrast, lickt = lickt, lickx = lickx)
    else:
        f = np.load(ff[0], allow_pickle = True)
        sspikes1 = f['sspikes1']
        speed1 = f['speed1']
        spk1 = f['spk1'][()]
        good_cells = f['good_cells']
        indsnull = f['indsnull']
        speed = f['speed'][()]
#        pos_trial = f['pos_trial'][()]
        data_pos = f['data_pos'][()]
        posx = f['posx'][()]
        post = f['post'][()]
        posxx = f['posxx'][()]
        postt = f['postt'][()]
        postrial = f['postrial'][()]
        gain = f['gain'][()]
        contrast = f['contrast'][()]
        lickt = f['lickt'][()]
        lickx = f['lickx'][()]
        f.close()
    data_ensemble = np.sqrt(sspikes1[:,ind ==cmod])

    f = np.load(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '.npz', allow_pickle = True)
    dgms = f['dgms_all'][()][0][0]
    coords_ds = f['coords_ds_all'][()][0]
    indstemp = f['indstemp_all'][()][0]
    f.close()
    times_cube = np.where(speed1>10)[0]
    coords1 = get_coords_all(data_ensemble, coords_ds, times_cube, indstemp, dim = 7, bPred = False, bPCA = True)
    return (files, data_ensemble, speed1, spk1, good_cells, indsnull, speed, 
           data_pos, posx, post, posxx, postt, 
           postrial, gain, contrast, lickt, lickx, dgms, 
            coords_ds, indstemp, times_cube, coords1, e1)
    
    


def get_score(dgms, dim = 1, dd = 1):
    births = dgms[dim][:, 0] #the time of birth for the 1-dim classes
    deaths = dgms[dim][:, 1] #the time of death for the 1-dim classes
    deaths[np.isinf(deaths)] = 0
    lives = deaths-births
    lives_sort = np.argsort(lives)
    lives = lives[lives_sort]
    births = births[lives_sort]
    deaths = deaths[lives_sort]
    if dd == 0:
        gaps = np.diff(np.diff(lives[-25:]))
    else:
        gaps = np.diff(lives[-25:])-1
    gapsmax = np.argmax(np.flip(gaps))
    return gapsmax 


def fit_derivative(coords1, coords2, combs1, times = [], thresh = 0.1): 
    res2 = []    
    cctrial_temp = coords2.copy()
    cctrial = np.zeros_like(cctrial_temp)
    cctrial[0,:] = cctrial_temp[0,:].copy()            
    k1, k2 = 0, 0
    for cn  in range(len(cctrial_temp)-1):
        ctmp1 = cctrial_temp[cn+1]
        c_temp = [ctmp1 + (k1*2*np.pi, k2*2*np.pi), 
                  ctmp1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                  ctmp1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                  ctmp1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                  ctmp1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                  ctmp1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                  ctmp1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                  ctmp1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                  ctmp1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                 ]  
        cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
        cctrial[cn+1,:] = c_temp[cmin]
        k1 += ks[cmin][0]
        k2 += ks[cmin][1]          
    cs11 = CubicSpline(times, cctrial[:,:])
    dcs11 = cs11.derivative(1)
    angular_rate1 = np.arctan2(dcs11(times)[:,1],dcs11(times)[:,0])
    cs12 = CubicSpline(times, np.cos(angular_rate1)).derivative(1)
    cs13 = CubicSpline(times, np.sin(angular_rate1)).derivative(1)
    dtrajs2 = np.concatenate((cs12(times)[:,np.newaxis], cs13(times)[:,np.newaxis]),1)
    times0 = np.sum((dtrajs2<thresh) & (dtrajs2>-thresh),1)    

    for comb in combs1:
        coords_2_1 = align(coords1, comb, [0,0])
        cctrial_temp = coords_2_1
        cctrial = np.zeros_like(cctrial_temp)
        cctrial[0,:] = cctrial_temp[0,:].copy()            
        k1, k2 = 0, 0
        for cn  in range(len(cctrial_temp)-1):
            ctmp1 = cctrial_temp[cn+1]
            c_temp = [ctmp1 + (k1*2*np.pi, k2*2*np.pi), 
                      ctmp1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                      ctmp1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                      ctmp1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                      ctmp1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                      ctmp1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                      ctmp1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                      ctmp1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                      ctmp1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                     ]  
            cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
            cctrial[cn+1,:] = c_temp[cmin]
            k1 += ks[cmin][0]
            k2 += ks[cmin][1]
        cs11 = CubicSpline(times, cctrial[:,:])
        dcs11 = cs11.derivative(1)
        angular_rate1 = np.arctan2(dcs11(times)[:,1],dcs11(times)[:,0])
        cs12 = CubicSpline(times, np.cos(angular_rate1)).derivative(1)
        cs13 = CubicSpline(times, np.sin(angular_rate1)).derivative(1)
        dtrajs1 = np.concatenate((cs12(times)[:,np.newaxis], cs13(times)[:,np.newaxis]),1)
        times1 = np.sum((dtrajs1<thresh) & (dtrajs1>-thresh),1)
        times2 = np.sum((times0 + times1,1))==2
        dtrajs11 = preprocessing.scale(dtrajs1[times2,:])
        dtrajs22 = preprocessing.scale(dtrajs2[times2,:])
        Ltemp0 = np.diagonal(np.matmul(dtrajs11.T, dtrajs22))
        Ltemp1 = np.sqrt(np.sum(np.square(dtrajs11),0))
        Ltemp2 = np.sqrt(np.sum(np.square(dtrajs22),0))        
        res2.extend([np.sum(1- Ltemp0/np.multiply(Ltemp1,Ltemp2))])
    return res2
    
def get_pshift(coords1, comb, coords2):
    c1 = coords1[:,0].copy()
    c2 = coords1[:,1].copy()
    for j in comb[1:2]:
        if j == 0:
            c21 = c1.copy()
            c22 = c2.copy()
        elif j == 1:
            c21 = 2*np.pi-c1.copy()
            c22 = 2*np.pi-c2.copy()
            
        elif j == 2:
            c21 = c1.copy()
            c22 = 2*np.pi-c2.copy()
        elif j == 3:
            c21 = 2*np.pi-c1.copy()
            c22 = c2.copy()           
        for k in comb[2:3]:
            if k == 0:
                c31 = c21.copy()
                c32 = c22.copy()
            elif k == 1:
                c31 = c21.copy() - np.pi/3*c22
                c32 = c22.copy()
            elif k == 2:
                c31 = c21.copy() + np.pi/3*c22
                c32 = c22.copy()
            elif k == 3:
                c31 = c21.copy()
                c32 = c22.copy() - np.pi/3*c21
            elif k == 4:
                c31 = c21.copy()
                c32 = c22.copy() + np.pi/3*c21  
                
            for i in comb[0:1]:
                if i == 0:
                    c41 = c31.copy()
                    c42 = c32.copy()
                else:
                    c41 = c32.copy()
                    c42 = c31.copy()
                coords_2_1 = np.concatenate((c41[:,np.newaxis], c42[:,np.newaxis]),1)%(2*np.pi)
                pshift = np.arctan2(np.mean(np.sin(coords_2_1-coords2),0), np.mean(np.cos(coords_2_1 - coords2),0))
    return pshift

    
def align(coords1, comb, pshift):
    c1 = coords1[:,0].copy()
    c2 = coords1[:,1].copy()
    for j in comb[1:2]:
        if j == 0:
            c21 = c1.copy()
            c22 = c2.copy()
        elif j == 1:
            c21 = 2*np.pi-c1.copy()
            c22 = 2*np.pi-c2.copy()
            
        elif j == 2:
            c21 = c1.copy()
            c22 = 2*np.pi-c2.copy()
        elif j == 3:
            c21 = 2*np.pi-c1.copy()
            c22 = c2.copy()           
        for k in comb[2:3]:
            if k == 0:
                c31 = c21.copy()
                c32 = c22.copy()
            elif k == 1:
                c31 = c21.copy() - np.pi/3*c22
                c32 = c22.copy()
            elif k == 2:
                c31 = c21.copy() + np.pi/3*c22
                c32 = c22.copy()
            elif k == 3:
                c31 = c21.copy()
                c32 = c22.copy() - np.pi/3*c21
            elif k == 4:
                c31 = c21.copy()
                c32 = c22.copy() + np.pi/3*c21  
                
            for i in comb[0:1]:
                if i == 0:
                    c41 = c31.copy()
                    c42 = c32.copy()
                else:
                    c41 = c32.copy()
                    c42 = c31.copy()
                coords_2_1 = np.concatenate((c41[:,np.newaxis], c42[:,np.newaxis]),1)%(2*np.pi)
                coords_mod_1_2 = (coords_2_1 - pshift)%(2*np.pi)
    return coords_mod_1_2



def get_mean_acorr(coords, spk, bSaveFig = False, bins = 50, times = []):
    numbins = 51
    numangsint = 52
    numangsint_1 = numangsint-1
    mid = int((numangsint_1)/2)
    bins = np.linspace(0,2*np.pi, numbins+1)
    num_neurons = len(spk[0,:])
    acorr0 = {}
    for k in range(num_neurons):
        mtot = binned_statistic_2d(coords[:, 0], coords[:,1], spk[:,k],
                                   bins = bins, range = None, expand_binnumbers = True)[0]
        mtot[np.isnan(mtot)] = np.mean(mtot[~np.isnan(mtot)])
        acorr0[k] = pearson_correlate2d(mtot, mtot)
        acorr0[k][mid,mid] = -np.inf
        acorr0[k][mid,mid] = np.max(acorr0[k])
    acorrall0 = np.zeros_like(acorr0[0])
    for j in acorr0:
        acorrall0 += (acorr0[j]-np.mean(acorr0[j]))/np.std(acorr0[j])#acorr0[j]
    return acorrall0/num_neurons

def plot_mean_acorr(coords, spk, bSaveFig = False, bins = 50, times = [], fname = ''):
    if len(times) == 0:
        times = np.arange(len(coords[:,0]))
    acorrall0 = get_mean_acorr(coords, spk, bSaveFig, bins, times)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(acorrall0.T, origin = 'lower', extent = [0,2*np.pi,0, 2*np.pi], vmin = 0, vmax = np.max(acorrall0) *0.975)
    ax.set_aspect('equal', 'box') 
    ax.axis('off')
    if bSaveFig:
        plt.savefig(fname)
        plt.close()
    return acorrall0

def normit(xxx):
    xx = xxx-np.min(xxx)
    xx = xx/np.max(xx)
    return(xx)

@numba.njit(fastmath=True, parallel = False)  # benchmarking `parallel=True` shows it to *decrease* performance
def simulate_spk_hex(cc, mcstemp, t = 0.1, nums = 4):
    num_neurons = len(mcstemp)
    spk_sim = np.zeros((len(cc), num_neurons))
    numsall = np.arange(-nums,nums+1)
    for i in range(num_neurons):
        cctmp = ((cc -mcstemp[i,:])%(2*np.pi))/(2*np.pi)
#        cctmp = (cc -mcstemp[i,:])/(2*np.pi)
        for k in numsall:
            for l in numsall:
                spk_sim[:,i] += np.exp(-np.pi/t*2/np.sqrt(3)*((k+cctmp[:,0])**2 + 
                                                              (k+cctmp[:,0])*(l+cctmp[:,1]) + 
                                                              (l +cctmp[:,1])**2))
    return spk_sim

@numba.njit(fastmath=True, parallel = False)  # benchmarking `parallel=True` shows it to *decrease* performance
def simulate_spk_sqr(cc, mcstemp, t = 0.1, nums = 4):
    num_neurons = len(mcstemp)
    spk_sim = np.zeros((len(cc), num_neurons))
    numsall = np.arange(-nums,nums+1)
    for i in range(num_neurons):
        cctmp = ((cc -mcstemp[i,:])%(2*np.pi))/(2*np.pi)
#        cctmp = (cc -mcstemp[i,:])/(2*np.pi)
        for k in numsall:
            for l in numsall:
                spk_sim[:,i] += np.exp(-np.pi/t*((k+cctmp[:,0])**2 + 
                                                              (l +cctmp[:,1])**2))
    return spk_sim

def get_ratemaps_center(coords, spk, numbins = 15,bMcs = True):
    mid = int((numbins)/2)
    bins = np.linspace(0,2*np.pi, numbins+1)
    num_neurons = len(spk[0,:])    
    mcstemp = np.zeros((num_neurons,2))
    mtot_all = {}
    for n in range(num_neurons):
        mtot = binned_statistic_2d(coords[:, 0], coords[:,1], spk[:,n],
                                   bins = bins, range = None, expand_binnumbers = True)[0]
        mtot_all[n] = mtot.copy()
        
    if bMcs:
        xv, yv = np.meshgrid(bins[0:-1] + (bins[1:] -bins[:-1])/2, 
                     bins[0:-1] + (bins[1:] -bins[:-1])/2)
        pos  = np.concatenate((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]),1)
        ccos = np.cos(pos)
        csin = np.sin(pos)
        for n in range(num_neurons):
            mtot = mtot_all[n].T.flatten()
            nans  = ~np.isnan(mtot) 
            centcos = np.sum(np.multiply(ccos[nans,:],mtot[nans,np.newaxis]),0)
            centsin = np.sum(np.multiply(csin[nans,:],mtot[nans,np.newaxis]),0)
            mcstemp[n,:] = np.arctan2(centsin,centcos)%(2*np.pi)
    return mcstemp, mtot_all

def get_centered_ratemaps(coords, spk, mcstemp, numbins = 15,):
    mid = int((numbins)/2)
    bins = np.linspace(0,2*np.pi, numbins+1)
    num_neurons = len(spk[0,:])    
    mtot_all = {}
    for n in range(num_neurons):        
        coords_temp = (coords.copy() - mcstemp[n,:])%(2*np.pi)
        coords_temp = (coords_temp + (np.pi, np.pi))%(2*np.pi)

        mtot = binned_statistic_2d(coords_temp[:, 0], coords_temp[:,1], spk[:,n],
                                   bins = bins, range = None, expand_binnumbers = True)[0]
        mtot_all[n] = mtot.copy()
    return mtot_all

def get_sim(coords, mcstemp, numbins = 10, t = 0.1, nums = 1, simtype = ''): 
    """Simulate activity """
    if simtype == 'hex0':
        coords0 = coords.copy()
        coords0[:,0] = 2*np.pi - coords0[:,0]
        spk_sim = simulate_spk_hex(coords0, mcstemp, t = t, nums = nums)
    elif simtype == 'hex' :
        spk_sim = simulate_spk_hex(coords, mcstemp, t = t, nums = nums)
    else:
        spk_sim = simulate_spk_sqr(coords, mcstemp, t = t, nums = nums)
    return spk_sim



def get_acorrs(spikes, pos_trial, pos):
    ################### Get acorrs ####################  
    num_neurons = len(spikes[0,:])
    posx_trial = pos + (pos_trial-1)*400
    binsx = np.linspace(0,np.max(posx_trial)+1e-3, int(max(pos_trial)*400/4))
    spk = np.zeros((len(binsx)-1, num_neurons))
    for j in range(num_neurons): 
        spk[:, j] = binned_statistic(posx_trial, spikes[:,j], bins = binsx)[0]

    lencorr = 300
    spk[np.isnan(spk)] = 0
    acorrs = np.zeros((len(spk[0,:]), lencorr))
    for i in range(len(spk[0,:])):
        spktemp = np.concatenate((spk[:,i], np.zeros(lencorr)))
        lenspk = len(spk[:,i])
        for t1 in range(lencorr):
            acorrs[i,t1] = np.sum(np.dot(spk[:,i],np.roll(spktemp, t1)[:lenspk]))
        acorrs[i,:] /= acorrs[i,0]
    return acorrs

def plot_sim_acorr(data_ensemble, coords, posxx, pos_trial, files, folder, speed = [],sp = -np.inf, theta = False, numbins = 10):
    coords0 = coords.copy()
    simtype = 'hex' 
    if theta:
        coords0[:,0] = 2*np.pi-coords0[:,0]     
        simtype = 'hex0'    
    
    mcstemp, mtot_all = get_ratemaps_center(coords0, data_ensemble[:,:], numbins = numbins)    
    spk_sim = get_sim(coords, mcstemp, numbins = numbins, simtype = simtype) 

    ################### Get acorrs ####################    
    acorr_sig = 2
    data_dir = 'giocomo_data'
    num_neurons = len(data_ensemble[0,:])
    acorr_real = {}
    acorr_sim = {}
    acorr_corr = {}
    t0 = 0    
    cs = ['#1f77b4', '#ff7f0e', '#2ca02c']
    if len(speed)== 0:
        speed = {}
        for fi in files:
            speed[fi] = np.ones(len(posxx[fi]))
            
    for fi in files[:]:
        finame = fi.replace('giocomo_data\\', '').replace('.mat', '')
        
        times = np.arange(t0,t0+len(posxx[fi]))
        times = times[speed[fi]>sp]
        t0 +=len(posxx[fi])
        
        acorr_real[fi] = get_acorrs(data_ensemble[times,:], pos_trial[fi][speed[fi]>sp], posxx[fi][speed[fi]>sp])
        acorr_sim[fi] = get_acorrs(spk_sim[times,:], pos_trial[fi][speed[fi]>sp], posxx[fi][speed[fi]>sp])        

        acorr_corr[fi] = np.zeros(num_neurons)
        for i in range(num_neurons):
            fi1 = fi.replace('\\', '/')
            acorr_corr[fi][i] = pearsonr(acorr_real[fi][i,1:], acorr_sim[fi][i,1:])[0] 

        print(np.mean(acorr_corr[fi]), np.std(acorr_corr[fi])/np.sqrt(len(acorr_corr[fi])))
        
        fig, ax = plt.subplots(1,1)        
        acorr_mean_real = acorr_real[fi] - acorr_real[fi].mean(1)[:,np.newaxis]
        acorrmean = acorr_mean_real.mean(0)
        acorrstd = 1*acorr_real[fi].std(0)/np.sqrt(len(acorr_real[fi][:,0]))
        ax.plot(acorrmean, lw = 2, c= cs[0])
        ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean + acorrstd,
                        lw = 0, color= cs[0], alpha = 0.3)
        ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean - acorrstd,
                        lw = 0, color= cs[0], alpha = 0.3)        

        acorr_mean_sim = acorr_sim[fi] - acorr_sim[fi].mean(1)[:,np.newaxis]
        acorrmean = acorr_mean_sim.mean(0)
        acorrstd = 1*acorr_sim[fi].std(0)/np.sqrt(len(acorr_real[fi][:,0]))
        ax.plot(acorrmean, lw = 2, c= cs[1])
        ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean + acorrstd,
                        lw = 0, color= cs[1], alpha = 0.3)
        ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean - acorrstd,
                        lw = 0, color= cs[1], alpha = 0.3)        
        ax.set_aspect(1/ax.get_data_ratio())
        #plt.xticks([0,50,100,150,200], ['', '', '', '',''])
        #plt.yticks(np.arange(0,3,1)/100, ['','',''])
        plt.gca().axes.spines['top'].set_visible(False)
        plt.gca().axes.spines['right'].set_visible(False)
        plt.savefig(folder + '/acorr_simreal_' + finame, bbox_inches='tight', pad_inches=0.1, transparent = True)
#        plt.savefig('acorr_classes' + str(fi) + '.pdf', bbox_inches='tight', pad_inches=0.1, transparent = True)
        plt.ylim([-0.2, 0.5])
        plt.close()        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(gaussian_filter1d(acorr_mean_real, axis = 1, sigma = 1),vmin = 0.0, vmax = 0.1)
        ax.set_aspect(1/ax.get_data_ratio())
        fig.tight_layout()
        fig.savefig(folder + '/_acorr_real' + finame, transparent = True)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(acorr_mean_sim,vmin = 0, vmax = 0.1)
        ax.set_aspect(1/ax.get_data_ratio())
        fig.tight_layout()
        fig.savefig(folder + '/_acorr_sum' + finame, transparent = True)
        plt.close()
    return acorr_real, acorr_sim, acorr_corr



def plot_distance_cells(mouse_sess, indsnull, e1, folder): 
    dist_name = "GiocomoLab-Campbell_Attinger_CellReports-d825378\\intermediate_data\\dark_distance\\dist_tun.mat"
    dist_tuning = sio.loadmat(dist_name)
    names = np.array(['Mouse', 'Date', 'MouseDate', 'Session', 'CellID', 'UniqueID', 'BrainRegion', 'InsertionDepth', 'DepthOnProbe',
             'DepthFromSurface', 'peak', 'period', 'prom', 'pval', 'peak_shuf', 'DepthAdjusted'])
    dist_tun = {}
    for i, names in enumerate(names):
        dist_tun[names] = dist_tuning['b']['data'][0,0][0,i]

    pval_cutoff = 0.01
    min_prom = 0.1
    if len(np.where((dist_tun['MouseDate'] == mouse_sess))[0]) == 0:
        print('Campbell not computed')
    else:
        cIdx_dist = np.where((dist_tun['MouseDate'] == mouse_sess) & (dist_tun['BrainRegion']=='MEC'))[0]
        ids = (dist_tun['pval'][cIdx_dist]<pval_cutoff) & (dist_tun['prom'][cIdx_dist]>min_prom)
        if len(ids)>len(indsnull):
            print('Campbell ids larger')
        else:
            print(np.sum(ids[indsnull,0][e1]), sum(e1))

            fig, ax = plt.subplots(1,1)
            ax.scatter(dist_tun['pval'][cIdx_dist],dist_tun['prom'][cIdx_dist])
            ax.scatter(dist_tun['pval'][cIdx_dist][indsnull][e1],dist_tun['prom'][cIdx_dist][indsnull][e1])
            ax.set_xlim([-0.05,1.05])
            ax.set_ylim([-0.03,0.7])
            ax.set_aspect(1/ax.get_data_ratio())
            fig.savefig(folder + '/distance_scores')
            plt.close()



def compute_trial_gain(angspeed, runspeed, trials_all, trial_range, gain_trials, 
                       bSaveFigs = False, ax1 = None, folder = ''):
    
    gains = np.unique(gain_trials)
    cc_gains_all = {}
    for i, gain in enumerate(gains):
        trial_curr = trial_range[gain_trials==gain]
        cc_gains = np.zeros(len(trial_curr))            
        for j, cctrial in enumerate(trial_curr):       
            cc_gains[j] = np.sum(runspeed[trials_all == cctrial])/np.sum(angspeed[trials_all == cctrial])
        cc_gains_all[i] = cc_gains
    
    for i in cc_gains_all:
        cc_gains_all[i] /= np.mean(cc_gains_all[len(gains)-1])
        
    gains_mean = [cc_gains_all[i].mean() for i in cc_gains_all]
    gains_std = [cc_gains_all[i].std()/np.sqrt(len(cc_gains_all[i])) for i in cc_gains_all]
    return gains_mean, gains_std


def plot_gain(coords1, postt, posxx, postrial, speed, gain, contrast, sp = -np.inf, folder = '', files = []):
    cs =   {}
    t0 = 0
    gains_means = {}
    gains_stds = {}
    for fi in files:
        times = np.arange(t0,t0+sum(speed[fi]>sp))
        cs[fi] = coords1[times,:]
        t0+=sum(speed[fi]>sp)

        if fi.find('gain')>-1:
            lens_temp = [0,]
            sp = -np.inf

            fig1 = plt.figure()
            contrast_curr = 100

            times = np.arange(len(cs[fi]))
            cs11 = CubicSpline(times, gaussian_filter1d(np.sin(cs[fi]),axis = 0, sigma = 10))
            cc11 = CubicSpline(times,  gaussian_filter1d(np.cos(cs[fi]), axis = 0, sigma = 10))
            angular_rate1 = np.sum(np.sqrt(np.square(cs11(times,1)) + np.square(cc11(times,1))),1)

            traj_keep = (contrast[fi] ==  contrast_curr)        
            trial_range = np.unique(postrial[fi])[traj_keep]

            gain_trials = gain[fi][traj_keep]
            sp_sess = speed[fi][speed[fi]>sp].copy()
            if len(np.unique(gain_trials))>1:
                ax1 = fig1.add_subplot(111)
                gains = np.unique(gain_trials)
                gains_mean, gains_std = compute_trial_gain(angular_rate1, sp_sess, 
                                   postrial[fi][speed[fi]>sp], 
                                   trial_range, gain_trials,
                                   bSaveFigs = False, ax1 = ax1)
                gains_means[fi] = gains_mean.copy()
                gains_stds[fi] = gains_std.copy()
                ax1.scatter(gains, gains_mean, marker = 'o', s = 100, lw = 0.1)
                ax1.set_xticks(gains)  
                ax1.set_ylim([0.25,1.1])
                ax1.set_xlim([gains[0]-0.1,gains[-1]+0.1])
                ax1.plot(plt.gca().get_xlim(), np.ones(2), c = 'k', ls = '--')
                ax1.errorbar(gains, gains_mean, gains_std,0, ls = ':')


                fig1.tight_layout()
                finame = fi.replace('giocomo_data\\', '').replace('.mat', '')
                fig1.savefig(folder + '/gain' +  finame, bbox_inches='tight', pad_inches=0.1, transparent = True)
                plt.close()
    return gains_means, gains_stds



def plot_cluster_trials(coords_all,  
                        trial_range,
                        postrial,
                        lbls, 
                        pos_curr,
                        bSaveFigs = False,
                       folder = '',
                        fname = '',
                        num_rhomb = -1
                       ):
    ks = np.array([[0,0], [1,0], [0,1], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])
    cs1 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    bRhomb = False
    if num_rhomb ==-1:
        num_rhomb = 0
        bRhomb = True
    for ii in np.unique(lbls):        
        fig = plt.figure(figsize = (20,10))    
        plt.axis('off')        
        ax = fig.add_subplot(111)          
        for i, trial in enumerate(trial_range[lbls == ii]):
            valid_trialsSpike = np.in1d(postrial,np.array([trial]))
            cctrial_temp = coords_all.copy()[valid_trialsSpike, :]
            cctrial_temp -= cctrial_temp[0,:]
            cctrial_temp += np.pi
            
            cctrial = np.zeros_like(cctrial_temp)
            cctrial[0,:] = cctrial_temp[0,:].copy()            
            k1, k2 = 0, 0
            for cn  in range(len(cctrial_temp)-1):
                c1 = cctrial_temp[cn+1]
                c_temp = [c1 + (k1*2*np.pi, k2*2*np.pi), 
                          c1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                          c1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                          c1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                          c1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                          c1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                          c1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                          c1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                          c1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                         ]  
                cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
                cctrial[cn+1,:] = c_temp[cmin]
                k1 += ks[cmin][0]
                k2 += ks[cmin][1]
            if bRhomb: num_rhomb = max(num_rhomb, int(np.max(cctrial)/(2*np.pi))+1)                    
            posx_trial = pos_curr[valid_trialsSpike]
            lm_inds = np.concatenate(([np.argmin(np.abs(posx_trial-0))],
                                      [np.argmin(np.abs(posx_trial-80))],
                                      [np.argmin(np.abs(posx_trial-160))],
                                      [np.argmin(np.abs(posx_trial-240))],
                                      [np.argmin(np.abs(posx_trial-320))],
                                      [np.argmin(np.abs(posx_trial-400))],))

            ax.scatter(cctrial[:,0], cctrial[:,1], 
                       s = 1, alpha = 0.7, c = preprocessing.minmax_scale(range(len(cctrial[:,0]))), 
                       vmin = 0, vmax = 1, zorder = -3)

            ax.scatter(cctrial[lm_inds,0], cctrial[lm_inds,1], 
                       marker = 'X', lw = 0.5, s = 10, c =cs1[:len(lm_inds)], zorder = -1)

                    
        for i in range(num_rhomb):
            for j in range(num_rhomb):
                ax.plot([2*np.pi*i,2*np.pi*(i+1)],[2*np.pi*j,2*np.pi*j], c = 'k', ls = '--', lw = 1, zorder = -2)
                ax.plot([2*np.pi*i,2*np.pi*i],[2*np.pi*j,2*np.pi*(j+1)], c = 'k', ls = '--', lw = 1, zorder = -2)
        i += 1
        for j in range(i):
            ax.plot([2*np.pi*i,2*np.pi*i],[2*np.pi*j,2*np.pi*(j+1)], c = 'k', ls = '--', lw = 1, zorder = -2)
        j += 1
        for i in range(j):
            ax.plot([2*np.pi*i,2*np.pi*(i+1)],[2*np.pi*j,2*np.pi*j], c = 'k', ls = '--', lw = 1, zorder = -2)


        ax.set_aspect(1/ax.get_data_ratio())
        r_box = transforms.Affine2D().skew_deg(15,15)
        for x in ax.images + ax.lines + ax.collections:
            trans = x.get_transform()
            x.set_transform(r_box+trans) 
            if isinstance(x, PathCollection):
                transoff = x.get_offset_transform()
                x._transOffset = r_box+transoff     
        ax.set_xlim(-np.pi, (num_rhomb)*2*np.pi + 8*3*np.pi/5)
        ax.set_ylim(-np.pi, (num_rhomb)*2*np.pi + 8*3*np.pi/5)
        ax.set_aspect('equal', 'box') 
        ax.axis('off')
        fig.tight_layout(h_pad = 0, w_pad = 0)
        
        if bSaveFigs:            
            fig.savefig(folder + '/' + fname + '_' + str(ii), transparent = True)
            plt.close()
        else:
            plt.show()




def plot_stats(stats_tor, stats_space = [], lbls = [], statname='', sess_name=''):
    if statname == 'dist':
        stats_tor = np.array(stats_tor)/(2*np.pi)*360
        stats_space = np.array(stats_space)/(2*np.pi)*360
    num_mods = len(stats_tor)
    bSems = False
    if lbls[1][-3:] == 'SEM':
        bSems = True
        num_mods = int(num_mods/2)
    fig = plt.figure(figsize=(2,6))
    ax = fig.add_subplot(111)
    xs = [0.01, 0.09]
    if len(stats_space):
        if len(stats_space):
            for i in range(num_mods):
                if bSems:
                    i*=2

                ax.scatter(xs, [stats_tor[i],stats_space[i]], s = 250, marker='.', #lw = 3, 
                    zorder=-1)
                ax.plot(xs, [stats_tor[i],stats_space[i]], lw = 4, alpha = 0.5, zorder = -2)

        if bSems:
            for i in range(num_mods):
                i*=2
                ax.errorbar(xs, [stats_tor[i],stats_space[i]], lw = 4, yerr=[stats_tor[i+1],stats_space[i+1]], 
                    fmt='none', alpha = 1, zorder=-2)

    ax.set_xlim([0,0.1])
    ax.set_ylim(ylims[statname])
    
    ax.set_xticks(xs)
#    ax.set_xticklabels(np.zeros(len(xs),dtype=str))
    ax.xaxis.set_tick_params(width=1, length =7)
    ax.set_yticks(ytics[statname])
#    ax.set_yticklabels(np.array(ytics[statname],dtype=str))
#    ax.set_yticklabels(np.zeros(len(ytics[statname]),dtype=str))
    ax.yaxis.set_tick_params(width=1, length =7)

    
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/(abs(y1-y0))*3.5)
    
    plt.gca().axes.spines['top'].set_visible(False)
    plt.gca().axes.spines['right'].set_visible(False)
    




def plot_centered_ratemaps(coords, data, mcstemp, numbins, ax = None):
    mtot_temp = get_centered_ratemaps(coords, data, mcstemp, numbins = numbins)
    mtot_mean = np.zeros_like(mtot_temp[0])
    num_neurons = len(mtot_temp)
    for n in range(num_neurons):
        nans = np.isnan(mtot_temp[n])
        mmean = np.mean(mtot_temp[n][~nans])
        mstd = np.std(mtot_temp[n][~nans])
        mtot_temp[n][nans] = mmean
        mtot_mean += (mtot_temp[n] - mmean)/mstd
    if not ax:
        fig,ax = plt.subplots(1,1)

    ax.imshow(mtot_mean, vmin = 0, vmax = mtot_mean.flatten()[np.argsort(mtot_mean.flatten())[int(len(mtot_mean.flatten())*0.99)]])
    ax.axis('off')
    


def get_sim_corr(coords, data, numbins = 10, bSqr = True, folder = '', fname = ''):    
    num_neurons = len(data[0,:])
    coords0 = coords.copy()
    coords0[:,0] = 2*np.pi-coords0[:,0]     
    mcstemp, mtot_all = get_ratemaps_center(coords, data, numbins = numbins,)
    mcstemp0, mtot_all0 = get_ratemaps_center(coords0, data, numbins = numbins,)    
    intervals = np.linspace(0, 2*np.pi, 100)
    intervals = intervals[1:] - (intervals[1]-intervals[0])/2
    c1,c2 = np.meshgrid(intervals,intervals)
#    coords_uniform = np.random.rand(len(coords),2)*2*np.pi
    coords_uniform = np.concatenate((c1.flatten()[:,np.newaxis], c2.flatten()[:,np.newaxis]), 1)
    
    spk_sim_hex = get_sim(coords_uniform, mcstemp, numbins = numbins, simtype = 'hex')
    spk_sim_hex0 = get_sim(coords_uniform, mcstemp0, numbins = numbins, simtype = 'hex0')
        
    ps1_hex = np.zeros(num_neurons)
    ps1_hex0 = np.zeros(num_neurons)
    ps1_sqr = np.zeros(num_neurons)
    

    __, mtot_all_hex = get_ratemaps_center(coords_uniform, spk_sim_hex, numbins = numbins, bMcs = False)
    __, mtot_all_hex0 = get_ratemaps_center(coords_uniform, spk_sim_hex0, numbins = numbins, bMcs = False)
    if bSqr:
        spk_sim_sqr = get_sim(coords_uniform, mcstemp, numbins = numbins, simtype = 'sqr')
        __, mtot_all_sqr = get_ratemaps_center(coords_uniform, spk_sim_sqr, numbins = numbins, bMcs = False)

    for n in range(num_neurons):        
        mtot_curr = mtot_all[n].copy()
        mtot_curr[np.isnan(mtot_curr)] = np.mean(mtot_curr[~np.isnan(mtot_curr)])
        
        mtot_hex = mtot_all_hex[n].copy()
        mtot_hex[np.isnan(mtot_hex)] = np.mean(mtot_hex[~np.isnan(mtot_hex)])
        
        mtot_hex0 = mtot_all_hex0[n].copy()
        mtot_hex0[np.isnan(mtot_hex0)] = np.mean(mtot_hex0[~np.isnan(mtot_hex0)])
        
        ps1_hex[n] = pearsonr(mtot_hex.flatten(), mtot_curr.flatten())[0]        
        ps1_hex0[n] = pearsonr(mtot_hex0.flatten(), mtot_curr.flatten())[0]
        
        if bSqr:
            mtot_sqr = mtot_all_sqr[n].copy()
            mtot_sqr[np.isnan(mtot_sqr)] = np.mean(mtot_sqr[~np.isnan(mtot_sqr)])
            ps1_sqr[n] = pearsonr(mtot_sqr.flatten(), mtot_curr.flatten())[0]
    theta = np.median(ps1_hex0)>np.median(ps1_hex)
    if len(folder) >0:
        fig,ax = plt.subplots(1,2)
        print(theta)
        plot_centered_ratemaps(coords, data, mcstemp, numbins, ax[0])
        if theta:
    #        plot_centered_ratemaps(coords0, data, mcstemp0, numbins, ax[0])
            plot_centered_ratemaps(coords_uniform, spk_sim_hex0, mcstemp, numbins, ax[1])
        else:
            plot_centered_ratemaps(coords_uniform, spk_sim_hex, mcstemp, numbins, ax[1])
            
        fig.tight_layout()
        fig.savefig(folder + '/stacked_ratemap' + fname, transparent = True)
        plt.close()
    
    print(np.median(ps1_sqr), np.median(ps1_hex), np.median(ps1_hex0))

    return (ps1_sqr, ps1_hex, ps1_hex0, theta)
from sklearn.cluster import AgglomerativeClustering


def get_combs(theta, theta1, thresh = 0.1):         
    if theta1:    
        if theta:
            combs1 = combs_all[2]
        else:
            combs1 = combs_all[0]
    else:
        if theta:
            combs1 = combs_all[1]
        else:
            combs1 = combs_all[3]   
    return combs1




def get_ind(mouse_sess, data_dir = 'giocomo_figures_final12', files = [], good_cells = [], indsnull = [], posxx = [], nbs = -1, lentmp = 0):
    indfile = glob.glob(data_dir + '/' + mouse_sess + '_mods.npz')
    if len(indfile)>0:
        f = np.load(indfile[0], allow_pickle = True)
        ind = f['ind'][()]
        f.close()
    else:
        ################### Get crosscorr stats ####################  
        crosscorr_train = get_cross(mouse_sess)
        num_neurons = len(good_cells)
        crosscorrs = np.zeros((num_neurons,num_neurons))
        for fi in files: 
            fi1 = fi.replace('\\', '/')
            crosscorrs_tmp = crosscorr_train[fi1].copy()
            num_neurons = len(crosscorrs_tmp[:,0,0])
            for i in range(num_neurons):
                for j in np.arange(i+1, num_neurons):
                    a = crosscorrs_tmp[i,j,:lentmp]
                    b = crosscorrs_tmp[j,i,:lentmp]
                    c = np.concatenate((a,b))
                    if np.min(c)>0:
                        crosscorrs[i,j] +=  np.square(np.min(c)/np.max(c))/len(files)
                    crosscorrs[j,i] = crosscorrs[i,j]

        num_neurons = np.sum(indsnull)
        crosscorr_tmp = crosscorrs[indsnull,:]
        crosscorr_tmp = crosscorr_tmp[:,indsnull]
        X1  = crosscorr_tmp
        X1[np.isnan(X1)] = 1
        X1[np.isinf(X1)] = 1
        agg = AgglomerativeClustering(n_clusters=None,affinity='precomputed', linkage='average', 
                                      distance_threshold=nbs)
        ind = agg.fit(X1).labels_
#        np.savez(data_dir + '/' + mouse_sess + '_mods', ind = ind)
    return ind


def load_data_all(mouse_sess, cmod, data_dir, files, data_dir1 = 'giocomo_analyses_250722'):
    f = np.load(data_dir1 + '/' + mouse_sess + '_mods.npz',allow_pickle = True)
    ind = f['ind']
    f.close()
    e1 = ind == cmod
    print('')
    print(mouse_sess, 'ind ' + str(cmod), sum(e1))
    
    ff = glob.glob(data_dir1 + '/' + mouse_sess + '_data.npz')

    if len(ff) == 0:
        (sspikes1, speed1, sspk1, spk1, good_cells, indsnull, speed, 
           pos_trial,data_pos, posx, post, posxx, postt, 
           postrial, gain, contrast, lickt, lickx) =  get_data(files)
        np.savez(data_dir + '/' + mouse_sess + '_data', sspikes1 = sspikes1, speed1 = speed1, spk1 = spk1, good_cells = good_cells, indsnull = indsnull, 
                 speed = speed, pos_trial = pos_trial, data_pos = data_pos, posx = posx, post = post, posxx = posxx, postt = postt, postrial = postrial, gain = gain, contrast = contrast, lickt = lickt, lickx = lickx)
    else:
        f = np.load(ff[0], allow_pickle = True)
        sspikes1 = f['sspikes1']
        speed1 = f['speed1']
        spk1 = f['spk1'][()]
        good_cells = f['good_cells']
        indsnull = f['indsnull']
        speed = f['speed'][()]
#        pos_trial = f['pos_trial'][()]
        data_pos = f['data_pos'][()]
        posx = f['posx'][()]
        post = f['post'][()]
        posxx = f['posxx'][()]
        postt = f['postt'][()]
        postrial = f['postrial'][()]
        gain = f['gain'][()]
        contrast = f['contrast'][()]
        lickt = f['lickt'][()]
        lickx = f['lickx'][()]
        f.close()
    t0 = 0
    f = np.load(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '.npz', allow_pickle = True)
    dgms = f['dgms_all'][()]
    coords_ds = f['coords_ds_all'][()]
    indstemp = f['indstemp_all'][()]
    f.close()
    coords1 = {}
    data_ensemble = {}
    times_cube = {}
    for it, fi in enumerate(files):
        times = np.arange(t0,t0+len(posxx[fi]))
        data_ensemble[fi] = np.sqrt(sspikes1[times,:][:,ind ==cmod])
        t0 += len(posxx[fi])
        times_cube[fi] = np.where(speed[fi]>10)[0]
        coords1[fi] = get_coords_all(data_ensemble[fi], coords_ds[it], times_cube[fi], indstemp[it], 
            dim = 7, bPred = False, bPCA = True)
    return (files, data_ensemble, speed1, spk1, good_cells, indsnull, speed, 
           data_pos, posx, post, posxx, postt, 
           postrial, gain, contrast, lickt, lickx, dgms, 
            coords_ds, indstemp, times_cube, coords1, e1)



def addonein(iii, jjj, pairs):
    if(len(pairs)==0):
        return [iii, jjj]
    for i in range(len(pairs)):
        if ((iii==pairs[i][0] and jjj==pairs[i][1]) or 
            (iii==pairs[i][1] and jjj==pairs[i][0])):
            return
    return [iii, jjj]


def get1dspatialpriorpairs(nn, periodicprior):
    pairs = []
    for i in range(nn):
        for j in range(nn):
            p = None
            if periodicprior and (abs(i-j)==nn-1):
                p = addonein(i, j, pairs)
            elif (abs(i-j)==1):
                p = addonein(i, j, pairs)
            if p:
                pairs.append(p) 
                    
    pairs = np.array(pairs)
    sortedpairs = []
    for i in range(nn):
        kpairs = []
        for j in range(len(pairs[:,0])):
            ii, jj = pairs[j,:]
            if(i == ii or i == jj):
                kpairs.append(pairs[j,:])
        kpairs = np.array(kpairs)
        sortedpairs.append(kpairs)
    return(pairs, sortedpairs)

def get2dspatialpriorpairs(nn, periodicprior):
    pairs = []
    for i in range(nn):
        for j in range(nn):
            for m in range(nn):
                for n in range(nn):
                    if periodicprior:
                        if ((abs(i-m)==nn-1 and (j-n)==0) or 
                            (abs(i-m)==0 and abs(j-n)==nn-1) or
                            (abs(i-m)==nn-1 and abs(j-n)==nn-1) or  
                            (abs(i-m)==1 and abs(j-n)==nn-1) or
                            (abs(i-m)==nn-1 and abs(j-n)==1)):
                            p = addonein(i*nn+j, m*nn+n, pairs)
                            if p:
                                pairs.append(p)  
                            continue
                    if ((abs(i-m)==1 and (j-n)==0) or  
                        (abs(i-m)==0 and abs(j-n)==1) or 
                        (abs(i-m)==1 and abs(j-n)==1)):
                        p = addonein(i*nn+j, m*nn+n, pairs)
                        if p:
                            pairs.append(p) 
                    
    pairs = np.array(pairs)
    sortedpairs = []
    for i in range(nn*nn):
        kpairs = []
        for j in range(len(pairs[:,0])):
            ii, jj = pairs[j,:]
            if(i == ii or i == jj):
                kpairs.append(pairs[j,:])
        kpairs = np.array(kpairs)
        sortedpairs.append(kpairs)
    return(pairs, sortedpairs)

def getpoissonsaturatedloglike(S):
    Sguys = S[S>0.00000000000001] ## note this is same equation as below with Sguy = exp(H)
    return np.sum(np.ravel(Sguys*np.log(Sguys) - Sguys)) - np.sum(np.ravel(np.log(factorial(S))))

def singleiter(vals, covariates, GoGaussian, finthechat, BC, y, LAM, sortedpairs):
    P = np.ravel(vals)
    H = np.dot(P,covariates)
    num_cov, T = np.shape(covariates)

    if(GoGaussian):
        guyH = H
    else:
        guyH = np.exp(H)

    dP = np.zeros(num_cov)
    for j in range(num_cov):
        pp = 0.
        if LAM > 0:
            if(len(np.ravel(sortedpairs))>0):
                kpairs = sortedpairs[j]
                if(len(np.ravel(kpairs))>0):
                    for k in range(len(kpairs[:,0])):
                        ii, jj = kpairs[k,:]
                        if(j == ii):
                            pp += LAM*(P[ii] - P[jj])
                        if(j == jj):
                            pp += -1.*LAM*(P[ii] - P[jj])
                            
        dP[j] = BC[j] - np.mean(guyH*covariates[j,:]) - pp/T

    if(GoGaussian):
        L = -np.sum( (y-guyH)**2 ) 
    else:
        L = np.sum(np.ravel(y*H - guyH)) - finthechat 
    return -L, -dP

def simplegradientdescent(vals, numiters, covariates, GoGaussian, finthechat, BC, y, LAM, sortedpairs):
    P = vals
    for i in range(0,numiters,1):
        L, dvals = singleiter(vals, covariates, GoGaussian, finthechat, BC, y, LAM, sortedpairs)
        P -= 0.8 * dvals
    return P, L

def fitmodel(y, covariates, GoGaussian = False, LAM = 0, sortedpairs = []):
    num_cov = np.shape(covariates)[0]
    T = len(y)
    BC = np.zeros(num_cov)
    for j in range(num_cov):
        BC[j] = np.mean(y * covariates[j,:])
    if GoGaussian:
        finthechat = 0
    else:
        finthechat = np.sum(np.ravel(np.log(factorial(y))))

    vals, Lmod = simplegradientdescent(np.zeros(num_cov), 2, covariates, GoGaussian, finthechat, BC, y, LAM, sortedpairs)
    res = opt.minimize(singleiter, vals, (covariates, GoGaussian, finthechat, BC, y, LAM, sortedpairs), 
        method='L-BFGS-B', jac = True, options={'ftol' : 1e-5, 'disp': False})
    vals = res.x + 0.
    vals, Lmod = simplegradientdescent(vals, 2, covariates, GoGaussian, finthechat, BC, y, LAM, sortedpairs)
    return vals

def preprocess_dataX(Xin, num_bins):
    Xin = np.transpose(Xin)
    num_dim, num_times = np.shape(Xin)

    tmp = np.linspace(-0.001, 1.001, num_bins+1)
    if num_dim == 1: 
        dig = (np.digitize(np.array(Xin), tmp)-1)
    elif num_dim == 2:
        dig = (np.digitize(np.array(Xin[0,:]), tmp)-1)*num_bins
        dig += (np.digitize(np.array(Xin[1,:]), tmp)-1)
   
    X = np.zeros((num_times, np.power(num_bins,num_dim)))
    X[range(num_times), dig] = 1
    return np.transpose(X)

def glm(xxss, ys, num_bins, GoGaussian, cv_folds, LAM = 0, periodicprior = False):
    T, dim = np.shape(xxss)
    tmp = np.floor(T/cv_folds)
    
    xxss = preprocessing.minmax_scale(xxss,axis =0)
    xvalscores = np.zeros(cv_folds)
    P = np.zeros((np.power(num_bins, dim), cv_folds))
    LL = np.zeros(T)
    yt = np.zeros(T)
    tmp = np.floor(T/cv_folds)
    if(LAM==0):
        sortedpairs = []
    else:
        if dim == 1:
            pairs, sortedpairs = get1dspatialpriorpairs(num_bins, periodicprior)
        elif dim == 2:
            pairs, sortedpairs = get2dspatialpriorpairs(num_bins, periodicprior)
    for i in range(cv_folds):
        fg = np.ones(T)
        if cv_folds == 1:
            fg = fg==1
            nonfg = fg
        else:
            if(i<cv_folds):
                fg[int(tmp*i):int(tmp*(i+1))] = 0
            else:
                fg[-(int(tmp)):] = 0
            fg = fg==1
            nonfg = ~fg
        X_space = preprocess_dataX(xxss[fg,:], num_bins)

        P[:, i] = fitmodel(ys[fg], X_space, GoGaussian, LAM, sortedpairs)
        X_test = preprocess_dataX(xxss[nonfg,:], num_bins)
        H = np.dot(P[:, i], X_test)
        if(GoGaussian):
            yt[nonfg] = H
            LL[nonfg] = -np.sum( (ys[nonfg]-yt[nonfg])**2 )             
        else:
            expH = np.exp(H)
            yt[nonfg] = expH
            finthechat = (np.ravel(np.log(factorial(ys[nonfg]))))
            LL[nonfg] = (np.ravel(ys[nonfg]*H - expH)) - finthechat
    if GoGaussian:
        leastsq = np.sum( (ys-yt)**2)
        ym = np.mean(ys)
        expl_deviance =  (1. - leastsq/np.sum((ys-ym)**2))
    else:
        LLnull = np.zeros(T)
        P_null = np.zeros((1, cv_folds))
        for i in range(cv_folds):
            fg = np.ones(T)
            if cv_folds == 1:
                fg = fg==1
                nonfg = fg
            else:
                if(i<cv_folds):
                    fg[int(tmp*i):int(tmp*(i+1))] = 0
                else:
                    fg[-(int(tmp)):] = 0
                fg = fg==1
                nonfg = ~fg
            
            X_space = np.transpose(np.ones((sum(fg),1)))
            X_test = np.transpose(np.ones((sum(nonfg),1)))
            P_null[:, i] = fitmodel(ys[fg], X_space, GoGaussian)

            H = np.dot(P_null[:, i], X_test)
            expH = np.exp(H)
            finthechat = (np.ravel(np.log(factorial(ys[nonfg]))))
            LLnull[nonfg] = (np.ravel(ys[nonfg]*H - expH)) - finthechat        
        LS = getpoissonsaturatedloglike(ys[~np.isinf(LL)]) 
        expl_deviance = 1 - (np.sum(LS) - np.sum(LL[~np.isinf(LL)]))/(np.sum(LS) - np.sum(LLnull[~np.isinf(LL)]))
        print(expl_deviance)
    return yt, LL, expl_deviance

def load_glm_data(mouse_sess, cmod, data_dir, bTor, bSess, files):
    data_dir1 = 'giocomo_analyses_250722'
    f = np.load(data_dir1 + '/' + mouse_sess + '_mods.npz',allow_pickle = True)
    ind = f['ind']
    f.close()
    e1 = ind == cmod
    print('')
    print(mouse_sess, 'ind ' + str(cmod), sum(e1))
    
    ff = glob.glob(data_dir1 + '/' + mouse_sess + '_data.npz')
    coords_all = {}

    if len(ff) == 0:
        (sspikes1, speed1, sspk1, spk1, good_cells, indsnull, speed, 
           pos_trial,data_pos, posx, post, posxx, postt, 
           postrial, gain, contrast, lickt, lickx) =  get_data(files)
        np.savez(data_dir + '/' + mouse_sess + '_data', sspikes1 = sspikes1, speed1 = speed1, spk1 = spk1, good_cells = good_cells, indsnull = indsnull, 
                 speed = speed, pos_trial = pos_trial, data_pos = data_pos, posx = posx, post = post, posxx = posxx, postt = postt, postrial = postrial, gain = gain, contrast = contrast, lickt = lickt, lickx = lickx)
    else:
        f = np.load(ff[0], allow_pickle = True)
        sspikes1 = f['sspikes1']
        speed1 = f['speed1']
        speed = f['speed'][()]
        spk1 = f['spk1'][()]
        indsnull = f['indsnull']
        posxx = f['posxx'][()]
        postrial = f['postrial'][()]
        gain = f['gain'][()]
        contrast = f['contrast'][()]

        f.close()
    if bTor:
        if bSess:
            coords_f = glob.glob(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '_coords_all_sess.npz')
            if len(coords_f)>0:
                f = np.load(coords_f[0], allow_pickle = True)
                coords_all = f['coords_all'][()]
                f.close()
            else:
                f = np.load(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '.npz', allow_pickle = True)
                coords_ds = f['coords_ds_all'][()]
                indstemp = f['indstemp_all'][()]
                f.close()
                coords1 = {}
                data_ensemble = {}
                times_cube = {}
                t0 = 0
                for it, fi in enumerate(files):
                    times = np.arange(t0,t0+len(posxx[fi]))
                    data_ensemble[fi] = np.sqrt(sspikes1[times,:][:,ind ==cmod])
                    t0 += len(posxx[fi])
                    times_cube[fi] = np.where(speed[fi]>10)[0]
                    num_neurons = len(data_ensemble[fi][0,:])
                for fi in files:
                    for it2, fi2 in enumerate(files):
                        if fi2 != fi:
                            coords_all[fi + str(it2)] = {}
                            for nn in range(num_neurons):
                                inds = np.ones(num_neurons, dtype = bool)
                                inds[nn] = False
                                coords_all[fi + str(it2)][nn] = get_coords_all(data_ensemble[fi2][:,inds], 
                                                                                coords_ds[it2], 
                                                                                times_cube[fi2], 
                                                                                indstemp[it2], 
                                                                                spk2 = data_ensemble[fi][:,inds],  
                                                                                bPred = False, bPCA = False)

                np.savez(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '_coords_all_sess.npz', coords_all = coords_all)
        else:   
            coords_f = glob.glob(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '_coords_all.npz')
            if len(coords_f)>0:
                f = np.load(coords_f[0], allow_pickle = True)
                coords_all = f['coords_all'][()]
                f.close()
            else:
                data_ensemble = np.sqrt(sspikes1[:,ind ==cmod])
                f = np.load(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '.npz', allow_pickle = True)
                coords_ds = f['coords_ds_all'][()][0]
                indstemp = f['indstemp_all'][()][0]
                f.close()
                times_cube = np.where(speed1>10)[0]
                num_neurons = len(data_ensemble[0,:])
                for nn in range(num_neurons):
                    inds = np.ones(num_neurons, dtype = bool)
                    inds[nn] = False
                    coords_all[nn] = get_coords_all(data_ensemble[:,inds], coords_ds, 
                                                    times_cube, indstemp, dim = sum(inds), 
                                                    bPred = False, bPCA = False)
                np.savez(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '_coords_all.npz', coords_all = coords_all)
    return spk1, posxx, indsnull, e1, coords_all, gain, contrast, postrial


def run_GLM(mouse_sess, cmod, data_dir, 
            num_bins_all = [15], LAM_all = [0], 
            bTor = True, cv_folds = 3, 
            GoGaussian = False, bSess = False, files = [],
            files_dec = [], gain_score = 1, contrast_score = 100):

    spk1, posxx, indsnull, e1, coords_all, gain, contrast, postrial = load_glm_data(mouse_sess, cmod, data_dir, bTor, bSess, files)
    files_dec
    GLMscores = {}
    t0 = 0
    if len(files_dec) == 0:
        files_dec = files
    elif len(files_dec)<len(files):
        files_temp = []
        for fi in files:
            if fi in files_dec:
                files_temp.extend([fi])
            else:
                files_temp.extend([''])
        files_dec = files_temp

    for fi in files:
        times = np.arange(t0, t0+len(posxx[fi]))
        t0 += len(posxx[fi])
        if (fi.find('baseline')==-1) & (fi.find('gain')==-1) & (fi.find('contrast')==-1):
            continue
        poscurr = (gain[fi][postrial[fi]-1] == gain_score) & (contrast[fi][postrial[fi]-1] == contrast_score)
        times = times[poscurr]
        spk = spk1[fi][:,indsnull][:,e1][poscurr,:].copy()
        __, num_neurons = np.shape(spk)

        if bSess:
            GLMscores[fi] = np.zeros((num_neurons, len(files), len(num_bins_all), len(LAM_all)))
        else:
            GLMscores[fi] = np.zeros((num_neurons, len(num_bins_all), len(LAM_all)))
            
        for i1, num_bins in enumerate(num_bins_all):
            for i2, LAM in enumerate(LAM_all):            
                for n in np.arange(0, num_neurons, 1): 
                    if bTor:
                        if bSess:
                            for itfi2, fi2 in enumerate(files_dec):
                                if (fi2 != fi) & (len(fi2)>0):
                                    (__, __, GLMscores[fi][n, itfi2, i1, i2]) = glm(coords_all[fi + str(itfi2)][n][poscurr, :2].copy(), 
                                                                        spk[:,n], 
                                                                        num_bins, 
                                                                        GoGaussian, 
                                                                        cv_folds, LAM)
                        else:                            
                            (__, __, GLMscores[fi][n, i1, i2]) = glm(coords_all[n][times, :2], 
                                                                spk[:,n], 
                                                                num_bins, 
                                                                GoGaussian, 
                                                                cv_folds, LAM)
                    else:
                         (__, __, GLMscores[fi][n, i1, i2]) = glm(posxx[fi][poscurr, np.newaxis], 
                                                           spk[:,n], 
                                                           num_bins,
                                                           GoGaussian, 
                                                           cv_folds, LAM)
    return GLMscores
