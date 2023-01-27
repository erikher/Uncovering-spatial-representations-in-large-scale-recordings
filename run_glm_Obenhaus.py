import h5py
import sys
import os 
import cv2 as cv
import numpy as np
import functools
import glob
import numba
from scipy.interpolate import interp1d
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
from sklearn.cluster import AgglomerativeClustering


from ripser import ripser
from gtda.homology import VietorisRipsPersistence
import h5py

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

def getbernoullisaturatedloglike(S):
    Sguys = S[S>0.00000000000001] ## note this is same equation as below with Sguy = exp(H)
    return np.sum(np.ravel(Sguys*np.log(Sguys) - np.log(1+ Sguys))) 


def singleiter(vals, covariates, GoGaussian, GoBernoulli, finthechat, BC, y, LAM, sortedpairs):
    P = np.ravel(vals)
    H = np.dot(P,covariates)
    num_cov, T = np.shape(covariates)

    if GoGaussian:
        guyH = H
    elif GoBernoulli:
        expH = np.exp(H) 
        guyH = expH / (1.+ expH)   
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

    if GoGaussian:
        L = -np.sum( (y-guyH)**2 ) 
    elif GoBernoulli:
        L = np.sum(np.ravel(y*H - np.log(1. + expH)))
    else:
        L = np.sum(np.ravel(y*H - guyH)) - finthechat 
    return -L, -dP


def simplegradientdescent(vals, numiters, covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs):
    P = vals
    for i in range(0,numiters,1):
        L, dvals = singleiter(vals, covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs)
        P -= 0.8 * dvals
    return P, L

def fitmodel(y, covariates, GoGaussian = False,  GoBernoulli = False, LAM = 0, sortedpairs = []):
    num_cov = np.shape(covariates)[0]
    T = len(y)
    BC = np.zeros(num_cov)
    for j in range(num_cov):
        BC[j] = np.mean(y * covariates[j,:])
    if GoGaussian:
        finthechat = 0
    elif GoBernoulli:
        finthechat = 0
    else:
        finthechat = np.sum(np.ravel(np.log(factorial(y))))

    vals, Lmod = simplegradientdescent(np.zeros(num_cov), 2, covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs)
    res = opt.minimize(singleiter, vals, (covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs), 
        method='L-BFGS-B', jac = True, options={'ftol' : 1e-5, 'disp': False})
    vals = res.x + 0.
    vals, Lmod = simplegradientdescent(vals, 2, covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs)
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

def glm(xxss, ys, num_bins, GoGaussian,  GoBernoulli, cv_folds, LAM = 0, periodicprior = False):
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

        P[:, i] = fitmodel(ys[fg], X_space, GoGaussian,  GoBernoulli, LAM, sortedpairs)
        X_test = preprocess_dataX(xxss[nonfg,:], num_bins)
        H = np.dot(P[:, i], X_test)
        if(GoGaussian):
            yt[nonfg] = H
            LL[nonfg] = -np.sum( (ys[nonfg]-yt[nonfg])**2 )             
        elif(GoBernoulli):
            expH = np.exp(H)
            yt[nonfg] = np.log(1. + expH)
            LL[nonfg] = np.sum(np.ravel(ys[nonfg]*H - np.log(1. + expH)))
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
            P_null[:, i] = fitmodel(ys[fg], X_space, GoGaussian, GoBernoulli)
            H = np.dot(P_null[:, i], X_test)
            expH = np.exp(H)
            if GoBernoulli:
                LLnull[nonfg] = np.sum(np.ravel(ys[nonfg]*H - np.log(1. + expH)))
            else:
                finthechat = (np.ravel(np.log(factorial(ys[nonfg]))))
                LLnull[nonfg] = (np.ravel(ys[nonfg]*H - expH)) - finthechat        
        if GoBernoulli:
            LS = getbernoullisaturatedloglike(ys[~np.isinf(LL)]) 
        else:        
            LS = getpoissonsaturatedloglike(ys[~np.isinf(LL)]) 
        expl_deviance = 1 - (np.sum(LS) - np.sum(LL[~np.isinf(LL)]))/(np.sum(LS) - np.sum(LLnull[~np.isinf(LL)]))
    return yt, LL, expl_deviance



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
        coords_mod1 = np.zeros((len(dspk1), num_circ))
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

def load_glm_data(mouse, sess, data_dir, bTor):
    import pandas as pd
    pd.set_option('display.max_rows', 10)

    AngularRate_Stats = pd.read_pickle('functional/AngularRate_Stats')
    BorderScore = pd.read_pickle('functional/BorderScore')
    BVScore = pd.read_pickle('functional/BVScore')
    FilteredSpikes = pd.read_pickle('functional/FilteredSpikes')
    Session = pd.read_pickle('functional/Session')
    GridScore = pd.read_pickle('functional/GridScore')
    GridScore_Stats = pd.read_pickle('functional/GridScore_Stats')
    NumberCellTypes_MEC = pd.read_pickle('functional/NumberCellTypes_MEC')
    NumberCellTypes_PAS = pd.read_pickle('functional/NumberCellTypes_PAS')
    Tracking_Linear = pd.read_pickle('functional/Tracking_Linear')
    Tracking_OpenField = pd.read_pickle('functional/Tracking_OpenField')
    OVCScore = pd.read_pickle('functional/OVCScores')
    MECcells = pd.read_pickle('functional/MECcells')
    PAScells = pd.read_pickle('functional/PAScells')
    FilteredCells = pd.read_pickle('functional/FilteredCells')
    sess_name = sess

    if  sess_name in ('c43d9bd004db772b','9190f2fccd52497e'):
        sess_name0 = '59825ec5641c94b4'
    elif sess_name == 'd5a06b6a7630bb11':
        sess_name0 ='7e888f1d8eaab46b' 
    elif sess_name == '26fd0fbe1e205255':
        sess_name0 ='1f20835f09e28706'
    elif sess_name == '419c1c6b319d0ddf':
        sess_name0 ='5b92b96313c3fc19'
    else:
        sess_name0 = sess_name

    spk = []
    it = 0
    for i in FilteredSpikes[FilteredSpikes['session_name'] == sess_name]['filtered_spikes']:
        spk.append(i)
        it += 1
    spk = np.array(spk)

    cell_id = np.array(FilteredSpikes[FilteredSpikes['session_name'] == sess_name0]['cell_id'])
    MEC_cellids = np.array(MECcells[MECcells['session_name']==sess_name0]['cell_id'])
    filtered_cellids = np.array(FilteredCells[FilteredCells['session_name']==sess_name0]['cell_id'])
    gs_id = np.array(GridScore[(GridScore['session_name'] == sess_name0) &
              (GridScore['signal_type'] == 'spikes') & 
              (GridScore['map_params_id'] == 'B')
             ]['cell_id'])
    del FilteredSpikes


    MEC_bool = np.zeros(len(cell_id), dtype = bool)
    for it, i in enumerate(cell_id):
        MEC_bool[it] = i in MEC_cellids

    filtered_bool = np.zeros(len(cell_id), dtype = bool)
    for it, i in enumerate(cell_id):
        filtered_bool[it] = i in filtered_cellids

    gs_bool = np.zeros(len(cell_id), dtype = bool)
    for it, i in enumerate(cell_id):
        gs_bool[it] = i in gs_id

    OFs = ('59825ec5641c94b4','c43d9bd004db772b','9190f2fccd52497e',
        '7e888f1d8eaab46b', '1f20835f09e28706', '5b92b96313c3fc19',)

    Ws = ('d5a06b6a7630bb11','26fd0fbe1e205255','419c1c6b319d0ddf',)
    if sess_name in OFs:
        xx0, yy0, speed_all, hd0, tt0 = [], [], [], [], []
        for i in Tracking_OpenField[Tracking_OpenField['session_name'] == sess_name]['x_pos']:
            xx0.extend([i])
        xx0 = np.array(xx0[0])
        for i in Tracking_OpenField[Tracking_OpenField['session_name'] == sess_name]['y_pos']:
            yy0.extend([i])
        yy0 = np.array(yy0[0])
        for i in Tracking_OpenField[Tracking_OpenField['session_name'] == sess_name]['speed']:
            speed_all.extend([i])
        speed_all = np.array(speed_all[0])
        for i in Tracking_OpenField[Tracking_OpenField['session_name'] == sess_name]['head_angle']:
            hd0.extend([i])
        hd0 = np.array(hd0[0])
        for i in Tracking_OpenField[Tracking_OpenField['session_name'] == sess_name]['timestamps']:
            tt0.extend([i])
        tt0 = np.array(tt0[0])
        times = np.round(np.linspace(0, len(speed_all)-1, len(spk[0,:]))).astype(int)
        headpos = np.concatenate((xx0[:,np.newaxis],yy0[:,np.newaxis]),1)[times, :]
    elif sess_name in Ws:
        pos, pos1, pos2, speed_all, hd, tt = [], [], [], [], [], []
        for i in Tracking_Linear[Tracking_Linear['session_name'] == sess_name]['pos']:
            pos.extend([i])
        pos = np.array(pos[0])    
        for i in Tracking_Linear[Tracking_Linear['session_name'] == sess_name]['lap_pos']:
            pos1.extend([i])
        pos1 = np.array(pos1[0])    
        for i in Tracking_Linear[Tracking_Linear['session_name'] == sess_name]['rel_pos']:
            pos2.extend([i])
        pos2 = np.array(pos2[0])    
        for i in Tracking_Linear[Tracking_Linear['session_name'] == sess_name]['speed']:
            speed_all.extend([i])
        speed_all = np.array(speed_all[0])
        for i in Tracking_Linear[Tracking_Linear['session_name'] == sess_name]['timestamps']:
            tt.extend([i])
        tt = np.array(tt[0])
        times = np.round(np.linspace(0, len(speed_all)-1, len(spk[0,:]))).astype(int)
        headpos = pos1[:,np.newaxis][times,:]


    acorrs = np.array(GridScore[(GridScore['session_name'] == sess_name0) &
              (GridScore['signal_type'] == 'spikes') & 
              (GridScore['map_params_id'] == 'B')
             ]['acorr'])

    acorrs_all = np.zeros((len(acorrs), len(acorrs[0])**2))
    for i in range(len(acorrs)):
        acorrs_all[i,:] = acorrs[i].flatten()

    MEC_bool = np.ones(len(filtered_bool), dtype = bool)
    acorrs_all = acorrs_all[filtered_bool & gs_bool & MEC_bool,:]

    dd = squareform(pdist(preprocessing.scale(acorrs_all[:,:],axis = 1), metric = 'cityblock'))
    ################### Get crosscorr stats ####################

    agg = AgglomerativeClustering(n_clusters=None,affinity='precomputed', linkage='average', distance_threshold=2000)
    ind = agg.fit(dd).labels_    

    xy_all = {}
    movetimes_all = {}
    spk_all = {}
    coords_all = {}

    sp = 100
    ii = np.argmax(np.bincount(ind))
    sig = 2

    tsname = Session[Session['session_name']==sess_name]['timeseries_name']
    tsname = np.array(tsname)[0]
    indscurr = np.where(filtered_bool & gs_bool & MEC_bool)[0][ind == ii]
    sspikes = spk[indscurr,:].T
    sspikes[np.isnan(sspikes)] = 0
    spksum = np.mean(sspikes,0)
    indssort = np.where((spksum>0))[0]
    movetimes0 = np.where((speed_all[times]>sp))[0]  
    sspikes = sspikes[:, indssort]
    sspk1 = gaussian_filter1d(sspikes,sigma = sig, axis = 0)[movetimes0,:]
    sspk1 = np.sqrt(sspk1)
    spknull = np.sum(sspk1,1)>0
    sspk1 = sspk1[spknull,:]
    movetimes0 = movetimes0[spknull]
    coords_all = {}
    sspk2 = gaussian_filter1d(sspikes,sigma = sig, axis = 0)
    sspk2 = np.sqrt(sspk2)
    if bTor: 
        coords_f = glob.glob(data_dir + '/' + mouse + '/' + sess + '_coords_all.npz')
        if len(coords_f)>0:
            f = np.load(coords_f[0], allow_pickle = True)
            coords_all = f['coords_all'][()]
            f.close()
        else:

            f = np.load(data_dir + '/' + mouse + '/' + sess + '_decoding_data.npz', allow_pickle = True)
            coords_ds = f['coords_ds']
            indstemp = f['indstemp']
            f.close()
            num_neurons = len(sspk1[0,:])
            for nn in range(num_neurons):
                inds = np.ones(num_neurons, dtype = bool)
                inds[nn] = False
                coords_all[nn] = get_coords_all(sspk1[:,inds], coords_ds, 
                                                np.arange(len(sspk1)), indstemp, dim = sum(inds), 
                                                bPred = False, bPCA = False, spk2 = sspk2[:, inds])
            np.savez(data_dir + '/' + mouse + '/' + sess + '_coords_all.npz', coords_all = coords_all)
    headpos = headpos[:, :]
    return sspikes, headpos, coords_all

def run_GLM(mouse, sess, data_dir, 
            num_bins_all = [15], LAM_all = [0], cv_folds = 1,
            bTor = True, GoGaussian = False, GoBernoulli = False, bSess = False):
    spk, pos, coords_all = load_glm_data(mouse, sess, data_dir, bTor)
    num_times, num_neurons = np.shape(spk)
    GLMscores = np.zeros((num_neurons, len(num_bins_all), len(LAM_all)))
    covar = pos.copy()
    periodicprior = False
    for i1, num_bins in enumerate(num_bins_all):
        for i2, LAM in enumerate(LAM_all):            
            for n in np.arange(0, num_neurons, 1): 
                if bTor:
                    periodicprior = True
                    covar = coords_all[n][:, :2] 
                spkcurr = spk[:,n].copy()
                spkmean = np.mean(spkcurr)
                spkcurr[spkcurr<spkmean] = 0                             
                spkcurr[spkcurr>spkmean] = 1
                (__, __, GLMscores[n, i1, i2]) = glm(covar,
                                                    spkcurr, 
                                                    num_bins, 
                                                    GoGaussian,
                                                    GoBernoulli, 
                                                    cv_folds, 
                                                    LAM)
    return GLMscores

mouse = ''
sess = sys.argv[1]

data_dir = 'functional' 

OFs = ('59825ec5641c94b4','c43d9bd004db772b','9190f2fccd52497e',
    '7e888f1d8eaab46b', '1f20835f09e28706', '5b92b96313c3fc19',)
Ws = ('d5a06b6a7630bb11','26fd0fbe1e205255','419c1c6b319d0ddf',)

numbins_all = [10,]
LAMs = [1, ]
GoGaussian = False
GoBernoulli = True
tor3 = run_GLM(mouse, sess, data_dir, numbins_all, LAMs, 
    cv_folds = 3, bTor = True, GoGaussian = GoGaussian, GoBernoulli = GoBernoulli)
if sess in OFs:
    space3 = run_GLM(mouse, sess, data_dir, numbins_all, LAMs, 
      cv_folds = 3, bTor = False, GoGaussian = GoGaussian, GoBernoulli = GoBernoulli)
else:
    space3 = np.zeros_like(tor3)
data_dir = 'Horst' 
np.savez(data_dir + '/' + sess + '_glm', 
    space3 = space3, tor3 = tor3)
