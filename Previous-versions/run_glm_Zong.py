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
    NAT = h5py.File(data_dir + '/' + mouse + '/' + sess + '/' + 'NAT.mat')
    nat_all = NAT[NAT['NAT'][()][0][0]][()]
    filtered_events = nat_all[np.arange(15,len(nat_all), 4),:]
    tt = nat_all[0,:]
    headpos = nat_all[1:3,:].T
    headdirection = nat_all[3,:]    
    speed = nat_all[4,:]
    sspikes = np.zeros(np.shape(filtered_events)).T
    for i in range(len(filtered_events)):
        if np.sum(np.isnan(filtered_events[i,:]))== len(tt):
            continue
        if np.isnan(filtered_events[i,0]):
            sspikes[1:-1,i] = interp1d(tt[np.arange(1,len(tt),2)], filtered_events[i,np.arange(1,len(tt),2)])(tt[1:-1])
        else:
            sspikes[1:-1,i] = interp1d(tt[np.arange(0,len(tt),2)], filtered_events[i,np.arange(0,len(tt),2)])(tt[1:-1])    
    sspikes = sspikes[1:-1,:]
    tt = tt[1:-1]
    headpos = headpos[1:-1,:]
    speed = speed[1:-1]
    if mouse == '97046':
        NeuronInformation = h5py.File(data_dir + '/' + mouse + '/' + sess + '/' + 'NeuronInformation.mat')['NeuronInformation']
        repremove = np.ones(len(sspikes[0,:]), dtype = bool)
        repremove[NeuronInformation['RepeatCell'][()][0].astype(int)-1] = False
    else:
        NeuronInformation = sio.loadmat(data_dir + '/' + mouse + '/' + sess + '/' + 'NeuronInformation.mat')['NeuronInformation']
        repremove = np.ones(len(sspikes[0,:]), dtype = bool)
        repremove[NeuronInformation['RepeatCell'][()][0,0][0].astype(int)-1] = False
    sspikes = sspikes[:, repremove]
    sspikes[np.isnan(sspikes)] = 0
    sspikes[sspikes<0.001] = 0
    spksum = np.mean(sspikes,0)
    indssort = np.where((spksum>0) & (spksum<10))   [0]    
    movetimes0 = np.where(speed>5)[0]
    sspk1 = sspikes[:,indssort][movetimes0,:]#,sigma = 1, axis = 0)[:,indssort][movetimes0,:]
    spknull0 = sspk1.sum(0)>0
    sspk1 = sspk1[:, spknull0 ]
    spknull = np.sum(sspk1,1)>0
    sspk1 = sspk1[spknull,:]
    movetimes0 = movetimes0[spknull]
    coords_all = {}
    if bTor:
        coords_f = glob.glob(data_dir + '/' + mouse + '/' + sess + '/coords_all.npz')
        if len(coords_f)>0:
            f = np.load(coords_f[0], allow_pickle = True)
            coords_all = f['coords_all'][()]
            f.close()
        else:

            f = np.load(data_dir + '/' + mouse + '/' + sess + '/decoding_data.npz', allow_pickle = True)
            coords_ds = f['coords_ds']
            indstemp = f['indstemp']
            f.close()
            num_neurons = len(sspk1[0,:])
            for nn in range(num_neurons):
                inds = np.ones(num_neurons, dtype = bool)
                inds[nn] = False
                coords_all[nn] = get_coords_all(sspk1[:,inds], coords_ds, 
                                                np.arange(len(sspk1)), indstemp, dim = sum(inds), 
                                                bPred = False, bPCA = False)
            np.savez(data_dir + '/' + mouse + '/' + sess + '/coords_all.npz', coords_all = coords_all)
    sspikes = sspikes[movetimes0,:][:, indssort]
    headpos = headpos[movetimes0, :]
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

mouse, sess = sys.argv[1], sys.argv[2]

numbins_all = [10,]
LAMs = [1, ]
data_dir = 'weijan' 
GoGaussian = False
GoBernoulli = True
tor3 = run_GLM(mouse, sess, data_dir, numbins_all, LAMs, 
    cv_folds = 3, bTor = True, GoGaussian = GoGaussian, GoBernoulli = GoBernoulli)

space3 = run_GLM(mouse, sess, data_dir, numbins_all, LAMs, 
  cv_folds = 3, bTor = False, GoGaussian = GoGaussian, GoBernoulli = GoBernoulli)

np.savez('weijan/' + mouse + '_' + sess + 'glm_bernoulli', 
    space3 = space3, tor3 = tor3)