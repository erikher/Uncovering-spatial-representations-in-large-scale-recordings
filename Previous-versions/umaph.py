import glob


from scipy.stats import binned_statistic
from scipy.interpolate import CubicSpline


from mpl_toolkits.mplot3d import Axes3D
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
import time
import numba
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib import animation, cm, transforms, pyplot as plt,  gridspec as grd
from scipy.stats import pearsonr

def pca(data, dim=2):
    if dim < 2:
        return data, [0]
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eig(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dim]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    tot = np.sum(evals)
    var_exp = [(i / tot) * 100 for i in sorted(evals[:dim], reverse=True)]
    components = np.dot(evecs.T, data.T).T
    return components, evecs, evals[:dim]


def sample_denoising(data, k=10, num_sample=500, omega=1, metric='euclidean'):
    n = data.shape[0]

    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists,
                                                    sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    X = result.toarray()
    F = np.sum(X, 1)
    Fs = np.zeros(num_sample)
    Fs[0] = np.max(F)
    i = np.argmax(F)
    inds_all = np.arange(n)
    inds_left = inds_all > -1
    inds_left[i] = False
    inds = np.zeros(num_sample, dtype=int)
    inds[0] = i
    for j in np.arange(1, num_sample):
        F -= omega * X[i, :]
        Fmax = np.argmax(F[inds_left])
        Fs[j] = F[Fmax]
        i = inds_all[inds_left][Fmax]

        inds_left[i] = False
        inds[j] = i
    d = np.zeros((num_sample, num_sample))

    for j, i in enumerate(inds):
        d[j, :] = X[i, inds]
    return inds, d, Fs


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

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


@numba.njit(
    fastmath=True
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=0.0,
                    bandwidth=1.0):
    target = np.log2(k) * bandwidth

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

                else:
                    psum += 1.0

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


def plot_diagrams(
        diagrams,
        plot_only=None,
        title=None,
        xy_range=None,
        labels=None,
        colormap="default",
        size=20,
        ax_color=np.array([0.0, 0.0, 0.0]),
        diagonal=True,
        lifetime=False,
        legend=True,
        show=False,
        ax=None,
        torus_colors=[],
        lw=2.5,
        cs=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

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

    if len(plot_only) > 0:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
    aspect = 'equal'
    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

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
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none",
                   c=c)
        i += 1
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if len(torus_colors) > 0:
        deaths1 = diagrams[1][:, 1]  # the time of death for the 1-dim classes
        deaths1[np.isinf(deaths1)] = 0
        inds1 = np.argsort(deaths1)
        ax.scatter(diagrams[1][inds1[-1], 0], diagrams[1][inds1[-1], 1],
                   10 * size, linewidth=lw, edgecolor=torus_colors[0],
                   facecolor="none")
        ax.scatter(diagrams[1][inds1[-2], 0], diagrams[1][inds1[-2], 1],
                   10 * size, linewidth=lw, edgecolor=torus_colors[1],
                   facecolor="none")

        deaths2 = diagrams[2][:, 1]  # the time of death for the 1-dim classes
        deaths2[np.isinf(deaths2)] = 0
        inds2 = np.argsort(deaths2)
        ax.scatter(diagrams[2][inds2[-1], 0], diagrams[2][inds2[-1], 1],
                   10 * size, linewidth=lw, edgecolor=torus_colors[2],
                   facecolor="none")

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect(aspect, 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="upper right")

    if show is True:
        plt.show()
    return


def get_coords(cocycle, threshold, num_sampled, dists, coeff):
    zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
    cocycle[zint, 2] = cocycle[zint, 2] - coeff
    d = np.zeros((num_sampled, num_sampled))
    d[np.tril_indices(num_sampled)] = np.NaN
    d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
    d[dists > threshold] = np.NaN
    d[dists == 0] = np.NaN
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
    f = lsmr(Aw, Bw)[0] % 1
    return f, verts

def giotto_to_ripser(dgm):
    dgm_ret = []
    for i in range(int(np.max(dgm[:, 2]))+1):
        dgm_ret.append(dgm[dgm[:,2]==i, :2])
    return dgm_ret

####### topological_denoising may be a downsampling/denoising method to compare with
def topological_denoising(data, num_sample = None, num_iters=100, inds=[],  sig = None, w = None, c= None, metric = 'euclidean'):
    n = np.float64(data.shape[0])
    d = data.shape[1]
    if len(inds)==0:
        inds = np.unique(np.floor(np.arange(0,n, n/num_sample)).astype(int))
    else:
        num_sample = len(inds)
    S = data[inds, :] 
    if not sig:
        sig = np.sqrt(np.var(S))
    if not c:
        c = 0.05*max(pdist(S, metric = metric)) 
    if not w:
        w = 0.3

    dF1 = np.zeros((len(inds), d), float)
    dF2 = np.zeros((len(inds), d), float)

    for i in range(num_sample):
        dF1[i, :] = np.dot((data - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], data, metric = metric), 2) / (2*sig)).T)[:, 0]
        dF2[i, :] = np.dot((S - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], S, metric = metric), 2) / (2*sig)).T)[:, 0]

    dF = 1/sig*(1/n * dF1 - (w / num_sample) * dF2)
    M = dF.max()
    for k in range(num_iters):
        S += c*dF/M
        for i in range(num_sample):
            dF1[i, :] = np.dot((data - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], data, metric = metric), 2) / (2*sig)).T)[:, 0]
            dF2[i, :] = np.dot((S - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], S, metric = metric), 2) / (2*sig)).T)[:, 0]
        dF = 1/sig*(1/n * dF1 - (w / num_sample) * dF2)
    data_denoised = S
    return data_denoised, inds


class UMAPH:
    def __init__(
            self,
            pca_dim=6,  # Number of principal components kept
            n_points=1200,  # Number of points to fuzzy downsample to
            k=1000,  # Number of neighbours for "fuzzy downsampling"
            metric='cosine',  # Choose initial choice of metric
            metric_ds='',
            maxdim=1,  # Number of homology dimensions
            coeff=47,
            # Homology coefficients, random prime number chosen, relevant if one suspects "torsion" group in homology
            active_times=10000,
            # Number of points with high mean population activity kept after downsampling
            nbs=800,
            # Number of neighbours in final fuzzy metric computation for homology computation
            bSqrt=False,
            smoothing=3,  # sigma of gaussian filter
            num_times=3,
            # temporal downsampling, keep every "num_times"-th point
            dec_tresh=0.99,
            # ratio of cocycle life  determining the "size"/scale of Vietoris-rips complex higher percentage = more edges/simplices
            ph_classes=[0, 1],
            # compute circular coordinates for top ph_classes cocycles
            compute_h2=False,  # if True, compute H2s with Giotto
            compute_h3=False,  # if True, compute H2s with Giotto
            n_points_h2=None,  # 1024 take 114s with Giotto
            decoding_analysis=True,
            get_tuning_curves=False,
            bPlot_coords = False,
            extrapolate_closest = False,
            extrapolate_nbs = 10,
            bUMA = True,
            verbose=True,
    ):
        self.pca_dim = pca_dim
        self.n_points = n_points
        self.k = k
        self.metric = metric
        if len(metric_ds) == 0:
            self.metric_ds = metric
        else:
            self.metric_ds = metric_ds
        self.maxdim = maxdim
        self.coeff = coeff
        self.active_times = active_times
        self.nbs = nbs
        self.smoothing = smoothing
        self.bSqrt = bSqrt
        self.num_times = num_times
        self.dec_tresh = dec_tresh
        self.ph_classes = ph_classes
        self.compute_h2 = compute_h2
        self.compute_h3 = compute_h3
        self.n_points_h2 = n_points_h2
        self.decoding_analysis = decoding_analysis
        self.good_ind = None
        self.get_tuning_curves = get_tuning_curves
        self.verbose = verbose
        self.bPlot_coords = bPlot_coords
        self.extrapolate_closest = extrapolate_closest
        self.extrapolate_nbs = extrapolate_nbs
        self.bUMA = bUMA

    def fit_transform(self, data, label=None, good_ind=None):
#            assert data.min() >= 0
        self.good_ind = good_ind  # external quality filter (like running ind)
        if label is None:
            self.data = data.copy()
            print('data shape', data.shape)
        else:
            assert len(label) == data.shape[0]
            self.data_all = data.copy()
            self.data = data[label]
            self.data_other = data[np.logical_not(label)]
            print('all data shape', self.data_all.shape)
            print('ensemble data shape', self.data.shape)
            print('other data shape', self.data_other.shape)


        # run UMAPH pipeline
        self.data_prepro, self.data_spikes, self.data_move = self.preprocessing(
            self.data)
        if self.pca_dim > 0:
            self.data_dimred = self.dim_red(self.data_prepro)
        else:
            self.data_dimred = self.data_prepro.copy()
        if self.n_points < len(self.data_dimred):
            self.indstemp, self.data_fuzzy = self.fuzzy_downsample(self.data_dimred)
        else:
            self.indstemp = np.arange(len(self.data_dimred))
            self.data_fuzzy = self.data_dimred.copy()
        if self.bUMA:
            self.distance_matrix = self.uniform_manifold_approximation(self.data_fuzzy)
        else:
            self.distance_matrix = squareform(pdist(self.data_fuzzy, metric = self.metric))
        self.rips_real, self.thresh = self.persistent_homology(
            self.distance_matrix)

        if self.compute_h2:
            print('Computing H2 with %s points (takes approx 2min.)' % self.n_points_h2)
            if self.n_points_h2 is None:
                self.giotto_diagrams = self.giotto_h2(self.distance_matrix)
            else:
                # redo from Fuzzy Downsampling with less points
                original_n_points = self.n_points
                self.n_points = self.n_points_h2
                self.indstemp_h2, self.data_fuzzy_h2 = self.fuzzy_downsample(
                    self.data_dimred)
                self.distance_matrix_h2 = self.uniform_manifold_approximation(
                    self.data_fuzzy_h2)
                self.giotto_diagrams = self.giotto_h2(self.distance_matrix_h2)
                self.n_points = original_n_points

        if self.compute_h3:
            print('Computing H2 with %s points (takes approx 2min.)' % self.n_points_h2)
            self.giotto_diagrams = self.giotto_h3(self.distance_matrix)

        if self.decoding_analysis:
            self.decoded = self.decoding(self.rips_real)
            if self.get_tuning_curves:
                print('Analyze decoding 0')
                self.decoding_tuning_curve_analysis(self.decoded[:, 0])
                print('Analyze decoding 1')
                self.decoding_tuning_curve_analysis(self.decoded[:, 1])

    ################### Preprocessing ####################
    def preprocessing(self, data):
        t0 = time.time()
        if self.bSqrt:
            assert data.min() >= 0
            sspikes1 = np.sqrt(data.T.copy())
        else:
            sspikes1 = data.T.copy()
        if self.smoothing>0:
            sspikes1 = gaussian_filter1d(sspikes1, self.smoothing, axis=0)
        if self.good_ind is not None:  # external quality filter
            sspikes1 = sspikes1[self.good_ind, :]
            print('external quality filter down to', sspikes1.shape)
        times_cube = np.arange(0, len(sspikes1[:, 0]), self.num_times)
        movetimes = np.sort(np.argsort(np.sum(
            sspikes1[times_cube, :], 1))[-self.active_times:])
        movetimes = times_cube[movetimes]
        scaled = preprocessing.scale(sspikes1[movetimes, :])
        print('after Preprocessing data shape', scaled.shape)
        print('Preprocessing', 'took', time.time() - t0)
        return scaled, sspikes1, movetimes

    ################### Dimension reduce ####################
    def dim_red(self, scaled):
        t0 = time.time()
        dim_red_spikes_move_scaled, _, e1 = pca(scaled, dim=self.pca_dim)
        print('Eigenvalues', e1)
        print('after PCA data shape', dim_red_spikes_move_scaled.shape)
        print('Dimension reduce', 'took', time.time() - t0)
        return dim_red_spikes_move_scaled

    ################### Fuzzy Downsampling ####################
    def fuzzy_downsample(self, dim_red_spikes_move_scaled):
        t0 = time.time()
        indstemp, dd, fs = sample_denoising(
            dim_red_spikes_move_scaled, self.k, self.n_points,
            metric=self.metric_ds)
        dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp, :]
        print('after Fuzzy Downsampling data shape',
              dim_red_spikes_move_scaled.shape)
        print('Fuzzy Downsampling', 'took', time.time() - t0)
        return indstemp, dim_red_spikes_move_scaled

    ################### UMA ####################
    def uniform_manifold_approximation(self, dim_red_spikes_move_scaled):
        t0 = time.time()
        X = squareform(pdist(dim_red_spikes_move_scaled, self.metric))
        knn_indices = np.argsort(X)[:, :self.nbs]
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
        # It's possible to force connectivity between neighbours
        sigmas, rhos = smooth_knn_dist(
            knn_dists, self.nbs, local_connectivity=0)
        rows, cols, vals = compute_membership_strengths(
            knn_indices, knn_dists, sigmas, rhos)
        result = coo_matrix((vals, (rows, cols)),
                            shape=(X.shape[0], X.shape[0]))
        result.eliminate_zeros()
        transpose = result.transpose()
        prod_matrix = result.multiply(transpose)
        result = (result + transpose - prod_matrix)
        result.eliminate_zeros()
        d = result.toarray()
        d = - np.log(d)
        np.fill_diagonal(d, 0)
        if self.verbose:
            print('distance_matrix', d.shape)
            print('UMA', 'took', time.time() - t0)
        return d

    ################### PH ####################
    def persistent_homology(self, d):
        t0 = time.time()
        thresh = np.max(d[~np.isinf(d)])
        rips_real = ripser(
            d, maxdim=self.maxdim, coeff=self.coeff, do_cocycles=True,
            distance_matrix=True, thresh=thresh)
        if self.verbose:
            print('PH lifetime thresh', thresh)
            plt.figure()
            plot_diagrams(rips_real["dgms"], plot_only=np.arange(self.maxdim + 1),
                lifetime=True)
            plt.show()
            print('PH', 'took', time.time() - t0)
        return rips_real, thresh

    ################### Giotto? ####################
    def giotto_h2(self, d):
        t0 = time.time()
        VR = VietorisRipsPersistence(
            homology_dimensions=[0, 1, 2],
            metric='precomputed',
            coeff=self.coeff,
            max_edge_length=self.thresh,
            collapse_edges=False,  # True faster?
            n_jobs=None  # -1 faster?
        )
        diagrams = VR.fit_transform([d])
        fig = plot_diagram(diagrams[0])
        fig.show()
        print('Giotto', 'took', time.time() - t0)
        return diagrams

    def giotto_h3(self, d):
        t0 = time.time()
        VR = VietorisRipsPersistence(
            homology_dimensions=[0, 1, 2, 3],
            metric='precomputed',
            coeff=self.coeff,
            max_edge_length=self.thresh,
            collapse_edges=False,  # True faster?
            n_jobs=None  # -1 faster?
        )
        diagrams = VR.fit_transform([d])
        fig = plot_diagram(diagrams[0])
        fig.show()
        print('Giotto', 'took', time.time() - t0)
        return diagrams

    ################### Decoding ####################
    def decoding(self, rips_real):
        t0 = time.time()
        diagrams = rips_real[
            "dgms"]  # the multiset describing the lives of the persistence classes
        cocycles = rips_real["cocycles"][
            1]  # the cocycle representatives for the 1-dim classes
        dists_land = rips_real[
            "dperm2all"]  # the pairwise distance between the points
        births1 = diagrams[1][:, 0]  # the time of birth for the 1-dim classes
        deaths1 = diagrams[1][:, 1]  # the time of death for the 1-dim classes
        lives1 = deaths1 - births1  # the lifetime for the 1-dim classes
        iMax = np.argsort(lives1)
        num_circ = len(self.ph_classes)
        coords1 = np.zeros((num_circ, len(self.indstemp)))
        coords1[:] = np.nan
        if np.max(self.ph_classes) > len(lives1):
            print('There are not enough 1-dimensional cohomology classes')
            return []
        for i, c in enumerate(self.ph_classes):
            cocycle = cocycles[iMax[-(c + 1)]]
            threshold = births1[iMax[-(c + 1)]] + (
                        deaths1[iMax[-(c + 1)]] - births1[
                    iMax[-(c + 1)]]) * self.dec_tresh
            if np.isinf(threshold):
                print('Threshold is infinite, using birth*1.5 as threshold')
                threshold = births1[iMax[-(c + 1)]]*2
            coordstemp, inds = get_coords(
                cocycle, threshold, len(self.indstemp), self.distance_matrix, self.coeff)
            coords1[i, inds] = coordstemp
            if len(self.data_fuzzy[0,:])>2:
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(self.data_fuzzy[:, 0],
                           self.data_fuzzy[:, 1],
                           self.data_fuzzy[:, 2],
                           c=np.cos(2 * np.pi * coords1[i, :]))
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(self.data_fuzzy[:, 0],
                           self.data_fuzzy[:, 1],
                           c=np.cos(2 * np.pi * coords1[i, :]))

            plt.show()

            plt.plot(coords1[i, np.argsort(self.data_move[self.indstemp])[:100]], '.-')

            plt.title('Cycle %s, temporal continuity?' % i)
            plt.show()
        self.coords1 = coords1
        if len(self.data_spikes[:,0]) == len(coords1[0,:]):
            return coords1.T
        else:

            return self.extrapolate_coords(self.data_spikes, t0)


        return coords_d
    def extrapolate_coords(self, spk, t0 = -1):
        if t0 < 0:
            t0 = time.time()
        num_circ = len(self.ph_classes)
        if self.extrapolate_closest:
            num_tot = len(spk[:,0])
            coords_d = np.zeros((num_tot, num_circ))
            num_batch = 20000 # Number of points to compute distance of simultaneously 
            for c in range(num_circ):
                circ_coord_dist = np.zeros(num_tot)
                circ_coord_tmp = self.coords1[c,:]*2*np.pi
                j = -1
                for j in range(int(num_tot/num_batch)):
                    dist_landmarks = cdist(spk[j*num_batch:(j+1)*num_batch, :],
                                           self.data_spikes[self.data_move[self.indstemp], :], 
                                           metric = self.metric)
                    closest_landmark = np.argsort(dist_landmarks, 1)[:,:self.extrapolate_nbs]
                    weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,-1]] for i in range(num_batch)])
                    nans = np.where(np.sum(weights,1)==0)[0]
                    if len(nans)>0:
                        weights[nans,:] = 1
                    weights /= np.sum(weights, 1)[:,np.newaxis] 

                    sincirc = [np.dot(np.sin(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(num_batch)]
                    coscirc = [np.dot(np.cos(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(num_batch)]
                    coords_d[j*num_batch:(j+1)*num_batch,c] = np.arctan2(sincirc, coscirc)%(2*np.pi)
                dist_landmarks = cdist(spk[(j+1)*num_batch:, :], self.data_spikes[self.data_move[self.indstemp], :], metric = self.metric)
                closest_landmark = np.argsort(dist_landmarks, 1)[:,:self.extrapolate_nbs]
                lenrest = len(closest_landmark[:,0])
                weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,-1]] for i in range(lenrest)])
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
                coords_d[(j+1)*num_batch:, c] = np.arctan2(sincirc, coscirc)%(2*np.pi)
        else:
            num_neurons = len(self.data_spikes[0, :])
            centcosall = np.zeros((num_neurons, num_circ, len(self.indstemp)))
            centsinall = np.zeros((num_neurons, num_circ, len(self.indstemp)))
            dspk = preprocessing.scale(self.data_spikes[self.data_move[self.indstemp], :])
            for neurid in range(num_neurons):
                spktemp = dspk[:, neurid].copy()
                centcosall[neurid, :, :] = np.multiply(np.cos(self.coords1[:, :] * 2 * np.pi), spktemp)
                centsinall[neurid, :, :] = np.multiply(np.sin(self.coords1[:, :] * 2 * np.pi), spktemp)
            plot_times = np.arange(0, len(spk[:,0]), 1)
            dspk1 = preprocessing.scale(spk[plot_times, :])
            a = np.zeros((len(plot_times), num_circ, num_neurons))

            for n in range(num_neurons):
                a[:, :, n] = np.multiply(dspk1[:, n:n + 1],
                                         np.sum(centcosall[n, :, :], 1))

            c = np.zeros((len(plot_times), num_circ, num_neurons))
            for n in range(num_neurons):
                c[:, :, n] = np.multiply(dspk1[:, n:n + 1],
                                         np.sum(centsinall[n, :, :], 1))

            mtot2 = np.sum(c, 2)
            mtot1 = np.sum(a, 2)
            coords_d = np.arctan2(mtot2, mtot1) % (2 * np.pi)
        if self.bPlot_coords:
            print('coords_d shape', coords_d.shape)

            N = 50
            plt.plot(coords_d[:N, 0], coords_d[:N, 1], alpha=.1)
            plt.scatter(coords_d[:N, 0], coords_d[:N, 1], c=np.arange(N), s=5)
            plt.colorbar()
            plt.show()

            shift = int(0.5 * coords_d.shape[0])
            plt.plot(coords_d[shift:shift + N, 0], coords_d[shift:shift + N, 1],alpha=.1)
            plt.scatter(coords_d[shift:shift + N, 0],coords_d[shift:shift + N, 1], c=np.arange(N), s=5)
            plt.colorbar()
            plt.show()

            shift = int(0.9 * coords_d.shape[0])
            plt.plot(coords_d[shift:shift + N, 0], coords_d[shift:shift + N, 1],alpha=.1)
            plt.scatter(coords_d[shift:shift + N, 0],coords_d[shift:shift + N, 1], c=np.arange(N), s=5)
            plt.colorbar()
            plt.show()
        print('Decoding', 'took', time.time() - t0)
        return coords_d
    ################### Tuning Curve Analysis ####################
    def get_design_matrix(self, decoding, num_bins=20):
        assert len(decoding.shape) == 1
        bins = np.linspace(
            decoding.min() - 1e-6, decoding.max() + 1e-6, num_bins + 1)
        digital = np.digitize(decoding[:, None], bins) - 1
        plt.bar(*np.unique(digital, return_counts=True))
        plt.title('Distribution over decoded variable')
        plt.show()
        X = np.zeros((digital.shape[0], num_bins))
        X[np.arange(digital.shape[0]), digital[:, 0]] = 1
        X -= np.mean(X, 0, keepdims=True)
        X /= np.linalg.norm(X, axis=0, keepdims=True)
        print('design matrix', X.shape)
        plt.imshow(X, extent=[0, 1, 0, 1])
        plt.show()
        return X

    def least_squares(self, X, responses):
        Y = responses.T.copy()
        Y -= np.mean(Y, 0, keepdims=True)
        Y /= np.std(Y, 0, keepdims=True)
        beta = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(Y)).T
        Y_ = X.dot(beta.T)
        correlations = [pearsonr(y, y_)[0] for y, y_ in zip(Y.T, Y_.T)]
        return beta, correlations

    def plot_tuning_curves(self, beta, corrs, title=''):
        num_show = np.min((int(beta.shape[0] ** .5), 4))
        plt.figure(figsize=(10, 10))
        for i in range(num_show ** 2):
            plt.subplot(num_show, num_show, i + 1)
            plt.plot(beta[i])
            plt.axis('off')
            plt.title(r'$\rho=%.2f$' % corrs[i])
        plt.subplot(num_show, num_show, 1)
        plt.axis('on')
        plt.text(35, 1.7 * beta[0].max(), title)
        plt.xlabel('Decode X')
        plt.ylabel('Response')
        plt.tight_layout()
        plt.show()

    def decoding_tuning_curve_analysis(self, decoding):
        X = self.get_design_matrix(decoding, num_bins=20)

        # In ensemble
        if self.good_ind is not None:
            responses = self.data[:, self.good_ind]
        else:
            responses = self.data
        self.beta_in, self.correlations_in = self.least_squares(X, responses)
        self.plot_tuning_curves(
            self.beta_in, self.correlations_in, 'Tuning Curves (in ensemble)')

        if hasattr(self, 'data_other'):
            # Out ensemble
            if self.good_ind is not None:
                responses = self.data_other[:, self.good_ind]
            else:
                responses = self.data_other
            self.beta_out, self.correlations_out = self.least_squares(
                X, responses)
            self.plot_tuning_curves(self.beta_out, self.correlations_out,
                                    'Tuning Curves (out ensemble)')

            # Compare
            plt.figure(figsize=(4, 4))
            plt.violinplot([self.correlations_in, self.correlations_out])
            plt.xticks((1, 2), ['in ensemble', 'out ensemble'])
            plt.ylabel('Correlation (decoding, response)')
            plt.show()

    def plot_barcode(self, file_name = ''):
        try: #see if giotto has been computed
            persistence = giotto_to_ripser(self.giotto_diagrams[0,:,:])
        except:
            persistence = self.rips_real['dgms']

        diagrams_roll = {}
        filenames=glob.glob('Results/Roll/' + file_name + '_H2_roll_*')
        for i, fname in enumerate(filenames): 
            f = np.load(fname, allow_pickle = True)
            diagrams_roll[i] = list(f['diagrams'])
            f.close() 

        alpha=1
        inf_delta=0.1
        legend=True
        maxdim = len(persistence)-1
        cs = np.repeat([[0,0.55,0.2]],maxdim+1).reshape(3,maxdim+1).T
        colormap=cs
        dims =np.arange(maxdim+1)
        num_rolls = len(diagrams_roll)

        if num_rolls>0:
            diagrams_all = np.copy(diagrams_roll[0])
            for i in np.arange(1,num_rolls):
                for d in dims:
                    diagrams_all[d] = np.concatenate((diagrams_all[d], diagrams_roll[i][d]),0)
            infs = np.isinf(diagrams_all[0])
            diagrams_all[0][infs] = 0
            diagrams_all[0][infs] = np.max(diagrams_all[0])
            infs = np.isinf(diagrams_all[0])
            diagrams_all[0][infs] = 0
            diagrams_all[0][infs] = np.max(diagrams_all[0])


        min_birth, max_death = 0,0     

        for dim in dims:

            persistence_dim = persistence[dim][~np.isinf(persistence[dim][:,1]),:]
            if len(persistence_dim)== 0:
                continue
            min_birth = min(min_birth, np.min(persistence_dim))
            max_death = max(max_death, np.max(persistence_dim))
        delta = (max_death - min_birth) * inf_delta
        infinity = max_death + delta
        axis_start = min_birth - delta            
        plotind = (dims[-1]+1)*100 + 10 +1
        fig = plt.figure()
        gs = grd.GridSpec(len(dims),1)

        indsall =  0
        labels = ["$H_0$", "$H_1$", "$H_2$"]
        for dit, dim in enumerate(dims):
            axes = plt.subplot(gs[dim])
            axes.axis('off')
            d = np.copy(persistence[dim])
            d[np.isinf(d[:,1]),1] = infinity
            dlife = (d[:,1] - d[:,0])
            dinds = np.argsort(dlife)[-30:]
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
            indsall = max(indsall, len(dinds))
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

            axes.plot([0,0], [0, max(indsall,10)], c = 'k', linestyle = '-', lw = 1)
            axes.plot([0,indsall],[0,0], c = 'k', linestyle = '-', lw = 1)
            axes.set_xlim([0, infinity])
            