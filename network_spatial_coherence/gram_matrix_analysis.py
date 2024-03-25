import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import eigh
from structure_and_args import GraphArgs
from spatial_constant_analysis import *
from utils import read_position_df
import scipy.stats as stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


from numpy import prod, zeros, sqrt
from numpy.random import randn
from scipy.linalg import qr
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from scipy.linalg import svd
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from numpy.linalg import matrix_rank

# plt.style.use(['science', 'nature'])
# # plt.rcParams.update({'font.size': 16, 'font.family': 'serif'})
# font_size = 24
# plt.rcParams.update({'font.size': font_size})
# plt.rcParams['axes.labelsize'] = font_size
# plt.rcParams['axes.titlesize'] = font_size + 6
# plt.rcParams['xtick.labelsize'] = font_size
# plt.rcParams['ytick.labelsize'] = font_size
# plt.rcParams['legend.fontsize'] = font_size - 10

def classical_mds(distance_matrix, dimensions=2):
    """Perform Classical MDS on a given distance matrix."""
    # Number of points
    n = distance_matrix.shape[0]

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Double centered distance matrix
    B = -H.dot(distance_matrix ** 2).dot(H) / 2

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eigh(B, eigvals=(n - dimensions, n - 1))

    # Sorting eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Computing the coordinates
    coordinates = eigenvectors * np.sqrt(eigenvalues)

    return coordinates[:, :dimensions], eigenvalues


def compute_gram_matrix_eigenvalues(distance_matrix):
    """
       Computes the eigenvalues of the Gram matrix derived from a given distance matrix. The Gram matrix is
       calculated based on the distance matrix, which is then used to compute its eigenvalues.

       Args:
           distance_matrix: A numpy array representing the pairwise distances between nodes in a graph,
                            from which the Gram matrix is derived.

       Returns:
           numpy.ndarray: An array of eigenvalues of the Gram matrix.

       Note:
           The function `distance_matrix_to_gram` is used to convert the distance matrix into a Gram matrix
           before computing its eigenvalues. This step might need optimization for large matrices.
       """
    # TODO: increase efficiency for large matrices
    B = distance_matrix_to_gram(distance_matrix)
    eigenvalues = compute_matrix_eigenvalues(B)
    return eigenvalues

def compute_matrix_eigenvalues(matrix):
    eigenvalues, eigenvectors = eigh(matrix)
    idx = eigenvalues.argsort()[::-1]  # sort in descending order
    return eigenvalues[idx]

def distance_matrix_to_gram(distance_matrix):
    n = distance_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -H.dot(distance_matrix ** 2).dot(H) / 2
    return B

def gram_to_distance_matrix(G):
    "Gram matrix to distance matrix transformation"
    norms = np.diag(G)
    distance_matrix_squared = norms[:, np.newaxis] - 2 * G + norms[np.newaxis, :]
    distance_matrix = np.sqrt(np.maximum(distance_matrix_squared, 0))  # Ensure non-negativity
    return distance_matrix

def test_statistic_singular_values(distance_matrix, dim):

    dims = dim
    D = distance_matrix**2
    n = D.shape[0]  # shape of EDM
    # double center EDM to retrive the corresponding Gram matrix
    J = np.eye(n) - (1. / n) * np.ones((n, n))
    G = -0.5 * J.dot(D).dot(J)

    # perform singular value decomposition
    U, S, Vh = np.linalg.svd(G)

    # calculate detection test statistic
    test_statistic = S[dims] * (sum(S[dims:]) / float(len(S[dims:]))) / S[0]  # when < 1 it is good
    return test_statistic

def edm_fde(D, dims, max_faults=None, edm_threshold = 1.0,
            verbose=False):
    """Performs EDM-based fault detection and exclusion (FDE).

    See [1]_ for more detailed explanation of algorithm.

    Parameters
    ----------
    D : np.array
        Euclidean distance matrix (EDM) of shape n x n where n is the
        number of satellites + 1.
    dims : int
        Dimensions of the state space.
    max_faults : int
        Maximum number of faults to exclude (corresponds to fault
        hypothesis). If set to None, then no limit is set.
    edm_threshold : float
        EDM-based FDE thresholding parameter. For an explanation of the
        detection threshold see [1]_.
    verbose : bool
        If true, prints a variety of helpful debugging statements.

    Returns
    -------
    tri : list
        indexes that should be exluded from the measurements

    References
    ----------
    ..  [1] D. Knowles and G. Gao. "Euclidean Distance Matrix-based
        Rapid Fault Detection and Exclusion." ION GNSS+ 2021.

    """

    ri = None                   # index to remove
    tri = []                    # removed indexes (in transmitter frame)
    reci = 0                    # index of the receiver
    oi = np.arange(D.shape[0])  # original indexes

    while True:

        if ri != None:
            if verbose:
                print("removing index: ",ri)

            # add removed index to index list passed back
            tri.append(oi[ri]-1)
            # keep track of original indexes (since deleting)
            oi = np.delete(oi,ri)
            # remove index from EDM
            D = np.delete(D,ri,axis=0)
            D = np.delete(D,ri,axis=1)


        n = D.shape[0]  # shape of EDM

        # stop removing indexes either b/c you need at least four
        # satellites or if maximum number of faults has been reached
        if n <= 5 or (max_faults != None and len(tri) >= max_faults):
            break


        # double center EDM to retrive the corresponding Gram matrix
        J = np.eye(n) - (1./n)*np.ones((n,n))
        G = -0.5*J.dot(D).dot(J)

        # perform singular value decomposition
        U, S, Vh = np.linalg.svd(G)

        # calculate detection test statistic
        warn = S[dims]*(sum(S[dims:])/float(len(S[dims:])))/S[0]
        if verbose:
            print("\nDetection test statistic:",warn)

        if warn > edm_threshold:
            ri = None

            u_mins = set(np.argsort(U[:,dims])[:2])
            u_maxes = set(np.argsort(U[:,dims])[-2:])
            v_mins = set(np.argsort(Vh[dims,:])[:2])
            v_maxes = set(np.argsort(Vh[dims,:])[-2:])

            def test_option(ri_option):
                # remove option
                D_opt = np.delete(D.copy(),ri_option,axis=0)
                D_opt = np.delete(D_opt,ri_option,axis=1)

                # reperform double centering to obtain Gram matrix
                n_opt = D_opt.shape[0]
                J_opt = np.eye(n_opt) - (1./n_opt)*np.ones((n_opt,n_opt))
                G_opt = -0.5*J_opt.dot(D_opt).dot(J_opt)

                # perform singular value decomposition
                _, S_opt, _ = np.linalg.svd(G_opt)

                # calculate detection test statistic
                warn_opt = S_opt[dims]*(sum(S_opt[dims:])/float(len(S_opt[dims:])))/S_opt[0]

                return warn_opt


            # get all potential options
            ri_options = u_mins | v_mins | u_maxes | v_maxes
            # remove the receiver as a potential fault
            ri_options = ri_options - set([reci])
            ri_tested = []
            ri_warns = []

            ui = -1
            while np.argsort(np.abs(U[:,dims]))[ui] in ri_options:
                ri_option = np.argsort(np.abs(U[:,dims]))[ui]

                # calculate test statistic after removing index
                warn_opt = test_option(ri_option)

                # break if test statistic decreased below threshold
                if warn_opt < edm_threshold:
                    ri = ri_option
                    if verbose:
                        print("chosen ri: ", ri)
                    break
                else:
                    ri_tested.append(ri_option)
                    ri_warns.append(warn_opt)
                ui -= 1

            # continue searching set if didn't find index
            if ri == None:
                ri_options_left = list(ri_options - set(ri_tested))

                for ri_option in ri_options_left:
                    warn_opt = test_option(ri_option)

                    if warn_opt < edm_threshold:
                        ri = ri_option
                        if verbose:
                            print("chosen ri: ", ri)
                        break
                    else:
                        ri_tested.append(ri_option)
                        ri_warns.append(warn_opt)

            # if no faults decreased below threshold, then remove the
            # index corresponding to the lowest test statistic value
            if ri == None:
                idx_best = np.argmin(np.array(ri_warns))
                ri = ri_tested[idx_best]
                if verbose:
                    print("chosen ri: ", ri)

        else:
            break

    return tri

def stress_measure(original_distances, embedded_distances):
    """Calculate the stress measure between original and embedded distances."""
    return np.sqrt(np.sum((original_distances - embedded_distances) ** 2) / np.sum(original_distances ** 2))


def analyze_network(args, shortest_path_matrix):
    """Analyze the network to distinguish between 'good' and 'bad' networks."""
    # Perform Classical MDS
    embedded_coords, eigenvalues_dim = classical_mds(shortest_path_matrix, dimensions=args.dim)

    ## No prior knowledge of dimension
    eigenvalues, eigenvectors = compute_gram_matrix_eigenvalues(shortest_path_matrix)

    determine_network_dimension(args, eigenvalues)
    # Calculate embedded distances
    embedded_distances = pdist(embedded_coords, metric='euclidean')
    original_distances = squareform(shortest_path_matrix)

    # Interpretation based on eigenvalues and stress
    positive_eigenvalues = np.sum(eigenvalues_dim > 0)  #TODO: this is biased as it will always be equal to the dimension
    print(f"MDS eigenvalues found {eigenvalues_dim}")
    # Calculate stress measure
    stress = stress_measure(original_distances, embedded_distances)

    print(f"Stress Measure: {stress}")
    print(f"Dimensionality Measure (number of positive eigenvalues): {positive_eigenvalues}")


    if stress < 0.1 and positive_eigenvalues <= 3:  # Thresholds can be adjusted
        print("This appears to be a 'good' network with a low-dimensional, efficient structure")
    else:
        print("This network might be 'bad' or 'pathological' due to high stress or high dimensionality.")

    return embedded_coords

# def determine_network_dimension(eigenvalues):
#     print("eigenvalues", eigenvalues)
#     total_variance = np.sum(eigenvalues)
#     cumulative_variance = np.cumsum(eigenvalues) / total_variance
#     significant_dimension = np.argmax(cumulative_variance >= 0.95) + 1  # Adjust the threshold as needed
#     print(f"Network dimension based on cumulative variance threshold: {significant_dimension}")
#     return significant_dimension


def determine_network_dimension(args, eigenvalues, variance_threshold=0.7):
    """Determine the network dimension based on eigenvalues."""
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    network_dimension = len(positive_eigenvalues)
    cumulative_variance = np.cumsum(positive_eigenvalues) / np.sum(positive_eigenvalues)

    predicted_dim = 0
    # Find the number of dimensions needed to reach the desired variance threshold
    for i, variance_covered in enumerate(cumulative_variance):
        print("dim, variance covered", i, variance_covered)
        if variance_covered >= variance_threshold and predicted_dim == 0:
            print(f"Network dimension based on cumulative variance threshold: {i+1}")
            predicted_dim = i + 1

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(positive_eigenvalues) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.axvline(x=predicted_dim, color='g', linestyle='--')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Cumulative Variance Covered')
    plt.title('Cumulative Variance by Eigenvalues')
    plt.legend(['Cumulative Variance', 'Variance Threshold', 'Determined Dimension'])

    plt.xscale('log')
    plt.ylim(0, 1)

    plot_folder = args.directory_map['mds_dim']
    plt.savefig(f'{plot_folder}/mds_dim_eigenvectors_{args.args_title}.svg')
    if args.show_plots:
        plt.show()
    plt.close()

    return len(positive_eigenvalues)  # If threshold not met, return all positive dimensions


def matrix_rank(A, tol=None):
    U, S, V = np.linalg.svd(A, full_matrices=False)
    if tol is None:
        tol = S.max() * max(A.shape) * np.finfo(S.dtype).eps
    rank = (S > tol).sum()
    # print("S", S)
    print("rank", rank)
    return rank

def compute_correlation_between_distance_matrices(matrix1, matrix2):
    """
    Compute the Pearson correlation coefficient between two distance matrices.

    :param matrix1: First distance matrix.
    :param matrix2: Second distance matrix.
    :return: Pearson correlation coefficient.
    """
    # Flatten the matrices
    flat_matrix1 = matrix1.flatten()
    flat_matrix2 = matrix2.flatten()

    # Compute Pearson correlation
    correlation, _ = stats.pearsonr(flat_matrix1, flat_matrix2)
    return correlation


def godec(X, rank=1, card=None, iterated_power=1, max_iter=100, tol=0.001):
    """
    GoDec - Go Decomposition (Tianyi Zhou and Dacheng Tao, 2011)

    The algorithm estimate the low-rank part L and the sparse part S of a matrix X = L + S + G with noise G.

    Parameters
    ----------
    X : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S
        and a low-rank matrix L.

    rank : int >= 1, optional
        The rank of low-rank matrix. The default is 1.

    card : int >= 0, optional
        The cardinality of the sparse matrix. The default is None (number of array elements in X).

    iterated_power : int >= 1, optional
        Number of iterations for the power method, increasing it lead to better accuracy and more time cost. The default is 1.

    max_iter : int >= 0, optional
        Maximum number of iterations to be run. The default is 100.

    tol : float >= 0, optional
        Tolerance for stopping criteria. The default is 0.001.

    Returns
    -------
    L : array-like, low-rank matrix.

    S : array-like, sparse matrix.

    LS : array-like, reconstruction matrix.

    RMSE : root-mean-square error.

    References
    ----------
    Zhou, T. and Tao, D. "GoDec: Randomized Lo-rank & Sparse Matrix Decomposition in Noisy Case", ICML 2011.
    """
    iter = 1
    RMSE = []
    card = prod(X.shape) if card is None else card

    X = X.T if (X.shape[0] < X.shape[1]) else X
    m, n = X.shape

    # Initialization of L and S
    L = X
    S = zeros(X.shape)
    LS = zeros(X.shape)

    while True:
        # Update of L
        Y2 = randn(n, rank)
        for i in range(iterated_power):
            Y1 = L.dot(Y2)
            Y2 = L.T.dot(Y1)
        Q, R = qr(Y2, mode='economic')
        L_new = (L.dot(Q)).dot(Q.T)

        # Update of S
        T = L - L_new + S
        L = L_new
        T_vec = T.reshape(-1)
        S_vec = S.reshape(-1)
        idx = abs(T_vec).argsort()[::-1]
        S_vec[idx[:card]] = T_vec[idx[:card]]
        S = S_vec.reshape(S.shape)

        # Reconstruction
        LS = L + S

        # Stopping criteria
        error = sqrt(mean_squared_error(X, LS))
        RMSE.append(error)

        print("iter: ", iter, "error: ", error)
        if (error <= tol) or (iter >= max_iter):
            break
        else:
            iter = iter + 1

    return L, S, LS, RMSE

def denoise_distmat(D, dim, p=2):
    if p == 2:
        # Perform SVD
        U, S, Vt = svd(D)
        # Low-rank approximation
        return lowrankapprox(U, S, Vt, dim+2)
    elif p == 1:
        # Placeholder for rpca; in practice, you should use an rpca implementation here
        # This example will just use PCA as a simple stand-in for the concept
        pca = PCA(n_components=dim+2, svd_solver='full')
        D_denoised = pca.fit_transform(D)
        D_reconstructed = pca.inverse_transform(D_denoised)
        return D_reconstructed
    else:
        raise ValueError("p must be 1 or 2")

def lowrankapprox(U, S, Vt, r):
    # Use only the first r singular values to reconstruct the matrix
    S_r = np.zeros((r, r))
    np.fill_diagonal(S_r, S[:r])
    return U[:, :r] @ S_r @ Vt[:r, :]


def edm_test_statistic(args, matrix, d, variance_threshold=0.99, similarity_threshold=0.1, near_zero_threshold=1e-5,
                       original=False):
    # Step 1: Compute SVD
    U, S, Vt = np.linalg.svd(matrix)

    # Step 2: Variance Captured by First d Singular Values
    total_variance = np.sum(S ** 2)
    variance_first_d = np.sum(S[:d] ** 2) / total_variance
    variance_check = variance_first_d >= variance_threshold

    # Step 3: Similarity Among First d Singular Values
    cv_first_d = np.std(S[:d]) / np.mean(S[:d])
    similarity_check = cv_first_d <= similarity_threshold

    # Step 4: Near-Zero Check for Remaining Singular Values
    near_zero_check = np.all(S[d:] < near_zero_threshold)

    # Combine Checks into a Single Score or Metric
    # Here, we simply return a boolean for simplicity, but you could design a more nuanced scoring system
    plot_cumulative_eigenvalue_contribution(args, singular_values=S, dimension=d, original=original)

    # Important here would be : variance_first_d   AND difference between the 3
    return variance_check and similarity_check and near_zero_check

def plot_cumulative_eigenvalue_contribution(args, eigenvalues, original, first_10_eigenvalues=False):
    """
    Plots the cumulative contribution of eigenvalues to the total variance and saves the plot. It can also
    display the contribution of just the first 10 eigenvalues if specified.

    Args:
        args: An object containing configuration parameters, including the expected dimensionality of the graph
              (`dim`) and directory mappings (`directory_map`) for saving plots.
        eigenvalues: An array of eigenvalues whose contributions are to be plotted.
        original (bool): Flag indicating whether the original Euclidean distances are used. Affects plot titling.
        first_10_eigenvalues (bool): If True, only the first 10 eigenvalues are considered in the plot. Defaults to False.

    Returns:
        float: The cumulative variance explained by the first `d` eigenvalues, where `d` is the dimensionality
               specified in `args.dim`.

    Note:
        The plot illustrates both individual and cumulative variance contributions of the eigenvalues, highlighting
        the significance of the first `d` dimensions. The plot is saved in the directory specified by
        `args.directory_map['mds_dim']`, with the filename reflecting whether it's based on original distances or
        the shortest path matrix, and whether it's limited to the first 10 eigenvalues.
    """
    d = args.dim
    S = eigenvalues

    if first_10_eigenvalues:
        S = S[:10]

    total_variance = np.sum(S)
    variance_proportion = S / total_variance
    cumulative_variance = np.cumsum(variance_proportion)
    cumulative_variance_first_d_eigenvalues = cumulative_variance[d-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(S) + 1), variance_proportion, alpha=0.7, label='Individual Contribution')
    plt.plot(range(1, len(S) + 1), cumulative_variance, '-o', color='r', label='Cumulative Contribution')
    plt.axvline(x=d, color='g', linestyle='--', label=f'Dimension {d} significance')
    # Display cumulative variance for first d eigenvalues
    plt.text(d, cumulative_variance_first_d_eigenvalues, f'{cumulative_variance_first_d_eigenvalues:.2f}', color='g',
             verticalalignment='bottom')

    plt.xlabel('Eigenvalue Rank')
    plt.ylabel('Eigenvalue Contribution')
    plt.xscale('log')

    plt.legend()
    plot_folder = args.directory_map['mds_dim']
    if original:
        title = "euclidean"
    else:
        title = "sp_matrix"

    if first_10_eigenvalues:
        title = title + "_first_10_eigen"
    plt.savefig(f'{plot_folder}/mds_cumulative_singular_values_{args.args_title}_{title}.svg')

    plot_folder2 = args.directory_map['spatial_coherence']
    plt.savefig(f"{plot_folder2}/gram_matrix_rank_analysis_{args.args_title}_{title}.svg")
    if args.show_plots:
        plt.show()
    plt.close()

    return cumulative_variance_first_d_eigenvalues


def calculate_eigenvalue_metrics(eigenvalues, d, first_10_eigenvalues=False):
    ratios = eigenvalues[1:6] / eigenvalues[0]

    if first_10_eigenvalues:
        eigenvalues = eigenvalues[:10]


    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues / total_variance)
    cumulative_variance_d = cumulative_variance[d - 1]  # d-1 because indexing starts at 0

    return ratios, cumulative_variance_d


def visualize_simulation_results(args, results_list):
    # Initialize lists to store data for plotting
    ratios_data = {'euclidean': [], 'SP_correct': [], 'SP_false': []}
    cumulative_variance_data = {'euclidean': [], 'SP_correct': [], 'SP_false': []}

    # Extract data from results_list
    for result in results_list:
        category = result['category']
        ratios = result['ratios']
        cumulative_variance = result['cumulative_variance_d']

        # Append ratios and cumulative_variance to respective category lists
        ratios_data[category].append(ratios)
        cumulative_variance_data[category].append(cumulative_variance)

    # Plot boxplots for ratios
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    for i, (category, ratios_list) in enumerate(ratios_data.items()):
        for j, ratios in enumerate(ratios_list):
            axs[0, i].boxplot(ratios, positions=[j+1], widths=0.6, patch_artist=True)
            axs[0, i].set_title('Relative Size of Eigenvalues')
            axs[0, i].set_xlabel('Eigenvalue Rank')
            axs[0, i].set_ylabel('Ratio to 1st Eigenvalue')

    # Plot boxplot for cumulative variance
    for i, (category, cumulative_variance_list) in enumerate(cumulative_variance_data.items()):
        axs[1, i].boxplot(cumulative_variance_list, positions=[1], widths=0.6, patch_artist=True)
        axs[1, i].set_title('Cumulative Variance by Dimension')
        axs[1, i].set_xlabel('Cumulative Variance')
        axs[1, i].set_ylabel('Category')

    plt.tight_layout()
    plot_folder = args.directory_map['mds_dim']
    plt.savefig(f'{plot_folder}/several_iterations.svg')
    if args.show_plots:
        plt.show()
    plt.close()


def plot_ratios(args, results_list, categories_to_compare, single_mode=False):
    # Filter results for selected categories
    selected_results = [result for result in results_list if result['category'] in categories_to_compare]

    # Initialize subplots
    num_categories = len(categories_to_compare)
    fig, axs = plt.subplots(1, num_categories, figsize=(6 * num_categories, 6))
    if num_categories == 1:
        axs = [axs]  # Ensure axs is always iterable

    # Extract and plot ratios data for each category
    for i, category in enumerate(categories_to_compare):
        ratios_data = [result['ratios'] for result in selected_results if result['category'] == category]

        # Reorganize ratios data by position
        num_ratios = len(ratios_data[0])  # Assuming all ratios lists have the same length
        ratios_by_position = [[] for _ in range(num_ratios)]
        for ratios_list in ratios_data:
            for j, ratio in enumerate(ratios_list):
                ratios_by_position[j].append(ratio)

        if single_mode:
            # Use bar plots for single data series
            positions = np.arange(1, num_ratios + 1)
            means = [np.mean(ratios) for ratios in ratios_by_position]
            axs[i].bar(positions, means, color='C0', width=0.6)
        else:
            # Use box plots for multiple data series
            positions = np.arange(2, num_ratios + 2)
            for j, ratios in enumerate(ratios_by_position):
                axs[i].boxplot([ratios], positions=[positions[j]], widths=0.6,
                               patch_artist=True, boxprops=dict(facecolor='C{}'.format(j)))

        axs[i].set_title(f'{category}')
        axs[i].set_xlabel('Eigenvalue Rank')
        axs[i].set_ylabel('Ratio to 1st Eigenvalue')
        axs[i].set_ylim(0, 1)  # Set y-axis limit from 0 to 1

    plt.tight_layout()
    plot_folder = args.directory_map['mds_dim']
    plt.savefig(f'{plot_folder}/{"experimental" if single_mode else "several"}_iterations_ratios.svg')
    if args.show_plots:
        plt.show()
    plt.close()



def plot_cumulative_variance(args, results_list, categories_to_compare, single_mode=False):
    # Filter results for selected categories
    selected_results = [result for result in results_list if result['category'] in categories_to_compare]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6 * len(categories_to_compare), 6))

    # Define color map for the plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories_to_compare)))

    if single_mode:
        # In single_mode, plot cumulative variance as bars in a single plot
        bar_positions = np.arange(len(categories_to_compare))
        mean_variances = []

        for i, category in enumerate(categories_to_compare):
            variance_data = [result['cumulative_variance_d'] for result in selected_results if
                             result['category'] == category]
            # Assuming variance_data contains lists of cumulative variances, calculate the mean of each list
            print("VARIANCE DATA SHOULD BE ONLY 1 VALUE", variance_data, category)
            mean_variance = [np.mean(variance) for variance in variance_data]
            mean_variances.append(np.mean(mean_variance))

        ax.bar(bar_positions, mean_variances, color=colors, width=0.6)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(categories_to_compare)

        ax.set_ylabel('D-Eigenvalues Contribution')
        ax.set_ylim(0, 1)  # Set y-axis limit from 0 to 1
    else:
        # For multiple data series per category, use boxplots as before
        for i, category in enumerate(categories_to_compare):
            variance_data = [result['cumulative_variance_d'] for result in selected_results if
                             result['category'] == category]
            ax.boxplot(variance_data, positions=[i], widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor=colors[i]))
        ax.set_title('Cumulative First-D Eigenvalue Contribution')

        ax.set_ylabel('D-Eigenvalues Contribution')
        ax.set_xticklabels(categories_to_compare)
        ax.set_ylim(0, 1)  # Set y-axis limit from 0 to 1

    plt.tight_layout()
    plot_folder = args.directory_map['mds_dim']
    plt.savefig(f'{plot_folder}/{"experimental" if single_mode else "several"}_iterations_cumulative.svg')
    if args.show_plots:
        plt.show()
    plt.close()

def custom_sort(category):
    if category == 'euclidean':
        return (0, 0)
    elif category == 'SP_correct':
        return (1, 0)
    elif category.startswith('SP_false_'):
        num_false_edges = int(category.split('_')[-1])
        return (2, num_false_edges)
def iteration_analysis(num_points_list, proximity_mode_list, intended_av_degree_list, dim_list, false_edges_list, first_10_eigenvalues=False):
    euclidean_eigenvalues_cache = set()  # Cache for memoizing Euclidean eigenvalues
    results_list = []
    args = GraphArgs()  # caution, this is just to get plot folders but specific graph values are default
    args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args
    # Iterate over all combinations of parameters
    for num_points, proximity_mode, dim, false_edges in itertools.product(num_points_list, proximity_mode_list, dim_list, false_edges_list):

        if (num_points, dim) not in euclidean_eigenvalues_cache:  # Write Original Euclidean properties
            intended_av_degree = intended_av_degree_list[0]
            result = perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges,
                                        euclidean=True, first_10_eigenvalues=first_10_eigenvalues)
            results_list.append(result)
            euclidean_eigenvalues_cache.add((num_points, dim))

        if proximity_mode == "delaunay_corrected" or proximity_mode == "lattice":  # Do not compute several proximity graphs when intended degree cannot change
            # For delaunay_corrected, use only the first value in intended_av_degree_list
            intended_av_degree = intended_av_degree_list[0]
            result = perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges, first_10_eigenvalues=first_10_eigenvalues)
            results_list.append(result)

        else:
            for intended_av_degree in intended_av_degree_list:
                print("FALSE EDGES", false_edges)
                result = perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges, first_10_eigenvalues=first_10_eigenvalues)
                results_list.append(result)

    categories_to_compare = list(set(result['category'] for result in results_list))
    categories_to_compare.sort(key=custom_sort)
    plot_ratios(args, results_list, categories_to_compare=categories_to_compare)
    plot_cumulative_variance(args, results_list, categories_to_compare=categories_to_compare)
    # visualize_simulation_results(args, results_list)


def generate_case_nicknames(file_nicknames, weight_specifications):
    case_nicknames = {}
    for filename, nickname in file_nicknames.items():
        # If the file requires weight specifications
        if filename in weight_specifications:
            weights = weight_specifications[filename]
            if isinstance(weights, range):  # If a range is provided
                case_nicknames[filename] = {weight: f"{nickname}_weight={weight}" for weight in weights}
            elif isinstance(weights, list):  # If a specific list of weights is provided
                case_nicknames[filename] = {weight: f"{nickname}_weight={weight}" for weight in weights}
            else:  # If a single weight is provided
                case_nicknames[filename] = {weights: f"{nickname}_weight={weights}"}
        else:  # If no special handling is needed
            case_nicknames[filename] = nickname
    return case_nicknames

def iteration_analysis_experimental(edge_list_and_weights_dict, first_10_eigenvalues=False):
    results_list = []
    args = GraphArgs()  # caution, this is just to get plot folders but specific graph values are default
    args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args
    # Iterate over all combinations of parameters

    file_nicknames = {
        "weinstein_data_corrected_february.csv": "Weinstein",
        "pixelgen_processed_edgelist_Sample04_Raji_Rituximab_treated_cell_3_RCVCMP0001806.csv": "Pixelgen",
        "subgraph_8_nodes_160_edges_179_degree_2.24.pickle": "HL-Simon",
        "subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle": "HL-Erik"
    }
    weight_specifications = {
        "weinstein_data_corrected_february.csv": range(30)  # Example: Generate all spanning weights till 30
    }

    case_nicknames = generate_case_nicknames(file_nicknames, weight_specifications)

    # # Normal SP iteration
    result = perform_simulation(num_points=1000, proximity_mode="knn", intended_av_degree=8, dim=2, false_edges=0)
    results_list.append(result)

    for edge_list_name, weight_list in edge_list_and_weights_dict.items():
        args1 = GraphArgs()  # caution, this is just to get plot folders but specific graph values are default
        args1.directory_map = create_project_structure()  # creates folder and appends the directory map at the args
        args1.edge_list_title = edge_list_name
        args1.proximity_mode = "experimental"



        if weight_list:
            for weight_threshold in weight_list:
                print("WEIGHT THRESHOLD", weight_threshold)
                args1.edge_list_title = edge_list_name
                sparse_graph, _ = load_graph(args1, load_mode='sparse', weight_threshold=weight_threshold)
                sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))
                eigenvalues_sp_matrix = compute_gram_matrix_eigenvalues(distance_matrix=sp_matrix)

                ratios, cumulative_variance_d = calculate_eigenvalue_metrics(eigenvalues_sp_matrix, args1.dim, first_10_eigenvalues=first_10_eigenvalues)

                category = case_nicknames[edge_list_name][weight_threshold]
                result = {'category': category,
                          'ratios': ratios,
                          'cumulative_variance_d': cumulative_variance_d}
                results_list.append(result)

        else:
            if os.path.splitext(args1.edge_list_title)[1] == ".pickle":
                write_nx_graph_to_edge_list_df(args1)  # activate if format is .pickle file

            sparse_graph, _ = load_graph(args1, load_mode='sparse')
            sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))
            eigenvalues_sp_matrix = compute_gram_matrix_eigenvalues(distance_matrix=sp_matrix)
            ratios, cumulative_variance_d = calculate_eigenvalue_metrics(eigenvalues_sp_matrix, args1.dim, first_10_eigenvalues=first_10_eigenvalues)
            category = case_nicknames[edge_list_name]
            result = {'category': category,
                      'ratios': ratios,
                      'cumulative_variance_d': cumulative_variance_d}
            results_list.append(result)

        plot_cumulative_eigenvalue_contribution(args1, eigenvalues=eigenvalues_sp_matrix, original=False)

    categories_to_compare = list(set(result['category'] for result in results_list))
    categories_to_compare = sorted(categories_to_compare)
    plot_ratios(args, results_list, categories_to_compare=categories_to_compare, single_mode=True)
    plot_cumulative_variance(args, results_list, categories_to_compare=categories_to_compare, single_mode=True)



def perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges, euclidean=False, first_10_eigenvalues=False):

    args1 = GraphArgs()
    args1.num_points = num_points
    args1.proximity_mode = proximity_mode
    args1.intended_av_degree = intended_av_degree
    args1.dim = dim
    args1.false_edges_count = false_edges
    if euclidean:
        create_proximity_graph.write_proximity_graph(args=args1)
        original_positions = read_position_df(args=args1)
        original_dist_matrix = compute_distance_matrix(original_positions)
        eigenvalues_euclidean = compute_gram_matrix_eigenvalues(distance_matrix=original_dist_matrix)
        ratios, cumulative_variance_d = calculate_eigenvalue_metrics(eigenvalues_euclidean, dim, first_10_eigenvalues=first_10_eigenvalues)
        category = 'euclidean'

    else:
        create_proximity_graph.write_proximity_graph(args=args1)
        sparse_graph = load_graph(args1, load_mode='sparse')

        if false_edges:
            sparse_graph = add_random_edges_to_csrgraph(sparse_graph, num_edges_to_add=false_edges)

        sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))
        eigenvalues_sp_matrix = compute_gram_matrix_eigenvalues(distance_matrix=sp_matrix)
        ratios, cumulative_variance_d = calculate_eigenvalue_metrics(eigenvalues_sp_matrix, dim, first_10_eigenvalues=first_10_eigenvalues)

        category = "SP_correct" if false_edges == 0 else f"SP_false_{false_edges}"

    result = {'category': category,
              'ratios': ratios,
              'cumulative_variance_d': cumulative_variance_d}

    return result


def calculate_eigenvalue_entropy(eigenvalues):
    probabilities = eigenvalues / np.sum(eigenvalues)
    # probabilities = probabilities[probabilities > 0]  # TODO: careful with this step, as we have negative eigenvalues
    probabilities = np.abs(probabilities)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def eigval_th(D, r):
    """
    Threshold eigenvalues of D, setting all but the largest 'r' to 0.
    """
    # Calculate eigenvalues and eigenvectors
    values, vectors = eigh(D)
    # Set all but the largest 'r' eigenvalues to 0
    values[:-r] = 0
    # Reconstruct the matrix
    D_th = vectors @ np.diag(values) @ vectors.T
    return D_th, {'values': values, 'vectors': vectors}


def rankcomplete_distmat(D, dim, iters=100, tol=1e-6, verbose=True):
    """
    Complete and denoise a Euclidean Distance Matrix (EDM) using OptSpace method.
    """
    assert np.all(np.diag(D) == 0), "The diagonal of D should always be 0"

    n = D.shape[0]
    D2 = D.copy()
    Dold = D.copy()

    for iter in range(1, iters + 1):
        D2, E = eigval_th(D2, dim + 2)
        # Since your matrices are complete, we skip the missing entries part
        np.fill_diagonal(D2, 0)
        D2 = np.maximum(D2, 0)

        # Calculate the change/error
        err = np.sqrt(np.sum((Dold - D2) ** 2)) / np.linalg.norm(D2)
        if verbose:
            print(f"Iter {iter} Change: {err}")
        if err < tol:
            break
        Dold = D2.copy()

    return D2, E


def plot_spectral_gap_and_analyze_negatives(args, eigenvalues):
    # Sort the eigenvalues in descending order

    # Sort the eigenvalues in descending order
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    # Keep only the positive eigenvalues and limit to the first few for analysis
    positive_eigenvalues = eigenvalues_sorted[eigenvalues_sorted > 0][:args.dim + 2]

    # Calculate spectral gap ratios ((eigenvalue_i - eigenvalue_{i+1}) / eigenvalue_i) for positive eigenvalues
    spectral_gaps = (positive_eigenvalues[:-1] - positive_eigenvalues[1:]) / positive_eigenvalues[:-1]

    # Initialize the plot with improved aesthetics
    plt.figure(figsize=(12, 6))
    plt.plot(spectral_gaps, marker='o', linestyle='-', color='#009ADE', linewidth=2, markersize=8,
             markerfacecolor='darkblue', label='Observed Spectral Gaps')

    # Highlight expected spectral gaps for ideal data up to args.dim
    ideal_gaps = np.zeros(args.dim)
    if args.dim > 1:
        ideal_gaps[:args.dim - 1] = 0  # Ideal gaps near 0 for dimensions before args.dim
        ideal_gaps[args.dim - 1] = 1  # Expected large gap at args.dim (indexing from 0)
    # plt.plot(range(args.dim), ideal_gaps, marker='x', linestyle='--', color='red', linewidth=2, markersize=10,
    #          label='Expected for Euclidean Data')
    plt.scatter(args.dim - 1, 1, marker='x', color='red', s=10, label='Expected for Euclidean Data')

    # Enhancing the plot
    plt.title('Spectral Gap Analysis', fontweight='bold')
    plt.xlabel('Eigenvalue Rank', )
    plt.ylabel('Spectral Gap Ratio',)
    plt.xticks()
    plt.yticks()
    plt.legend()




    negative_eigenvalues = eigenvalues_sorted[eigenvalues_sorted < 0]


    if len(positive_eigenvalues) > 0:
        negative_proportion = np.sum(np.abs(negative_eigenvalues)) / np.sum(positive_eigenvalues)
    else:
        negative_proportion = np.inf

    print(f"Proportion of the sum of negative eigenvalues to positive eigenvalues: {negative_proportion:.4f}")

    print("Spectral gap score:", spectral_gaps[args.dim - 1])

    plot_folder2 = args.directory_map['spatial_coherence']
    plt.savefig(f"{plot_folder2}/gram_matrix_spectral_gap_{args.args_title}.svg")
    if args.show_plots:
        plt.show()
    plt.close()
    return spectral_gaps[args.dim - 1]




def plot_gram_matrix_eigenvalues(args, shortest_path_matrix):
    """
    Plots the cumulative eigenvalue contribution of a graph's shortest path matrix after converting it to a Gram matrix.
    It computes the eigenvalues of the Gram matrix derived from the shortest path matrix, then plots and saves the
    cumulative contribution of these eigenvalues to understand the variance explained by the principal components.

    Args:
        args: An object containing configuration parameters and options for the analysis, including
              the expected dimensionality of the graph (`dim`) and directory mappings (`directory_map`)
              for saving plots.
        shortest_path_matrix: A numpy array representing the pairwise shortest path distances between nodes
                              in the graph.

    Returns:
        float: The cumulative variance contribution of the first `d` eigenvalues, where `d` is the dimensionality
               specified in `args.dim`.

    Note:
        This function relies on `compute_gram_matrix_eigenvalues` to compute the eigenvalues of the Gram matrix
        corresponding to the shortest path matrix and `plot_cumulative_eigenvalue_contribution` to generate and save
        a plot of the eigenvalues' cumulative contribution. The plot is saved in the directory specified by
        `args.directory_map['mds_dim']` with a naming convention that reflects the analysis type and graph properties.
    """
    eigenvalues_sp_matrix = compute_gram_matrix_eigenvalues(distance_matrix=shortest_path_matrix)

    # 1 - Contribution
    # # this plots the total contribution with negative eigenvalues
    first_d_values_contribution = plot_cumulative_eigenvalue_contribution(args, eigenvalues=eigenvalues_sp_matrix, original=False)

    # # this plots the contribution of the first 5 eigenvalues
    first_d_values_contribution_5_eigen = plot_gram_matrix_first_eigenvalues_contribution(args, eigenvalues=eigenvalues_sp_matrix)
    print("First d values contribution", first_d_values_contribution)

    # 2 - Spectral Gap
    spectral_gap = plot_spectral_gap_and_analyze_negatives(args, eigenvalues=eigenvalues_sp_matrix)


    # 3 - Negative eigenvalues contribution
    return first_d_values_contribution, first_d_values_contribution_5_eigen, spectral_gap


def plot_gram_matrix_euclidean_and_shortest_path_comparative(args, eigenvalues_euclidean, eigenvalues_shortest_path,
                                                             useful_plot_folder):

    # TODO: so far we consider only the 5st largest positive eigenvalues
    # TODO: I get better results when considering all the eigenvalues, but they include the negative ones
    dim = args.dim
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey='row')  # Share the y-axis across subplots

    for idx, eigenvalues in enumerate([eigenvalues_euclidean, eigenvalues_shortest_path]):
        if idx==0:
            color = '#00CD6C'
        else:
            color = '#009ADE'
        eigenvalues = eigenvalues[eigenvalues > 0][:5]  # keep only positive eigenvalues, up to the first 5
        total_variance = np.sum(eigenvalues)
        variance_proportion = eigenvalues / total_variance
        cumulative_variance = np.cumsum(variance_proportion)

        # For annotation of total contribution at args.dim
        if dim <= len(eigenvalues):
            cumulative_variance_first_d_eigenvalues = cumulative_variance[dim - 1]
        else:
            cumulative_variance_first_d_eigenvalues = cumulative_variance[-1]

        bars = axs[idx].bar(range(1, len(eigenvalues) + 1), variance_proportion * 100, alpha=0.7,
                            label='Individual Contribution', color=color)
        line, = axs[idx].plot(range(1, len(eigenvalues) + 1), cumulative_variance * 100, '-o', color='r',
                              label='Cumulative Contribution')

        # Annotate each bar with its percentage
        for bar, proportion in zip(bars, variance_proportion):
            axs[idx].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{proportion * 100:.1f}%',
                          ha='center', va='bottom')

        # Annotate the plot with the total contribution at args.dim
        axs[idx].text(dim, cumulative_variance_first_d_eigenvalues * 100,
                      f'{cumulative_variance_first_d_eigenvalues * 100:.2f}%', ha='center', va='bottom', color='blue')

        axs[idx].set_xlabel('Eigenvalue Rank')
        if idx == 0:
            axs[idx].set_title('Euclidean')
            axs[idx].set_ylabel('Eigenvalue Contribution (%)')
        else:
            axs[idx].set_title('Network')
        axs[idx].set_xticks(range(1, 6))  # Ensure x-axis labels go from 1 to 5

    handles, labels = axs[0].get_legend_handles_labels()
    # Place a single legend outside the rightmost subplot
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.tight_layout()

    plot_folder2 = args.directory_map['spatial_coherence']
    plt.savefig(f"{plot_folder2}/gram_matrix_comparative_eucl_and_sp_{args.args_title}.svg", format="svg",
                bbox_inches="tight")
    plt.savefig(f"{useful_plot_folder}/gram_matrix_comparative_eucl_and_sp_{args.args_title}.svg", format="svg",
                bbox_inches="tight")
    if args.show_plots:
        plt.show()
    plt.close()

def plot_gram_matrix_first_eigenvalues_contribution(args, eigenvalues):
    dim = args.dim
    color = '#009ADE'
    # Consider only positive eigenvalues, up to the first 5
    eigenvalues = eigenvalues[eigenvalues > 0][:5]
    total_variance = np.sum(eigenvalues)
    variance_proportion = eigenvalues / total_variance
    cumulative_variance = np.cumsum(variance_proportion)
    cumulative_variance_first_d_eigenvalues = cumulative_variance[dim - 1]



    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(range(1, len(eigenvalues) + 1), variance_proportion * 100, alpha=0.7, color=color)
    line, = ax.plot(range(1, len(eigenvalues) + 1), cumulative_variance * 100, '-o', color='r', label=f'Contribution at '
                                                                                                      f'Dim={args.dim}: '
                                                                                                      f'{cumulative_variance_first_d_eigenvalues * 100:.2f}%')

    # Annotate each bar with its percentage
    for bar, proportion in zip(bars, variance_proportion):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{proportion * 100:.1f}%', ha='center', va='bottom')

    # Annotate the plot with the total contribution at specified dimension
    ax.text(dim, cumulative_variance_first_d_eigenvalues * 100, f'{cumulative_variance_first_d_eigenvalues * 100:.2f}%',
            ha='center', va='bottom', color='blue')

    # Draw a vertical line for the spectral gap score and annotate
    if len(eigenvalues) > dim:
        mean_d_eigenvalues_normalized = np.mean(variance_proportion[:dim]) * 100
        d_plus_one_eigenvalue_normalized = variance_proportion[dim] * 100

        gap_score_normalized = (mean_d_eigenvalues_normalized - d_plus_one_eigenvalue_normalized) / mean_d_eigenvalues_normalized

        # Custom legend handle for the arrow

        arrow_handle = Line2D([0], [0], color='purple', marker='>', markersize=10, label='Spectral Gap Score',
                              linestyle='None')
        ax.annotate('', xy=(dim + 0.5, d_plus_one_eigenvalue_normalized), xytext=(dim + 0.5, mean_d_eigenvalues_normalized),
                    arrowprops=dict(arrowstyle="<->", color='purple'), label="Spectral Gap Score")
        ax.text(dim + 0.5, (mean_d_eigenvalues_normalized + d_plus_one_eigenvalue_normalized) / 2, f'{gap_score_normalized:.2f}',
                ha='left', va='center', color='purple', fontsize=9, label='Spectral Gap Score')

        ax.axhline(y=mean_d_eigenvalues_normalized, color='purple', linestyle='--', label="Mean Eigenvalue")

    ax.set_xlabel('Eigenvalue Rank')
    ax.set_ylabel('Eigenvalue Contribution (%)')
    ax.set_xticks(range(1, 6))  # Ensure x-axis labels go from 1 to 5

    # Add custom legend handle
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles.append(arrow_handle)  # Add the custom handle for the arrow
    ax.legend(handles=legend_handles, labels=legend_labels + ['Spectral Gap Score'])

    # ax.legend()

    plt.tight_layout()

    plot_folder2 = args.directory_map['spatial_coherence']
    plt.savefig(f"{plot_folder2}/gram_matrix_first_eigenvalues_contribution{args.args_title}.svg", format="svg",
                bbox_inches="tight")
    if args.show_plots:
        plt.show()
    plt.close()
    return cumulative_variance_first_d_eigenvalues


def make_comparative_gram_matrix_plot_euc_sp(useful_plot_folder):

    ## Setting up the data
    args = GraphArgs()
    args.dim = 2
    args.intended_av_degree = 6
    args.num_points = 1000
    args.false_edges_count = 0
    args.proximity_mode = "knn"
    create_proximity_graph.write_proximity_graph(args)
    sparse_graph = load_graph(args, load_mode='sparse')
    sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))
    original_positions = read_position_df(args=args)
    original_dist_matrix = compute_distance_matrix(original_positions)

    ### Compute Gram Matrix Eigenvalues
    eigenvalues_sp_matrix = compute_gram_matrix_eigenvalues(distance_matrix=sp_matrix)
    eigenvalues_euclidean = compute_gram_matrix_eigenvalues(distance_matrix=original_dist_matrix)
    plot_gram_matrix_euclidean_and_shortest_path_comparative(args, eigenvalues_euclidean, eigenvalues_sp_matrix, useful_plot_folder)



def main():
    # Parameters
    args = GraphArgs()
    args.dim = 2
    args.intended_av_degree = 6
    args.num_points = 1000

    ### Add random edges? See efect in the dimensionality here
    args.false_edges_count = 0

    args.proximity_mode = "knn"


    simulation_or_experiment = "simulation"
    load_mode = 'sparse'
    first_10_eigenvalues = False  # Compute only the 1st 10 eigenvalues for the metrics   #TODO: Not clear if this is the way to go, weird results were sp gets lower values than experimental. Is sparser data improving the score??


    if simulation_or_experiment == "experiment":

        args.edge_list_title = "weinstein_data_corrected_february.csv"
        # args.edge_list_title = "mst_N=1024_dim=2_lattice_k=15.csv"  # Seems to have dimension 1.5

        weighted = True
        weight_threshold = 3

        if os.path.splitext(args.edge_list_title)[1] == ".pickle":
            write_nx_graph_to_edge_list_df(args)  # activate if format is .pickle file

        if not weighted:
            sparse_graph, _ = load_graph(args, load_mode='sparse')
        else:
            sparse_graph, _ = load_graph(args, load_mode='sparse', weight_threshold=weight_threshold)
        # plot_graph_properties(args, igraph_graph_original)  # plots clustering coefficient, degree dist, also stores individual spatial constant...

    elif simulation_or_experiment == "simulation":
        # # # 1 Simulation
        create_proximity_graph.write_proximity_graph(args)
        sparse_graph = load_graph(args, load_mode='sparse')
        ## Original data    edge_list = read_edge_list(args)
        original_positions = read_position_df(args=args)
        # plot_original_or_reconstructed_image(args, image_type="original", edges_df=edge_list)
        original_dist_matrix = compute_distance_matrix(original_positions)
    else:
        raise ValueError("Please input a valid simulation or experiment mode")


    # Simple simulation to test stuff
    num_points_list = [500, 1000, 1500]
    proximity_mode_list = ["knn", "knn_bipartite", "delaunay_corrected"]
    intended_av_degree_list = [6, 10, 15]
    false_edges_list = [0, 2, 5, 10, 50]
    dim_list = [3]

    # # # Iteration analysis simulation
    # iteration_analysis(num_points_list, proximity_mode_list, intended_av_degree_list, dim_list, false_edges_list, first_10_eigenvalues)


    # # # Experimental data iteration
    # edge_names_and_weights_dict = {"weinstein_data_corrected_february.csv": [5,10,15],
    #                                "pixelgen_processed_edgelist_Sample04_Raji_Rituximab_treated_cell_3_RCVCMP0001806.csv": None,
    #                                "subgraph_8_nodes_160_edges_179_degree_2.24.pickle": None,
    #                                "subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle": None}
    # iteration_analysis_experimental(edge_list_and_weights_dict=edge_names_and_weights_dict, first_10_eigenvalues=first_10_eigenvalues)


    ## Only 1 iteration



    # Compute shortest path matrix
    sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))

    eigenvalues_sp_matrix = compute_gram_matrix_eigenvalues(distance_matrix=sp_matrix)
    eigenvalues_euclidean = compute_gram_matrix_eigenvalues(distance_matrix=original_dist_matrix)
    plot_spectral_gap_and_analyze_negatives(args, eigenvalues_euclidean)
    plot_gram_matrix_euclidean_and_shortest_path_comparative(args, eigenvalues_euclidean, eigenvalues_sp_matrix)


    plot_cumulative_eigenvalue_contribution(args, eigenvalues=eigenvalues_sp_matrix, original=False)
    plot_cumulative_eigenvalue_contribution(args, eigenvalues=eigenvalues_euclidean, original=True)
    plot_cumulative_eigenvalue_contribution(args, eigenvalues=eigenvalues_sp_matrix, original=False, first_10_eigenvalues=True)
    plot_cumulative_eigenvalue_contribution(args, eigenvalues=eigenvalues_euclidean, original=True, first_10_eigenvalues=True)


    # entropy_simulation = calculate_eigenvalue_entropy(eigenvalues_sp_matrix)
    # entropy_euclidean = calculate_eigenvalue_entropy(eigenvalues_euclidean)
    #
    # print("entropy euclidean", entropy_euclidean)
    # print("entropy simulation", entropy_simulation)


    ## ----------------------------------------------


    # edm_test_statistic(args=args, matrix=original_dist_matrix, d=args.dim, original=True)
    # edm_test_statistic(args=args, matrix=sp_matrix, d=args.dim, original=False)


    # # Square them for some algorithms
    # sp_matrix = np.square(sp_matrix)
    # original_dist_matrix = np.square(original_dist_matrix)


    # matrix_rank(sp_matrix, tol=0.5)

    # # analyze_network(args, sp_matrix)
    # analyze_network(args, original_dist_matrix)


    # edm_fde(D=sp_matrix, dims=args.dim, verbose=True)  # This is for the statistical test of being an EDM based on the eigenvalues



    # L, S, LS, RMSE = godec(sp_matrix, rank=args.dim+2, card=None)
    #
    #
    # print("shape L", L.shape)
    print("correlation noise", compute_correlation_between_distance_matrices(original_dist_matrix, sp_matrix))
    # print("correlation denoised GoDec full", compute_correlation_between_distance_matrices(original_dist_matrix, LS))
    # print("correlation denoised lowrank", compute_correlation_between_distance_matrices(original_dist_matrix, L))



    # # Initialize NMF
    # model = NMF(n_components=args.dim, init='random', random_state=0)
    # # Fit the model to X
    # W = model.fit_transform(sp_matrix)  # Basis matrix (features)
    # H = model.components_       # Coefficients matrix (components)
    # # Reconstruct the original matrix
    # X_approx = np.dot(W, H)
    # print("correlation denoised NMF", compute_correlation_between_distance_matrices(original_dist_matrix, X_approx))
    #
    # EDM paper algorithms from julia github  #TODO: this is the best denoiser, actually having a positive result
    denoised_mat = denoise_distmat(sp_matrix, dim=args.dim, p=1)
    print("correlation denoised EDM", compute_correlation_between_distance_matrices(original_dist_matrix, denoised_mat))
    #
    #
    # # analyze_network(args, sp_matrix)
    # # print("Denoised MAT")
    # # analyze_network(args, denoised_mat)
    #
    # print("Rank sp_matrix", matrix_rank(sp_matrix))
    # print("Rank denoised MAT", matrix_rank(denoised_mat))
    # print("Rank original MAT", matrix_rank(original_dist_matrix))


    # denoised_mat, values = rankcomplete_distmat(sp_matrix, dim=args.dim, iters=10)  ## This uses SpaceOpt in theory
    # print("correlation SP mat", compute_correlation_between_distance_matrices(original_dist_matrix, sp_matrix))
    # print("correlation denoised EDM", compute_correlation_between_distance_matrices(original_dist_matrix, denoised_mat))

    # eigenvalues_sp_matrix = compute_gram_matrix_eigenvalues(distance_matrix=L)
    # plot_cumulative_eigenvalue_contribution(args, eigenvalues=eigenvalues_sp_matrix, original=False)



    # ## Eigenvalues of the Gram matrix
    # eigenvalues_matrix = compute_matrix_eigenvalues(matrix=L)
    # plot_cumulative_eigenvalue_contribution(args, eigenvalues=eigenvalues_matrix, original=False)

    ## Eigenvalues of the Gram matrix after denoising with the best denoiser (I get things above 1)
    # G_mat = distance_matrix_to_gram(denoised_mat)
    # eigen_G = compute_matrix_eigenvalues(G_mat)
    # plot_cumulative_eigenvalue_contribution(args, eigenvalues=eigen_G, original=False)

if __name__ == '__main__':
    main()