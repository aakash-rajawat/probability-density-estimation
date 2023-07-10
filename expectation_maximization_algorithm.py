import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from scipy import misc

def getLogLikelihood(means, weights, covariances, X):
    """
    Estimates the Log Likelihood.
    :param means: Mean for each Gaussian KxD
    :param weights: Weight vector 1xK for K Gaussians
    :param covariances: Covariance matrices for each gaussian DxDxK
    :param X: Input data NxD
    :return:
        logLikelihood: log-likelihood
    """
    if len(X.shape) > 1:
        N, D = X.shape
    else:
        N = 1
        D = X.shape[0]

    # get number of gaussians
    K = len(weights)

    logLikelihood = 0
    for i in range(N):  # For each of the data points
        # probability p
        p = 0
        for j in range(K):  # For each of the mixture components
            if N == 1:
                meansDiff = X - means[j]
            else:
                meansDiff = X[i,:] - means[j]

            covariance = covariances[:, :, j].copy()
            norm = 1. / float(((2 * np.pi) ** (float(D) / 2.)) * np.sqrt(np.linalg.det(covariance)))

            p += weights[j] * norm * np.exp(-0.5 * ((meansDiff.T).dot(np.linalg.lstsq(covariance.T, meansDiff.T)[0].T)))
        logLikelihood += np.log(p)
    #####End Subtask#####
    return logLikelihood


def EStep(means, covariances, weights, X):
    """
    Expectation step of the EM Algorithm
    :param means: Mean for each Gaussian KxD
    :param covariances: Weight vector 1xK for K Gaussians
    :param weights: Covariance matrices for each Gaussian DxDxK
    :param X: Input data NxD
    :return:
        logLikelihood  : Log-likelihood (a scalar).
        gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    """
    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    n_training_samples, dim = X.shape
    K = len(weights)

    gamma = np.zeros((n_training_samples, K))
    for i in range(n_training_samples):
        for j in range(K):
            means_diff = X[i] - means[j]
            covariance = covariances[:, :, j].copy()
            norm = 1. / float(((2 * np.pi) ** (float(dim) / 2)) * np.sqrt(np.linalg.det(covariances[:, :, j])))
            gamma[i, j] = weights[j] * norm * np.exp(
                -0.5 * (means_diff.T.dot(np.linalg.lstsq(covariance.T, means_diff.T)[0].T)))
        gamma[i] /= gamma[i].sum()

    return [logLikelihood, gamma]


def MStep(gamma, X):
    """
    Maximization step of the EM Algorithm.
    :param gamma: NxK matrix of responsibilities for N datapoints and K Gaussians.
    :param X: Input data (NxD matrix for N datapoints of dimension D).
    :return:
        logLikelihood  : Log-likelihood (a scalar).
        means          : Mean for each gaussian (KxD).
        weights        : Vector of weights of each gaussian (1xK).
        covariances    : Covariance matrices for each component(DxDxK).
    """
    # Get the sizes
    n_training_samples, dim = X.shape
    K = gamma.shape[1]

    # Create matrices
    means = np.zeros((K, dim))
    covariances = np.zeros((dim, dim, K))

    # Compute the weights
    Nk = gamma.sum(axis=0)
    weights = Nk / n_training_samples

    means = np.divide(gamma.T.dot(X), Nk[:, np.newaxis])

    for i in range(K):
        auxSigma = np.zeros((dim, dim))
        for j in range(n_training_samples):
            meansDiff = X[j] - means[i]
            auxSigma = auxSigma + gamma[j, i] * np.outer(meansDiff.T, meansDiff)
        covariances[:, :, i] = auxSigma/Nk[i]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    return weights, means, covariances, logLikelihood


def regularize_cov(covariance, epsilon):
    """
    Regularize a covariance matrix, by enforcing a minimum
    value on its singular values.
    :param covariance: matrix
    :param epsilon: minimum value for singular values
    :return:
        regularized_cov: reconstructed matrix
    """
    n, m = covariance.shape
    regularized_cov = covariance + epsilon * np.eye(n, m)

    # makes sure matrix is symmetric upto 1e-15 decimal
    regularized_cov = (regularized_cov + regularized_cov.conj().transpose()) / 2

    return regularized_cov


def estGaussMixEM(data, K, n_iters, epsilon):
    """
    EM algorithm for estimation gaussian mixture mode
    :param data: input data, N observations, D dimensional
    :param K: number of mixture components (modes)
    :param n_iters: Number of iterations
    :param epsilon: minimum value for singular values
    :return:
        weights        : mixture weights - P(j) from lecture
        means          : means of gaussians
        covariances    : covariancesariance matrices of gaussians
        logLikelihood  : log-likelihood of the data given the model
    """
    n_dim = data.shape[1]

    # initialize weights and covariances
    weights = np.ones(K) / K
    covariances = np.zeros((n_dim, n_dim, K))

    # Use k-means for initializing the EM-Algorithm.
    # cluster_idx: cluster indices
    # means: cluster centers
    kmeans = KMeans(n_clusters = K, n_init = 10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create initial covariance matrices
    for j in range(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in range(K):
            # compute sum of distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(n_dim) * min_dist

    # Run EM
    for idx in range(n_iters):

        oldLogLi, gamma = EStep(means, covariances, weights, data)
        weights, means, covariances, newLogli = MStep(gamma, data)

        # regularize covariance matrix
        for j in range(K):
            covariances[:, :, j] = regularize_cov(covariances[:, :, j], epsilon)

        # termination criterion
        if abs(oldLogLi - newLogli) < 1:
            break

    return [weights, means, covariances]


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    """
    Skin Color detector
    :param ndata: data for non-skin color
    :param sdata: data for skin-color
    :param K: number of modes
    :param n_iter: number of iterations
    :param epsilon: regularization parameter
    :param theta: threshold
    :param img: input image
    :return:
        result: Result of the detector for every image pixel
    """
    print('creating GMM for non-skin')
    weight_nonskin, means_nonskin, cov_nonskin = estGaussMixEM(ndata, K, n_iter, epsilon)
    print('GMM for non-skin completed')
    print('creating GMM for skin')
    weight_skin, means_skin, cov_skin = estGaussMixEM(sdata, K, n_iter, epsilon)
    print('GMM for skin completed')

    height, width, _ = img.shape

    noSkin = np.ndarray((height, width))
    skin = np.ndarray((height, width))

    for h in range(height):
        for w in range(width):
            noSkin[h, w] = np.exp(
                getLogLikelihood(means_nonskin, weight_nonskin, cov_nonskin, np.array([img[h, w, 0], img[h, w, 1],
                                                                                       img[h, w, 2]])))
            skin[h, w] = np.exp(
                getLogLikelihood(means_skin, weight_skin, cov_skin, np.array([img[h, w, 0], img[h, w, 1],
                                                                              img[h, w, 2]])))

    # calculate ration and threshold
    result = skin / noSkin
    result = np.where(result > theta, 1, 0)
    return result


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def plotModes(means, covMats, X):
    plt.subplot()
    plt.scatter(X[:, 0], X[:, 1])
    M = means.shape[1]

    for i in range(M):
        plotGaussian(means[:, i], covMats[:, :, i])


def plotGaussian(mu, sigma):
    dimension = mu.shape[0]
    if len(mu.shape) > 1:
        n_components = mu.shape[1]
    else:
        n_components = 1
    plt.subplot()
    if dimension == 2:
        if n_components == 1 and sigma.shape == (2, 2):
            n = 36
            phi = np.arange(0, n, 1) / (n-1) * 2 * np.pi
            epoints = np.sqrt(np.abs(sigma)).dot([np.cos(phi), np.sin(phi)]) + mu[:, np.newaxis]
            plt.plot(epoints[0, :], epoints[1, :], 'r')
        else:
            print('ERROR: size mismatch in mu or sigma\n')
    else:
        raise ValueError('Only dimension 2 is implemented.')


epsilon = 0.0001 # regularization
K = 3  # number of desired clusters
n_iter = 5  # number of iterations
skin_n_iter = 5
skin_epsilon = 0.0001
skin_K = 3
theta = 2.0  # threshold for skin detection

print('Question: Expectation Maximization Algorithm for GMMs')

# load datasets
data = [[], [], []]
data[0] = np.loadtxt('data1')
data[1] = np.loadtxt('data2')
data[2] = np.loadtxt('data3')

# test getLogLikelihood
print('(a) testing getLogLikelihood function')
weights = [0.341398243018411, 0.367330235091507, 0.291271521890082]
means = [
    [3.006132088737974,  3.093100568285389],
    [0.196675859954268, -0.034521603109466],
    [-2.957520528756456,  2.991192198151507]
]
covariances = np.zeros((2, 2, 3))
covariances[:, :, 0] = [
    [0.949104844872119, -0.170637132238246],
    [-0.170637132238246,  2.011158266600814]
]
covariances[:, :, 1] = [
    [0.837094104536474, 0.044657749659523],
    [0.044657749659523, 1.327399518241827]
]
covariances[:, :, 2] = [
    [1.160661833073708, 0.058151801834449],
    [0.058151801834449, 0.927437098385088]
]

loglikelihoods = [-1.098653352229586e+03, -1.706951862352565e+03, -1.292882804841197e+03]
for idx in range(3):
    ll = getLogLikelihood(means, weights, covariances, data[idx])
    diff = loglikelihoods[idx] - ll
    print('LogLikelihood is {0}, should be {1}, difference: {2}\n'.format(ll, loglikelihoods[idx], diff))

# test EStep
print('\n')
print('(b) testing EStep function')
# load gamma values
testgamma = [[], [], []]
testgamma[0] = np.loadtxt('gamma1')
testgamma[1] = np.loadtxt('gamma2')
testgamma[2] = np.loadtxt('gamma3')
for idx in range(3):
    _, gamma = EStep(means, covariances, weights, data[idx])
    absdiff = testgamma[idx] - gamma
    print('Sum of difference of gammas: {0}\n'.format(np.sum(absdiff)))

# test MStep
print('\n')
print('(c) testing MStep function')
# load gamma values
testparams = np.ndarray((3, 3), dtype=object)
# means
testparams[0, 0] = [
    [3.018041988488699,  3.101046000178649],
    [0.198328683921772, -0.019449541135746],
    [-2.964974332415026,  2.994362963328281]
]
testparams[0, 1] = [
    [3.987646604627858, -0.056285481712672],
    [0.064528352867431, -0.046345896337489],
    [-3.244342020825232,  0.164140465045744]
]
testparams[0, 2] = [
    [3.951117305917324, -0.913396187074355],
    [0.121144018117729, -0.040037587868608],
    [-3.054802211026562,  1.969195200268656]
]
# weights
testparams[1, 0] = [0.339408153353897, 0.370303288436004, 0.290288558210099]
testparams[1, 1] = [0.336051939551412, 0.432073585981995, 0.231874474466593]
testparams[1, 2] = [0.257806471569113, 0.379609598797200, 0.362583929633687]
# covariances
testparams[2, 0] = np.ndarray((2, 2, 3))
testparams[2, 0][:, :, 0] = [
    [0.928530520617187, -0.186093601749430],
    [-0.186093601749430,  2.005901936462142]
]
testparams[2, 0][:, :, 1] = [
    [0.838623744823879, 0.045317199218797],
    [0.045317199218797, 1.352200524531750]
]
testparams[2, 0][:, :, 2] = [
    [1.146594581079395, 0.064658231773354],
    [0.064658231773354, 0.925324018684456]
]
testparams[2, 1] = np.ndarray((2, 2, 3))
testparams[2, 1][:, :, 0] = [
    [0.333751473448182, -0.036902134347530],
    [-0.036902134347530,  0.249019229685320]
]
testparams[2, 1][:, :, 1] = [
    [2.790985903869931, 0.180319331359206],
    [0.180319331359206, 0.208102949332177]
]
testparams[2, 1][:, :, 2] = [
    [0.211697922392049, 0.052177894905363],
    [0.052177894905363, 0.221516522642614]
]
testparams[2,2] = np.ndarray((2, 2, 3))
testparams[2,2][:, :, 0] = [
    [0.258550175253901, -0.018706579394884],
    [-0.018706579394884,  0.102719055240694]
]
testparams[2,2][:, :, 1] = [
    [0.467180426168570, -0.153028946058116],
    [-0.153028946058116,  0.657684560660198]
]
testparams[2,2][:, :, 2] = [
    [0.559751011345552, 0.363911891484002],
    [0.363911891484002, 0.442160603656823]
]
for idx in range(3):
    weights, means, covariances, _ = MStep(testgamma[idx], data[idx])
    absmeandiff = abs(means - testparams[0, idx])
    absweightdiff = abs(weights - testparams[1, idx])
    abscovdiff = abs(covariances - testparams[2, idx])

    print('Sum of difference of means:       {0}\n'.format(np.sum(absmeandiff)))
    print('Sum of difference of weights:     {0}\n'.format(np.sum(absweightdiff)))
    print('Sum of difference of covariances: {0}\n'.format(np.sum(abscovdiff)))

# test regularization
print('\n')
print('(c) testing regularization of covariances')
regularized_cov = np.ndarray((2, 2, 3))
regularized_cov[:, :, 0] = [
    [0.938530520617187, -0.186093601749430],
    [-0.186093601749430,  2.015901936462142]
]
regularized_cov[:, :, 1] = [
    [0.848623744823879, 0.045317199218797],
    [0.045317199218797, 1.362200524531750]
]
regularized_cov[:, :, 2] = [
    [1.156594581079395, 0.064658231773354],
    [0.064658231773354, 0.935324018684456]
]
for idx in range(3):
    covariance = regularize_cov(testparams[2, 0][:, :, idx], 0.01)
    absdiff = abs(covariance - regularized_cov[:, :, idx])
    print('Sum of difference of covariances: {0}\n'.format(np.sum(absdiff)))


# compute GMM on all 3 datasets
print('\n')
print('(f) evaluating EM for GMM on all datasets')
for idx in range(3):
    print('evaluating on dataset {0}\n'.format(idx+1))

    # compute GMM
    weights, means, covariances = estGaussMixEM(data[idx], K, n_iter, epsilon)

    # plot result
    plt.subplot()
    plotModes(np.transpose(means), covariances, data[idx])
    plt.title('Data {0}'.format(idx+1))
    plt.show()


# uncomment following lines to generate the result
# for different number of modes k plot the log likelihood for data3
num = 14
logLikelihood = np.zeros(num)
for k in range(num):
    # compute GMM
    weights, means, covariances = estGaussMixEM(data[2], k+1, n_iter, epsilon)
    logLikelihood[k] = getLogLikelihood(means, weights, covariances, data[2])

# plot result
plt.subplot()
plt.plot(range(num),logLikelihood)
plt.title('Loglikelihood for different number of k on Data 3')
plt.show()

# skin detection
print('\n')
print('(g) performing skin detection with GMMs')
sdata = np.loadtxt('skin.dat')
ndata = np.loadtxt('non-skin.dat')

img = im2double(misc.imread('faces.png'))

skin = skinDetection(ndata, sdata, skin_K, skin_n_iter, skin_epsilon, theta, img)
plt.imshow(skin)
plt.show()
misc.imsave('skin_detection.png', skin)
