"""
NIQE: Natural Image Quality Evaluator
An excellent no-reference image quality metric.  
Code below is roughly similar on Matlab code graciously provided by Anish Mittal, but is written from scratch (since most of the Matlab functions there don't have numpy equivalents).
Training the model is not implemented yet, so we rely on pre-trained model parameters in file modelparameters.mat.
Cite:
Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik. "Making a completely blind image quality analyzer." Signal Processing Letters, IEEE 20.3 (2013): 209-212.
"""
import numpy
import numpy.linalg
from scipy.special import gamma
from scipy.ndimage import gaussian_filter
import imageio.v3 as imageio
from scipy.io import loadmat
import skimage.transform
from skimage.color import rgb2gray
from skimage.util import view_as_blocks
"""
Generalized Gaussian distribution estimation.
Cite: 
Dominguez-Molina, J. Armando, et al. "A practical procedure to estimate the shape parameter in the generalized Gaussian distribution.", 
  available through http://www. cimat. mx/reportes/enlinea/I-01-18_eng. pdf 1 (2001).
"""

"""
Estimate the parameters of an asymmetric generalized Gaussian distribution
"""
def estimate_aggd_params(vec):
    ###MATLAB   gam   = 0.2:0.001:10
    gam=numpy.linspace(0.2,10,num=9801,endpoint=True)
    r_gam = ((gamma(2/gam))**2)/(gamma(1/gam)*gamma(3/gam))


    leftstd            = numpy.sqrt(numpy.mean((vec[vec<0])**2))
    rightstd           = numpy.sqrt(numpy.mean((vec[vec>0])**2))

    gammahat           = leftstd/rightstd;
    rhat               = (numpy.mean(numpy.abs(vec)))**2/numpy.mean((vec)**2)
    rhatnorm           = (rhat*(gammahat**3 +1)*(gammahat+1))/((gammahat**2 +1)**2)
    min_difference, array_position = numpy.min((r_gam - rhatnorm)**2), numpy.argmin((r_gam - rhatnorm)**2)
    alpha              = gam[array_position]

    betal              = leftstd *numpy.sqrt(gamma(1/alpha)/gamma(3/alpha))
    betar              = rightstd*numpy.sqrt(gamma(1/alpha)/gamma(3/alpha))
    return alpha,betal,betar

def compute_features(img_norm):
    features = []
    alpha, beta_left, beta_right = estimate_aggd_params(img_norm)

    features.extend([ alpha, (beta_left+beta_right)/2 ])

    for x_shift, y_shift in ((0,1), (1,0), (1,1), (1,-1)):
        img_pair_products  = img_norm * numpy.roll(numpy.roll(img_norm, y_shift, axis=0), x_shift, axis=1)
        alpha, beta_left, beta_right = estimate_aggd_params(img_pair_products)
        eta = (beta_right - beta_left) * (gamma(2.0/alpha) / gamma(1.0/alpha))
        features.extend([ alpha, eta, beta_left, beta_right ])

    return numpy.array(features)

def normalize_image(img, sigma=7/6):
    mu  = gaussian_filter(img, sigma, mode='nearest')
    mu_sq = mu * mu
    sigma = numpy.sqrt(numpy.abs(gaussian_filter(img * img, sigma, mode='nearest') - mu_sq))
    img_norm = (img - mu) / (sigma + 1)
    return img_norm

def get_divisor(shp):
    for i in range(4,7):
        if shp%i==0:
            return i
    return None
def niqe(img):
    divisor=get_divisor(img.shape[0]) #getting a divisor 3 to 6 to divide the image in blocks
    model_mat = loadmat('modelparameters.mat') #modelparameters.mat is mat file provided by original author of matlab
    model_mu = model_mat['mu_prisparam']
    model_cov = model_mat['cov_prisparam']
    if len(img.shape)>2:
        img=rgb2gray(img)
    features = None
    img_scaled = img
    for scale in [1,2]:

        if scale != 1:
            img_scaled = skimage.transform.rescale(img, 1/scale)
            #img_scaled = scipy.misc.imresize(img_norm, 0.5)

        # print img_scaled
        img_norm = normalize_image(img_scaled)

        scale_features = []
        block_size = (img.shape[0]//divisor)//scale
        feature_shape=img_norm.shape[0]//block_size*img_norm.shape[1]//block_size
        #below code break the image using block shape, if image size is 480X480 and block size
        #is 96 then break the image 480//96=5. so 5X5X96X96 5X5 nos of 96X96 size blocks each
        View=view_as_blocks(img_norm,block_shape=(block_size,block_size))
        # below code python vectorize a function with signature. In signature portion, before -> it is input and after -> it is output
        # so it take input of (block_size,block_size) each time and return output of length 18, call it 5X5= 25 time and generate
        #final output of size 25X18
        vect_compute=numpy.vectorize(compute_features,signature='(block_size,block_size)->(18)')
        scale_features=numpy.append(scale_features,vect_compute(View),axis=None).reshape((feature_shape,18))
        if features is None:
            features = scale_features.copy()  #first time it copies 
            # print features.shape
        else:
            features = numpy.hstack([features, scale_features])
            # print features.shape
    features=numpy.nan_to_num(features)  # reset NaN or Inf values in array to 0 
    features_mu =numpy.nanmean(features, axis=0) # mean calculation including NaN// although its not requred
    features_cov =numpy.cov(features.T)
    #below code Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) 
    #and including all large singular values
    pseudoinv_of_avg_cov  = numpy.linalg.pinv((model_cov + features_cov)/2)
    niqe_quality = numpy.sqrt( (model_mu - features_mu).dot( pseudoinv_of_avg_cov.dot( (model_mu - features_mu).T ) ) )
    return niqe_quality.item()

if __name__=='__main__':
    img=numpy.random.uniform(0,255,(10,10))
    #img = imageio.imread('lena.png').astype(float)
    print (f"NIQE = {niqe(img)}" )