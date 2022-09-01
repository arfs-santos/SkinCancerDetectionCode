import os
import time
import sys
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

tf.disable_v2_behavior() 

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def heat_conv(input, kernel):
  """A simplified 2D convolution operation for Heat Equation"""
  input = tf.expand_dims(tf.expand_dims(input, 0), -1)

  result = tf.nn.depthwise_conv2d(input, kernel,
                                    [1, 1, 1, 1],
                                    padding='SAME')
  return result[0, :, :, 0]

def show_viz(i,original, masked, mask, inpainted):
    """Show Image using matplotlib"""
    plt.figure(i)
    plt.subplot(221), plt.imshow(original, 'gray')
    plt.title('original image')
    plt.subplot(222), plt.imshow(masked, 'gray')
    plt.title('source image')
    plt.subplot(223), plt.imshow(mask, 'gray')
    plt.title('mask image')
    plt.subplot(224), plt.imshow(inpainted, 'gray')
    plt.title('inpaint result')

    plt.tight_layout()
    plt.draw()
    
def show_ssim(original, masked, inpainted):
    """Show SSIM Difference"""
    print("SSIM : ")
    print("  Original vs. Original  : ", ssim(original,original))
    print("  Original vs. Masked    : ", ssim(original,masked))
    print("  Original vs. Inpainted : ", ssim(original,inpainted))

    
def inpaint(masked, mask):
    # Init variable
    N = 2000
    ROOT_DIR = os.getcwd()

    # Create variables for simulation state
    U = tf.Variable(masked)
    G = tf.Variable(masked)
    M = tf.Variable(np.multiply(mask,1))
    K = make_kernel([[0.0, 1.0, 0.0],
                     [1.0, -4., 1.0],
                     [0.0, 1.0, 0.0]])

    dt = tf.placeholder(tf.float32, shape=())

    """Discretized PDE update rules"""
    """u[i,j] = u[i,j] + dt * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) - dt * lambda_m[i,j]*(u[i,j]-g[i,j])"""

    #Tensorflow while_loop function, iterate the PDE N times.
    index_summation = (tf.constant(1), U, M, G, K)
    def condition(i, U, M, G, K):
        return tf.less(i, 2000)

    def body(i,U,M,G,K):
        U_ = U + 0.1 * heat_conv(U,K) - 0.1 * M * (U-G)
        # i = tf.Print(i, [i])
        return tf.add(i, 1), U_, M, G, K

    #Tensorflow Session
    with tf.Session():
        # Initialize state to initial conditions
        tf.global_variables_initializer().run()

        #Run PDE using tensorflow while_loop
        t = time.time()
        uf=tf.while_loop(condition, body, index_summation)[1]
        U = uf.eval()

    print("Execution Time : {} s".format(time.time()-t))

    return U



img_dir = os.listdir('D:/Bases de Imagens/ISIC2020/ISIC_2020_Training_JPEG')

out_dir = 'D:/Bases de Imagens/ISIC2020/ISIC_2020_hair_removal/'

images=[]

scale = 65


kernel = cv2.getStructuringElement(1,(23,23))
    
i = 0

for name in img_dir:
    img = os.path.join('D:/Bases de Imagens/ISIC2020/ISIC_2020_Training_JPEG/', 
                    name)
    images.append(img)

for im in images:
    orig = cv2.imread(im, cv2.IMREAD_COLOR)
    width = int(orig.shape[1] * scale / 100)
    height = int(orig.shape[0] * scale / 100)
    dsize = (width, height)
    orig = cv2.resize(orig, dsize)
    grayScale = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    orig_r, orig_g, orig_b = cv2.split(orig)
    masked_r = cv2.normalize(orig_r, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    masked_g = cv2.normalize(orig_g, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    masked_b = cv2.normalize(orig_b, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    blackhat = cv2.dilate(grayScale,kernel,iterations = 10)
    blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_BLACKHAT, kernel)
    ret,mask = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    mask = 1-cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    inpanted_r = inpaint(masked_r, mask)
    inpanted_g = inpaint(masked_g, mask)
    inpanted_b = inpaint(masked_b, mask)
    im_masked = cv2.merge([inpanted_r, inpanted_g, inpanted_b])
    aux = os.path.split(images[i])
    i = i+1
    path = (out_dir + str(aux[1]))
    cv2.imwrite(path, im_masked)
    
    
    





