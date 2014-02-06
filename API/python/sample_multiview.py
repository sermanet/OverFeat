import overfeat
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize

def multiview_image(image):
    # resize and crop into a 256x256 image
    DIM = 256
    dim = 231

    h0, w0 = image.shape[:2]
    d0 = float(min(h0, w0))
    h1 = int(round(DIM*h0/d0))
    w1 = int(round(DIM*w0/d0))
    d1 = DIM
    image = imresize(image, (h1, w1)).astype(np.float32)
    h_start = int(round((h1-d1)/2.))
    w_start = int(round((w1-d1)/2.))
    image = image[h_start:h_start+DIM, w_start:w_start+DIM, :]
    # numpy loads image with colors as last dimension, transpose tensor
    h,w,c = image.shape
    image = image.reshape(w*h, c)
    image = image.transpose()
    image = image.reshape(c, h, w)

    # crop four corners and the center, and then flip them
    # total 10 231x231 images
    dim_diff = DIM - dim
    start_points = [(0, 0),  (0, dim_diff),
                    (dim_diff, 0), (dim_diff, dim_diff),
                    (dim_diff /2., dim_diff /2.)
                    ]
    images = np.zeros((10, c, dim, dim), dtype=np.float32)
    for i, (h_start, w_start) in enumerate(start_points):
        images[i,:] = image[:, h_start:h_start+dim, w_start:w_start+dim]
        images[i+5,:] = images[i,:,::-1]
    return images

# read image
image = imread('../../samples/bee.jpg')
images = multiview_image(image)

# initialize overfeat. Note that this takes time, so do it only once if possible
overfeat.init('../../data/default/net_weight_0', 0)
b_avg = None
for image in images:
    # run overfeat on the image
    b = overfeat.fprop(image)
    b = overfeat.soft_max(b)
    b = b.flatten()
    if b_avg is None:
        b_avg = b
    else:
        b_avg += b
print 'Average Prediction of 4 corners + center + flip'
# average the prediction and then display top 5 classes
b_avg /= 10
top = [(b_avg[i], i) for i in xrange(len(b_avg))]
top.sort()
print "Top classes :"
for i in xrange(5):
    print(overfeat.get_class_name(top[-(i+1)][1]))