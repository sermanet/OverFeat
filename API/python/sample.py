import overfeat
import numpy
from scipy.ndimage import imread
from scipy.misc import imresize

# read image
image = imread('../../samples/bee.jpg')

# resize and crop into a 231x231 image
DIM = 231
h0, w0 = image.shape[:2]
d0 = float(min(h0, w0))
h1 = int(round(DIM*h0/d0))
w1 = int(round(DIM*w0/d0))
d1 = DIM
image = imresize(image, (h1, w1)).astype(numpy.float32)
image = image[int(round((h1-d1)/2.)):int(round((h1-d1)/2.)+DIM),
              int(round((w1-d1)/2.)):int(round((w1-d1)/2.)+DIM), :]

# numpy loads image with colors as last dimension, transpose tensor
h,w,c = image.shape
image = image.reshape(w*h, c)
image = image.transpose()
image = image.reshape(c, h, w)
print "Image size :", image.shape

# initialize overfeat. Note that this takes time, so do it only once if possible
overfeat.init('../../data/default/net_weight_0', 0)

# run overfeat on the image
b = overfeat.fprop(image)

# display top 5 classes
b = b.flatten()
top = [(b[i], i) for i in xrange(len(b))]
top.sort()
print "\nTop classes :"
for i in xrange(5):
    print(overfeat.get_class_name(top[-(i+1)][1]))
