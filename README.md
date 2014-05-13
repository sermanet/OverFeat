OverFeat
========

OverFeat is a Convolutional Network-based image classifier and feature extractor.

OverFeat was trained on the ImageNet dataset and participated in the ImageNet 2013 competition.

This package allows researchers to use OverFeat to recognize images and extract features.

A library with C++ source code is provided for running the OverFeat convolutional network, together with wrappers in various scripting languages (Python, Lua, Matlab coming soon).

OverFeat was trained with the Torch7 package ( http://www.torch.ch ). The OverFeat package provides tools to run the network in a standalone fashion. The training code is not distributed at this time.


CREDITS, LICENSE, CITATION
--------------------------

OverFeat is Copyright NYU 2013. Authors of the present package are Michael Mathieu, Pierre Sermanet, and Yann LeCun.

The OverFeat system is by Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann LeCun.

Please refer to the LICENSE file in the same directory as the present file for licensing information.

If you use OverFeat in your research, please cite the following paper: 

"OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks", Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, Yann LeCun http://arxiv.org/abs/1312.6229


INSTALLATION:
-------------

Download the archive from http://cilvr.cs.nyu.edu/doku.php?id=software:overfeat:start

Extract the files:
```
tar xvf overfeat-vXX.tgz
cd overfeat
```

Overfeat uses external weight files. Since these files are large and do not change often, they are not included in the archive.
We provide a script to automatically download the weights :
```
./download_weights.py
```
The weight files should be in the folder data/default in the overfeat directory.

Overfeat can run without BLAS, however it would be very slow. We *strongly*
advice you to install openblas on linux (on MacOS, Accelerate should be available
without any installation). On Ubuntu/Debian you should compile it (it might take
a while, but it is worth it) :
```
sudo apt-get install build-essential gcc g++ gfortran git libgfortran3
cd /tmp
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make NO_AFFINITY=1 USE_OPENMP=1
sudo make install
```
For some reason, on 32 bits Ubuntu, libgfortran doesn't create the correct symlink.
If you have issues linking with libgfortran, locate where libgfortran is installed
(for instance /usr/lib/i386-linux-gnu) and create the correct symlink :
```
cd <folder_containing_libgfortran.so.3>
sudo ln -sf libgfortran.so.3 libgfortran.so
```
The precompiled binaries use BLAS. If you don't want to (or can't, for some reason)
use BLAS, you must recompile overfeat.

RUNNING THE PRE-COMPILED BINARIES

Pre-compiled binaries are provided for Ubuntu Linux (32 bits and 64 bits) and Mac OS. The pre-requisites are python and imagemagick, which are installed by default on most popular Linux distros.

**Important note:** OverFeat compiled from source on your computer will run faster
than the pre-compiled binaries.

Example of image classification, printing the 6 highest-scoring categories:
```
bin/YOUR_OS/overfeat -n 6 samples/bee.jpg
```
where YOUR_OS can be either linux_64, linux_32, or macos.

Running the webcam demo:
```
bin/YOUR_OS/webcam
```

GPU PRE-COMPILED BINARIES (EXPERIMENTAL)

We are providing precompiled binaries to run overfeat on GPU. Because the code
is not released yet, we do not provide the source for now. The GPU release is
experimental and for now only runs on linux 64bits. It requires a Nvidia GPU
with CUDA architecture >= 2.0 (that covers all recent GPUs from Nvidia).

You will need openblas to run the GPU binaries.

The binaries are located in
```
bin/linux_64/cuda
```
And work the same way as the CPU versions. You can include the static library the
same way as the CPU version.

COMPILING FROM SOURCE

Install dependencies : python, imagemagick, git, gcc, cmake (pkg-config and opencv required for the webcam demo).
On Ubuntu/Debian :
```
apt-get install g++ git python imagemagick cmake
```
For the webcam demo :
```
apt-get install pkg-config libopencv-dev libopencv-highgui-dev
```

Here are the instructions to build the OverFeat library and tools:

Go to the src folder :
```
cd src
```

Build the tensor library (TH), OverFeat and the command-line tools:
```
make all
```

Build the webcam demo (OpenCV required) :
```
make cam
```

On Mac OS, the default gcc doesn't support OpenMP. We strongly recommend to install
a gcc version with OpenMP support. With MacPort :
```
sudo port install gcc48
```
Which will provide g++-mp-48 . If you don't install this version, you will have to
change the two corresponding lines in the Makefile.

UPDATING

A git repository is provided with the archive. You can update by typing
```
git pull
```
from the overfeat directory.

HIGH LEVEL INTERFACE:
---------------------

The feature extractor requires a weight file, containing the weights of the
network. We provide a weight file located in data/default/net_weight .
The software we provide should be able to locate it automatically. In case
it doesn't, the option -d can be used to manually provide a path.

Overfeat can use two sizes of network. By default, it uses the smaller one.
For more accuracy, the option -l can be used to use a larger, but slower, network.

CLASSIFICATION:

In order to get the top <N> (by default, <N>=5) classes from a number of images :
```
bin/linux_64/overfeat [-n <N>] [-d <path_to_weights>] [-l] path_to_image1 [path_to_image2 [path_to_image3 [... ] ] ]
```

To use overfeat online (feeding an image stream),
feed its stdin stream with a sequence of ppm images (ended
by end of file ('\0') character). In this case, please use
option -p. For instance :
```
convert image1.jpg image2.jpg -resize 231x231 ppm:- | ./overfeat [-n <N>] [-d <path_to_weights>] [-l] -p
```
Please note that to get the classes from an image, the image size should be 231x231.
The image will be cropped if one dimension is larger than 231, and the network
won't be able to work if both dimension are larger.
For feature extraction without classification, it can be any size greater or
equal to 231x231 for the small network, and 221x221 for the large network .

FEATURE EXTRACTION:

In order to extract the features instead of classifying, use -f option. For instance :
```
bin/linux_64/overfeat [-d <path_to_weights>] [-l] -f image1.png image2.jpg
```
It is compatible with option -p.
The option -L (overrides -f) can be used to return the output of any layer.
For instance
```
bin/linux_64/overfeat [-d <path_to_weights>] [-l] -L 12 image1.png
```
returns the output of layer 12. The option -f corresponds to layer 19 for the small layer
and 22 for the large one.

It writes the features on stdout as a sequence. Each feature starts with three integers
separated by spaces, the first is the number of features (n),
the second is the number of rows (h) and the last is the number of columns (w).
It is followed by a end of line ('\n') character. Then follows n*h*w floating point
numbers (written in ascii) separated by spaces. The feature is the first dimension
(so that to obtain the next feature, you must add w*h to your index), followed by the
row (to obtain the next row, add w to your index).
That means that if you want the features corresponding to the top-left window, you need
to read pixels i*h*w for i=0..4095 .

The output is going to be a 3D tensor. The first dimension correspond to the features,
while dimensions 2 and 3 are spatial (y and x respectively).
The spatial dimension is reduced at each layer, and with the default network, using
option -f, the output has size nFeatures * h * w where
  - for the small network, 
    - nFeatures = 4096
    - h = ((H-11)/4 + 1)/8-6
    - w = ((W-11)/4 + 1)/8-6
  - for the large network,
    - nFeatures = 4096
    - h = ((H-7)/2 + 1)/18-5
    - w = ((W-7)/2 + 1)/18-5
if the input has size 3*H*W . Each pixel in the feature map corresponds to a
localized window in the input. With the small network, the windows are 231x231
pixels, overlapping so that the i-th window begins at pixel 32*i, while for the
large network, the windows are 221x221, and the i-th window begins at pixel 36*i.

WEBCAM:

We provide a live classifier based on the webcam. It reads images from the webcam,
and displays the most likely classes along with the probabilities.
It can be run with
```
bin/linux_64/webcam [-d <path_to_weights>] [-l] [-w <webcam_idx>]
```

BATCH:

We also provide an easy way to process a whole folder :
```
./bin/linux_64/overfeat_batch [-d <path_to_weights>] [-l] -i <input_dir> -o <output_dir>
```

It process each image in the input folder and produces a corresponding
file in the output directory, containing the features,in the same format
as before.

EXAMPLES:

Classify image samples/bee.jpg, getting the 3 most likely classes :
```
bin/linux_64/overfeat -n 3 samples/bee.jpg
```

Extract features from samples/pliers.jpg with the large network :
```
bin/linux_64/overfeat -f -l samples/pliers.jpg
```

Extract the features from all files in samples :
```
./bin/linux_64/overfeat_batch -i samples -o samples_features
```

Run the webcam demo with the large network :
```
bin/linux_64/webcam -l
```

ADVANCED:

The true program is actually overfeatcmd, where overfeat is only a python script calling
overfeatcmd. overfeatcmd is not designed to be used by itself, but can be if necessary.
It taked three arguments :
```
bin/linux_64/overfeatcmd <path_to_weights> <N> <I> <L>
```
If <N> is positive, it is, as before, the number of top classes to display.
If <N> is nonpositive, the features are going to be the output. The option <L>
specifies from which layer the features are obtained
(by default, <L>=16, corresponding to the last layer before the classifier).
<I> corresponds to the size of the network : 0 for small, 1 for large.

APIs:
----

C++:

The library is written in C++. It consists of one static library named liboverfeat.a .
The corresponding header is overfeat.hpp . It uses the low level torch tensor
library (TH). Sample code can be found in overfeatcmd.cpp and webcam.cpp.

The library provides several functions in the namespace overfeat :

- `void init(const std::string & weight_file_path, int net_idx)` : 
This function must be called once before using the feature extractor.
It reads the weights and must be passed a path to the weight files.
It must also be passed the size of the network (net_idx), which should be 0,
or 1, respectively for small or large networks. Note that the
weight file must correspond to the size of the network.

- `void free()` : This function releases the ressources and should be called when the
feature extractor is no longer used.

- `THTensor* fprop(THTensor* input)` : 
This is the main function. It takes an image stored in a THTensor* and runs the network on it.
It returns a pointer to a THTensor containing the output of the classifier. 
If the input is 3*H*W, the output is going to be nClasses * h * w, where
  - for the small network :
    - nClasses = 1000
    - h = ((H-11)/4 + 1)/8 - 6
    - w = ((W-11)/4 + 1)/8 - 6
  - for the large network :
    - nClasses = 1000
    - h = ((H-7)/2 + 1)/18 - 5
    - w = ((W-7)/2 + 1)/18 - 5
Each pixel of the output corresponds to a 231x231 window on the input for the
small network, and 221x221 for the large network. The windows overlap in the same
way as described earlier for the feature extraction.
Each class gets a score, but they are not probabilities (they are not normalized).

- `THTensor* get_output(int i)` : Once fprop has been computed, this function returns the 
output of any layer. For instance, in the default network, layer 16 corresponds to the
final features before the classifier.

- `int get_n_layers()` : Returns the total number of layers of the network.

- `void soft_max(THTensor* input, THTensor* output)` : This function converts 
the output to probabilities. It only works if h = w = 1 (only one output pixel).

- ` std::string get_class_name(int i)` : This function returns the string corresponding to the i-th class.

- `std::vector<std::pair<std::string, float> > get_top_classes(THTensor* probas, int n)` : 
Given a vector with nClasses elements containing scores or
probabilities, this function returns the names of the top `n` classes, along with their
score/probabilities.

When compiling code using liboverfeat.a, the code must also be linked against
libTH.a, the tensor library. The file libTH.a will have been produced when
compiling torch.


Torch7:

We have bindings for torch, in the directory API/torch. The file
API/torch/README contains more details.


Python:

The bindings for python are in API/python. See API/python/README .
