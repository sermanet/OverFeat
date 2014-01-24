#!/bin/sh

if [ ! -d "torch7-distro" ]; then
    git clone https://github.com/torch/torch7-distro.git
    #cd torch7-distro/lib/TH
    #patch < ../../../patch.diff
    #cd ../../..
fi
cd torch7-distro
mkdir -p build
mkdir -p installed
cd build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/../installed ..
#cmake -DCMAKE_INSTALL_PREFIX=$HOME/local ..
make -j8
make install