OS:=$(shell uname -s)
ifeq (${OS}, Darwin)
# Mac Users : You might have to change these lines
CC=gcc-mp-4.8
CXX=g++-mp-4.8
BLAS_LINK=-framework Accelerate
else
CC=gcc
CXX=g++
BLAS_LINK=-L/opt/OpenBLAS/lib -lopenblas
endif

OMP_FLAG=-fopenmp
FLAGS=${OMP_FLAG} -O3
# -Wl,--hash-style=both -static-libgcc

LIB_FLAGS=${FLAGS} -fPIC
LIB_CPP=overfeat.cpp
LIB_HEADERS=modules.hpp overfeat.hpp
LIB_TARGET=liboverfeat.a
LIB_INCLUDE=-I`if [ ! -d "data/default" ]; then echo "../data/default"; else echo "data/default"; fi`

CMD_LINK=liboverfeat.a libTH.a ${BLAS_LINK}
CMD_CPP=overfeatcmd.cpp tools/ppm.cpp
CMD_HEADERS=tools/ppm.hpp
CMD_TARGET=overfeatcmd

WEBCAM_FLAGS=${FLAGS}
WEBCAM_LINK=`pkg-config --libs opencv` liboverfeat.a libTH.a ${BLAS_LINK}
WEBCAM_CPP=webcam.cpp
WEBCAM_HEADERS=overfeat.hpp tools/cv2TH.hpp
WEBCAM_TARGET=webcamcmd

THLIB=libTH.a

all:
	make libTH
	make lib
	make cmd

libTH:
	cd TH; mkdir -p build; cd build; rm -f libTH.a; cmake -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} ..; make; cp libTH.a ../..; cp THGeneral.h ..

lib: ${LIB_CPP} ${LIB_HEADERS}
	${CXX} -c ${LIB_FLAGS} ${LIB_INCLUDE} ${LIB_CPP} -o overfeat.o
	rm -f ${LIB_TARGET}
	ar rcs ${LIB_TARGET} overfeat.o

cmd: ${CMD_CPP} ${CMD_HEADERS}
	${CXX} ${FLAGS} ${CMD_CPP} ${CMD_LINK} -o ${CMD_TARGET}

cam: ${WEBCAM_CPP} ${WEBCAM_HEADERS}
	${CXX} ${WEBCAM_FLAGS} ${WEBCAM_CPP} ${WEBCAM_LINK} -o ${WEBCAM_TARGET}

clean:
	rm -rf *.o ${LIB_TARGET} ${CMD_TARGET} ${WEBCAM_TARGET} TH/build libTH.a
