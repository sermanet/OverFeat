CXX=g++
OS:=$(shell uname -s)
ifeq (${OS}, Darwin)
OMP_FLAG=-openmp
else
OMP_FLAG=-fopenmp
endif
#OMP_FLAG=`if [ "\`gcc -lgomp 2>&1 | grep library\ not\ found\`" == "" ]; then echo ""; else echo "-fopenmp"; fi`
#OMP_FLAG=`if [ \`uname\` == 'Darwin' ]; then echo "-openmp"; else echo "-fopenmp"; fi`
FLAGS=${OMP_FLAG} -O3

LIB_FLAGS=${OMP_FLAG} -O3 -fPIC -c
LIB_LINK=-Ltorch7-distro/installed/lib -lTH
LIB_CPP=overfeat.cpp
LIB_HEADERS=modules.hpp overfeat.hpp
LIB_TARGET=liboverfeat.a
LIB_INCLUDE=-I`if [ ! -d "data/default" ]; then echo "../data/default"; else echo "data/default"; fi`

CMD_LINK=-Ltorch7-distro/installed/lib -lTH liboverfeat.a
CMD_CPP=overfeatcmd.cpp tools/ppm.cpp
CMD_HEADERS=tools/ppm.hpp
CMD_TARGET=overfeatcmd

WEBCAM_LINK=-Ltorch7-distro/installed/lib -lTH `pkg-config --libs opencv` liboverfeat.a
WEBCAM_CPP=webcam.cpp
WEBCAM_HEADERS=overfeat.hpp tools/cv2TH.hpp tools/print_tensor.hpp
WEBCAM_TARGET=webcamcmd

THLIB=libTH.*
#`if [ \`uname\` == 'Darwin' ]; then echo "libTH.dylib"; else echo "libTH.so"; fi`

all:
	make lib
	make cmd
	chmod +x overfeat
	cp torch7-distro/installed/lib/${THLIB} .

lib: ${LIB_CPP} ${LIB_HEADERS}
	${CXX} ${LIB_FLAGS} ${LIB_INCLUDE} ${LIB_CPP} -o overfeat.o
	rm -f ${LIB_TARGET}
	ar -cvq ${LIB_TARGET} overfeat.o

cmd: ${CMD_CPP} ${CMD_HEADERS}
	${CXX} ${FLAGS} ${CMD_CPP} ${CMD_LINK} -o ${CMD_TARGET}

cam: ${WEBCAM_CPP} ${WEBCAM_HEADERS}
	${CXX} ${FLAGS} ${WEBCAM_CPP} ${WEBCAM_LINK} -o ${WEBCAM_TARGET}

clean:
	rm -rf *.o ${LIB_TARGET} ${CMD_TARGET} ${WEBCAM_TARGET}
