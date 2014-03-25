#ifndef __OVERFEAT_THCUDATENSOR_HPP__
#define __OVERFEAT_THCUDATENSOR_HPP__

#include "THC.h"
//#include "TH.h"

#define CUDA_CHECK(a)							\
  {									\
    cudaError_t status = (a);						\
    if (status != cudaSuccess) {					\
      std::cerr << "Error file " << __FILE__ <<				\
	" line " << __LINE__ << std::endl;				\
      std::cerr << cudaGetErrorString(status) << std::endl;		\
      exit(0);								\
    }									\
  }

#define CUDA_LOOK_FOR_ERROR()                                           \
  {                                                                     \
    cudaError_t err;							\
    if ((err = cudaGetLastError()) != cudaSuccess) {			\
      std::cerr << "Error in file " << __FILE__ <<			\
	" before line " << __LINE__ <<					\
	" : " << cudaGetErrorString(err) << std::endl;			\
      exit(0);								\
    }									\
  }


#endif
