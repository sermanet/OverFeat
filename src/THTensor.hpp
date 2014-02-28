#ifndef __OVERFEAT_THTENSOR_HPP__
#define __OVERFEAT_THTENSOR_HPP__

//#include "torch7-distro/installed/include/torch/TH/TH.h"
#include "TH/TH.h"

#ifdef THTensor_
#undef THTensor_
#endif
#ifdef THStorage_
#undef THStorage_
#endif
#ifdef real
#undef real
#endif
#ifdef accreal
#undef accreal
#endif
#ifdef THTensor
#undef THTensor
#endif
#ifdef THStorage
#undef THStorage
#endif
typedef float real;
typedef double accreal;
typedef THFloatTensor THTensor;
typedef THFloatStorage THStorage;
#define THTensor_(x) THFloatTensor_##x
#define THStorage_(x) THFloatStorage_##x

#endif
