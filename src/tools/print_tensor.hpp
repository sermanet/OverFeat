#ifndef __OVERFEAT_PRINT_TENSOR_HPP__
#define __OVERFEAT_PRINT_TENSOR_HPP__

#include "../THTensor.hpp"
#include <cstdio>

inline void print_tensor(const THTensor* a) {
  int nDim = a->nDimension;
  real* data = THTensor_(data)(a);
  switch (nDim) {
  case(0):
    printf("THTensor with no dimension\n");
    break;
  case(1):
    {
      int d1 = a->size[0];
      for (int i = 0; i < d1; ++i)
	printf("%f ", data[i]);
      printf("\nTensor with dimensions {%d}\n", d1);
    }
  case(2):
    {
      int d1 = a->size[0];
      int d2 = a->size[1];
      long n = 0;
      for (int i = 0; i < d1; ++i) {
	for (int j = 0; j < d2; ++j)
	  printf("%f ", data[n++]);
	printf("\n");
      }
      printf("Tensor with dimensions {%dx%d}\n", d1, d2);
    }
  case(3):
    {
      int d1 = a->size[0];
      int d2 = a->size[1];
      int d3 = a->size[2];
      long n = 0;
      for (int i = 0; i < d1; ++i) {
	for (int j = 0; j < d2; ++j) {
	  for (int k = 0; k < d3; ++k)
	    printf("%f ", data[n++]);
	  printf("\n");
	}
	printf("\n");
      }
      printf("Tensor with dimensions {%dx%dx%d}\n", d1, d2, d3);
    }
  }
}

inline void format_tensor_display(THTensor* a, THTensor* out) {
  int nDim = a->nDimension;
  real* a_data = THTensor_(data)(a);
  if (nDim < 3) {
    THTensor_(resizeAs)(out, a);
    THTensor_(copy)(out, a);
  } else if (nDim == 3) {
    int n = a->size[0];
    if (n == 3) {
      THTensor_(resizeAs)(out, a);
      THTensor_(copy)(out, a);
    } else {
      int nw = ceil(sqrt((float)(n)));
      int nh = ceil((float)n / nw);
      THTensor_(resize2d)(out, (a->size[1])*nh, (a->size[2])*nw);
      real* out_data = THTensor_(data)(out);
      int stride = nw*(a->size[2]);
      for (int i = 0; i < n; ++i) {
	int iw = i % nw, ih = i / nw;
	int sw = iw * (a->size[2]), sh = ih * (a->size[1]);
	for (int y = 0; y < a->size[1]; ++y)
	  for (int x = 0; x < a->size[2]; ++x) {
	    out_data[(sh + y)*(out->size[1]) + (sw + x)] =
	      a_data[(i*(a->size[1]) + y)*(a->size[2]) + x];
	  }
      }
    }
  } else {
    assert(0);
  }
  real* out_data = THTensor_(data)(out);
  real m = out_data[0], M = out_data[0];
  for (int i = 0; i < THTensor_(nElement)(out); ++i) {
    if (m > out_data[i])
      m = out_data[i];
    if (M < out_data[i])
      M = out_data[i];
  }
  THTensor_(add)(out, out, -m);
  THTensor_(div)(out, out, (float)(M-m)/255.);
  //printf("m=%f, M=%f\n", (float)m, (float)M);
}

#endif
