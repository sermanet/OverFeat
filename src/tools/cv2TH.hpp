#ifndef __OVERFEAT_CV2TH_HPP__
#define __OVERFEAT_CV2TH_HPP__

#include <cstring>
#include <cassert>
//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include "../THTensor.hpp"

typedef ::real scalar;

/*
THTensor* cv2TH(const cv::Mat_<scalar> & M) {
  int w = M.size().width, h = M.size().height;
  THTensor* out = THTensor_(newWithSize3d)(1, h, w);
  memcpy(M.data, THTensor_(data)(out), w*h*sizeof(scalar));
  return out;
}
*/

THTensor* cv2TH(const cv::Mat_<float> & M, THTensor* output = NULL) {
  int w = M.size().width, h = M.size().height;
  if (output)
    THTensor_(resize3d)(output, 1, h, w);
  else
    output = THTensor_(newWithSize3d)(1, h, w);
  scalar* data = THTensor_(data)(output);
  for (int i = 0; i < w*h; ++i)
    data[i] = ((float*)(M.data))[i];
  return output;
}


THTensor* cv2TH(const cv::Mat_<cv::Vec3b> & M, THTensor* output = NULL) {
  int w = M.size().width, h = M.size().height;
  if (output)
    THTensor_(resize3d)(output, 3, h, w);
  else
    output = THTensor_(newWithSize3d)(3, h, w);
  scalar* data = THTensor_(data)(output);
  for (int i = 0; i < w*h; ++i) {
    data[i      ] = ((cv::Vec3b*)(M.data))[i][0];
    data[i+w*h  ] = ((cv::Vec3b*)(M.data))[i][1];
    data[i+2*w*h] = ((cv::Vec3b*)(M.data))[i][2];
  }
  return output;
}

cv::Mat TH2cv_byte(const THTensor* T) {
  int c, h, w;
  if (T->nDimension == 3) {
    c = T->size[0];
    h = T->size[1];
    w = T->size[2];
  } else if (T->nDimension == 2) {
    c = 1;
    h = T->size[0];
    w = T->size[1];
  } else {
    assert(0);
  }
  scalar* data = THTensor_(data)(T);
  if (c == 1) {
    /*
    cv::Mat_<scalar> out(h, w);
    memcpy(data, out.data, h*w*sizeof(scalar));
    return out;
    */
    cv::Mat_<unsigned char> out(h, w);
    for (int i = 0; i < w*h; ++i)
      ((unsigned char*)(out.data))[i] = (unsigned char)(data[i]);
    //memcpy(data, out.data, h*w*sizeof(scalar));
    return out;
  } else if (c == 3) {
    cv::Mat_<cv::Vec3b> out(h, w);
    for (int i = 0; i < w*h; ++i)
      ((cv::Vec3b*)(out.data))[i] = cv::Vec3b(data[i], data[i+w*h], data[i+2*w*h]);
    return out;
  }
  assert(0);
}

#endif
