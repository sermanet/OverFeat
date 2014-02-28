#ifndef __OVERFEAT_OVERFEAT_CUDA_HPP__
#define __OVERFEAT_OVERFEAT_CUDA_HPP__

#include "THCudaTensor.hpp"
#include <string>
#include <vector>
#include <utility>

namespace overfeat_cuda {
  // Call this function once before any other call to overfeat
  // weight_file_path must be set to the path to the weight file.
  // The default weight file is located at data/default/net_weight
  void init(const std::string & weight_file_path, int net_idx);
  
  // Call this function to release resources
  void free();

  void copyDeviceToHost(THCudaTensor* src, THFloatTensor* dst);
  void copyHostToDevice(THFloatTensor* src, THCudaTensor* dst);
  
  // This function computes the feature extraction
  //  input should be a 3xHxW THTensor*
  //  the function returns a nClasses x h x w tensor
  //  see README for more details
  // The returned tensor is a THCudaTensor (ie. stored in the gpu memory).
  //  It must NOT be freed by the user.
  THCudaTensor* fprop(THFloatTensor* input);
  THCudaTensor* fprop(THCudaTensor* input);
  
  // This function computes the soft max, transforming the output of the network
  //  input probabilities. See README for more details
  void soft_max(THCudaTensor* input, THCudaTensor* output);
  
  // Returns the number of layers of the network
  int get_n_layers();
  
  // Returns the output of the i-th layer in a THCudaTensor
  // (stored in the gpu memory)
  THCudaTensor* get_output(int i);
  
  // Returns a string corresponding of the name of the i-th class
  std::string get_class_name(int i);
  
  // Returns a vector of pairs (name, probability), corresponding to the
  //  n most likely classes, in decreasing order.
  // The tensor probas should correspond to probabilities
  //  (ie. it should have been through soft_max), otherwise the probabilities
  //  will be wrong (alghouth the ranking whould be ok)
  std::vector<std::pair<std::string, float> >
  get_top_classes(THCudaTensor* probas, int n);
}


#endif
