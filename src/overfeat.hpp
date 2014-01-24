#ifndef __OVERFEAT_OVERFEAT_HPP__
#define __OVERFEAT_OVERFEAT_HPP__

#include "THTensor.hpp"
#include <string>
#include <vector>
#include <utility>

namespace overfeat {
  // Call this function once before any other call to overfeat
  // weight_file_path must be set to the path to the weight file.
  // The default weight file is located at data/default/net_weight
  void init(const std::string & weight_file_path, int net_idx);
  
  // Call this function to release resources
  void free();
  
  // This function computes the feature extraction
  //  input should be a 3xHxW THTensor*
  //  the function returns a nClasses x h x w THTensor*
  //  see README for more details
  // The output tensor must NOT be freed by the user
  THTensor* fprop(THTensor* input);
  
  // This function computes the soft max, transforming the output of the network
  //  input probabilities. See README for more details
  void soft_max(THTensor* input, THTensor* output);
  
  // Returns the number of layers of the network
  int get_n_layers();
  
  // Returns the output of the i-th layer
  THTensor* get_output(int i);
  
  // Returns a string corresponding of the name of the i-th class
  std::string get_class_name(int i);
  
  // Returns a vector of pairs (name, probability), corresponding to the
  //  n most likely classes, in decreasing order.
  // The tensor probas should correspond to probabilities
  //  (ie. it should have been through soft_max), otherwise the probabilities
  //  will be wrong (alghouth the ranking whould be ok)
  std::vector<std::pair<std::string, float> >
  get_top_classes(THTensor* probas, int n);
}


#endif
