#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "overfeat.hpp"
using namespace std;

// these paths must be in the include path
#define INIT_FILE "net_init.hpp"
#define FPROP_FILE "net_fprop.hpp"

// this path is defined when init is called
//#define WEIGHT_FILE "../data/default/net_weight"
#define WEIGHT_FILE (overfeat::weight_file_path_g.c_str())

#define argcheck(test, narg, message) {if (!(test)) fprintf(stderr, "Error in file %s at line %d, argument %d : %s\n", __FILE__, __LINE__, (narg), (message));}

#include "modules.hpp"

namespace overfeat {
  
  string weight_file_path_g;
  int net_idx_g;
  FILE* weight_file = NULL;
  size_t weight_file_pos = 0;
  THTensor* load_tensor(int istart, int d1, int d2=-1, int d3=-1, int d4=-1) {
    if (!weight_file) {
      cout << "Error : can't read weight file" << endl;
    }
    THTensor* out;
    if (d4 != -1)
      out = THTensor_(newWithSize4d)(d1, d2, d3, d4);
    else if (d3 != -1) {
      out = THTensor_(newWithSize3d)(d1, d2, d3);
      d4 = 1;
    } else if (d2 != -1) {
      out = THTensor_(newWithSize2d)(d1, d2);
      d4 = d3 = 1;
    } else {
      out = THTensor_(newWithSize1d)(d1);
      d4 = d3 = d2 = 1;
    }
    real* out_data = THTensor_(data)(out);
    const int elemsize = sizeof(float);
    assert(sizeof(real) == elemsize); //FIXME
    if (weight_file_pos != istart*elemsize)
      fseek(weight_file, istart*elemsize, SEEK_SET);
    assert(fread(out_data, elemsize, d1*d2*d3*d4, weight_file) == d1*d2*d3*d4);
    weight_file_pos += d1*d2*d3*d4*elemsize;
    return out;
  }

  #include INIT_FILE
  void init(const string & weight_file_path, int net_idx) {
    weight_file_path_g = weight_file_path;
    net_idx_g = net_idx;
    memset(outputs, 0, nModules(net_idx)*sizeof(THTensor*));
    memset(weights, 0, nModules(net_idx)*sizeof(THTensor*));
    memset(bias   , 0, nModules(net_idx)*sizeof(THTensor*));
    for (int i = 0; i < nModules(net_idx); ++i)
      outputs[i] = THTensor_(new)();
    init1(net_idx);
  }

  void free() {
    for (int i = 0; i < nModules(net_idx_g); ++i) {
      if (outputs[i])
        THTensor_(free)(outputs[i]);
      if (weights[i])
        THTensor_(free)(weights[i]);
      if (bias[i])
        THTensor_(free)(bias[i]);
    }
  }

  void soft_max(THTensor* input, THTensor* output) {
    SoftMax_updateOutput(input, output);
  }
  
  int get_n_layers() {
    return nModules(net_idx_g);
  }

  THTensor* get_output(int i_layer) {
    assert((i_layer >= 0) && (i_layer < nModules(net_idx_g)));
    return outputs[i_layer];
  }

  string get_class_name(int i_class) {
    assert((i_class >= 0) && (i_class < nClasses));
    return class_names[i_class];
  }

  vector<pair<string, float> > get_top_classes(THTensor* net_output, int n) {
    real* net_output_data = THTensor_(data)(net_output);
    vector<pair<float, int> > tosort(nClasses);
    for (int i = 0; i < nClasses; ++i)
      tosort[i] = pair<float, int>(net_output_data[i], i);
    sort(tosort.begin(), tosort.end());
    vector<pair<string, float> > out(n);
    for (int i = 0; i < n; ++i)
      out[i] = pair<string, float>(get_class_name(tosort[nClasses-i-1].second),
				   tosort[nClasses-i-1].first);
    return out;
  }
    
  #include FPROP_FILE
  THTensor* fprop(THTensor* input) {
    return fprop1(input, net_idx_g);
  }

}
