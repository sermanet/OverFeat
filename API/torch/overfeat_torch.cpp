extern "C" {
#ifdef __APPLE__
#include<luaT.h>
#include<TH/TH.h>
#else
#include<torch/luaT.h>
#include<torch/TH/TH.h>
#endif
}
#include"overfeat.hpp"
#include<cmath>
#include<cassert>
#include<vector>
#include<algorithm>
#include<iostream>
using namespace std;

#ifdef assert
#undef assert
#endif
#define assert(x) {if (!x) {std::cerr << "Assertion failed file " << __FILE__ << ", line " << __LINE__ << std::endl; exit(0);}}

static int Overfeat_init(lua_State* L) {
  string weight_file_path = lua_tostring(L, 1);
  int net_idx = lua_tointeger(L, 2);
  overfeat::init(weight_file_path, net_idx);
  return 0;
}

static int Overfeat_free(lua_State* L) {
  overfeat::free();
  return 0;
}

static int Overfeat_fprop(lua_State* L) {
  const char* idfloat = "torch.FloatTensor";
  THFloatTensor* input        = (THFloatTensor*)luaT_checkudata(L, 1, idfloat);
  THFloatTensor* output       = (THFloatTensor*)luaT_checkudata(L, 2, idfloat);
  THTensor* output2 = overfeat::fprop(input);
  THTensor_(resizeAs)(output, output2);
  THTensor_(copy)(output, output2);
  return 0;
}

static int Overfeat_get_n_layers(lua_State* L) {
  lua_pushinteger(L, overfeat::get_n_layers());
  return 1;
}

static int Overfeat_get_output(lua_State* L) {
  const char* idfloat = "torch.FloatTensor";
  THFloatTensor* output       = (THFloatTensor*)luaT_checkudata(L, 1, idfloat);
  int i = lua_tointeger(L, 2);
  THTensor* output2 = overfeat::get_output(i);
  THTensor_(resizeAs)(output, output2);
  THTensor_(copy)(output, output2);
  return 0;
}

static int Overfeat_get_class_name(lua_State* L) {
  int i = lua_tointeger(L, 1);
  lua_pushstring(L, overfeat::get_class_name(i).c_str());
  return 1;
}

static const struct luaL_reg liboverfeat_torch[] = {
  {"init", Overfeat_init},
  {"free", Overfeat_free},
  {"fprop", Overfeat_fprop},
  {"get_n_layers", Overfeat_get_n_layers},
  {"get_output", Overfeat_get_output},
  {"get_class_name", Overfeat_get_class_name},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_liboverfeat_torch(lua_State *L) {
  luaL_openlib(L, "liboverfeat_torch", liboverfeat_torch, 0);
  return 1;
}
