#ifndef __OVERFEAT_PPM_HPP__
#define __OVERFEAT_PPM_HPP__

#include "../THTensor.hpp"
#include <cstdio>

bool readPPM(FILE* stream, THTensor* output);
void writePPM(THTensor* input, FILE* stream);

#endif
