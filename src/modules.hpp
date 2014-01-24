/*
  Code from Torch 7
  http://torch.ch/
*/

#ifndef __OVERFEAT_MODULES_HPP__
#define __OVERFEAT_MODULES_HPP__

#include "THTensor.hpp"
#include <cassert>

inline void Threshold_updateOutput(THTensor* input, real val, real threshold,
				   THTensor* output) {
  THTensor_(resizeAs)(output, input);
  TH_TENSOR_APPLY2(real, output, real, input, \
                  *output_data = (*input_data > threshold) ? *input_data : val;);
}

inline void SpatialConvolution_updateOutput(THTensor* input, int dH, int dW,
					    THTensor* weight, THTensor* bias,
					    THTensor* output) {
  int dimw = 2;
  int dimh = 1;

  argcheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  if (input->nDimension == 4) {
    dimw++;
    dimh++;
  }

  {
    long nOutputPlane = weight->size[0];
    long kW           = weight->size[3];
    long kH           = weight->size[2];
    long inputWidth   = input->size[dimw];
    long inputHeight  = input->size[dimh];
    long outputWidth  = (inputWidth - kW) / dW + 1;
    long outputHeight = (inputHeight - kH) / dH + 1;

    if (input->nDimension == 3)
    {
      long i;
      real* bias_data;
      real* output_data;

      THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
      /* add bias */
      bias_data = THTensor_(data)(bias);
      output_data = THTensor_(data)(output);

#pragma omp parallel for private(i)
      for (i=0; i<bias->size[0]; i++)
      {
        /*THTensor_(select)(outn,output,0,i);*/
        /*TH_TENSOR_APPLY(real,outn, *outn_data = bias_data[i];);*/
        real *ptr_output = output_data + i*outputWidth*outputHeight;
        long j;
        for(j = 0; j < outputWidth*outputHeight; j++)
          ptr_output[j] = bias_data[i];
      }
      /*THTensor_(free)(outn);*/
      
      /* do convolutions */
      THTensor_(conv2Dmv)(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
    }
    else
    {
      real* bias_data;
      real* output_data; 
      long p;

      THTensor_(resize4d)(output, input->size[0], nOutputPlane, outputHeight, outputWidth);
      
      bias_data = THTensor_(data)(bias);
      output_data = THTensor_(data)(output);
      
#pragma omp parallel for private(p)
      for (p=0; p<input->size[0]; p++)
      {
        /* BIAS */
        long i;
        for (i=0; i<bias->size[0]; i++)
        {
          real *ptr_output = output_data + p*nOutputPlane*outputWidth*outputHeight + i*outputWidth*outputHeight;
          long j;
          for(j = 0; j < outputWidth*outputHeight; j++)
            ptr_output[j] = bias_data[i];
        }
      }
      
      /* do convolutions */
      THTensor_(conv2Dmm)(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
    }
  }
}

static void SpatialMaxPooling_updateOutput_frame(real *input_p, real *output_p,
						 real *indx_p, real *indy_p,
						 long nslices,
						 long iwidth, long iheight,
						 long owidth, long oheight,
						 int kW, int kH, int dW, int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        /* local pointers */
        real *ip = input_p   + k*iwidth*iheight + i*iwidth*dH + j*dW;
        real *op = output_p  + k*owidth*oheight + i*owidth + j;
        real *indyp = indy_p + k*owidth*oheight + i*owidth + j;
        real *indxp = indx_p + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        long maxindex = -1;
        real maxval = -THInf;
        long tcntr = 0;
        int x,y;
        for(y = 0; y < kH; y++)
        {
          for(x = 0; x < kW; x++)
          {
            real val = *(ip + y*iwidth + x);
            if (val > maxval)
            {
              maxval = val;
              maxindex = tcntr;
            }
            tcntr++;
          }
        }

        /* set output to local max */
        *op = maxval;

        /* store location of max (x,y) */
        *indyp = (int)(maxindex / kW)+1;
        *indxp = (maxindex % kW) +1;
      }
    }
  }
}

inline void SpatialMaxPooling_updateOutput(THTensor* input, int kH, int kW, int dH,
					   int dW, THTensor* indices,
					   THTensor* output) {
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  long nslices;
  long iheight;
  long iwidth;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;
  real *indices_data;


  argcheck(input->nDimension == 3 || input->nDimension == 4 , 2, "3D or 4D (batch mode) tensor expected");

  if (input->nDimension == 4) 
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }
  argcheck(input->size[dimw] >= kW && input->size[dimh] >= kH, 2, "input image smaller than kernel size");

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  oheight = (iheight - kH) / dH + 1;
  owidth = (iwidth - kW) / dW + 1;

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  /* resize output */
  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);
    /* indices will contain i,j locations for each output point */
    THTensor_(resize4d)(indices, 2, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

    SpatialMaxPooling_updateOutput_frame(input_data, output_data,
					 indices_data+nslices*owidth*oheight,
					 indices_data,
					 nslices,
					 iwidth, iheight,
					 owidth, oheight,
					 kW, kH, dW, dH);
  }
  else
  {
    long p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);
    /* indices will contain i,j locations for each output point */
    THTensor_(resize5d)(indices, 2, nbatch, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      SpatialMaxPooling_updateOutput_frame(input_data+p*nslices*iwidth*iheight,
					   output_data+p*nslices*owidth*oheight,
					   indices_data+(p+nbatch)*nslices*owidth*oheight,
					   indices_data+p*nslices*owidth*oheight,
					   nslices,
					   iwidth, iheight,
					   owidth, oheight,
					   kW, kH, dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(input);
}

inline void SpatialZeroPadding_updateOutput(THTensor* input, int pad_l,
					    int pad_r, int pad_t, int pad_b,
					    THTensor* output) {
  if (input->nDimension == 3) {
    
    int h = input->size[1] + pad_t + pad_b;
    int w = input->size[2] + pad_l + pad_r;
    assert((w > 0) && (h > 0));
    THTensor_(resize3d)(output, input->size[0], h, w);
    THTensor_(zero)(output);
    THTensor* c_input = THTensor_(newWithTensor)(input);
    if (pad_t < 0)
      THTensor_(narrow)(c_input, c_input, 1, - pad_t, c_input->size[1] + pad_t);
    if (pad_b < 0)
      THTensor_(narrow)(c_input, c_input, 1, 0, c_input->size[1] + pad_b);
    if (pad_l < 0)
      THTensor_(narrow)(c_input, c_input, 2, - pad_l, c_input->size[2] + pad_l);
    if (pad_r < 0)
      THTensor_(narrow)(c_input, c_input, 2, 0, c_input->size[2] + pad_r);
    THTensor* c_output = THTensor_(newWithTensor)(output);
    if (pad_t > 0)
      THTensor_(narrow)(c_output, c_output, 1, pad_t, c_output->size[1] - pad_t);
    if (pad_b > 0)
      THTensor_(narrow)(c_output, c_output, 1, 0, c_output->size[1] - pad_b);
    if (pad_l > 0)
      THTensor_(narrow)(c_output, c_output, 2, pad_l, c_output->size[2] - pad_l);
    if (pad_r > 0)
      THTensor_(narrow)(c_output, c_output, 2, 0, c_output->size[2] - pad_r);
    THTensor_(copy)(c_output, c_input);
    THTensor_(free)(c_output);
    THTensor_(free)(c_input);

  } else if (input->nDimension == 4) {
    
    int h = input->size[2] + pad_t + pad_b;
    int w = input->size[3] + pad_l + pad_r;
    assert((w > 0) && (h > 0));
    THTensor_(resize4d)(output, input->size[0], input->size[1], h, w);
    THTensor_(zero)(output);
    THTensor* c_input = THTensor_(newWithTensor)(input);
    if (pad_t < 0)
      THTensor_(narrow)(c_input, c_input, 2, - pad_t, c_input->size[1] + pad_t);
    if (pad_b < 0)
      THTensor_(narrow)(c_input, c_input, 2, 0, c_input->size[1] + pad_b);
    if (pad_l < 0)
      THTensor_(narrow)(c_input, c_input, 3, - pad_l, c_input->size[2] + pad_l);
    if (pad_r < 0)
      THTensor_(narrow)(c_input, c_input, 3, 0, c_input->size[2] + pad_r);
    THTensor* c_output = THTensor_(newWithTensor)(output);
    if (pad_t > 0)
      THTensor_(narrow)(c_output, c_output, 2, pad_t, c_output->size[1] - pad_t);
    if (pad_b > 0)
      THTensor_(narrow)(c_output, c_output, 2, 0, c_output->size[1] - pad_b);
    if (pad_l > 0)
      THTensor_(narrow)(c_output, c_output, 3, pad_l, c_output->size[2] - pad_l);
    if (pad_r > 0)
      THTensor_(narrow)(c_output, c_output, 3, 0, c_output->size[2] - pad_r);
    THTensor_(copy)(c_output, c_input);
    THTensor_(free)(c_output);
    THTensor_(free)(c_input);
    
  } else {
    assert(0);
  }
}

inline void Normalization_updateOutput(THTensor* input, real mean, real std,
				       THTensor* output) {
  THTensor_(resizeAs)(output, input);
  THTensor_(add)(output, input, -mean);
  THTensor_(div)(output, output, std);
}

inline void SoftMax_updateOutput(THTensor* input, THTensor* output) {
  real *input_data, *output_data;
  long nframe = 0, dim = 0;
  long t, d;

  if(input->nDimension == 1) {
    nframe = 1;
    dim = input->size[0];
  } else if(input->nDimension == 2) {
    nframe = input->size[0];
    dim = input->size[1];
  } else
    THArgCheck(0, 2, "For now, output must be a vector or matrix"); //TODO

  input = THTensor_(newContiguous)(input);
  THTensor_(resizeAs)(output, input);

  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);
  for(t = 0; t < nframe; t++) {
    real inputMax = -THInf;
    accreal sum;

    for(d = 0; d < dim; d++) {
      if (input_data[d] >= inputMax) inputMax = input_data[d];
    }

    sum = 0;
    for(d = 0; d < dim; d++) {
      real z = THExpMinusApprox(inputMax - input_data[d]);
      output_data[d] = z;
      sum += z;
    }

    for(d = 0; d < dim; d++) {
      output_data[d] *= 1/sum;
    }

    input_data += dim;
    output_data += dim;
  }

  THTensor_(free)(input);
}

#endif
