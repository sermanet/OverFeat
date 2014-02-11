#include "ppm.hpp"

#define RGB_COMPONENT_COLOR 255

bool readPPM(FILE* stream, THTensor* output) {
  char buff[16];
  int c, rgb_comp_color;

  //read image format
  buff[0] = getc(stream);
  if ((buff[0] == '\0') || (buff[0] == -1))
    return false;
  if (!fgets(buff+1, sizeof(buff)-sizeof(char), stream)) {
    fprintf(stderr, "Unreadable PPM file\n");
    exit(1);
  }

  //check the image format
  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  //check for comments
  c = getc(stream);
  while (c == '#') {
    while (getc(stream) != '\n') ;
    c = getc(stream);
  }

  ungetc(c, stream);
  //read image size information
  int w, h;
  if (fscanf(stream, "%d %d", &w, &h) != 2) {
    fprintf(stderr,"%d %d\n", w, h);
    fprintf(stderr, "Invalid image size\n");
    exit(1);
  }

  //resize tensor
  THTensor_(resize3d)(output, 3, h, w);

  //read rgb component
  if (fscanf(stream, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component\n");
    exit(1);
  }

  //check rgb component depth
  if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
    fprintf(stderr, "Image does not have 8-bits components\n");
    exit(1);
  }

  while (fgetc(stream) != '\n') ;

  //read pixel data from file
  real* data = THTensor_(data)(output);
  long
    sc = output->stride[0],
    sh = output->stride[1],
    sw = output->stride[2];
  unsigned char buf[3];
  for(int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      if (fread(buf, sizeof(unsigned char), 3, stream) != 3) {
	fprintf(stderr, "Error reading image\n");
	exit(1);
      }
      data[       x*sw + y*sh] = (float)(buf[0]);
      data[  sc + x*sw + y*sh] = (float)(buf[1]);
      data[2*sc + x*sw + y*sh] = (float)(buf[2]);
    }
  return true;
}
