#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
//#include <opencv/highgui.h>
#include "overfeat.hpp"
#include "tools/cv2TH.hpp"
#include <signal.h>
using namespace std;
using namespace cv;

CvFont global_hershey_font;

// This function reads and resizes a frame from the camera, using opencv
VideoCapture* cam = NULL;
Mat_<Vec3b> getCameraFrame_tmp;
THTensor* getCameraFrame(int webcamidx, THTensor* output, int w = -1, int h = -1) {
  if (!cam)
    cam = new VideoCapture(webcamidx);
  Mat_<Vec3b> im;
  // flush camera buffer
  cam->grab();
  cam->grab();
  cam->grab();
  // get up to date image
  cam->read(im);
  int cam_w = im.size().width;
  int cam_h = im.size().height;
  if (w == -1) w = cam_w;
  if (h == -1) h = cam_h;
  // crop a square from the webcam
  Rect myROI(cam_w/2 - cam_h/2, 0, cam_h, cam_h);
  im = im(myROI);
  resize(im, getCameraFrame_tmp, Size(w, h), 0, 0, INTER_CUBIC);
  cv2TH(getCameraFrame_tmp, output);
}

// Simple display of a tensor
void display(const THTensor* todisp, bool wait = false) {
  Mat M = TH2cv_byte(todisp);
  cvNamedWindow("display");
  IplImage Mimg = M;
  cvShowImage("display", &Mimg);
  cvWaitKey((wait) ? 0 : 1);
}

// Displays an image, along with classes and probabilities
void displayWithConf(const THTensor* im, vector<pair<string, float> > classes) {
  int c = im->size[0], h = im->size[1], w = im->size[2];
  int wextra = 300;
  assert(c == 3);
  Mat_<Vec3b> M(h, w + wextra, 0.0f);
  scalar* im_data = THTensor_(data)(im);
  for (int y = 0, i = 0; y < h; ++y)
    for (int x = 0; x < w; ++x, ++i)
      M(y, x) = Vec3b(im_data[i], im_data[i+w*h], im_data[i+2*w*h]);

  char buffer[128];
  for (int i = 0; i < classes.size(); ++i) {
    pair<string, float> & cl = classes[i];
    IplImage Mimg = M;
    cvPutText(&Mimg, cl.first.c_str(), Point2i(w, 20+40*i),
	      &global_hershey_font, cvScalarAll(255));
    sprintf(buffer, "  %f", cl.second);
    rectangle(M, Point2i(w+10, 30+40*i),
	      Point2i(w+(wextra-20)*cl.second, 40+40*i),
	      Scalar(0, 64, 0), CV_FILLED);
    cvPutText(&Mimg, buffer, Point2i(w, 40+40*i),
	      &global_hershey_font, cvScalarAll(255));
  }
  cvNamedWindow("display");
  IplImage Mimg = M;
  cvShowImage("display", &Mimg);
  cvWaitKey(1);
}

// Releases the camera when Ctrl-C is pressed
void killed(int s) {
  if (cam) {
    cam->release();
  }
  exit(0);
}

int main(int argc, char* argv[]) {
  // Attach the "killed" function to Ctrl-C and Ctrl-\ interrupts
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = killed;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
  sigaction(SIGQUIT, &sigIntHandler, NULL);
  sigaction(SIGSEGV, &sigIntHandler, NULL);

  if (argc < 2) {
    fprintf(stderr, "Missing argument : path to weight file\n");
      exit(0);
  }
  if (argc < 3) {
    fprintf(stderr, "Missing argument : network size\n");
    exit(0);
  }
  if (argc < 4) {
    fprintf(stderr, "Missing argument : webcam idx\n");
    exit(0);
  }

  string weight_file_path = argv[1];
  int net_idx = atoi(argv[2]);
  int webcamidx = atoi(argv[3]);

  cvInitFont(&global_hershey_font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, 8);

  try {
    
    // Initialization
    overfeat::init(weight_file_path, net_idx);
    THTensor* input = THTensor_(new)();
    THTensor* probas = THTensor_(new)();
    
    // Process 100 images from the webcam
    //for (int i = 0; i < 100; ++i) {
    // loop
    while(true) {
      // Read image from webcam
      // minimum input size : 3x231x231
      getCameraFrame(webcamidx, input, 231, 231);
      
      // Extract features and classify
      THTensor* output = overfeat::fprop(input);
      
      // Convert output into probabilities
      assert((output->size[1] == 1) && (output->size[2] == 1));
      output->nDimension = 1;
      overfeat::soft_max(output, probas);
      
      // Display
      vector<pair<string, float> > top_classes = overfeat::get_top_classes(probas, 5);
      displayWithConf(input, top_classes);
    }

    THTensor_(free)(input);
    THTensor_(free)(probas);
    overfeat::free();

  } catch (cv::Exception & e) {
    cout << "OpenCV error" << endl;
    killed(0);
  }
  if (cam)
    cam->release();
  return 0;
}
