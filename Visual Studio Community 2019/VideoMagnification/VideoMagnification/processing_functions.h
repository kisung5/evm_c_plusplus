#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int amplify_spatial_lpyr_temporal_ideal(string inFile, string outDir, int alpha,
    int lambda_c, double fl, double fh, int samplingRate, int chromAttenuation);

int maxPyrHt(int frameWidth, int frameHeight, int filterSizeX, int filterSizeY);

vector<Mat> build_Lpyr_stack(Mat image, int levels);

vector<Mat> ideal_bandpassing(vector<Mat> input, int dim, double wl, double wh, int samplingRate);