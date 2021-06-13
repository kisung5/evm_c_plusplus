#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int amplify_spatial_lpyr_temporal_ideal(string inFile, string outDir, int alpha,
    int lambda_c, double fl, double fh, int samplingRate, int chromAttenuation);

int maxPyrHt(int frameWidth, int frameHeight, int filterSizeX, int filterSizeY);

vector<Mat> buildLpyr(Mat image, int levels);

Mat buildLpyr2(Mat image, int levels);

Mat buildLpyr3(Mat image, int levels);

vector<Mat> buildLpyr4(Mat image, int levels);

Mat upConv(Mat image, Mat filter, int widthStep, int heightStep);

Mat corrDn(Mat image, Mat filter, int heightStep, int widthStep);

vector<vector<Mat>> build_Lpyr_stack(string vidFile, int startIndex, int endIndex);

vector<vector<Mat>> ideal_bandpassing_lpyr(vector<vector<Mat>>& input, int dim, double wl, double wh, int samplingRate);

