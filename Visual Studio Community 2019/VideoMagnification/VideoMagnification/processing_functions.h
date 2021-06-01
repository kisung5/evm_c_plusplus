#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*
Video processing functions
*/

int amplify_spatial_Gdown_temporal_ideal(string inFile, string outDir, double alpha, int level, 
	double f1, double fh, int samplingRate, double chromAttenuation);

int amplify_spatial_lpyr_temporal_butter(string inFile, string outDir, double alpha, double lambda_c,
	double fl, double fh, int samplingRate, double chromAttenuation);

int amplify_spatial_lpyr_temporal_ideal(string inFile, string outDir, int alpha, int lambda_c, 
	double fl, double fh, int samplingRate, int chromAttenuation);

/*
Spatial filter functions
*/

vector<Mat> build_GDown_stack(string vidFile, int startIndex, int endIndex, int level);

vector<Mat> buildLpyr(Mat image, int levels);

Mat buildLpyr2(Mat image, int levels);

Mat buildLpyr3(Mat image, int levels);

vector<Mat> buildLpyr4(Mat image, int levels);

Mat upConv(Mat image, Mat filter, int widthStep, int heightStep);

Mat corrDn(Mat image, Mat filter, int heightStep, int widthStep);

vector<vector<Mat>> build_Lpyr_stack(string vidFile, int startIndex, int endIndex);

int maxPyrHt(int frameWidth, int frameHeight, int filterSizeX, int filterSizeY);

Mat reconLpyr(vector<Mat> pyr);

/*
Temporal filter functions
*/

vector<Mat> ideal_bandpassing(vector<Mat> input, int dim, double wl, double wh, int samplingRate);

vector<vector<Mat>> ideal_bandpassing_lpyr(vector<vector<Mat>> input, int dim, double wl, double wh, int samplingRate);