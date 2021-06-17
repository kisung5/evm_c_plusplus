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

int amplify_spatial_lpyr_temporal_ideal(string inFile, string outDir, double alpha,
    double lambda_c, double fl, double fh, double samplingRate, double chromAttenuation);

int amplify_spatial_lpyr_temporal_iir(string inFile, string outDir, double alpha,
    double lambda_c, double r1, double r2, double chromAttenuation);

/*
Spatial filter functions
*/

vector<Mat> build_GDown_stack(string vidFile, int startIndex, int endIndex, int level);

int maxPyrHt(int frameWidth, int frameHeight, int filterSizeX, int filterSizeY);

vector<Mat> buildLpyrFromGauss(Mat image, int levels);

vector<vector<Mat>> build_Lpyr_stack(string vidFile, int startIndex, int endIndex);

Mat reconLpyr(vector<Mat> pyr);

/*
Temporal filter functions
*/

vector<vector<Mat>> ideal_bandpassing_lpyr(vector<vector<Mat>>& input, int dim, double wl, double wh, double samplingRate);

