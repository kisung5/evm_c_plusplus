#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <numeric>

#include "processing_functions.h"
#include "im_conv.h"

/**
* Spatial Filtering : Gaussian blurand down sample
* Temporal Filtering : Ideal bandpass
*
* Copyright(c) 2021 Tecnologico de Costa Rica.
*
* Authors: Eduardo Moya Bello, Ki - Sung Lim
* Date : April 2021
*
* This work was based on a project EVM
*
* Original copyright(c) 2011 - 2012 Massachusetts Institute of Technology,
* Quanta Research Cambridge, Inc.
*
* Original authors : Hao - yu Wu, Michael Rubinstein, Eugene Shih,
* License : Please refer to the LICENCE file (MIT license)
* Original date : June 2012
**/

constexpr auto MAX_FILTER_SIZE = 5;
constexpr auto PYR_BORDER_TYPE = 2;

using namespace cv;
using namespace std;

int amplify_spatial_lpyr_temporal_ideal(string inFile, string outDir, int alpha,
    int lambda_c, double fl, double fh, int samplingRate, int chromAttenuation) {
    // Creates the result video name
    string outName = outDir + inFile + "-ideal-from-" + to_string(fl) + "-to-" +
        to_string(fh) + "-alpha-" + to_string(alpha) + "-lambda_c-" + to_string(lambda_c) +
        "-chromAtn-" + to_string(chromAttenuation) + ".avi";

    setBreakOnError(true);
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    //VideoCapture video(0);
    VideoCapture video(inFile);

    // Check if video opened successfully
    if (!video.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Extract video info
    int vidHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = video.get(CAP_PROP_FRAME_WIDTH);
    int nChannels = 3;
    int fr = video.get(CAP_PROP_FPS);
    int len = video.get(CAP_PROP_FRAME_COUNT);

    // Compute maximum pyramid height for every frame
    int max_ht = 1 + maxPyrHt(vidWidth, vidHeight, MAX_FILTER_SIZE, MAX_FILTER_SIZE);

    // Define variables
    Mat frame;
    Mat newFrame;
    vector<Mat> pyr_stack;

    while (1) {
        // Capture frame-by-frame
        video >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Modify the frame for the implementation
        cvtColor(frame, newFrame, COLOR_BGR2RGB);
        newFrame = im2double(frame);
        newFrame = rgb2ntsc(frame);

        pyr_stack = build_Lpyr_stack(newFrame, max_ht);

        //vector<Mat> filtered_stack = ideal_bandpassing(pyr_stack, 3, fl, fh, samplingRate);

        //const Mat lFrame = lpyr[0];
        //double value = lFrame.at<double>(1, 0);

        // Display the resulting frame
        imshow("Frame", frame);
        imshow("New Frame", newFrame);
        imshow("Lpyr lvl 0", pyr_stack[0]);
        //
        /*
        imshow("Lpyr lvl 1", lpyr[1]);
        imshow("Lpyr lvl 2", lpyr[2]);
        imshow("Lpyr lvl 3", lpyr[3]);
        imshow("Lpyr lvl 4", lpyr[4]);
        imshow("Lpyr lvl 5", lpyr[5]);
        imshow("Lpyr lvl 6", lpyr[6]);
        imshow("Lpyr lvl 7", lpyr[7]);
        */

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(25);
        if (c == 27)
            break;

    }

    // When everything done, release the video capture object
    video.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}


int maxPyrHt(int frameWidth, int frameHeight, int filterSizeX, int filterSizeY) {
    // 1D image
    if (frameWidth == 1 || frameHeight == 1) {
        frameWidth = frameHeight = 1;
        filterSizeX = filterSizeY = filterSizeX * filterSizeY;
    }
    // 2D image
    else if (filterSizeX == 1 || filterSizeY == 1) {
        filterSizeY = filterSizeX;
    }
    // Stop condition
    if (frameWidth < filterSizeX || frameWidth < filterSizeY ||
        frameHeight < filterSizeX || frameHeight < filterSizeY)
    {
        return 0;
    }
    // Next level
    else {
        return 1 + maxPyrHt(frameWidth / 2, frameHeight / 2, filterSizeX, filterSizeY);
    }
}


vector<Mat> build_Lpyr_stack(Mat image, int levels) {
    vector<Mat> laplacianPyramid;
    Mat up, down, lap;
    for (int l = 0; l < levels - 1; l++) {
        pyrDown(image, down);
        pyrUp(down, up, Size(image.cols, image.rows));
        lap = image - up;
        laplacianPyramid.push_back(lap);
        image = down;
    }
    laplacianPyramid.push_back(image);
    return laplacianPyramid;
}

/**
* FILTERED = ideal_bandpassing(INPUT, DIM, WL, WH, SAMPLINGRATE)
*
* Apply ideal band pass filter on INPUT along dimension DIM.
*
* WL: lower cutoff frequency of ideal band pass filter
* WH : higher cutoff frequency of ideal band pass filter
* SAMPLINGRATE : sampling rate of INPUT
*
* Copyright(c) 2011 - 2012 Massachusetts Institute of Technology,
*  Quanta Research Cambridge, Inc.
*
* Authors : Hao - yu Wu, Michael Rubinstein, Eugene Shih,
* License : Please refer to the LICENCE file
* Date : June 2012
*
* --Update--
* Code translated to C++
* Author: Ki - Sung Lim
* Date: May 2021
**/
vector<Mat> ideal_bandpassing(vector<Mat> input, int dim, double wl, double wh, int samplingRate)
{

    // Comprobation of the dimentions
    if (dim > 1 + input[0].channels()) {
        cout << "Exceed maximun dimension" << endl;
        exit(1);
    }

    cout << "F1: " + to_string(wl) + " F2: " + to_string(wh) << endl;

    vector<Mat> filtered;

    int n = (int)input.size();

    vector<int> Freq_temp(n);
    iota(begin(Freq_temp), end(Freq_temp), 0); //0 is the starting number

    Mat Freq(Freq_temp, false);
    double alpha = (double)samplingRate / (double)n;
    Freq.convertTo(Freq, CV_64FC1, alpha);

    // Testing the values in Freq [OK]
    //cout << Freq.at<double>(0, 0) << endl;
    //cout << Freq.at<double>(1, 0) << endl;
    //cout << Freq.at<double>(69, 0) << endl;
    //cout << Freq.at<double>(70, 0) << endl;
    //cout << Freq.at<double>(71, 0) << endl;
    //cout << Freq.at<double>(79, 0) << endl;
    //cout << Freq.at<double>(80, 0) << endl;

    Mat mask = (Freq > wl) & (Freq < wh);

    // Testing the values in the mask [OK]
    //cout << mask.at<bool>(0, 0) << endl;
    //cout << mask.at<bool>(1, 0) << endl;
    //cout << mask.at<bool>(69, 0) << endl;
    //cout << mask.at<bool>(70, 0) << endl;
    //cout << mask.at<bool>(71, 0) << endl;
    //cout << mask.at<bool>(79, 0) << endl;
    //cout << mask.at<bool>(80, 0) << endl;

    int total_rows = input[0].rows * input[0].cols * input[0].channels();

    Mat temp_dft(total_rows, n, CV_64F);

    int pos_temp = 0;
    for (int x = 0; x < input[0].rows; x++) {
        for (int y = 0; y < input[0].cols; y++) {

            for (int i = 0; i < n; i++) {
                Vec3d pix_colors = input[i].at<Vec3d>(x, y);
                temp_dft.at<double>(pos_temp, i) = pix_colors[0];
                temp_dft.at<double>(pos_temp + 1, i) = pix_colors[1];
                temp_dft.at<double>(pos_temp + 2, i) = pix_colors[2];
            }

        }
    }

    // Testing the values in temp_dft [OK]
    //cout << temp_dft.at<double>(0, 0) << endl;
    //cout << temp_dft.at<double>(1, 0) << endl;
    //cout << temp_dft.at<double>(2, 0) << endl;
    //cout << temp_dft.at<double>(3, 0) << endl;
    //cout << temp_dft.at<double>(4, 0) << endl;
    //cout << temp_dft.at<double>(5, 0) << endl;
    //cout << temp_dft.at<double>(6, 0) << endl;
    //cout << temp_dft.at<double>(7, 0) << endl;
    //cout << temp_dft.at<double>(8, 0) << endl;

    //cout << input[0].at<Vec3d>(0, 0) << endl;
    //cout << input[0].at<Vec3d>(0, 1) << endl;
    //cout << input[0].at<Vec3d>(0, 2) << endl;

    //cout << temp_dft.at<double>(total_rows-9, n - 1) << endl;
    //cout << temp_dft.at<double>(total_rows-8, n - 1) << endl;
    //cout << temp_dft.at<double>(total_rows-7, n - 1) << endl;
    //cout << temp_dft.at<double>(total_rows-6, n - 1) << endl;
    //cout << temp_dft.at<double>(total_rows-5, n - 1) << endl;
    //cout << temp_dft.at<double>(total_rows-4, n - 1) << endl;
    //cout << temp_dft.at<double>(total_rows-3, n - 1) << endl;
    //cout << temp_dft.at<double>(total_rows-2, n - 1) << endl;
    //cout << temp_dft.at<double>(total_rows-1, n - 1) << endl;

    //cout << input[n-1].at<Vec3d>(input[0].rows-1, input[0].cols - 3) << endl;
    //cout << input[n-1].at<Vec3d>(input[0].rows-1, input[0].cols - 2) << endl;
    //cout << input[n-1].at<Vec3d>(input[0].rows-1, input[0].cols - 1) << endl;

    Mat input_dft, input_idft;

    dft(temp_dft, input_dft, DFT_ROWS | DFT_COMPLEX_OUTPUT);

    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < n; j++) {
            if (!mask.at<bool>(j, 0)) {
                Vec2d temp_zero_vector(0.0f, 0.0f);
                input_dft.at<Vec2d>(i, j) = temp_zero_vector;
            }
        }
    }

    //cout << input_dft.at<Vec2d>(0, 0) << endl;
    //cout << input_dft.at<Vec2d>(0, 1) << endl;
    //cout << input_dft.at<Vec2d>(0, 2) << endl;
    //cout << input_dft.at<Vec2d>(0, 3) << endl;
    //cout << input_dft.at<Vec2d>(0, 68) << endl;
    //cout << input_dft.at<Vec2d>(0, 69) << endl;
    //cout << input_dft.at<Vec2d>(0, 70) << endl;
    //cout << input_dft.at<Vec2d>(0, 71) << endl;
    //cout << input_dft.at<Vec2d>(0, 72) << endl;
    //cout << input_dft.at<Vec2d>(0, 73) << endl;
    //cout << input_dft.at<Vec2d>(0, 74) << endl;
    //cout << input_dft.at<Vec2d>(0, 75) << endl;
    //cout << input_dft.at<Vec2d>(0, 76) << endl;
    //cout << input_dft.at<Vec2d>(0, 77) << endl;
    //cout << input_dft.at<Vec2d>(0, 78) << endl;
    //cout << input_dft.at<Vec2d>(0, 79) << endl;
    //cout << input_dft.at<Vec2d>(0, 80) << endl;
    //cout << input_dft.at<Vec2d>(0, 81) << endl;

    idft(input_dft, input_idft, DFT_ROWS);

    // Testing the values for the transformation [OK]
    cout << input_idft.at<Vec2d>(0, 0) << endl;
    cout << input_idft.at<Vec2d>(1, 0) << endl;
    cout << input_idft.at<Vec2d>(2, 0) << endl;
    cout << input_idft.at<Vec2d>(3, 0) << endl;
    cout << input_idft.at<Vec2d>(4, 0) << endl;
    cout << input_idft.at<Vec2d>(5, 0) << endl;
    cout << input_idft.at<Vec2d>(6, 0) << endl;


    filtered.reserve(n);

    for (int i = 0; i < n; i++) {
        Mat temp_filtframe(input[0].rows, input[0].cols, CV_64FC3);
        pos_temp = 0;
        for (int x = 0; x < input[0].rows; x++) {
            for (int y = 0; y < input[0].cols; y++) {


                Vec3d pix_colors;
                pix_colors[0] = input_idft.at<Vec2d>(pos_temp, i)[0];
                pix_colors[1] = input_idft.at<Vec2d>(pos_temp + 1, i)[0];
                pix_colors[2] = input_idft.at<Vec2d>(pos_temp + 2, i)[0];
                temp_filtframe.at<Vec3d>(x, y) = pix_colors;

                pos_temp += 3;
            }
        }

        filtered.push_back(temp_filtframe.clone());
    }

    // Testing the values for the filtered

    //cout << input_idft.at<Vec2d>(0, 0) << endl;
    //cout << input_idft.at<Vec2d>(1, 0) << endl;
    //cout << input_idft.at<Vec2d>(2, 0) << endl;
    //cout << input_idft.at<Vec2d>(3, 0) << endl;
    //cout << input_idft.at<Vec2d>(4, 0) << endl;
    //cout << input_idft.at<Vec2d>(5, 0) << endl;
    //cout << input_idft.at<Vec2d>(6, 0) << endl;
    //cout << input_idft.at<Vec2d>(7, 0) << endl;
    //cout << input_idft.at<Vec2d>(8, 0) << endl;

    //cout << filtered[0].at<Vec3d>(0, 0) << endl;
    //cout << filtered[0].at<Vec3d>(0, 1) << endl;
    //cout << filtered[0].at<Vec3d>(0, 2) << endl;

    return filtered;
}