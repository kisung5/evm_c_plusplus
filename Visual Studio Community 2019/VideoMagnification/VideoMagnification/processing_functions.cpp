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

using namespace std;
using namespace cv;

int amplify_spatial_Gdown_temporal_ideal(string inFile, string outDir, int alpha,
    int level, double fl, double fh, int samplingRate, int chromAttenuation) 
{

    /// <summary>
    /// 
    /// </summary>
    /// <param name="inFile"></param>
    /// <param name="outDir"></param>
    /// <param name="alpha"></param>
    /// <param name="level"></param>
    /// <param name="fl"></param>
    /// <param name="fh"></param>
    /// <param name="samplingRate"></param>
    /// <param name="chromAttenuation"></param>
    /// <returns></returns>


    // Creates the result video name
    string outName = outDir + inFile + "-ideal-from-" + to_string(fl) + "-to-" +
        to_string(fh) + "-alpha-" + to_string(alpha) + "-level-" + to_string(level) +
        "-chromAtn-" + to_string(chromAttenuation) + ".avi";

    // Read video
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(inFile);

    // Check if camera opened successfully
    if (!video.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Extracting video info
    int vidHeight = (int)video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = (int)video.get(CAP_PROP_FRAME_WIDTH);
    int nChannels = 3;
    int fr = (int)video.get(CAP_PROP_FPS);
    int len = (int)video.get(CAP_PROP_FRAME_COUNT);

    int startIndex = 0;
    int endIndex = len - 10;

    // Testing values
    //cout << vidHeight << endl;
    //cout << vidWidth << endl;
    //cout << fr << endl;
    cout << len << endl;

    // Write video
    // Define the codec and create VideoWriter object
    VideoWriter videoOut(outName, VideoWriter::fourcc('M', 'J', 'P', 'G'), fr, 
        Size(vidWidth, vidHeight));

    
    // Compute Gaussian blur stack
    cout << "Spatial filtering... " << endl;
    vector<Mat> Gdown_stack = build_GDown_stack(inFile, startIndex, endIndex, level);
    cout << "Finished" << endl;

    // Getting the final Gdown Stack sizes same as Matlab code
    cout << "GDown Stack Size: len-" + to_string(Gdown_stack.size()) + 
        " rows-" + to_string(Gdown_stack[0].rows) +
        " cols-" + to_string(Gdown_stack[0].cols) + 
        " channels-" + to_string(Gdown_stack[0].channels()) << endl;

    // Testing spatial filtering values
    //cout << Gdown_stack[0].at<Vec3d>(0, 0) << endl;
    //cout << Gdown_stack[0].at<Vec3d>(0, 1) << endl;

    // Temporal filtering
    cout << "Temporal filtering... " << endl;
    vector<Mat> filtered_stack = ideal_bandpassing(Gdown_stack, 1, fl, fh, samplingRate);
    cout << "Finished" << endl;

    // Testing temporal filtering values
    //cout << filtered_stack[0].at<Vec3d>(0, 0) << endl;
    //cout << filtered_stack[0].at<Vec3d>(0, 1) << endl;
    //cout << filtered_stack[0].at<Vec3d>(0, 2) << endl;

    //for (int i = startIndex; i < endIndex; i++) {

    //    Mat frame = Gdown_stack[i];
    //    // Capture frame-by-frame
    //    //video >> frame;

    ////    //cout << "Channels: " + to_string(frame.channels()) << endl;
    ////    //cout << "Size: ";
    ////    //cout << frame.size << endl;
    ////    //cout << frame.type() << endl;

    //    // If the frame is empty, break immediately
    //    if (frame.empty())
    //        break;

    ////    // Display the resulting frame
    //    imshow("Frame", frame);

    //    // Press  ESC on keyboard to exit
    //    char c = (char)waitKey(25);
    //    if (c == 27)
    //        break;
    //    
    //    //break;
    //}

    // When everything done, release the video capture and write object
    video.release();
    videoOut.release();

    // Closes all the frames
    destroyAllWindows();
    return 0;
}


/**
* GDOWN_STACK = build_GDown_stack(VID_FILE, START_INDEX, END_INDEX, LEVEL)
*
* Apply Gaussian pyramid decomposition on VID_FILE from START_INDEX to
*  END_INDEX and select a specific band indicated by LEVEL
*
* GDOWN_STACK: stack of one band of Gaussian pyramid of each frame
*  the first dimension is the time axis
*  the second dimension is the y axis of the video
*  the third dimension is the x axis of the video
*  the forth dimension is the color channel
*
* Copyright(c) 2011 - 2012 Massachusetts Institute of Technology,
*  Quanta Research Cambridge, Inc.
* 
* Authors: Hao - yu Wu, Michael Rubinstein, Eugene Shih,
* License : Please refer to the LICENCE file
* Date : June 2012
* 
* --Update--
* Code translated to C++
* Author: Ki - Sung Lim
* Date: May 2021
**/
vector<Mat> build_GDown_stack(string vidFile, int startIndex, int endIndex, int level) {

    // Read video
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(vidFile);

    // Check if camera opened successfully
    if (!video.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        exit(1);
    }

    // Extracting video info
    int vidHeight = (int) video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = (int) video.get(CAP_PROP_FRAME_WIDTH);
    int nChannels = 3;
    int t_size = endIndex - startIndex + 1;

    vector<Mat> GDown_stack;
    GDown_stack.reserve(t_size);

    for (int i = startIndex; i < endIndex; i++) {
        Mat frame, rgbframe, ntscframe;
        vector<Mat> pyr_output;
        // Capture frame-by-frame
        video >> frame;

        cvtColor(frame, rgbframe, COLOR_BGR2RGB);

        //cout << rgbframe.at<Vec3b>(0, 0) << endl;
        //cout << rgbframe.at<Vec3b>(0, 1) << endl;
        //cout << rgbframe.at<Vec3b>(0, 2) << endl;

        rgbframe = im2double(rgbframe);
        //cout << rgbframe.at<Vec3d>(0, 0) << endl;
        //cout << rgbframe.at<Vec3d>(0, 1) << endl;
        //cout << rgbframe.at<Vec3d>(0, 2) << endl;

        ntscframe = rgb2ntsc(rgbframe);
        //cout << ntscframe.at<Vec3d>(0, 0) << endl;
        //cout << rgbframe.at<Vec3b>(0, 1) << endl;
        //cout << rgbframe.at<Vec3b>(0, 2) << endl;

        buildPyramid(ntscframe, pyr_output, level+1, BORDER_REFLECT101);

        //Mat bgrframe;
        //cvtColor(pyr_output[0], bgrframe, COLOR_RGB2BGR);
        //imshow("1", pyr_output[level]);
        //imshow("2", pyr_output[6]);
        //imshow("original", bgrframe);
        //waitKey(0);

        GDown_stack.push_back(pyr_output[level].clone());
    }

    video.release();

    return GDown_stack;
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

    int n = (int) input.size();

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

    Mat temp_dft(total_rows, n, CV_64FC1);

    int pos_temp = 0;
    for (int x = 0; x < input[0].rows; x++) {
        for (int y = 0; y < input[0].cols; y++) {
            
            for (int i = 0; i < n; i++) {
                Vec3d pix_colors = input[i].at<Vec3d>(x, y);
                temp_dft.at<double>(pos_temp, i) = pix_colors[0];
                temp_dft.at<double>(pos_temp+1, i) = pix_colors[1];
                temp_dft.at<double>(pos_temp+2, i) = pix_colors[2];
            }

            pos_temp += 3;
            
        }
    }
    
    // Testing the values in temp_dft [OK - PASSED]
    //int test_index = n-1;
    //cout << "-----Testing vectorizing values in " + to_string(test_index) + "-------" << endl;
    //for (int i = total_rows-3; i < total_rows; i++) {
    //    cout << temp_dft.at<double>(i, test_index) << endl;
    //}
    //cout << input[test_index].at<Vec3d>(5, 9) << endl;
    //cout << "----------End of tests----------" << endl;


    Mat input_dft, input_idft;

    dft(temp_dft, input_dft, DFT_ROWS | DFT_COMPLEX_OUTPUT);

    // Testing the values in the transformation DFT [OK - Passed]
    //int test_index = total_rows-1;
    //cout << "-----Testing DFT values in " + to_string(test_index) + "-------" << endl;
    //for (int i = 0; i < 10; i++) {
    //    cout << input_dft.at<Vec2d>(test_index, i) << endl;
    //}
    //for (int i = 879; i < 890; i++) {
    //    cout << input_dft.at<Vec2d>(test_index, i) << endl;
    //}
    //cout << "----------End of tests----------" << endl;

    // Filtering the video matrix with a mask
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < n; j++) {
            if (!mask.at<bool>(j, 0)) {
                Vec2d temp_zero_vector(0.0f, 0.0f);
                input_dft.at<Vec2d>(i, j) = temp_zero_vector;
            }
        }
    }

    idft(input_dft, input_idft, DFT_ROWS | DFT_COMPLEX_INPUT | DFT_SCALE);

    // Testing the values for the transformation IDFT [OK - Passed]
    //int test_index = total_rows-1;
    //cout << "-----Testing IDFT values in " + to_string(test_index) + "-------" << endl;
    //for (int i = 0; i < 10; i++) {
    //    cout << input_idft.at<Vec2d>(test_index, i) << endl;
    //}
    //for (int i = 879; i < 890; i++) {
    //    cout << input_idft.at<Vec2d>(test_index, i) << endl;
    //}
    //cout << "----------End of tests----------" << endl;


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

    // Testing the values for the filtered [OK - Passed]
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