#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <sstream>

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
    //cout << len << endl;

    // Write video
    // Define the codec and create VideoWriter object
    VideoWriter videoOut(outName, VideoWriter::fourcc('M', 'J', 'P', 'G'), fr, Size(vidWidth, vidHeight));

    
    // Compute Gaussian blur stack
    cout << "Spatial filtering... " << endl;
    vector<Mat> Gdown_stack = build_GDown_stack(inFile, startIndex, endIndex, level);
    cout << "Finished" << endl;

    // Testing spatial filtering values
    cout << Gdown_stack.size() << endl;
    cout << Gdown_stack[0].rows << endl;
    cout << Gdown_stack[0].cols << endl;
    cout << Gdown_stack[0].channels() << endl;
    cout << Gdown_stack[0].at<Vec3d>(0, 0) << endl;
    cout << Gdown_stack[0].at<Vec3d>(0, 1) << endl;

    // Temporal filtering
    cout << "Temporal filtering... " << endl;

    cout << "Finished" << endl;


    for (int i = startIndex; i < endIndex; i++) {

        Mat frame = Gdown_stack[i];
        // Capture frame-by-frame
        //video >> frame;

    //    //cout << "Channels: " + to_string(frame.channels()) << endl;
    //    //cout << "Size: ";
    //    //cout << frame.size << endl;
    //    //cout << frame.type() << endl;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

    //    // Display the resulting frame
        imshow("Frame", frame);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(25);
        if (c == 27)
            break;
        
        break;
    }

    // When everything done, release the video capture and write object
    video.release();
    videoOut.release();

    // Closes all the frames
    destroyAllWindows();
    return 0;
}


vector<Mat> build_GDown_stack(string vidFile, int startIndex, int endIndex, int level) {

    // Read video
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(vidFile);

    // Check if camera opened successfully
    //if (!video.isOpened()) {
    //    cout << "Error opening video stream or file" << endl;
    //    return GDown_stack;
    //}

    // Extracting video info
    int vidHeight = (int)video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = (int)video.get(CAP_PROP_FRAME_WIDTH);
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

        cout << rgbframe.at<Vec3b>(0, 0) << endl;
        //cout << rgbframe.at<Vec3b>(0, 1) << endl;
        //cout << rgbframe.at<Vec3b>(0, 2) << endl;

        rgbframe = im2double(rgbframe);
        cout << rgbframe.at<Vec3d>(0, 0) << endl;
        //cout << rgbframe.at<Vec3d>(0, 1) << endl;
        //cout << rgbframe.at<Vec3d>(0, 2) << endl;

        ntscframe = rgb2ntsc(rgbframe);
        cout << ntscframe.at<Vec3d>(0, 0) << endl;
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

        //video >> frame;
        //cout << frame.row(0).col(0) << endl;

        break;
    }

    video.release();

    return GDown_stack;
}


vector<Mat> ideal_bandpassing(vector<Mat> input, int dim, double wl, double wh,
    int samplingRate) 
{
    vector<Mat> filtered;

    return filtered;
}