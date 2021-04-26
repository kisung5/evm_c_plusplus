#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <sstream>

#include "processing_functions.h"

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
    int level, double fl, double fh, int samplingRate, int chromAttenuation) {

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
    int vidHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = video.get(CAP_PROP_FRAME_WIDTH);
    int nChannels = 3;
    int fr = video.get(CAP_PROP_FPS);
    int len = video.get(CAP_PROP_FRAME_COUNT);

    int startIndex = 0;
    int endIndex = len - 11;

    // Testing values
    cout << vidHeight << endl;
    cout << vidWidth << endl;
    cout << len << endl;

    // Write video
    // Define the codec and create VideoWriter object
    VideoWriter videoOut(outName, VideoWriter::fourcc('M', 'J', 'P', 'G'), fr, Size(vidWidth, vidHeight));

    
    // Compute Gaussian blur stack
    cout << "Spatial filtering... ";
    vector<Mat> Gdown_stack = build_GDown_stack(inFile, startIndex, endIndex, level);
    cout << "Finished" << endl;


    // Temporal filtering
    cout << "Temporal filtering... ";

    cout << "Finished" << endl;

    while (1) {

        Mat frame;
        // Capture frame-by-frame
        video >> frame;

        cout << "Channels: " + to_string(frame.channels()) << endl;
        cout << "Size: ";
        cout << frame.size << endl;
        cout << frame.type() << endl;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Display the resulting frame
        imshow("Frame", frame);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(25);
        if (c == 27)
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
    int vidHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = video.get(CAP_PROP_FRAME_WIDTH);
    int nChannels = 3;
    int t_size = endIndex - startIndex + 1;

    vector<Mat> GDown_stack;
    GDown_stack.reserve(t_size);

    for (int i = startIndex; i < endIndex; i++) {
        Mat frame;
        vector<Mat> pyr_output;
        // Capture frame-by-frame
        video >> frame;

        buildPyramid(frame, pyr_output, level+1, BORDER_REFLECT101);

        GDown_stack.push_back(pyr_output[level].clone());
    }

    video.release();

    return GDown_stack;
}