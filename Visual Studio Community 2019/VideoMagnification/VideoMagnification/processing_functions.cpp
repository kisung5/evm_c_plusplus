}


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

int amplify_spatial_Gdown_temporal_ideal(string inFile, string outDir, double alpha,
    int level, double fl, double fh, int samplingRate, double chromAttenuation) 
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

    string name;
    string delimiter = "/";

    size_t last = 0; size_t next = 0; 
    while ((next = inFile.find(delimiter, last)) != string::npos) { 

        last = next + 1; 
    } 

    name = inFile.substr(last);
    name = name.substr(0, name.find("."));
    cout << name << endl;

    // Creates the result video name
    string outName = outDir + name + "-ideal-from-" + to_string(fl) + "-to-" +
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

    // Amplify color channels in NTSC
    for (Mat frame : filtered_stack) {
        for (int x = 0; x < frame.rows; x++) {
            for (int y = 0; y < frame.cols; y++) {
                Vec3d this_pixel = frame.at<Vec3d>(x, y);
                this_pixel[0] = this_pixel[0] * alpha;
                this_pixel[1] = this_pixel[1] * alpha * chromAttenuation;
                this_pixel[2] = this_pixel[2] * alpha * chromAttenuation;
                frame.at<Vec3d>(x, y) = this_pixel;
            }
        }
    }

    // Render on the input video to make the output video
    cout << "Rendering... ";
    int k = 0;
    for (int i = startIndex; i < endIndex; i++) {

        Mat frame, rgbframe, ntscframe, filt_ind, filtered, out_frame;
        // Capture frame-by-frame
        video >> frame;

        // Color conversion GBR 2 NTSC
        cvtColor(frame, rgbframe, COLOR_BGR2RGB);
        rgbframe = im2double(rgbframe);
        ntscframe = rgb2ntsc(rgbframe);

        filt_ind = filtered_stack[k];

        Size img_size(vidWidth, vidHeight);//the dst image size,e.g.100x100
        resize(filt_ind, filtered, img_size, 0, 0, INTER_CUBIC);//resize image

        filtered = filtered + ntscframe;

        frame = ntsc2rgb(filtered);

        for (int x = 0; x < frame.rows; x++) {
            for (int y = 0; y < frame.cols; y++) {
                Vec3d this_pixel = frame.at<Vec3d>(x, y);
                for (int z = 0; z < 3; z++) {
                    if (this_pixel[z] > 1) {
                        this_pixel[z] = 1;
                    }
                    
                    if (this_pixel[z] < 0) {
                        this_pixel[z] = 0;
                    }
                }

                frame.at<Vec3d>(x, y) = this_pixel;
            }
        }

        rgbframe = im2uint8(frame);

        cvtColor(rgbframe, out_frame, COLOR_RGB2BGR);

        // Write the frame into the file 'outcpp.avi'
        videoOut.write(out_frame);

        k++;
    }

    // When everything done, release the video capture and write object
    video.release();
    videoOut.release();

    cout << "Finished" << endl;

    // Closes all the frames
    destroyAllWindows();
    return 0;
}

/**
* Spatial Filtering: Laplacian pyramid
* Temporal Filtering: substraction of two butterworth lowpass filters
*                     with cutoff frequencies fh and fl
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
int amplify_spatial_lpyr_temporal_butter(string inFile, string outDir, double alpha, int lambda_c,
    double fl, double fh, int samplingRate, double chromAttenuation) {

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
* 
* --Notes--
* The commented sections that say "test" or "testing" are manual tests that were applied to the
* method or algorithm before them. You can just ignore them or use them if you want to check
* the results.
**/
vector<Mat> ideal_bandpassing(vector<Mat> input, int dim, double wl, double wh, int samplingRate)
{

    /*
    Comprobation of the dimention 
    It is so 'dim' doesn't excede the actual dimension of the input

    In Matlab you can shift the dimentions of a matrix, for example, 3x4x3 can be shifted to 4x3x3
    with the same values stored in the correspondant dimension.
    Here (C++) it is not applied any shifting, yet.
    */
    if (dim > 1 + input[0].channels()) {
        cout << "Exceed maximun dimension" << endl;
        exit(1);
    }

    // Printing the cut frequencies
    cout << "F1: " + to_string(wl) + " F2: " + to_string(wh) << endl;

    vector<Mat> filtered;

    // Number of frames in the video
    // Represents time
    int n = (int) input.size();

    // Temporal vector that's constructed for the mask
    // iota is used to fill the vector with a integer sequence 
    // [0, 1, 2, ..., n]
    vector<int> Freq_temp(n);
    iota(begin(Freq_temp), end(Freq_temp), 0); //0 is the starting number

    // Initialize the cv::Mat with the temp vector and without copying values
    Mat Freq(Freq_temp, false);
    double alpha = (double)samplingRate / (double)n;
    Freq.convertTo(Freq, CV_64FC1, alpha); // alpha is mult to every value
    
    // Testing the values in Freq [OK]
    //cout << Freq.at<double>(0, 0) << endl;
    //cout << Freq.at<double>(1, 0) << endl;
    //cout << Freq.at<double>(69, 0) << endl;
    //cout << Freq.at<double>(70, 0) << endl;
    //cout << Freq.at<double>(71, 0) << endl;
    //cout << Freq.at<double>(79, 0) << endl;
    //cout << Freq.at<double>(80, 0) << endl;

    Mat mask = (Freq > wl) & (Freq < wh); // creates a boolean matrix/mask

    // Testing the values in the mask [OK]
    //cout << mask.at<bool>(0, 0) << endl;
    //cout << mask.at<bool>(1, 0) << endl;
    //cout << mask.at<bool>(69, 0) << endl;
    //cout << mask.at<bool>(70, 0) << endl;
    //cout << mask.at<bool>(71, 0) << endl;
    //cout << mask.at<bool>(79, 0) << endl;
    //cout << mask.at<bool>(80, 0) << endl;

    // Sum of rows, columns and color channels of a single cv::Mat in input
    int total_rows = input[0].rows * input[0].cols * input[0].channels();

     /*
     Temporal matrix that is constructed so the DFT method (Discrete Fourier Transform)
     that OpenCV provides can be used. The most common use for the DFT in image 
     processing is the 2-D DFT, in this case we want 1-D DFT for every pixel time vector.
     Every row of the matrix is the timeline of an specific pixel.
     The structure of temp_dft is:

     [
          [pixel_0000, pixel_1000, pixel_2000, ..., pixel_n000],
          [pixel_0001, pixel_1001, pixel_2001, ..., pixel_n001],
          [pixel_0002, pixel_1002, pixel_2002, ..., pixel_n002],
          [pixel_0010, pixel_1010, pixel_2010, ..., pixel_n010],
          .
          .
          .
          [pixel_0xy0, pixel_1xy0, pixel_2xy0, ..., pixel_nxy0],
          .
          .
          .
          [pixel_0xy3, pixel_1xy3, pixel_2xy3, ..., pixel_nxy0],
     ]
      
     If you didn't see it: pixel_time-row/x-col/y-colorchannel
     */
    Mat temp_dft(total_rows, n, CV_64FC1);

    int pos_temp = 0;
    // Here we populate the forementioned matrix
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
    
    Mat input_dft, input_idft; // For DFT input / output 

    // 1-D DFT applied for every row, complex output 
    // (2 value real-complex vector)
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

    // 1-D inverse DFT applied for every row, complex output 
    // Only the real part is importante
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

    filtered.reserve(n); // Reserving heap memory space for the output

    // Reording the matrix to a vector of matrixes, 
    // contrary of what was done for temp_dft
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