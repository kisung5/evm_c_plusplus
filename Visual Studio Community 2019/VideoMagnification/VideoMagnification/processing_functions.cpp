#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <numeric>
#include <omp.h>

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


vector<Mat> buildLpyr(Mat image, int levels) {
    vector<Mat> laplacianPyramid(levels);
    Mat up, down, lap;
    for (int l = 0; l < levels - 1; l++) {
        pyrDown(image, down, Size(image.cols / 2, image.rows / 2), BORDER_REFLECT101);
        pyrUp(down, up, Size(image.cols, image.rows), BORDER_REFLECT101);
        lap = image - up;
        laplacianPyramid[l] = lap.clone();
        image = down.clone();
    }
    int maxLevelIndex = levels - 1;
    laplacianPyramid[maxLevelIndex] = image.clone();
    return laplacianPyramid;
}

vector<vector<Mat>> build_Lpyr_stack(string vidFile, int startIndex, int endIndex) {
    // Read video
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(vidFile);

    // Extract video info
    int vidHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = video.get(CAP_PROP_FRAME_WIDTH);

    // Compute maximum pyramid height for every frame
    int max_ht = 1 + maxPyrHt(vidWidth, vidHeight, MAX_FILTER_SIZE, MAX_FILTER_SIZE);

    vector<vector<Mat>> pyr_stack(endIndex, vector<Mat>(max_ht));

    double start, end;
    for (int i = startIndex; i < endIndex; i++) {
        //start = omp_get_wtime();
        // Define variables
        Mat frame, rgbframe, ntscframe;

        // Capture frame-by-frame
        video >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        cvtColor(frame, rgbframe, COLOR_BGR2RGB);
        rgbframe = im2double(rgbframe);
        ntscframe = rgb2ntsc(rgbframe);
        vector<Mat> pyr_output = buildLpyr(ntscframe, max_ht);
        pyr_stack[i] = pyr_output;
        //end = omp_get_wtime();
        //cout << "Iteration " << i << " time = " << end - start << endl;
    }

    return pyr_stack;
}

vector<vector<Mat>> ideal_bandpassing_lpyr(vector<vector<Mat>> input, int dim, double wl, double wh, int samplingRate) {
    // Get channel count
    int channels = input[0][0].channels();

    // Comprobation of the dimentions
    if (dim > 1 + channels) {
        std::cout << "Exceed maximun dimension" << endl;
        exit(1);
    }

    vector<vector<Mat>> filtered = input;

    int n = input.size();

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

    int total_pixels = 0;
    int levels = input[0].size();

    #pragma omp parallel for
    for (int level = 0; level < levels; level++) {
        total_pixels += input[0][level].cols * input[0][level].rows * channels;
    }

    int pos_temp = 0;
    Mat tmp(total_pixels, n, CV_64FC1);

    // 0.155 s elapsed since start
 
    // Here we populate the forementioned matrix
    // 14.99 s
    // #pragma omp parallel for collapse(2)
    for (int level = 0; level < levels; level++) {
        for (int x = 0; x < input[0][level].rows; x++) {
            // #pragma omp parallel for reduction(+:pos_temp)
            for (int y = 0; y < input[0][level].cols; y++) {
                // #pragma omp parallel for
                for (int i = 0; i < n; i++) {
                    Vec3d pix_colors = input[i][level].at<Vec3d>(x, y);
                    tmp.at<double>(pos_temp, i) = pix_colors[0];
                    tmp.at<double>(pos_temp + 1, i) = pix_colors[1];
                    tmp.at<double>(pos_temp + 2, i) = pix_colors[2];
                }
                pos_temp += 3;
            }
        }
    }

    /*
    cout << "0: " << tmp.at<double>(0, 0) << endl;
    cout << "1: " << tmp.at<double>(0, 1) << endl;
    cout << "2: " << tmp.at<double>(0, 2) << endl; 
    cout << "3: " << tmp.at<double>(0, 3) << endl;
    cout << "4: " << tmp.at<double>(0, 4) << endl;
    cout << "5: " << tmp.at<double>(0, 5) << endl;
    cout << "6: " << tmp.at<double>(0, 6) << endl;
    cout << "7: " << tmp.at<double>(0, 7) << endl;
    cout << "8: " << tmp.at<double>(0, 8) << endl;
    cout << "9: " << tmp.at<double>(0, 9) << endl;
    cout << "10: " << tmp.at<double>(0, 10) << endl;
    */
    dft(tmp, tmp, DFT_ROWS | DFT_COMPLEX_OUTPUT);

    // Filtering the video matrix with a mask
    // 15.4 s
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < total_pixels; i++) {
        for (int j = 0; j < n; j++) {
            if (!mask.at<bool>(j, 0)) {
                Vec2d temp_zero_vector(0.0f, 0.0f);
                tmp.at<Vec2d>(i, j) = temp_zero_vector;
            }
        }
    }

    // 1-D inverse DFT applied for every row, complex output 
    // Only the real part is importante
    idft(tmp, tmp, DFT_ROWS | DFT_COMPLEX_INPUT | DFT_SCALE);

     // Reording the matrix to a vector of matrixes, 
    // contrary of what was done for temp_dft
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        vector<Mat> levelsVector(levels);
        for (int level = 0; level < levels; level++) {
            Mat temp_filtframe(input[0][level].rows, input[0][level].cols, CV_64FC3);
            pos_temp = 0;
            //#pragma omp parallel for reduction(+:pos_temp)
            for (int x = 0; x < input[0][level].rows; x++) {
                //#pragma omp parallel for
                for (int y = 0; y < input[0][level].cols; y++) {
                    Vec3d pix_colors;
                    pix_colors[0] = tmp.at<Vec2d>(pos_temp, i)[0];
                    pix_colors[1] = tmp.at<Vec2d>(pos_temp + 1, i)[0];
                    pix_colors[2] = tmp.at<Vec2d>(pos_temp + 2, i)[0];
                    temp_filtframe.at<Vec3d>(x, y) = pix_colors;
                    pos_temp += 3;
                }
            }
            levelsVector[level] = temp_filtframe.clone();
        }
        filtered[i] = levelsVector;
    }

    return filtered;
}

int amplify_spatial_lpyr_temporal_ideal(string inFile, string outDir, int alpha,
    int lambda_c, double fl, double fh, int samplingRate, int chromAttenuation) {

    double itime, spatial_time, temporal_time;
    // Creates the result video name
    string outName = "guitar-ideal-from-" + to_string(fl) + "-to-" +
        to_string(fh) + "-alpha-" + to_string(alpha) + "-lambda_c-" + to_string(lambda_c) +
        "-chromAtn-" + to_string(chromAttenuation) + ".avi";

    setBreakOnError(true);
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    //VideoCapture video(0);
    VideoCapture video(inFile);

    // Check if video opened successfully
    if (!video.isOpened()) {
        std::cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Extract video info
    int len = video.get(CAP_PROP_FRAME_COUNT);
    int startIndex = 0;
    int endIndex = len - 10;
    int vidHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = video.get(CAP_PROP_FRAME_WIDTH);
    int fr = video.get(CAP_PROP_FPS);

    // Compute maximum pyramid height for every frame
    int max_ht = 1 + maxPyrHt(vidWidth, vidHeight, MAX_FILTER_SIZE, MAX_FILTER_SIZE);

    itime = omp_get_wtime();

    vector<vector<Mat>> pyr_stack = build_Lpyr_stack(inFile, startIndex, endIndex);

    spatial_time = omp_get_wtime();

    std::cout << "Spatial filtering: " << spatial_time - itime << endl;

    vector<vector<Mat>> filteredStack = ideal_bandpassing_lpyr(pyr_stack, 3, fl, fh, samplingRate);

    temporal_time = omp_get_wtime();

    std::cout << "Temporal filtering: " << temporal_time - spatial_time << endl;

    cout << filteredStack[0][0].at<Vec3d>(0, 0) << endl;
    cout << filteredStack[0][0].at<Vec3d>(0, 1) << endl;
    cout << filteredStack[0][0].at<Vec3d>(1, 0) << endl;
    cout << filteredStack[1][0].at<Vec3d>(0, 0) << endl;
    cout << filteredStack[1][0].at<Vec3d>(0, 1) << endl;


    // Amplify color channels in NTSC
    for (vector<Mat> frame : filteredStack) {
        for (Mat levelFrame : frame) {
            for (int x = 0; x < levelFrame.rows; x++) {
                for (int y = 0; y < levelFrame.cols; y++) {
                    Vec3d this_pixel = levelFrame.at<Vec3d>(x, y);
                    this_pixel[0] = this_pixel[0] * alpha;
                    this_pixel[1] = this_pixel[1] * alpha * chromAttenuation;
                    this_pixel[2] = this_pixel[2] * alpha * chromAttenuation;
                    levelFrame.at<Vec3d>(x, y) = this_pixel;
                }
            }
        }
    }

    // Render on the input video to make the output video
    // Define the codec and create VideoWriter object
    VideoWriter videoOut(outName, VideoWriter::fourcc('M', 'J', 'P', 'G'), fr,
        Size(vidWidth, vidHeight));

    cout << "Rendering... ";
    int k = 0;
    for (int i = startIndex; i < endIndex; i++) {

        Mat frame, rgbframe, ntscframe, filt_ind, filtered, out_frame;
        // Capture frame-by-frame
        video >> frame;

        imshow("Original", frame);

        // Color conversion GBR 2 NTSC
        cvtColor(frame, rgbframe, COLOR_BGR2RGB);
        rgbframe = im2double(rgbframe);
        ntscframe = rgb2ntsc(rgbframe);

        imshow("Converted", ntscframe);

        filt_ind = filteredStack[k][0];
        imshow("Filtered stack", filt_ind);

        Size img_size(vidWidth, vidHeight);//the dst image size,e.g.100x100
        resize(filt_ind, filtered, img_size, 0, 0, INTER_CUBIC);//resize image

        filtered = filtered + ntscframe;
        imshow("Filtered", filtered);

        frame = ntsc2rgb(filtered);
        imshow("Frame", frame);

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
        imshow("Rgb frame", rgbframe);

        cvtColor(rgbframe, out_frame, COLOR_RGB2BGR);
        imshow("Out frame", out_frame);


        // Write the frame into the file 'outcpp.avi'
        videoOut.write(out_frame);

        k++;

        // Display the resulting frame
        
        
        //Press  ESC on keyboard to exit
        //char c = (char)waitKey(25);
        //if (c == 27)
            //break;
    }

    // When everything done, release the video capture and write object
    video.release();
    videoOut.release();

    cout << "Finished" << endl;

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}
