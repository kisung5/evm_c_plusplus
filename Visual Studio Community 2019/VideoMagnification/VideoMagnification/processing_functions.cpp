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

double FILTER_DATA[] = { 0.088388347648318, 0.353553390593274, 0.530330085889911, 0.353553390593274, 0.088388347648318 };
Mat FILTER(5, 1, CV_64FC1, FILTER_DATA);
Mat T_FILTER(1, 5, CV_64FC1, FILTER_DATA);

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

Mat corrDn(Mat image, Mat filter, int heightStep, int widthStep) {
    resize(image, image, Size(image.cols / widthStep, image.rows / heightStep), 0, 0, INTER_NEAREST);
    //filter2D(image, image, -1, filter, Point(-1, -1), 0, BORDER_REFLECT101);
    return image;
}

Mat upConv(Mat image, Mat filter, int widthStep, int heightStep) {
    //filter2D(image, image, -1, filter, Point(-1, -1), 0, BORDER_REFLECT101);
    resize(image, image, Size(image.cols * widthStep, image.rows * heightStep), 0, 0, INTER_NEAREST);
    return image;
}

vector<Mat> buildLpyr(Mat image, int levels) {
    vector<Mat> laplacianPyramid(levels);

    Mat lap, down, up;

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

Mat buildLpyr2(Mat image, int levels) {
    if (levels <= 1) {
        image.reshape(0, (int)image.total());
        return image;
    }
    else {
        Mat lo, lo2, hi, hi2, npyr, pyr;
        if (image.cols == 1) {
            lo2 = corrDn(image, FILTER, 2, 1);
            hi2 = upConv(lo2, FILTER, 2, 1);
        }
        else if (image.rows == 1) {
            lo2 = corrDn(image, T_FILTER, 1, 2);
            hi2 = upConv(lo2, T_FILTER, 1, 2);
        }
        else {
            lo = corrDn(image, T_FILTER, 1, 2);
            lo2 = corrDn(lo, FILTER, 2, 1);
            hi = upConv(lo2, FILTER, 2, 1);
            hi2 = upConv(hi, T_FILTER, 1, 2);
        }

        npyr = buildLpyr2(lo2, levels - 1);

        int maxWidth = cv::max(hi2.cols, image.cols);
        int maxHeight = cv::max(hi2.rows, image.rows);
        resize(image, image, Size(maxWidth, maxHeight));
        resize(hi2, hi2, Size(maxWidth, maxHeight));
        hi2 = image - hi2;

        hi2 = hi2.reshape(0, (int)hi2.total());
        npyr = npyr.reshape(0, (int)npyr.total());
        vconcat(hi2, npyr, pyr);

        return pyr.clone();
    }
}

Mat buildLpyr3(Mat image, int levels) {
    Mat lap, down, up, reshaped;
    pyrDown(image, down, Size((image.cols + 1) / 2, (image.rows + 1) / 2), BORDER_REFLECT101);
    pyrUp(down, up, Size(image.cols, image.rows), BORDER_REFLECT101);
    image = image - up;
    lap = image.reshape(0, (int)image.total());
    image = down.clone();

    for (int l = 1; l < levels - 1; l++) {
        pyrDown(image, down, Size((image.cols + 1) / 2, (image.rows + 1) / 2), BORDER_REFLECT101);
        pyrUp(down, up, Size(image.cols, image.rows), BORDER_REFLECT101);
        image = image - up;
        image = image.reshape(0, (int)image.total());
        vconcat(lap, image, lap);
        image = down.clone();
    }

    image = image.reshape(0, (int)image.total());
    vconcat(lap, image, lap);

    return lap;
}

vector<Mat> buildLpyr4(Mat image, int levels) {
    vector<Mat> gaussianPyramid(levels);
    vector<Mat> expandedPyramid(levels - 1);
    vector<Mat> laplacianPyramid(levels);

    gaussianPyramid[0] = image.clone();

    for (int l = 0; l < levels - 1; l++) {
        pyrDown(gaussianPyramid[l], gaussianPyramid[l + 1], Size((gaussianPyramid[l].cols + 1) / 2, (gaussianPyramid[l].rows + 1) / 2), BORDER_REFLECT101);
        pyrUp(gaussianPyramid[l + 1], expandedPyramid[l], Size(gaussianPyramid[l].cols, gaussianPyramid[l].rows), BORDER_REFLECT101);
        laplacianPyramid[l] = gaussianPyramid[l] - expandedPyramid[l];
    }
    laplacianPyramid[levels - 1] = gaussianPyramid[levels - 1];

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
    vector<Mat> pyr_stack3(endIndex);

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
        //start = omp_get_wtime();
        vector<Mat> pyr_output = buildLpyr4(ntscframe, max_ht);
        //pyr_stack3[i] = buildLpyr3(ntscframe, max_ht);
        //end = omp_get_wtime();
        //cout << "Build Lpyr for frame " << i << ". Elapsed time = " << end - start << endl;
        //cout << pyr_output[0].at<Vec3d>(0, 0) << endl;
        //cout << pyr_output[0].at<Vec3d>(1, 0) << endl;
        //cout << pyr_output[0].at<Vec3d>(69, 0) << endl;
        //cout << pyr_output[0].at<Vec3d>(70, 0) << endl;
        //cout << pyr_output[0].at<Vec3d>(71, 0) << endl;
        //cout << pyr_output[0].at<Vec3d>(79, 0) << endl;
        //cout << pyr_output[0].at<Vec3d>(80, 0) << endl;
        //start = omp_get_wtime();
        //Mat pyr_output2 = buildLpyr2(ntscframe, max_ht);
        //end = omp_get_wtime();
        //cout << "Build Lpyr2 for frame " << i << ". Elapsed time = " << end - start << endl;
        // Testing the values in pyr_output2 
        //cout << pyr_output2.at<Vec3d>(0, 0) << endl;
        //cout << pyr_output2.at<Vec3d>(1, 0) << endl;
        //cout << pyr_output2.at<Vec3d>(69, 0) << endl;
        //cout << pyr_output2.at<Vec3d>(70, 0) << endl;
        //cout << pyr_output2.at<Vec3d>(71, 0) << endl;
        //cout << pyr_output2.at<Vec3d>(79, 0) << endl;
        //cout << pyr_output2.at<Vec3d>(80, 0) << endl;
        //cout << "" << endl;
        pyr_stack[i] = pyr_output;
        //end = omp_get_wtime();
        //cout << "Iteration " << i << " time = " << end - start << endl;
    }

    return pyr_stack;
}

vector<vector<Mat>> ideal_bandpassing_lpyr(vector<vector<Mat>>& input, int dim, double wl, double wh, int samplingRate) {
    /*
    Comprobation of the dimention
    It is so 'dim' doesn't excede the actual dimension of the input
    In Matlab you can shift the dimentions of a matrix, for example, 3x4x3 can be shifted to 4x3x3
    with the same values stored in the correspondant dimension.
    Here (C++) it is not applied any shifting, yet.
    */
    if (dim > 1 + input[0][0].channels()) {
        std::cout << "Exceed maximun dimension" << endl;
        exit(1);
    }

    vector<vector<Mat>> filtered = input;

    // Number of frames in the video
    // Represents time
    int n = input.size();

    // Temporal vector that's constructed for the mask
    // iota is used to fill the vector with a integer sequence 
    // [0, 1, 2, ..., n]
    vector<int> Freq_temp(n);
    iota(begin(Freq_temp), end(Freq_temp), 0); //0 is the starting number

    // Initialize the cv::Mat with the temp vector and without copying values
    Mat Freq(Freq_temp, false);
    double alpha = (double)samplingRate / (double)n;
    Freq.convertTo(Freq, CV_64FC1, alpha);

    Mat mask = (Freq > wl) & (Freq < wh); // creates a boolean matrix/mask

    // Sum of total pixels to be processed
    int total_pixels = 0;
    int levels = input[0].size();
    #pragma omp parallel for
    for (int level = 0; level < levels; level++) {
        total_pixels += input[0][level].cols * input[0][level].rows * input[0][0].channels();
    }

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

    If you didn't get it: pixel_time-row/x-col/y-colorchannel
    */
    Mat tmp(total_pixels, n, CV_64FC1);

    // 0.155 s elapsed since start
 
    // Here we populate the forementioned matrix
    // 14.99 s
    #pragma omp parallel for
    for (int level = 0; level < levels; level++) {
        #pragma omp parallel for
        for (int x = 0; x < input[0][level].rows; x++) {
            #pragma omp parallel for
            for (int y = 0; y < input[0][level].cols; y++) {
                #pragma omp parallel for shared(input, tmp)
                for (int i = 0; i < n; i++) {
                    int pos_temp = 3 * (y + x * input[0][level].cols);
                    Vec3d pix_colors = input[i][level].at<Vec3d>(x, y);
                    tmp.at<double>(pos_temp, i) = pix_colors[0];
                    tmp.at<double>(pos_temp + 1, i) = pix_colors[1];
                    tmp.at<double>(pos_temp + 2, i) = pix_colors[2];
                }
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

    #pragma omp parallel for
    for (int i = 0; i < total_pixels; i++) {
        #pragma omp parallel for shared(tmp)
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
    #pragma omp parallel for shared(input)
    for (int i = 0; i < n; i++) {
        vector<Mat> levelsVector(levels);
        #pragma omp parallel for shared(levelsVector)
        for (int level = 0; level < levels; level++) {
            Mat temp_filtframe(input[0][level].rows, input[0][level].cols, CV_64FC3);
            #pragma omp parallel for
            for (int x = 0; x < input[0][level].rows; x++) {
                #pragma omp parallel for shared(tmp, temp_filtframe)
                for (int y = 0; y < input[0][level].cols; y++) {
                    int pos_temp = 3 * (y + x * input[0][level].cols);
                    
                    Vec3d pix_colors;
                    pix_colors[0] = tmp.at<Vec2d>(pos_temp, i)[0];
                    pix_colors[1] = tmp.at<Vec2d>(pos_temp + 1, i)[0];
                    pix_colors[2] = tmp.at<Vec2d>(pos_temp + 2, i)[0];
                    temp_filtframe.at<Vec3d>(x, y) = pix_colors;
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

    double itime, spatial_time, temporal_time, etime;

    string name;
    string delimiter = "/";

    size_t last = 0; size_t next = 0;
    while ((next = inFile.find(delimiter, last)) != string::npos) {
        last = next + 1;
    }

    name = inFile.substr(last);
    name = name.substr(0, name.find("."));
    cout << name << endl;
    cout << outDir << endl;

    // Creates the result video name
    string outName = outDir + name + "-ideal-from-" + to_string(fl) + "-to-" +
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

    Scalar colorAmp(alpha, alpha * chromAttenuation, alpha * chromAttenuation);

    /*
    cout << filteredStack[0][0].at<Vec3d>(0, 0) << endl;
    cout << filteredStack[0][0].at<Vec3d>(0, 1) << endl;
    cout << filteredStack[0][0].at<Vec3d>(1, 0) << endl;
    cout << filteredStack[1][0].at<Vec3d>(0, 0) << endl;
    cout << filteredStack[1][0].at<Vec3d>(0, 1) << endl;
    */


    // Amplify color channels in NTSC
    double ntsc_start = omp_get_wtime();

    #pragma omp parallel for shared(filteredStack, colorAmp)
    for (int frame = 0; frame < filteredStack.size(); frame++) {
        #pragma omp parallel for shared(filteredStack, colorAmp)
        for (int levelFrame = 0; levelFrame < filteredStack[frame].size(); levelFrame++) {
            multiply(filteredStack[frame][levelFrame], colorAmp, filteredStack[frame][levelFrame]);
        }
    }

    temporal_time = omp_get_wtime();

    std::cout << "NTSC: " << temporal_time - ntsc_start<< endl;
    std::cout << "Temporal filtering: " << temporal_time - spatial_time << endl;

    // Render on the input video to make the output video
    // Define the codec and create VideoWriter object
    VideoWriter videoOut(outName, VideoWriter::fourcc('M', 'J', 'P', 'G'), fr,
        Size(vidWidth, vidHeight));

    int k = 0;
    for (int i = startIndex; i < endIndex; i++) {

        Mat frame, rgbframe, ntscframe, filt_ind, filtered, out_frame;
        // Capture frame-by-frame
        video >> frame;

        // Color conversion GBR 2 NTSC
        cvtColor(frame, rgbframe, COLOR_BGR2RGB);
        rgbframe = im2double(rgbframe);
        ntscframe = rgb2ntsc(rgbframe);

        filt_ind = filteredStack[k][0];

        Size img_size(vidWidth, vidHeight);//the dst image size,e.g.100x100
        resize(filt_ind, filtered, img_size, 0, 0, INTER_CUBIC);//resize image

        filtered = filtered + ntscframe;

        frame = ntsc2rgb(filtered);

        #pragma omp parallel for collapse(3)
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

        // Display the resulting frame
        
        
        //Press  ESC on keyboard to exit
        //char c = (char)waitKey(25);
        //if (c == 27)
            //break;
    }

    etime = omp_get_wtime();

    std::cout << "Render: " << etime - temporal_time << endl;

    // When everything done, release the video capture and write object
    video.release();
    videoOut.release();

    cout << "Finished" << endl;

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}
