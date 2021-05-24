#include <omp.h>
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

extern "C" {
#include "ellf.h"
}

using namespace std;
using namespace std::chrono;
using namespace cv;

extern "C" int butter_coeff(int, int, double, double);

constexpr auto MAX_FILTER_SIZE = 5;
constexpr auto PYR_BORDER_TYPE = 2;


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
int amplify_spatial_Gdown_temporal_ideal(string inFile, string outDir, double alpha,
    int level, double fl, double fh, int samplingRate, double chromAttenuation)
{
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
    cout << "Video information: Height-" << vidHeight << " Width-" << vidWidth
        << " FrameRate-" << fr << " Frames-" << len << endl;

    // Write video
    // Define the codec and create VideoWriter object
    VideoWriter videoOut(outName, VideoWriter::fourcc('M', 'J', 'P', 'G'), fr,
        Size(vidWidth, vidHeight));

    // Compute Gaussian blur stack
    cout << "Spatial filtering... " << endl;
    // Get starting timepoint
    auto start = high_resolution_clock::now();
    vector<Mat> Gdown_stack = build_GDown_stack(inFile, startIndex, endIndex, level);
    // Get ending timepoint
    auto stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to 
    // get durarion. To cast it to proper unit
    // use duration cast method
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time spatial filtering: " << duration.count() << " microseconds" << endl;
    cout << "Finished" << endl;

    // Getting the final Gdown Stack sizes same as Matlab code
    cout << "GDown Stack Size: len-" + to_string(Gdown_stack.size()) +
        " rows-" + to_string(Gdown_stack[0].rows) +
        " cols-" + to_string(Gdown_stack[0].cols) +
        " channels-" + to_string(Gdown_stack[0].channels()) << endl;

    // Temporal filtering
    cout << "Temporal filtering... " << endl;
    // Get starting timepoint
    start = high_resolution_clock::now();
    vector<Mat> filtered_stack = ideal_bandpassing(Gdown_stack, 1, fl, fh, samplingRate);
    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to get durarion.
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time temporal filtering: " << duration.count() << " microseconds" << endl;
    cout << "Finished" << endl;

    // Amplify color channels in NTSC
    // Get starting timepoint
    start = high_resolution_clock::now();
    Scalar color_amp(alpha, alpha * chromAttenuation, alpha * chromAttenuation);

#pragma omp parallel for shared(color_amp, filtered_stack)
    for (int ind_amp = 0; ind_amp < filtered_stack.size(); ind_amp++) {
        Mat frame, frame_result;
        frame = filtered_stack[ind_amp];
        multiply(frame, color_amp, frame_result);
        filtered_stack[ind_amp] = frame_result;
    }

    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to get durarion.
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time amplify: " << duration.count() << " microseconds" << endl;

    // Render on the input video to make the output video
    cout << "Rendering... ";
    // Get starting timepoint
    start = high_resolution_clock::now();

    //vector<Mat> result_video;
    //result_video.reserve(endIndex);

    for (int i = startIndex; i < endIndex; i++) {
        Mat frame;
        video >> frame;
        Gdown_stack[i] = frame;
    }

#pragma omp parallel for shared(video, Gdown_stack, filtered_stack)
    for (int i = startIndex; i < endIndex; i++) {

        Mat frame, rgbframe, d_frame, ntscframe, filt_ind, filtered, out_frame;
        // Capture frame-by-frame

        // Color conversion GBR 2 NTSC
        cvtColor(Gdown_stack[i], rgbframe, COLOR_BGR2RGB);
        d_frame = im2double(rgbframe);
        ntscframe = rgb2ntsc(d_frame);

        filt_ind = filtered_stack[i];

        Size img_size(vidWidth, vidHeight);//the dst image size,e.g.100x100
        resize(filt_ind, filtered, img_size, 0, 0, INTER_CUBIC);//resize image

        filt_ind = filtered + ntscframe;

        frame = ntsc2rgb(filt_ind);

        threshold(frame, out_frame, 0.0f, 0.0f, THRESH_TOZERO);
        threshold(out_frame, frame, 1.0f, 1.0f, THRESH_TRUNC);

        rgbframe = im2uint8(frame);

        cvtColor(rgbframe, out_frame, COLOR_RGB2BGR);

        filtered_stack[i] = out_frame.clone();
    }

    for (int i = startIndex; i < endIndex; i++) {
        // Write the frame into the file 'outcpp.avi'
        videoOut.write(filtered_stack[i]);
    }

    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to get durarion.
    duration = duration_cast<microseconds>(stop - start);
    cout << "Finished" << endl;
    cout << "Time rendering: " << duration.count() << " microseconds" << endl;

    // When everything done, release the video capture and write object
    video.release();
    videoOut.release();


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

    // Coefficients for IIR butterworth filter
    //[low_a, low_b] = butter(1, fl / samplingRate, 'low');
    //[high_a, high_b] = butter(1, fh / samplingRate, 'low');

    butter_coeff(1, 1, samplingRate * 2, fl);
    Vec2d low_a(pp[0], pp[1]);
    Vec2d high_a(aa[0], aa[1]);
    //cout << pp[0] << " " << pp[1] << " " << aa[0] << " " << aa[1] << endl;

    butter_coeff(1, 1, samplingRate * 2, fh);
    Vec2d low_b(pp[0], pp[1]);
    Vec2d high_b(aa[0], aa[1]);
    //cout << pp[0] << " " << pp[1] << " " << aa[0] << " " << aa[1] << endl;

    // Out video preparation

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
    string outName = outDir + name + "-butter-from-" + to_string(fl) + "-to-" +
        to_string(fh) + "-alpha-" + to_string(alpha) + "-lambda_c-" + to_string(lambda_c) +
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
    cout << "Video information: Height-" << vidHeight << " Width-" << vidWidth
        << " FrameRate-" << fr << " Frames-" << len << endl;

    // Write video
    // Define the codec and create VideoWriter object
    VideoWriter videoOut(outName, VideoWriter::fourcc('M', 'J', 'P', 'G'), fr,
        Size(vidWidth, vidHeight));


    // First frame
    Mat frame, rgbframe, ntscframe;
    vector<Mat> pyr_output;
    // Capture frame-by-frame
    video >> frame;

    // BGR to NTSC frame color space
    cvtColor(frame, rgbframe, COLOR_BGR2RGB);
    rgbframe = im2double(rgbframe);
    ntscframe = rgb2ntsc(rgbframe);

    // Compute maximum pyramid height for every frame
    int max_ht = 1 + maxPyrHt(vidWidth, vidHeight, MAX_FILTER_SIZE, MAX_FILTER_SIZE);

    // Render on the input video to make the output video
    cout << "Rendering... ";
    int k = 0;
    // Get starting timepoint
    for (int i = startIndex + 1; i < endIndex; i++) {

        Mat filt_ind, filtered, out_frame;
        // Capture frame-by-frame
        video >> frame;

        // Color conversion GBR 2 NTSC
        cvtColor(frame, rgbframe, COLOR_BGR2RGB);
        rgbframe = im2double(rgbframe);
        ntscframe = rgb2ntsc(rgbframe);

        //filt_ind = filtered_stack[k];

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
    int vidHeight = (int)video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = (int)video.get(CAP_PROP_FRAME_WIDTH);
    int nChannels = 3;
    int t_size = endIndex - startIndex + 1;

    vector<Mat> GDown_stack;
    GDown_stack.reserve(t_size);

    for (int i = startIndex; i < endIndex; i++) {
        Mat frame;
        video >> frame;
        GDown_stack.push_back(frame);
    }

#pragma omp parallel for shared(GDown_stack)
    for (int i = startIndex; i < endIndex; i++) {
        Mat frame, rgbframe, ntscframe;
        vector<Mat> pyr_output;
        // Capture frame-by-frame
        //video >> frame;

        cvtColor(GDown_stack[i], rgbframe, COLOR_BGR2RGB);
        frame = im2double(rgbframe);
        ntscframe = rgb2ntsc(frame);

        buildPyramid(ntscframe, pyr_output, level + 1, BORDER_REFLECT101);

        GDown_stack[i] = pyr_output[level].clone();
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

    // Number of frames in the video
    // Represents time
    int n = (int)input.size();

    // Temporal vector that's constructed for the mask
    // iota is used to fill the vector with a integer sequence 
    // [0, 1, 2, ..., n]
    vector<int> Freq_temp(n);
    iota(begin(Freq_temp), end(Freq_temp), 0); //0 is the starting number

    // Initialize the cv::Mat with the temp vector and without copying values
    Mat Freq(Freq_temp, false);
    double alpha = (double)samplingRate / (double)n;
    Freq.convertTo(Freq, CV_64FC1, alpha); // alpha is mult to every value

    Mat mask = (Freq > wl) & (Freq < wh); // creates a boolean matrix/mask

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

    // Here we populate the forementioned matrix
#pragma omp parallel for
    for (int x = 0; x < input[0].rows; x++) {
#pragma omp parallel for
        for (int y = 0; y < input[0].cols; y++) {
#pragma omp parallel for shared(input, temp_dft)
            for (int i = 0; i < n; i++) {
                int pos_temp = 3 * (y + x * input[0].cols);

                Vec3d pix_colors = input[i].at<Vec3d>(x, y);
                temp_dft.at<double>(pos_temp, i) = pix_colors[0];
                temp_dft.at<double>(pos_temp + 1, i) = pix_colors[1];
                temp_dft.at<double>(pos_temp + 2, i) = pix_colors[2];
            }
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
#pragma omp parallel for
    for (int i = 0; i < total_rows; i++) {
#pragma omp parallel for shared(input_dft)
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

    // Reording the matrix to a vector of matrixes, 
    // contrary of what was done for temp_dft
#pragma omp parallel for shared(input)
    for (int i = 0; i < n; i++) {
        Mat temp_filtframe(input[0].rows, input[0].cols, CV_64FC3);
        //int pos_temp = 0;
#pragma omp parallel for 
        for (int x = 0; x < input[0].rows; x++) {
#pragma omp parallel for shared(input_idft, temp_filtframe)
            for (int y = 0; y < input[0].cols; y++) {

                int pos_temp = 3 * (y + x * input[0].cols);

                Vec3d pix_colors;
                pix_colors[0] = input_idft.at<Vec2d>(pos_temp, i)[0];
                pix_colors[1] = input_idft.at<Vec2d>(pos_temp + 1, i)[0];
                pix_colors[2] = input_idft.at<Vec2d>(pos_temp + 2, i)[0];
                temp_filtframe.at<Vec3d>(x, y) = pix_colors;

                //pos_temp += 3;
            }
        }

        input[i] = temp_filtframe;
    }

    return input;
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
    int vidHeight = (int)video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = (int)video.get(CAP_PROP_FRAME_WIDTH);

    // Compute maximum pyramid height for every frame
    int max_ht = 1 + maxPyrHt(vidWidth, vidHeight, MAX_FILTER_SIZE, MAX_FILTER_SIZE);

    vector<vector<Mat>> pyr_stack(endIndex, vector<Mat>(max_ht));

    //double start, end;
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

    int total_pixels = 0;
    int levels = (int)input[0].size();

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
    int len = (int)video.get(CAP_PROP_FRAME_COUNT);
    int startIndex = 0;
    int endIndex = len - 10;
    int vidHeight = (int)video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = (int)video.get(CAP_PROP_FRAME_WIDTH);
    int fr = (int)video.get(CAP_PROP_FPS);

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