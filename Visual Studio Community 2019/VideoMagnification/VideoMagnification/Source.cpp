#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
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

/*
* First attempt to build Lpyr
*/
/*
Mat buildLpyr(Mat frame) {
    Mat dst, dstL2, dstU1, laplacian;
    // Gaussian Pyramid downsampling
    pyrDown(frame, dst);
    pyrDown(dst, dstL2);

    // Gaussian Pyramid downsampling
    pyrUp(dstL2, dstU1);

    // Build Laplacian Pyramid
    subtract(dstU1, dst, laplacian);
    
    return laplacian;
}
*/

/*
* Second attempt to build Lpyr using Matlab code
*/
/*
Mat buildLpyr(Mat frame, int frameWidth, int frameHeight, int pyrHeight) {
    Mat lo, lo2, hi, hi2;
    if (pyrHeight <= 1) {
        return frame;
    }
    else {
        if (frameWidth == 1) {
            pyrDown(frame, lo2, Size(frame.cols, (frame.rows + 1) / 2), PYR_BORDER_TYPE);
        }
        else if (frameHeight == 1) {
            pyrDown(frame, lo2, Size((frame.cols + 1) / 2, frame.rows), PYR_BORDER_TYPE);
        }
        else {
            pyrDown(frame, lo, Size((frame.cols + 1) / 2, frame.rows), PYR_BORDER_TYPE);
            pyrDown(lo, lo2, Size(lo.cols, (lo.rows + 1) / 2), PYR_BORDER_TYPE);
            cout << "fin" << endl;
        }

        buildLpyr(lo2, lo.cols, lo.rows, pyrHeight - 1);

        if (frameWidth == 1) {
            pyrUp(lo2, hi2, Size((lo2.cols + 1) / 2, lo2.rows),
                PYR_BORDER_TYPE);
        }
        else if (frameHeight == 1) {
            pyrUp(lo2, hi2, Size(lo2.cols, (lo2.rows + 1) / 2),
                PYR_BORDER_TYPE);
        }
        else {
            pyrUp(lo2, hi, Size(lo2.cols, (lo2.rows + 1) / 2),
                PYR_BORDER_TYPE);
            pyrUp(hi, hi2, Size((hi.cols + 1) / 2, hi.rows),
                PYR_BORDER_TYPE);
        }
        subtract(frame, hi2, hi2);
        return hi2;
    }
}
*/

/*
* Third attempt to build Lpyr using pyrHeight
*/

vector<Mat> buildLpyr(Mat image, int levels) {
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

int main() {
    setBreakOnError(true);
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video(0);
    //VideoCapture video("vid/baby.mp4");

    // Extract video info
    int vidHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = video.get(CAP_PROP_FRAME_WIDTH);
    int nChannels = 3;
    int fr = video.get(CAP_PROP_FPS);
    int len = video.get(CAP_PROP_FRAME_COUNT);
    
    // Compute maximum pyramid height for every frame
    int max_ht = 1 + maxPyrHt(vidWidth, vidHeight, MAX_FILTER_SIZE, MAX_FILTER_SIZE);

    // Check if video opened successfully
    if (!video.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Define variables
    Mat frame;
    vector<Mat> lpyr;

    while (1) {
        // Capture frame-by-frame
        video >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        else

        lpyr = buildLpyr(frame, max_ht);

        // Display the resulting frame
        imshow("Frame", frame);
        imshow("Lpyr lvl 0", lpyr[0]);
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