#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
constexpr auto MAX_FILTER_SIZE = 5;

using namespace cv;
using namespace std;

// Global variables
int pyramidHeight;

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
        pyramidHeight = 0;
    }
    // Next level
    else {
        pyramidHeight = 1 + maxPyrHt(frameWidth / 2, frameHeight / 2, filterSizeX, filterSizeY);
    }
    return pyramidHeight;
}

int main() {

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    // VideoCapture video(0);
    VideoCapture video("vid/guitar.mp4");

    // Extract video info
    int vidHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = video.get(CAP_PROP_FRAME_WIDTH);
    int nChannels = 3;
    int fr = video.get(CAP_PROP_FPS);
    int len = video.get(CAP_PROP_FRAME_COUNT);
    int max_ht = 1 + maxPyrHt(vidWidth, vidHeight, MAX_FILTER_SIZE, MAX_FILTER_SIZE);

    cout << max_ht << endl;
    return 0;

    // Check if video opened successfully
    if (!video.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while (1) {
        // Define variables
        Mat frame, dst, dstL2, dstU1, laplacian;

        // Capture frame-by-frame
        video >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Gaussian Pyramid downsampling
        pyrDown(frame, dst);
        pyrDown(dst, dstL2);

        // Gaussian Pyramid downsampling
        pyrUp(dstL2, dstU1);

        // Build Laplacian Pyramid
        subtract(dstU1, dst, laplacian);

        // Display the resulting frame
        imshow("Frame", laplacian);

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