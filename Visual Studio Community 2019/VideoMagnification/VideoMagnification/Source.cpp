#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture video("vid/guitar.mp4");

    // Extract video info
    int vidHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    int vidWidth = video.get(CAP_PROP_FRAME_WIDTH);
    int nChannels = 3;
    int fr = video.get(CAP_PROP_FPS);
    int len = video.get(CAP_PROP_FRAME_COUNT);

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

