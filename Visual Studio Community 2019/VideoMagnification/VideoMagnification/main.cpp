#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <vector>

#include "processing_functions.h"

using namespace cv;
using namespace std;

int main() {
    string input1 = "./vid/guitar.mp4";
    string input2 = "./vid/baby.mp4";
    string output = "./Results/";

    if (utils::fs::createDirectory(output) != 0)
    {
        cout << "Not able to create the directory" << endl;
    }

    //Mat input(10, 10, CV_8U);


    //int status1 = amplify_spatial_lpyr_temporal_ideal(input1, output,
        //100, 10, 100, 120, 600, 0);

    int status2 = amplify_spatial_lpyr_temporal_iir(input2, output,
        10, 16, 0.4, 0.05, 0.1);

    if (status2 == -1) {
        return -1;
    }

    return 0;
}