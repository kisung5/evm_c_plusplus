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
    string dataDir = "./vid";
    string resultsDir = "./Results";

    if (utils::fs::createDirectory(resultsDir) != 0)
    {
        cout << "Not able to create the directory" << endl;
    }

    //Mat input(10, 10, CV_8U);


    int status1 = amplify_spatial_lpyr_temporal_ideal(dataDir + "/guitar.mp4", resultsDir,
        100, 10, 100, 120, 600, 0);

    if (status1 == -1) {
        return -1;
    }

    return 0;
}