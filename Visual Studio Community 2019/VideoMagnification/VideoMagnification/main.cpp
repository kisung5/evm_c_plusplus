#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "processing_functions.h"

using namespace std;
using namespace cv;

int main() {

    string dataDir = "./vid/";
    string resultsDir = "./Results/";

    if (utils::fs::createDirectory(resultsDir) != 0)
    {
        cout << "Not able to create the directory" << endl;
    }

    //amplify_spatial_Gdown_temporal_ideal(dataDir + "baby2.mp4", resultsDir, 
    //    150, 6, 140.0f / 60.0f, 160.0f / 60.0f, 30, 1);

    //amplify_spatial_Gdown_temporal_ideal(dataDir + "face.mp4", resultsDir, 
    //    50, 4, 50.0f / 60.0f, 60.0f / 60.0f, 30, 1);

    //amplify_spatial_Gdown_temporal_ideal(dataDir + "face2.mp4", resultsDir, 
    //    50, 6, 50.0f / 60.0f, 60.0f / 60.0f, 30, 1);

    amplify_spatial_lpyr_temporal_butter(dataDir + "baby.mp4", resultsDir,
        30, 16, 0.4f, 3.0f, 30, 0.1f);

    return 0;
}