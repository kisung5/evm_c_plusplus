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

    // Color Magnification

    int status1 = amplify_spatial_Gdown_temporal_ideal(dataDir + "baby2.mp4", resultsDir, 
        150, 6, 140.0f / 60.0f, 160.0f / 60.0f, 30, 1);

    int status2 = amplify_spatial_Gdown_temporal_ideal(dataDir + "face.mp4", resultsDir, 
        50, 4, 50.0f / 60.0f, 60.0f / 60.0f, 30, 1);

    int status3 = amplify_spatial_Gdown_temporal_ideal(dataDir + "face2.mp4", resultsDir, 
        50, 6, 50.0f / 60.0f, 60.0f / 60.0f, 30, 1);


    // Motion Magnification

    // Butter

    int status4 = amplify_spatial_lpyr_temporal_butter(dataDir + "baby.mp4", resultsDir,
        30.0f, 16.0f, 0.4f, 3.0f, 30, 0.1f);

    int status5 = amplify_spatial_lpyr_temporal_butter(dataDir + "camera.mp4", resultsDir,
        150.0f, 20.0f, 45.0f, 100.0f, 300, 0.0f);

    int status6 = amplify_spatial_lpyr_temporal_butter(dataDir + "subway.mp4", resultsDir,
        60.0f, 90.0f, 3.6f, 6.2f, 30, 0.3f);

    int status7 = amplify_spatial_lpyr_temporal_butter(dataDir + "wrist.mp4", resultsDir,
        30.0f, 16.0f, 0.4f, 3.0f, 30, 0.1f);

    int status8 = amplify_spatial_lpyr_temporal_butter(dataDir + "shadow.mp4", resultsDir,
        5.0f, 48.0f, 0.5f, 10.0f, 30, 0.0f);

    int status9 = amplify_spatial_lpyr_temporal_butter(dataDir + "face2.mp4", resultsDir,
        20.0f, 80.0f, 0.5f, 10.0f, 30, 0.0f);

    // Ideal

    int status10 = amplify_spatial_lpyr_temporal_ideal(dataDir + "guitar.mp4", resultsDir,
        50, 10, 72.0f, 92.0f, 600, 0);

    int status11 = amplify_spatial_lpyr_temporal_ideal(dataDir + "guitar.mp4", resultsDir,
        100, 10, 100.0f, 120.0f, 600, 0);

    // IIR

    int status12 = amplify_spatial_lpyr_temporal_iir(dataDir + "baby.mp4", resultsDir,
        10, 16, 0.4f, 0.05f, 0.1f);

    int status13 = amplify_spatial_lpyr_temporal_iir(dataDir + "wrist.mp4", resultsDir, 
        10, 16, 0.4, 0.05, 0.1);
  
    int status = status1 + status2 + status3 + status4 + status5 + status6 + status7 + status8 +
        status9 + status10 + status11 + status12 + status13;
    
    if (status < 0) {
        return -1;
    }
  
    return 0;
}