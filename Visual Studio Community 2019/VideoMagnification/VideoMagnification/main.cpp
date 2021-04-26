#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "processing_functions.h"

using namespace std;
using namespace cv;

int main() {

    string dataDir = "./vid";
    string resultsDir = "./Results";

    if (utils::fs::createDirectory(resultsDir) != 0)
    {
        cout << "Not able to create the directory" << endl;
    }

    int status1 = amplify_spatial_Gdown_temporal_ideal(dataDir+"/baby2.mp4", resultsDir, 
        150, 6, 140/60, 160/60, 30, 1);

    if (status1 == -1) {
        return -1;
    }

    return 0;
}