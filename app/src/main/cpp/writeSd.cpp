//
// Created by swan on 2023/9/13.
//
#include "opencv2/opencv.hpp"

// 获取文件路径
#define sdPath "/data/user/0/com.swan.study_opencv/files/"

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 256
void writeSd1(const cv::Mat &dst,std::string fileName) {
    std::string dPath = sdPath;
    dPath.append(fileName);
    imwrite(dPath, dst);
}
#endif

