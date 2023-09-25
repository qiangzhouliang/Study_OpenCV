#include <jni.h>
#include <string>
#include <android/bitmap.h>

#include "Log.cpp"
#include "opencv2/opencv.hpp"

#include "util/BitmapMatUtils.h"
#include "util/BitmapUtil.h"

// 获取文件路径
#define sdPath "/data/user/0/com.swan.study_opencv/files/"

using namespace std;
using namespace cv;

extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_OpenCVUtils_rotation(JNIEnv *env, jclass clazz, jobject bitmap) {
    // 旋转
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);

    int res_w = src.rows;
    int res_h = src.cols;
    Mat res(res_h, res_w, src.type());

    for (int rows = 0; rows < res_h; ++rows) {
        for (int cols = 0; cols < res_w; ++cols) {
            // 向右旋转 90 度
            int src_rows = cols;
            int src_cols = res_h - rows;
            res.at<int>(rows, cols) = src.at<int>(src_rows, src_cols);
        }
    }

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, res_w, res_h);
    BitmapMatUtils::mat2bitmap(env, newBitmap, res);
    return newBitmap;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_OpenCVUtils_warpAffine(JNIEnv *env, jclass clazz, jobject bitmap) {
    // 旋转
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);

    int res_w = src.cols;
    int res_h = src.rows;
    Mat res(src.size(), src.type());

    // 必须是一个 2 * 3 矩阵
    // 这几个值 应该怎么确定
    // [a0, a1, a2] 两个矩阵  [a0, a1]  [a2]    = [x] * [a0, a1] + [a2] = a0 * x + a1*x + a2
    // [b0, b1, b2]          [b0, b1]  [b2]      [y] * [b0, b1]   [b2] = b0 * y + b1*y + b2
    /*Mat M(2,3, CV_32FC1);
    M.at<float>(0, 0) = 0.5; // a0
    M.at<float>(0, 1) = 0; // a1
    M.at<float>(0, 2) = 0; // a2

    M.at<float>(1, 0) = 0; // b0
    M.at<float>(1, 1) = 0.5; // b1
    M.at<float>(1, 2) = 0; // b2*/

    // 旋转
    Point2f center(src.cols/2, src.rows/2);
    double angle = 45;
    double scale = 1;
    Mat M = getRotationMatrix2D(center, angle, scale);

    warpAffine(src, res, M, src.size());

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, res_w, res_h);
    BitmapMatUtils::mat2bitmap(env, newBitmap, res);
    return newBitmap;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_OpenCVUtils_reSize(JNIEnv *env, jclass clazz, jobject bitmap, jint width, jint height) {
    // 图片缩放
    // 上采样和降采样
    // 旋转
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);

    Mat res(height, width, src.type());

    float src_w = src.cols;
    float src_h = src.rows;

    // 最近领域插值
    for (int rows = 0; rows < res.rows; ++rows) {
        for (int cols = 0; cols < res.cols; ++cols) {
            int src_rows = rows * (src_h / height); // src 的高
            int src_cols = cols * (src_w / width);
            Vec4b pixels = src.at<Vec4b>(src_rows, src_cols);
            res.at<Vec4b>(rows, cols) = pixels;
        }
    }


    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, width, height);
    BitmapMatUtils::mat2bitmap(env, newBitmap, res);
    return newBitmap;

}

void myRemap(Mat &src, Mat &res, Mat &map_x, Mat &map_y) {
    res.create(src.size(), src.type());
    int res_w = src.cols;
    int res_h = src.rows;

    for (int row = 0; row < res_h; ++row) {
        for (int col = 0; col < res_w; ++col) {
            int x = map_x.at<float>(row, col);
            int y = map_y.at<float>(row, col);
            res.at<Vec4b>(row, col) = src.at<Vec4b>(y, x);
        }
    }
}
extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_OpenCVUtils_reMap(JNIEnv *env, jclass clazz, jobject bitmap) {
    // 重映射
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);

    int res_w = src.cols;
    int res_h = src.rows;
    Mat res;

    Mat map_x(src.size(), CV_32F);
    Mat map_y(src.size(), CV_32F);
    // 照片左右调换
    for (int row = 0; row < src.rows; ++row) {
        for (int col = 0; col < src.cols; ++col) {

            // 照片左右调换
//            map_x.at<float>(row,col) = src.cols - col - 1;
//            map_y.at<float>(row,col) = row;
            // 照片上下调换(倒影)
            map_x.at<float>(row,col) = col;
            map_y.at<float>(row,col) = src.rows - row - 1;
            // 缩小2倍
            /*if(col>src.cols*0.25 && col<src.cols*0.75 && row>src.rows*0.25 && row<src.rows*0.75)
            {
                map_x.at<float>(row,col)=static_cast<float>(2*(col-src.cols*0.25)+0.5);
                map_y.at<float>(row,col)=static_cast<float>(2*(row-src.rows*0.25)+0.5);
            }
            else
            {
                map_x.at<float>(row,col)=0;
                map_y.at<float>(row,col)=0;

            }*/
        }
    }

//    remap(src, res, map_x, map_y, 1);
    myRemap(src, res, map_x, map_y);

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, res_w, res_h);
    BitmapMatUtils::mat2bitmap(env, newBitmap, res);
    return newBitmap;
}