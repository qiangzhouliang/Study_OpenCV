//
// Created by swan on 2023/8/22.
//

#include "BitmapMatUtils.h"
#include <android/bitmap.h>

int BitmapMatUtils::bitmap2mat(JNIEnv* env, jobject &bitmap, cv::Mat &mat) {
    // 1. 锁定画布
    void *pixels; // 像素
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    // 构建mat对象，还要判断什么颜色通道 0 - 255
    // 获取 bitmap信息
    AndroidBitmapInfo bitmapInfo;
    AndroidBitmap_getInfo(env,bitmap,&bitmapInfo);

    // 返回三通道 CV_8UC4 -> argb CV_8UC2 -> rgb CV_8UC1 -> 黑白
    Mat createMat(bitmapInfo.height, bitmapInfo.width, CV_8UC4);
    // 判断颜色通道
    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888){ // mat 里面的四颜色通道 -> CV_8UC4
        Mat temp(bitmapInfo.height, bitmapInfo.width,CV_8UC4, pixels);
        temp.copyTo(createMat);

    } else if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565){ // mat 里面的三颜色通道 -> CV_8UC2
        Mat temp(bitmapInfo.height, bitmapInfo.width,CV_8UC2, pixels);
        // CV_8UC2 -> CV_8UC4
//        cvtColor(temp, createMat, COLOR_BGRA2BGR565);
        temp.copyTo(createMat, COLOR_BGR5652BGRA);
    }
    // 将 createMat 拷贝到 mat
    createMat.copyTo(mat);
    // 2. 解锁画布
    AndroidBitmap_unlockPixels(env, bitmap);

    return 0;
}

int BitmapMatUtils::mat2bitmap(JNIEnv* env, jobject bitmap, cv::Mat &mat) {
// 1 获取bitmap信息
    AndroidBitmapInfo info;
    void *pixels;
    AndroidBitmap_getInfo(env, bitmap,&info);

    // 锁定 bitmap 画布
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888){ // c4
        Mat temp(info.height, info.width, CV_8UC4, pixels);
        if (mat.type() == CV_8UC4){
            mat.copyTo(temp);
        } else if (mat.type() == CV_8UC2){
            cvtColor(mat, temp, COLOR_BGR5652BGRA);
        } else if (mat.type() == CV_8UC1){ // 灰度 mat
            cvtColor(mat, temp, COLOR_GRAY2RGBA);
        }
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565){ // c2
        Mat temp(info.height, info.width, CV_8UC2, pixels);
        if (mat.type() == CV_8UC4){
            cvtColor(mat, temp, COLOR_RGBA2BGR565);
        } else if (mat.type() == CV_8UC2){
            mat.copyTo(temp);
        } else if (mat.type() == CV_8UC1){ // 灰度 mat
            cvtColor(mat, temp, COLOR_GRAY2BGR565);
        }
    }
    // 其他要自己去转

    // 解锁bitmap画布
    AndroidBitmap_unlockPixels(env, bitmap);
    return 0;
}