//
// Created by swan on 2023/9/21.
//

#ifndef STUDY_OPENCV_BITMAPUTIL_H
#define STUDY_OPENCV_BITMAPUTIL_H
#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include "opencv2/opencv.hpp"


class BitmapUtil {
public:
    // 创建一个bitmap
    static jobject getBitMap(JNIEnv *env, int width, int height, int type = CV_8UC4);
};


#endif //STUDY_OPENCV_BITMAPUTIL_H
