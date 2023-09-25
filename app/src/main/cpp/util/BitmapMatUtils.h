//
// Created by swan on 2023/8/22.
//

#ifndef CARD_OCR_BITMAPMATUTILS_H
#define CARD_OCR_BITMAPMATUTILS_H
#include <jni.h>
#include "opencv2/opencv.hpp"

using namespace cv;

class BitmapMatUtils {
    // 开发项目增强，方法怎么写
    // Java中是吧想要的结果返回
    // c/c++ 结果参数传递，返回值一般返回是否成功
public:
    /**
     * bitmap2mat bitmap 转 map
     */
    static int bitmap2mat(JNIEnv* env, jobject bitmap, Mat &mat);

    /**
     * mat 转 bitmap
     */
    static int mat2bitmap(JNIEnv* env, jobject bitmap, Mat &mat);
};


#endif //CARD_OCR_BITMAPMATUTILS_H
