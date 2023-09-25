#include <jni.h>
#include <string>

#include "Log.cpp"

#include "util/BitmapMatUtils.h"
#include "util/BitmapUtil.h"

// 获取文件路径
#define sdPath "/data/user/0/com.swan.study_opencv/files/"

using namespace std;
using namespace cv;


extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_SdkUtils_mark(JNIEnv *env, jclass clazz, jobject bitmap) {
    // 掩膜操作
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);

    // 掩膜操作 - 卷积
    Mat final;
    Mat kernel = (Mat_<char>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);//定义掩膜
    //调用filter2D
    filter2D(src,final,src.depth(),kernel);

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, src.cols, src.rows);
    BitmapMatUtils::mat2bitmap(env, newBitmap, final);
    return newBitmap;
}
extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_SdkUtils_blur(JNIEnv *env, jclass clazz, jobject bitmap) {
    // 模糊操作
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);

    // 掩膜操作 - 卷积
    Mat final;
    Mat kernel = Mat::ones(Size(25, 25), CV_32FC1) / (25 * 25);//定义掩膜

    //调用filter2D
    filter2D(src,final,src.depth(),kernel);

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, src.cols, src.rows);
    BitmapMatUtils::mat2bitmap(env, newBitmap, final);
    return newBitmap;
}