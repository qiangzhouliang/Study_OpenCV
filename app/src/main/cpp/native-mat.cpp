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
JNIEXPORT jlong JNICALL
Java_com_swan_opencv2_Mat_nMat(JNIEnv *env, jobject thiz) {
    Mat *mat = new Mat();

    return reinterpret_cast<jlong>(mat);
}
extern "C"
JNIEXPORT jlong JNICALL
Java_com_swan_opencv2_Mat_nMatIII(JNIEnv *env, jobject thiz, jint rows, jint cols, jint type) {
    Mat *mat = new Mat(rows, cols, type);

    return reinterpret_cast<jlong>(mat);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_swan_opencv2_Mat_nputF(JNIEnv *env, jobject thiz, jlong mat_ptr, jint row, jint col,
                                jfloat value) {
    Mat *mat = reinterpret_cast<Mat *>(mat_ptr);
    mat->at<float>(row, col) = value;
}