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
JNIEXPORT void JNICALL
Java_com_swan_opencv2_Utils_nbitmap2mat(JNIEnv *env, jclass clazz, jobject bitmap, jlong mat_ptr) {
    Mat *mat = reinterpret_cast<Mat *>(mat_ptr);

    BitmapMatUtils::bitmap2mat(env, bitmap, *mat);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_swan_opencv2_Utils_nmat2bitmap(JNIEnv *env, jclass clazz, jlong mat_ptr, jobject bitmap) {
    Mat *mat = reinterpret_cast<Mat *>(mat_ptr);

    BitmapMatUtils::mat2bitmap(env, bitmap, *mat);
}