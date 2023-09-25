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
Java_com_swan_opencv2_Imgproc_nfilter2D(JNIEnv *env, jclass clazz, jlong src_ptr, jlong dst_ptr,
                                        jlong kernel_ptr) {
    Mat *srcMat = reinterpret_cast<Mat *>(src_ptr);
    Mat *dstMat = reinterpret_cast<Mat *>(dst_ptr);
    Mat *kernelMat = reinterpret_cast<Mat *>(kernel_ptr);

//    Mat bgr;
//    cvtColor(*srcMat, bgr, COLOR_BGRA2BGR);

    filter2D(*srcMat,*dstMat,srcMat->depth(),*kernelMat);

}