#include <jni.h>
#include <string>
#include <android/bitmap.h>

#include "Log.cpp"
#include "opencv2/opencv.hpp"
#include "mat_test.cpp"
#include "util/BitmapMatUtils.h"
#include "util/BitmapUtil.h"

// 获取文件路径
#define sdPath "/data/user/0/com.swan.study_opencv/files/"



using namespace std;

using namespace cv;

extern "C" JNIEXPORT jstring JNICALL
Java_com_swan_study_1opencv_Plugin_stringFromJNI(
        JNIEnv* env,
        jclass clazz) {

    std::string hello = "Hello from C++"+cv::getVersionString();
    return env->NewStringUTF(hello.c_str());
}
extern "C"
JNIEXPORT jstring JNICALL
Java_com_swan_study_1opencv_Plugin_setImg(JNIEnv *env, jclass clazz, jstring img_path) {
    string sPath = env->GetStringUTFChars(img_path, 0);
    // 读取一张图片 mat 图片
    Mat src = imread(sPath);
    // 判断读取是否正确
    if (src.empty()){
        LOGE("src imread error");
        return env->NewStringUTF("-1");
    }
    // 宽、高、颜色通道 rows 行（高） cols 列（宽），3（1个像素点 3 颜色通道，包含三个信息，BGR）
//    LOGE("宽度= %d, 高度= %d, 颜色通道 = %d", src.cols, src.rows, src.channels());

    // 创建Mat
//    matT::create_mat(src);
    // 像素操作
//    string imgPath = matT::pixel_operate(src);
    // 图像混合
//    string imgPath = matT::img_mix(src);
    // 饱和度、亮度、对比度
//    string imgPath = matT::img_degree(src);
    // 图像绘制
    string imgPath = matT::img_draw(src);


    return env->NewStringUTF(imgPath.c_str());
}


extern "C"
JNIEXPORT jint JNICALL
Java_com_swan_study_1opencv_BitmapUtils_gary3(JNIEnv *env, jclass clazz, jobject bitmap) {
    AndroidBitmapInfo bitmapInfo;
    int info_res = AndroidBitmap_getInfo(env, bitmap, &bitmapInfo);
    if (info_res != 0){
        return -1;
    }

    // void 指针？并不知道具体的类型
    void *pixels;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888){
        for (int i = 0; i < bitmapInfo.width * bitmapInfo.height; ++i) {
            uint32_t *pixel_p = static_cast<uint32_t *>(pixels)+i;
            uint32_t pixel = *pixel_p;

            int a = pixel >> 24 & 0xff;
            int r = pixel >> 16 & 0xff;
            int g = pixel >> 8 & 0xff;
            int b = pixel & 0xff;
            // f = 0.213f * r + 0.715f * g + 0.072f * b
            int gray = (int) (0.213f * r + 0.715f * g + 0.072f * b);
            *pixel_p = (a << 24) | (gray << 16) | (gray << 8) | gray;
        }
    } else if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565){
        for (int i = 0; i < bitmapInfo.width * bitmapInfo.height; ++i) {
            uint16_t *pixel_p = static_cast<uint16_t *>(pixels)+i;
            uint16_t pixel = *pixel_p;

            // 565 总共16位，需要高位的5位，所以 向右移 11 位，5 位 不够8位 左移补3位
            int r = (pixel >> 11 & 0x1f) << 3; // 5 位 不够8位 补3位
            int g = (pixel >> 5 & 0x3f)  << 2;
            int b = (pixel & 0x1f)  << 3;
            // f = 0.213f * r + 0.715f * g + 0.072f * b （这个事针对 32 位来讲的）
            int gray = (int) (0.213f * r + 0.715f * g + 0.072f * b);
            *pixel_p = ((gray >> 3) << 11) | ((gray >> 2) << 5) | (gray >> 3);
        }
    }


    AndroidBitmap_unlockPixels(env, bitmap);

    return 1;
}
extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_Plugin_operateBitm(JNIEnv *env, jclass clazz, jobject src_b) {
    Mat mat;
    BitmapMatUtils().bitmap2mat(env, src_b, mat);

    cvtColor(mat, mat, COLOR_BGR2GRAY);

    // 获取 bitmap信息
    AndroidBitmapInfo bitmapInfo;
    AndroidBitmap_getInfo(env,src_b,&bitmapInfo);
    jobject newBitmap = BitmapUtil::getBitMap(env, bitmapInfo.width, bitmapInfo.height);


    BitmapMatUtils().mat2bitmap(env, newBitmap, mat);
    return newBitmap;
}

/// 逆世界效果
extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_NDKBitmapUtils_againstWorld(JNIEnv *env, jclass clazz, jobject bitmap) {
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);

    // 结果图片
    Mat res(src.size(), src.type());

    // 获取图片的宽高
    int src_w = src.cols;
    int src_h = src.rows;

    // 一半的高度
    int mid_h = src_h >> 1;
    // 1/4 高度
    int a_h = mid_h >> 1;
    // 处理下半部分
    for (int row = 0; row < mid_h; ++row) {
        for (int col = 0; col < src_w; ++col) {
            // 4 自己，rgba, 应该需要判断 type
            res.at<Vec4b>(row + mid_h, col) = src.at<Vec4b>(row + a_h, col);
        }
    }

    // 处理上半部分
    for (int row = 0; row < mid_h; ++row) {
        for (int col = 0; col < src_w; ++col) {
            // 4 自己，rgba, 应该需要判断 type
            res.at<Vec4b>(row, col) = src.at<Vec4b>(src_h - a_h - row, col);
        }
    }

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, src_w, src_h);
    BitmapMatUtils::mat2bitmap(env, newBitmap, res);
    return newBitmap;

}

// 浮雕效果
extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_NDKBitmapUtils_anaglyph(JNIEnv *env, jclass clazz, jobject bitmap) {
    // 有立体感，突出了轮廓信息，OpenCV filter2D
    // 卷积 [1,0] [0,-1]
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);

    // 结果图片
    Mat res(src.size(), src.type());

    // 获取图片的宽高
    int src_w = src.cols;
    int src_h = src.rows;
    for (int row = 0; row < src_h - 1; ++row) {
        for (int col = 0; col < src_w - 1; ++col) {
            Vec4b pixels_p = src.at<Vec4b>(row, col);
            Vec4b pixels_n = src.at<Vec4b>(row + 1, col + 1);

            // bgra
            res.at<Vec4b>(row, col)[0] = saturate_cast<uchar>(pixels_p[0] - pixels_n[0] + 128);
            res.at<Vec4b>(row, col)[1] = saturate_cast<uchar>(pixels_p[1] - pixels_n[1] + 128);
            res.at<Vec4b>(row, col)[2] = saturate_cast<uchar>(pixels_p[2] - pixels_n[2] + 128);
        }
    }

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, src_w, src_h);
    BitmapMatUtils::mat2bitmap(env, newBitmap, res);
    return newBitmap;
}
// 马赛克
extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_NDKBitmapUtils_mosaic(JNIEnv *env, jclass clazz, jobject bitmap) {
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);
    // 获取图片的宽高
    int src_w = src.cols;
    int src_h = src.rows;
    // 省略 人脸识别
    int rows_s = src_h >> 2;
    int rows_e = src_h * 3 / 4;
    int cols_s = src_w >> 2;
    int cols_e = src_w * 3 / 4;
    // 马赛克大小
    int size = 10;

    for (int row = rows_s; row < rows_e; row += size) {
        for (int col = cols_s; col < cols_e; col += size) {
            int pixels = src.at<int>(row, col);
            // 10 * 10 的范围内都有第一个 像素值
            for (int m_rows = 1; m_rows < size; ++m_rows) {
                for (int m_cols = 0; m_cols < size; ++m_cols) {
                    src.at<int>(row + m_rows, col + cols_s) = pixels;
                }
            }
        }
    }

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, src_w, src_h);
    BitmapMatUtils::mat2bitmap(env, newBitmap, src);
    return newBitmap;
}

// 毛玻璃
extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_NDKBitmapUtils_groundGlass(JNIEnv *env, jclass clazz, jobject bitmap) {
    // 高斯模糊 ，毛玻璃（对某个区域取随机像素）
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);
    // 获取图片的宽高
    int src_w = src.cols;
    int src_h = src.rows;
    int size = 8;

    RNG rng(time(NULL));
    for (int row = 0; row < src_h - size; ++row) {
        for (int col = 0; col < src_w - size; ++col) {
            int random = rng.uniform(0, 20);
            src.at<int>(row, col) = src.at<int>(row + random, col + random);
        }
    }

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, src_w, src_h);
    BitmapMatUtils::mat2bitmap(env, newBitmap, src);
    return newBitmap;
}

// 油画效果
extern "C"
JNIEXPORT jobject JNICALL
Java_com_swan_study_1opencv_NDKBitmapUtils_oilPainting(JNIEnv *env, jclass clazz, jobject bitmap) {
    // 基于直方统计
    Mat src;
    BitmapMatUtils::bitmap2mat(env, bitmap, src);
    // 结果图片
    Mat res(src.size(), src.type());

    Mat gary;
    cvtColor(src, gary, COLOR_BGRA2GRAY);

    // 获取图片的宽高
    int src_w = src.cols;
    int src_h = src.rows;
    int size = 8;
    // 1 每个店需要分成 n*n 小块
    // 2 统计灰度等级
    // 3 选择灰度等级中最多的值
    // 4 找到所以得像素取平均值
    for (int row = 0; row < src_h - size; ++row) {
        for (int col = 0; col < src_w - size; ++col) {
            // g 灰度等级
            int g[8] = {0}, b_g[8] = {0}, g_g[8] = {0}, r_g[8] = {0};
            // 这个位置：64 次循环 才能处理一个像素点
            for (int o_rows = 0; o_rows < size; ++o_rows) {
                for (int o_cols = 0; o_cols < size; ++o_cols) {
                    uchar gery = gary.at<uchar>(row + o_rows, col + o_cols);
                    uchar index = gery / (254 / 7);
                    g[index] += 1;
                    // 等级的像素值之和
                    b_g[index] += src.at<Vec4b>(row + o_rows, col + o_cols)[0];
                    g_g[index] += src.at<Vec4b>(row + o_rows, col + o_cols)[1];
                    r_g[index] += src.at<Vec4b>(row + o_rows, col + o_cols)[2];
                }
            }
            // 最大的角标找出来
            int max_index = 0;
            int max = g[0];
            for (int i = 1; i < size; ++i) {
                if (g[max_index] < g[i]){
                    max_index = i;
                    max = g[i];
                }
            }
            res.at<Vec4b>(row , col)[0] = b_g[max_index] / max;
            res.at<Vec4b>(row , col)[1] = g_g[max_index] / max;
            res.at<Vec4b>(row , col)[2] = r_g[max_index] / max;
        }
    }

    // 获取 bitmap信息
    jobject newBitmap = BitmapUtil::getBitMap(env, src_w, src_h);
    BitmapMatUtils::mat2bitmap(env, newBitmap, res);
    return newBitmap;
}
