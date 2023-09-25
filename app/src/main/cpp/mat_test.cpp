#include "img_mix.cpp"
#include "Log.cpp"
#include "opencv2/opencv.hpp"
#include "writeSd.cpp"
#include "img_operate.cpp"
#include "util/BitmapMatUtils.h"



using namespace std;
using namespace cv;

namespace matT{
/**
     * 创建Mat
     */
    void create_mat(Mat src) {
        // 创建：CV_8UC1 一个颜色通道（0-255）
        // CV_8UC2 2个颜色通道（0 - 0xff）16位 RGB565
        // CV_8UC3 3个颜色通道（0 - 0xfff）24位
        // CV_8UC4 4个颜色通道（0 - 0xffff）32位 ARGB8888
        // 匹配上 Java Bitmap 的颜色通道 RGB565 ARGB8888
        // Scalar: 指定颜色
//        Mat mat(20, 20, CV_8UC1, Scalar(200));

        // 3个颜色通道
        Mat mat(20, 20, CV_8UC3, Scalar(0,0,255)); // BGR

        writeSd1(mat, "create_mat.jpg");
    }

/**
     * 像素操作
     * @param src
     */
    string pixel_operate(Mat src){
        return img_o::pixelOperate(src);
    }

    /**
     * 图像混合
     * @param src
     * @return
     */
    string img_mix(Mat src) {
        // 添加水印
        return img_mix::addWater(src);
    }


    string img_degree(Mat src) {
        // 饱和度
        return img_o::img_saturation(src);
    }

    // 图像绘制
    string img_draw(Mat src) {
        // 图像绘制
//        return img_o::img_draw(src);


        // 矩阵掩膜
//        return img_o::img_jzym(src);
//        return img_o::img_jzym1(src);
        // 图像模糊
//        return img_o::img_vague(src);
        // 膨胀与腐、、蚀
//        return img_o::img_erode_dilate(src);
        // 过滤验证码干扰
//        return img_o::get_img_code(src);
        // 提取水平与垂直线
//        return img_o::get_H_V_line(src);
        // 上采样与降采样、
//        return img_o::get_caiyang(src);

        // 图像边缘检测
        // 1 sobel
//        return img_o::txbyjc_sobel(src);
        // 2 laplance
//        return img_o::txbyjc_laplacian(src);
//        return img_o::txbyjc_canny(src);
        // 霍夫检测
//        return img_o::txbyjc_hf_check(src);
        // 直方图
//        return img_o::txbyjc_zft(src);
//        return img_o::txbyjc_data_zft(src);
        // 直方图比较
//        return img_o::txbyjc_zft_bj(src);
        // 直方图反向投射
//        return img_o::txbyjc_zft_fxts(src);
        // 直方图 - 模版匹配
//        return img_o::txbyjc_zft_mbpp(src);

        // 银行卡轮廓查找
//        return img_o::card_lunkuo(src);
        // 图形矩
//        return img_o::card_txj(src);
        // 图像风水岭切割
        return img_o::txfslqg(src);
    }
}
