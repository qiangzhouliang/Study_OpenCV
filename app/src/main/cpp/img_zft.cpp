#include "Log.cpp"
#include "opencv2/opencv.hpp"
#include "writeSd.cpp"

using namespace std;
using namespace cv;

// 手写直方图
namespace zft{
    // 亮度增强
    string txbyjc_zft_my_ld(Mat src) {
        // src -> 3通道 bgr
        Mat dst = src.clone();
        for (int row = 0; row < src.rows; ++row) {
            for (int col = 0; col < src.cols; ++col) {
                Vec3b pixels = src.at<Vec3b>(row, col);
                dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(pixels[0] +50);
                dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(pixels[1] +50);
                dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(pixels[2] +50);
            }
        }

        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    void calcHist(const Mat &mat, Mat &hist){
        // int 存
        hist.create(1, 256, CV_32S);

        for (int i = 0; i < hist.cols; ++i) {
            hist.at<int>(0, i) = 0;
        }

        for (int row = 0; row < mat.rows; ++row) {
            for (int col = 0; col < mat.cols; ++col) {
                // 获取灰度等级的下标
                int index = mat.at<uchar>(row, col);
                hist.at<int>(0, index) += 1;
            }
        }
    }

    // 1. 直方图统计
    // 2. 计算直方图中像素的概率
    // 3. 生成一张映射表
    // 4. 从映射表中查找赋值
    void equalizeHist(const Mat &src, Mat &dst){
        // 1. 直方图统计
        Mat hist;
        calcHist(src, hist);
        // 2. 计算直方图中像素的概率
        Mat prob_mat(hist.size(), CV_32FC1);
        // 图片像素点大小
        float image_size = src.cols * src.rows;
        for (int i = 0; i < hist.cols; ++i) {
            float prob = hist.at<int>(0, i) / image_size;
            prob_mat.at<float>(0, i) = prob;
        }
        // 计算累计概率 256
        float prob_sum = 0;
        for (int i = 0; i < prob_mat.cols; ++i) {
            float prob = prob_mat.at<float>(0, i);
            prob_sum += prob;
            prob_mat.at<float>(0, i) = prob_sum;
        }

        // 3. 生成 映射 表
        Mat map(hist.size(), CV_32FC1);
        for (int i = 0; i < prob_mat.cols; ++i) {
            float prob = prob_mat.at<float>(0, i);
            // 小的纠正：用累积概率 * 255
            map.at<float>(0, i) = prob * 255;
        }

        // 4. 从映射表中查找赋值
        dst.create(src.size(), src.type());
        for (int row = 0; row < src.rows; ++row) {
            for (int col = 0; col < src.cols; ++col) {
                uchar pixels = src.at<uchar>(row, col);
                dst.at<uchar>(row, col) = saturate_cast<uchar>( map.at<float>(0, pixels));
            }
        }
    }

    // 直方均衡图- 提升对比度
    string zft_zfjht(Mat src) {
        Mat dst;
//        Mat hsv;
//        cvtColor(src, hsv, COLOR_BGR2HSV);
//
//        vector<Mat> hsv_s;
//        split(hsv, hsv_s);
//
//        // 直方图均衡化: 均衡亮度通道
//        equalizeHist(hsv_s[2], hsv_s[2]);
//
//        merge(hsv_s, hsv);
//        cvtColor(hsv, dst, COLOR_HSV2BGR);

        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, dst);

        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }



    void normalize(const Mat &src, Mat &dst, int mMax){
        // 显示到 0 - mMax 之间
        int max_value = 0;
        for (int row = 0; row < src.rows; ++row) {
            for (int col = 0; col < src.cols; ++col) {
                int value = src.at<int>(row, col);
                max_value = max(value, max_value);
            }
        }
        dst.create(src.size(), src.type());
        for (int row = 0; row < src.rows; ++row) {
            for (int col = 0; col < src.cols; ++col) {
                int value = src.at<int>(row, col);
                dst.at<int>(row, col) = (1.0/max_value) * value * mMax;
            }
        }
    }

    // 直方图计算
    string zft_js(Mat src) {

        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);

        // 直方图，hist 没有宽高，是生成了一个空的数组
        Mat hist;
        calcHist(gray, hist);

        // 画直方图 - 归一化
        normalize(hist,hist, 255);

        // 画直方图图
        int bin_w = 5; // 画笔的大小
        int grad = 256; // 等级
        Mat hist_mat(grad, bin_w * 256, CV_8UC3);
        for (int i = 0; i < grad; ++i) {
            Point start(i*bin_w, hist_mat.rows);
            Point end(i*bin_w, hist_mat.rows - hist.at<int>(0, i));
            line(hist_mat, start, end, Scalar(255, 0, 255), bin_w, LINE_AA);
        }

        writeSd1(hist_mat, "mat_operate.jpg");
        return "mat_operate.jpg";
    }
}