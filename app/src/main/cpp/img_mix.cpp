#include "Log.cpp"
#include "opencv2/opencv.hpp"
#include "writeSd.cpp"
#include "util/BitmapMatUtils.h"
#include "util/BitmapUtil.h"

using namespace std;
using namespace cv;

/// 图片混合、图片美容
namespace img_mix{

    // 图像混合
    string addWater(Mat src) {
        Mat logo = imread("/data/user/0/com.swan.study_opencv/files/face.jpg");

        // 注意两张图片的大小必须得一致
        Mat dst;
        // 非常生硬，直接像素点相加
        //add(src, logo, dst);

        // dst(x) = saturate_cast(src(x)*alpha + logo(x)*beta + gamma)
//        addWeighted(src, 0.5, logo, 0.5, 0.0, dst);

        // 非得加一个 logo 怎么加
        Mat srcROI = src(Rect(0,0, logo.cols, logo.rows));
        // 并不适合去加水印，只是适合做图片混合
        addWeighted(srcROI, 0.5, logo, 0.5, 0.0, srcROI);


        writeSd1(src, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    int getBlockSum(Mat &sum_mat, int x0, int y0, int x1, int y1, int ch){
        // 获取四个点的值
        int lt = sum_mat.at<Vec3i>(y0, x0)[ch];
        int lb = sum_mat.at<Vec3i>(y1, x0)[ch];
        int rt = sum_mat.at<Vec3i>(y0, x1)[ch];
        int rb = sum_mat.at<Vec3i>(y1, x1)[ch];

        // 区块的合
        int sum = rb - rt - lb + lt;
        return sum;
    }
    // 获取平方和
    float getBlockSqSum(Mat &sqsum_mat, int x0, int y0, int x1, int y1, int ch){
        // 获取四个点的值
        float lt = sqsum_mat.at<Vec3f>(y0, x0)[ch];
        float lb = sqsum_mat.at<Vec3f>(y1, x0)[ch];
        float rt = sqsum_mat.at<Vec3f>(y0, x1)[ch];
        float rb = sqsum_mat.at<Vec3f>(y1, x1)[ch];

        // 区块的合
        float sqsum = rb - rt - lb + lt;
        return sqsum;
    }

    // 积分图模糊算法 size - 模糊的直径
    void fatsBilateralBlur(Mat &src, Mat &dst, int size, int sigma = 0){
        // size %2 == 1
        // 把原来进行填充，方便运算
        Mat mat;
        int radius = size / 2;
        copyMakeBorder(src, mat, radius, radius, radius, radius, BORDER_DEFAULT);
        // 求积分图
        Mat sum_mat, sqsum_mat;
        integral(mat, sum_mat, sqsum_mat, CV_32S, CV_32F);

        dst.create(src.size(), src.type());
        int imageH = src.rows;
        int imageW = src.cols;
        int area = size * size;

        // 求四个点，左上 左下 右上 右下
        int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        int channels = src.channels();
        for (int row = 0; row < imageH; ++row) {
            // 思考：x0，y0，x1，y1 都是和 sum_mat 有关
            y0 = row;
            y1 = y0 + size;
            for (int col = 0; col < imageW; ++col) {
                x0 = col;
                x1 = x0 + size;
                // 循环通道
                for (int i = 0; i < channels; ++i) {
                    // 区块的和
                    int sum = getBlockSum(sum_mat, x0, y0, x1, y1, i);
                    float sqsum = getBlockSqSum(sqsum_mat, x0, y0, x1, y1, i);
                    // 计算方差
                    float diff_sq = (sqsum - (sum * sum) / area) / area;
                    float k = diff_sq / (diff_sq + sigma);

                    int pixels = src.at<Vec3b>(row, col)[i];
                    pixels = (1 - k)*(sum / area) + k * pixels;

                    dst.at<Vec3b>(row, col)[i] = pixels;
                }
            }
        }
    }

    // 皮肤区域检测
    void skinDetect(const Mat &src, Mat &skinMask){
        skinMask.create(src.size(), CV_8UC1);
        int rows = src.rows;
        int cols = src.cols;

        Mat ycrcb;
        cvtColor(src, ycrcb, COLOR_BGR2YCrCb);

        for (int row = 0; row < rows; row++){
            for (int col = 0; col < cols; col++){
                Vec3b pixels = ycrcb.at<Vec3b>(row, col);
                uchar y = pixels[0];
                uchar cr = pixels[1];
                uchar cb = pixels[2];

                if (y > 80 && 85 < cb < 135 && 135 < cr < 180){
                    skinMask.at<uchar>(row, col) = 255;
                } else {
                    skinMask.at<uchar>(row, col) = 0;
                }
            }
        }
    }

    // 皮肤区域融合
    void fuseSkin(const Mat &src, const  Mat &blur_mat, Mat &dst, const Mat &mask){
        // 融合？
        dst.create(src.size(),src.type());
        // 将 mask 进行模糊
        GaussianBlur(mask, mask, Size(3, 3), 0.0);
        Mat mask_f;
        mask.convertTo(mask_f, CV_32F);
        // 将数据 归一化 在 0-1 之间
        normalize(mask_f, mask_f, 1.0, 0.0, NORM_MINMAX);

        int rows = src.rows;
        int cols = src.cols;
        int ch = src.channels();

        for (int row = 0; row < rows; row++){
            for (int col = 0; col < cols; col++){
                // mask_f (1-k)
                /*uchar mask_pixels = mask.at<uchar>(row,col);
                // 人脸位置
                if (mask_pixels == 255){
                    dst.at<Vec3b>(row, col) = blur_mat.at<Vec3b>(row, col);
                }
                else{
                    dst.at<Vec3b>(row, col) = src.at<Vec3b>(row, col);
                }*/


                // src ，通过指针去获取， 指针 -> Vec3b -> 获取
                uchar b1 = src.at<Vec3b>(row, col)[0];
                uchar g1 = src.at<Vec3b>(row, col)[1];
                uchar r1 = src.at<Vec3b>(row, col)[2];

                // blur_mat
                uchar b2 = blur_mat.at<Vec3b>(row, col)[0];
                uchar g2 = blur_mat.at<Vec3b>(row, col)[1];
                uchar r2 = blur_mat.at<Vec3b>(row, col)[2];

                // dst 254  1
                float k = mask_f.at<float>(row,col);

                dst.at<Vec3b>(row, col)[0] = b2*k + (1 - k)*b1;
                dst.at<Vec3b>(row, col)[1] = g2*k + (1 - k)*g1;
                dst.at<Vec3b>(row, col)[2] = r2*k + (1 - k)*r1;
            }
        }
    }

    // 图片美容
    jobject imgFacial(JNIEnv *env, jobject bitmap) {
        Mat src;
        BitmapMatUtils::bitmap2mat(env, bitmap, src);

        // 由于 bitmap是rgba 颜色通道，mat操作需要 bgr，=》 rgba -> bgr
        Mat bgr;
        cvtColor(src, bgr, COLOR_RGBA2BGR);

        // 获取图片的宽高
        int src_w = src.cols;
        int src_h = src.rows;
        // 高斯（模糊），计算高斯卷积和，卷积操作，在考虑像素之间的差值（更好的保留图像的边缘）
        // 2-3 秒，
        int size = 105;
        Mat dst;
        // 优化 - 积分图模糊算法
        fatsBilateralBlur(bgr, dst, size, size * size);
        // 皮肤区域检测
        Mat skinMask;
        skinDetect(bgr, skinMask);

        // 融合皮肤区域
        Mat fuseDst;
        fuseSkin(bgr, dst, fuseDst, skinMask);

        // 边缘的提升 (可有可无)
        Mat cannyMask;
        Canny(bgr, cannyMask, 150, 300, 3, false);

        // & 运算  0 ，255
        bitwise_and(bgr, bgr, fuseDst, cannyMask);

        // 稍微提升一下对比度(亮度)
        add(fuseDst, Scalar(10, 10, 10), fuseDst);

        cvtColor(fuseDst, fuseDst, COLOR_BGR2RGBA);
        // 获取 bitmap信息
        jobject newBitmap = BitmapUtil::getBitMap(env, src_w, src_h);
        BitmapMatUtils::mat2bitmap(env, newBitmap, fuseDst);
        return newBitmap;
    }
}