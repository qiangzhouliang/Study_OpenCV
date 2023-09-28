//
// 人脸识别
//
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

namespace ImgFaceCheck{
    /// hog 特征提取
    jobject hogBitmap(JNIEnv *env, jobject bitmap) {
        Mat src;
        BitmapMatUtils::bitmap2mat(env, bitmap, src);
        // 由于 bitmap是rgba 颜色通道，mat操作需要 bgr，=》 rgba -> bgr
        Mat dst, dst_gary;
        // 拿 hog 特征
        // 1. 分成 8*8 的网格 ceil，求取直方图
        // 2. 对 ceil 的 4 * 4 的块去合并直方图，每次必须要有 1/2 的重叠区域
        // 3. 然后获取直方图的数据

        resize(src, dst, Size(64, 128));
        cvtColor(dst, dst_gary, COLOR_BGRA2GRAY);

        HOGDescriptor hogDescriptor;
        vector<float> descriptors;
        vector<Point> locations;
        /**
         * CV_WRAP virtual void compute(InputArray img,
                         CV_OUT std::vector<float>& descriptors,
                         Size winStride = Size(), Size padding = Size(),
                         const std::vector<Point>& locations = std::vector<Point>()) const;
         */
        hogDescriptor.compute(dst_gary, descriptors, Size(), Size(), locations);


        LOGE("descriptors size: %d", descriptors.size());
        // 获取 bitmap信息
        jobject newBitmap = BitmapUtil::getBitMap(env, dst_gary.cols, dst_gary.rows);
        BitmapMatUtils::mat2bitmap(env, newBitmap, dst_gary);
        return newBitmap;
    }

    /// 行人检测
    jobject peopleCheck(JNIEnv *env, jobject bitmap) {
        Mat src;
        BitmapMatUtils::bitmap2mat(env, bitmap, src);

        Mat bgr;
        cvtColor(src, bgr, COLOR_RGBA2BGR);

        // 训练样本，直接拿来用
        HOGDescriptor descriptor;
        descriptor.setSVMDetector(descriptor.getDefaultPeopleDetector());
        /**
         * CV_WRAP virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
                                  CV_OUT std::vector<double>& foundWeights, double hitThreshold = 0,
                                  Size winStride = Size(), Size padding = Size(), double scale = 1.05,
                                  double finalThreshold = 2.0,bool useMeanshiftGrouping = false) const;
         */
        // 多维度检测
        vector<Rect> foundLocations;
        descriptor.detectMultiScale(bgr, foundLocations, 0, Size(10, 10));

        for (int i = 0; i < foundLocations.size(); ++i) {
            rectangle(bgr, foundLocations[i], Scalar(255, 0, 0), 2, LINE_AA);
        }

        Mat dst;
        cvtColor(bgr, dst, COLOR_BGR2RGBA);
        // 获取 bitmap信息
        jobject newBitmap = BitmapUtil::getBitMap(env, dst.cols, dst.rows);
        BitmapMatUtils::mat2bitmap(env, newBitmap, dst);
        return newBitmap;
    }

    /// lbp 特征提取
    jobject lbpBitmap(JNIEnv *env, jobject bitmap) {
        Mat src;
        BitmapMatUtils::bitmap2mat(env, bitmap, src);

        Mat bgr;
        cvtColor(src, bgr, COLOR_RGBA2BGR);
        // 自己手写，两套代码：1. 3 * 3、 2. 考虑角度和步长
        Mat gary;
        cvtColor(src, gary, COLOR_BGRA2GRAY);

        // 3*3 lbp 计算的特征数据
        Mat result = Mat::zeros(Size(src.cols - 1, src.rows -1), CV_8UC1);

        for (int row = 1; row < gary.rows - 1; ++row) {
            for (int col = 1; col < gary.cols - 1; ++col) {
                uchar pixels = gary.at<uchar>(row, col);
                int rPixels = 0;
                rPixels |= (pixels >= gary.at<uchar>(row - 1, col - 1)) << 0;
                rPixels |= (pixels >= gary.at<uchar>(row - 1, col)) << 1;
                rPixels |= (pixels >= gary.at<uchar>(row - 1, col + 1)) << 2;
                rPixels |= (pixels >= gary.at<uchar>(row, col - 1)) << 7;
                rPixels |= (pixels >= gary.at<uchar>(row, col + 1)) << 3;
                rPixels |= (pixels >= gary.at<uchar>(row + 1, col - 1)) << 6;
                rPixels |= (pixels >= gary.at<uchar>(row + 1, col)) << 5;
                rPixels |= (pixels >= gary.at<uchar>(row + 1, col + 1)) << 4;

                result.at<uchar>(row - 1, col - 1) = rPixels;
            }
        }


        Mat dst;
        cvtColor(bgr, dst, COLOR_BGR2RGBA);
        // 获取 bitmap信息
        jobject newBitmap = BitmapUtil::getBitMap(env, result.cols, result.rows);
        BitmapMatUtils::mat2bitmap(env, newBitmap, result);
        return newBitmap;
    }

    /// 均值
    jobject mean_value(JNIEnv *env, jobject bitmap) {
        Mat src = (Mat_<double>(3,3) << 50, 50, 50, 60, 60, 60, 70, 70, 70);
        // 求平均值怎么求？（50 + 50 + 。。。+70）/9 = 60
        // 方差怎么求 （(50 - 60)的平方 + (50 - 60)的平方 + (50 - 60)的平方 +...+(70 - 60)的平方）/ 9 = 开根号（66.666666）= 8.164
        Mat mean, stddev;
        meanStdDev(src, mean, stddev);
        // 协方差矩阵怎么求？
        Mat covar;
        calcCovarMatrix(src, covar, mean, COVAR_NORMAL | COVAR_ROWS);

        // 协方差矩阵再去求 特征和特征向量
        Mat src1 = (Mat_<double>(2,2) << 100, 100, 100, 100);
        Mat eigenvalues,eigenvectors;
        eigen(src1, eigenvalues, eigenvectors);
        for (int i = 0; i < eigenvalues.rows; ++i) {
            LOGE("eigenvectors %f", eigenvectors.at<float>(i));
        }
        return bitmap;
    }

    // PCA 原理与应用，降维
    jobject pca(JNIEnv *env, jobject bitmap) {
        Mat src;
        BitmapMatUtils::bitmap2mat(env, bitmap, src);

        Mat bgr;
        cvtColor(src, bgr, COLOR_RGBA2BGR);

        int  size = bgr.rows * bgr.cols;
        Mat data(size, 3, CV_8UC1);

        for (int i = 0; i < size; ++i) {
            int row = i / bgr.cols;
            int col = i - row * bgr.cols;

            data.at<uchar>(i,0) = bgr.at<Vec3b>(row, col)[0];
            data.at<uchar>(i,1) = bgr.at<Vec3b>(row, col)[1];
            data.at<uchar>(i,2) = bgr.at<Vec3b>(row, col)[2];
        }

        // 最终降维的数据
        PCA pca_analyze(data, Mat(), PCA::Flags::DATA_AS_ROW);
        LOGE("%f", pca_analyze.mean.at<float>(0,0));
        LOGE("%f", pca_analyze.mean.at<float>(0,1));
        LOGE("%f", pca_analyze.mean.at<float>(0,2));

        // 获取 bitmap信息
        jobject newBitmap = BitmapUtil::getBitMap(env, src.cols, src.rows);
        BitmapMatUtils::mat2bitmap(env, newBitmap, src);
        return newBitmap;
    }
}