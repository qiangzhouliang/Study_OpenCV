//
// 微信公众号二维码检测与识别
//
#ifndef STUDY_OPENCV_IMG_WX_QR_CODE_CPP
#define STUDY_OPENCV_IMG_WX_QR_CODE_CPP

#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
namespace qrCode {

    //  判断 X 方向上是否符合规则
    bool isXVerify(const Mat& qrROI){
        // 判断 x 方向从左到右的像素比例
        // 黑：白：黑：白：黑 = 1:1:3:1:1
        int cb = 0, lw = 0, rw = 0, lb = 0, rb = 0;

        int width = qrROI.cols;
        int height = qrROI.rows;
        int cx = width / 2;
        int cy = height / 2;
        uchar pixels = qrROI.at<uchar>(cy, cx);
        if (pixels == 255){
            return false;
        }

        // 求中心 黑色
        int start = 0, end = 0, offset = 0;
        bool findleft = false, findright = false;
        while (true){
            offset ++;
            if ((cx - offset) <= 0 || (cx + offset >= width - 1)){
                break;
            }
            // 左边扫
            pixels = qrROI.at<uchar>(cy, cx - offset);
            if (!findleft && pixels == 255){
                start = cx - offset;
                findleft = true;
            }
            // 右边扫
            pixels = qrROI.at<uchar>(cy, cx + offset);
            if (!findright && pixels == 255){
                end = cx + offset;
                findright = true;
            }

            if (findleft && findright){
                break;
            }
        }
        if (start == 0 || end == 0){
            return false;
        }

        cb = end - start;// 中间黑色
        // 相间的白色
        for (int col = end; col < width - 1; ++col) {
            pixels = qrROI.at<uchar>(cy, col);
            if (pixels == 0){
                break;
            }
            rw++;
        }
        for (int col = start; col > 0; --col) {
            pixels = qrROI.at<uchar>(cy, col);
            if (pixels == 0){
                break;
            }
            lw++;
        }
        if (lw == 0 || rw == 0 ){
            return false;
        }
        // 两边的黑色
        for (int col = end + rw; col < width - 1; ++col) {
            pixels = qrROI.at<uchar>(cy, col);
            if (pixels == 255){
                break;
            }
            rb++;
        }

        for (int col = start + lw; col > 0; --col) {
            pixels = qrROI.at<uchar>(cy, col);
            if (pixels == 255){
                break;
            }
            lb++;
        }
        if (lb == 0 || rb == 0){
            return false;
        }
        // 求比例 黑：白：黑：白：黑 = 1:1:3:1:1
        float sum = cb + lb + rb + lw + rw;
        cb = (cb / sum) * 7.0 + 0.5;
        lb = (lb / sum) * 7.0 + 0.5;
        rb = (rb / sum) * 7.0 + 0.5;
        lw = (lw / sum) * 7.0 + 0.5;
        rw = (rw / sum) * 7.0 + 0.5;
        if ((cb == 3 || cb == 4) && (lw == rw) && (lb == rb) && (lw == rb) && (lw == 1)){
            return true;
        }

        return false;
    }

//  判断 Y 方向上是否符合规则
    bool isYVerify(const Mat& qrROI){
        // y 方向上也可以按照 isXVerify 方法判断
        // 但我们也可以适当的写简单一些
        // 白色像素 * 2 < 黑色像素 && 黑色像 < 4 * 白色像素
        // 1. 统计白色像素点和黑色像素点
        int bp = 0, wp = 0;
        int width = qrROI.cols;
        int height = qrROI.rows;
        int cx = width / 2;

        // 2. 中心点是黑色
        int pv = 0;
        for (int row = 0; row < height; ++row) {
            pv = qrROI.at<uchar>(row, cx);
            if (pv == 0){
                bp ++;
            } else if (pv == 255){
                wp++;
            }
        }
        if (bp == 0 || wp == 0){
            return false;
        }
        if (wp * 2 > bp || bp > 4 * wp){
            return false;
        }

        return true;
    }

    // 转换倾斜，非正方形照片 为 正方形
    Mat warpTransfrom(const Mat &gary, const RotatedRect &rect){
        int width = rect.size.width;
        int height = rect.size.height;
        Mat res(Size(width, height), gary.type());
        // 矩阵
        vector<Point> srcPoints;
        // 取照片上的四个点
        Point2f pts[4];
        rect.points(pts);
        for (int i = 0; i < 4; ++i) {
            srcPoints.push_back(pts[i]);
        }
        // 对应目标图片上的四个点
        vector<Point> dstPoints;
        dstPoints.push_back(Point(0,0));
        dstPoints.push_back(Point(width,0));
        dstPoints.push_back(Point(width,height));
        dstPoints.push_back(Point(0,height));

        Mat M = findHomography(srcPoints, dstPoints);

        warpPerspective(gary, res, M, res.size());
        return res;
    }

// 微信公众号二维码检测与识别
    jobject wxQrCode(JNIEnv *env, jobject bitmap) {
        Mat src;
        BitmapMatUtils::bitmap2mat(env, bitmap, src);
        // 由于 bitmap是rgba 颜色通道，mat操作需要 bgr，=》 rgba -> bgr
        Mat bgr;
        cvtColor(src, bgr, COLOR_RGBA2BGR);
        // 对图像进行灰度转换
        Mat gary;
        cvtColor(bgr, gary, COLOR_BGR2GRAY);
        // 二值化
        threshold(gary, gary, 0.0, 255, THRESH_BINARY | THRESH_OTSU);
//        imshow("threshold", gary);
        // 1. 对其进行轮廓查找
        vector<vector<Point> > contours;
        vector<vector<Point> > resContours;
        findContours(gary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++){
            // 2. 对查找的到的轮廓进行初步过滤
            double area = contourArea(contours[i]);
            // 2.1 初步过滤面积 7*7 = 49
            if (area < 49){
                continue;
            }

            // 过滤宽高比和宽高大小
            RotatedRect rRect = minAreaRect(contours[i]);
            float w = rRect.size.width;
            float h = rRect.size.height;
            float ratio = min(w, h) / max(w, h);
            // 2.2 初步过滤宽高比大小
            if (ratio > 0.9 && w< gary.cols/2 && h< gary.rows/2){
                Mat qrROI = warpTransfrom(gary, rRect);
                // 3. 判断是否符合二维码的特征规则
                if (isYVerify(qrROI) && isXVerify(qrROI)) {
//                    char name[256];
//                    sprintf(name, "%d.jpg",i);
//                    writeSd1(qrROI, name);
                    drawContours(bgr, contours, i, Scalar(0, 0, 255), 4);
                }
            }
        }

        Mat dst;
        // bgr -> rgba
        cvtColor(bgr, dst, COLOR_BGR2RGBA);
        // 获取 bitmap信息
        jobject newBitmap = BitmapUtil::getBitMap(env, dst.cols, dst.rows);
        BitmapMatUtils::mat2bitmap(env, newBitmap, dst);
        return newBitmap;
    }
}
#endif //STUDY_OPENCV_IMG_WX_QR_CODE_H