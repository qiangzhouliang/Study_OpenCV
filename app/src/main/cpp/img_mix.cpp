#include "Log.cpp"
#include "opencv2/opencv.hpp"
#include "writeSd.cpp"

using namespace std;
using namespace cv;

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
}