#include "Log.cpp"
#include "opencv2/opencv.hpp"
#include "writeSd.cpp"

using namespace std;
using namespace cv;

int element_size = 1;
int max_Size = 21;

namespace img_o{
    string pixelOperate(Mat &src) {// 灰度图
        /*Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        int rows = gray.rows;
        int cols = gray.cols;
        int channel = gray.channels();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (channel == 3){
                    // 获取像素 at Vec3b 代表3通道
                    int b = gray.at<Vec3b>(i,j)[0];
                    int g = gray.at<Vec3b>(i,j)[1];
                    int r = gray.at<Vec3b>(i,j)[2];

                    // 修改像素(底片效果)
                    gray.at<Vec3b>(i,j)[0] = 255 - b;
                    gray.at<Vec3b>(i,j)[1] = 255 - g;
                    gray.at<Vec3b>(i,j)[2] = 255 - r;
                } else if (channel == 1){
                    // 获取像素 at Vec3b 代表3通道
                    uchar pixels = gray.at<uchar>(i,j);
                    // 修改像素(底片效果)
                    gray.at<uchar>(i,j) = 255 - pixels;
                }
            }
        }*/

        // 自己转灰度图
        int rows = src.rows;
        int cols = src.cols;
        int channel = src.channels();

        // 创建单通道 mat
        Mat gray(rows, cols, CV_8UC1);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (channel == 3){
                    // 获取像素 at Vec3b 代表3通道
                    int b = src.at<Vec3b>(i,j)[0];
                    int g = src.at<Vec3b>(i,j)[1];
                    int r = src.at<Vec3b>(i,j)[2];

                    // 处理像素，成为灰度图
                    gray.at<uchar>(i,j) = 0.11f * r + 0.59f * g + 0.30f * b;
                }
            }
        }

//        matT::writeSd(src, "src.jpg");
        writeSd1(gray, "mat_operate.jpg");
        return "mat_operate.jpg";
    }


    string img_saturation(Mat src) {
        // 滤镜 UI设计师 在调亮一点，饱和一点，对比度在调高一点 alpha 增大成比例去增 1:500  10:5000
        // alpha：调饱和度、对比度
        // beta：调 亮度
        // F(R) = alpha*R + beta;
        // F(G) = alpha*G + beta;
        // F(B) = alpha*B + beta;

        int rows = src.rows;
        int cols = src.cols;
        int channel = src.channels();

        int alpha = 1;
        int beta = 50;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (channel == 3){
                    // 获取像素 at Vec3b 代表3通道
                    int b = src.at<Vec3b>(i,j)[0];
                    int g = src.at<Vec3b>(i,j)[1];
                    int r = src.at<Vec3b>(i,j)[2];

                    // 处理像素，成为灰度图
                    src.at<Vec3b>(i,j)[0] = saturate_cast<uchar>(alpha * b + beta);
                    src.at<Vec3b>(i,j)[1] = saturate_cast<uchar>(alpha * g + beta);
                    src.at<Vec3b>(i,j)[2] = saturate_cast<uchar>(alpha * r + beta);
                }
            }
        }

        writeSd1(src, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 图像绘制
    string img_draw(Mat src) {
        // 线 line LINE_8 LINE_4 LINE_AA 之间的区别？
        line(src, Point(100, 100), Point(200,200), Scalar(255, 0, 0),2, LINE_8);

        // 椭圆 ellipse
        // center: 中心点 axes：size 第一个值是椭圆 x width 的半径，第二个。。。
        // angle: 椭圆的旋转角度
//        ellipse(src, Point(src.cols/2, src.rows/2), Size(src.cols/8, src.rows/4),
//                180,0, 360, Scalar(0, 255, 255),2);
        // 矩形 rectangle
        rectangle(src, Point(100, 100), Point(200,200), Scalar(0, 0, 255),2, LINE_8);

        // 圆 circle
//        circle(src, Point(src.cols/2, src.rows/2), 100, Scalar(0, 0, 255), 2);
        // 填充 fillPloy 多边形
        //fillPoly(Mat& img, const Point** pts,
        //         const int* npts, int ncontours,
        //         const Scalar& color, int lineType = LINE_8, int shift = 0,
        //         Point offset = Point() );

        Point pts[1][3];
        pts[0][0] = Point(100, 100);
        pts[0][1] = Point(100, 200);
        pts[0][2] = Point(200, 100);

        const Point* ptss[] = {pts[0]};
        const int npts[] = {3};

        fillPoly(src, ptss, npts, 1, Scalar(255, 0, 0));

        // 文字 putText
        putText(src, "hello OC", Point(100, 100), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1, LINE_AA);
        // 随机画 srand
        // opencv 做随机
        RNG rng(time(NULL));
        // 创建一张图，与 src 的宽度和颜色通道一致，里面的值都是 0
        Mat dst = Mat::zeros(src.size(), src.type());
        for (int i = 0; i < 1000; ++i) {
            Point sp;
            sp.x = rng.uniform(0, dst.cols);
            sp.y = rng.uniform(0, dst.rows);
            Point ep;
            ep.x = rng.uniform(0, dst.cols);
            ep.y = rng.uniform(0, dst.rows);

            line(dst, sp, ep, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),2);
        }

        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }


    string img_vague(Mat src) {
        // 均值模糊
        Mat dest;
        // 1. 均值模糊
        // Size: 宽高只能是奇数
//        blur(src, dest, Size(151, 151));

        //2. 高斯模糊
        // sigmaX：作用（）
        // sigmaY：不传代表和 sigmaX 一样
        // 如果 sigmaX >= 0,自己会计算 0.3*((ksize - 1)*0.5 - 1) + 0.8
        // 自己传得怎么传？有什么意义？
//        GaussianBlur(src, dest, Size(151, 151), 0);

        // 3. 中值滤波
//        medianBlur(src, dest, 7);
        // 4. 双边滤波 - 美容
//        bilateralFilter(src, dest, 15, 100, 5);

        //  掩膜操作 - 卷积
//        Mat final;
//        Mat kernel = (Mat_<char>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);//定义掩膜
//        //调用filter2D
//        filter2D(src,final,src.depth(),kernel);

        // 自定义线性滤波与图像模糊
        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        // 1. Robert 算子：
//        Mat kernel = (Mat_<char>(2,2)<< 1, 0, 0, -1);//定义掩膜
        // 2. Sobel算子：
//        Mat kernel = (Mat_<char>(3,3)<< -1, 0, 1, -2, 0, 2,-1, 0, 1);//定义掩膜
//        Mat kernel = (Mat_<char>(3,3)<< -1, -2, -1, 0, 0, 0,1, 2, 1);//定义掩膜

        // 3. 拉普拉斯：
//        Mat kernel = (Mat_<char>(3,3)<< 0, -1, 0, -1, 4, -1, 0, -1, 0);//定义掩膜
//        //调用filter2D
//        // depth 概念：type的精度层度
//        filter2D(gray,dest,gray.depth(),kernel);

        // 自定义 模糊
//        int size = 5;
//        Mat kernel = Mat::ones(size, size, CV_32F) / (size * size);
//        filter2D(src,dest,gray.depth(),kernel);

        // 图像二值化
//        threshold(gray, dest, 100, 255, THRESH_OTSU);
        // 部分区域取一个 thresh
        adaptiveThreshold(gray, dest, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 0);

        writeSd1(dest, "mat_operate.jpg");
        return "mat_operate.jpg";
    }


    string img_jzym(Mat src) {
        // 公式  I(i,j)=5∗I(i,j)−[I(i−1,j)+I(i+1,j)+I(i,j−1)+I(i,j+1)]
        Mat dest = Mat::zeros(src.size(),src.type());//生成一个和源图像大小相等类型相同的全0矩阵
        int cols = (src.cols-1)*src.channels();//获取图像的列数,一定不要忘记图像的通道数
        int rows = src.rows;//获取图像的行数
        int offsetx = src.channels();

        for (int row = 1; row < rows-1;row++){
            // 上一行
            uchar* previous = src.ptr<uchar>(row-1);
            // 当前行
            uchar* current = src.ptr<uchar>(row);
            // 下一行
            uchar* next = src.ptr<uchar>(row+1);
            // 输出
            uchar* output = dest.ptr<uchar>(row);
            for (int col = offsetx; col < cols; col++){
                output[col] = saturate_cast<uchar>(
                    5*current[col] - (current[col- offsetx ]+current[col+offsetx] + previous[col]+next[col]));
            }
        }

        writeSd1(dest, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    string img_jzym1(Mat src) {
        Mat dest;
        Mat kernel = (Mat_<char>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);//定义掩膜
        //调用filter2D
        filter2D(src,dest,src.depth(),kernel);
        writeSd1(dest, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 膨胀与腐蚀
    string img_erode_dilate(Mat src) {
        Mat dest;
        // 创建一个 kernel
        Mat kernel = getStructuringElement(MORPH_RECT,Size(10, 10));
        // 腐蚀
//        erode(src, dest, kernel);
        // 膨胀
//        dilate(src, dest, kernel);
        // 形态学操作
        // CV_MOP_OPEN 开操作：先腐蚀后膨胀
        //CV_MOP_CLOSE 闭操作：先膨胀后腐蚀
        //CV_MOP_GRADIENT 梯度：膨胀 - 腐蚀
        //CV_MOP_TOPHAT 顶帽：原图像 - 开图像
        //CV_MOP_BLACKHAT 黑帽：闭图像 - 原图像
        morphologyEx(src, dest, CV_MOP_CLOSE, kernel);
        writeSd1(dest, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 提取图片验证码
    string get_img_code(Mat src) {
        Mat dst;
        // 1. 把彩色照片变成灰白 二值化
        // 灰白：
        Mat gary;
        cvtColor(src, gary, COLOR_BGR2GRAY);
        // 二值化方法，自动阈值 gary 必须是单通道
        //~ 0 - 255 -> 255 - 0 取反
        Mat binary;
        adaptiveThreshold(~gary, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 0);

        // 创建一个 kernel
        Mat kernel = getStructuringElement(MORPH_RECT,Size(4, 4));
        // 腐蚀 取最大值
        erode(binary, dst, kernel);

        // 膨胀 取最小值
        dilate(dst, dst, kernel);

//        morphologyEx(src, dst, CV_MOP_CLOSE, kernel);
        // 取反 ~ 或
        bitwise_not(dst, dst);

        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }


    string get_H_V_line(Mat src) {
        Mat dst;
        // 灰白：
        Mat gary;
        cvtColor(src, gary, COLOR_BGR2GRAY);
        // 二值化方法，自动阈值 gary 必须是单通道
        //~ 0 - 255 -> 255 - 0 取反
        Mat binary;
        adaptiveThreshold(~gary, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 0);
        // 二值化有问题，做一下弥补
        // 创建一个 kernel
        Mat kernel = getStructuringElement(MORPH_RECT,Size(9, 9));
        dilate(binary, dst, kernel);

        // 腐蚀 取最小值
        erode(dst, dst, kernel);

        // 取水平
        Mat lLine = getStructuringElement(MORPH_RECT,Size(src.cols/16, 1));
        // 取垂直
        Mat vLine = getStructuringElement(MORPH_RECT,Size(1, src.rows/16));
        // 腐蚀 取最小值
        erode(dst, dst, vLine);
        dilate(dst, dst, vLine);

        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    string get_caiyang(Mat src) {
        Mat dst;
        // 上采样
//        pyrUp(src, dst, Size(src.cols * 2, src.rows*2));
        // 降采样 - 缩小  ，高斯 - 比较慢
        pyrDown(src, dst, Size(src.cols / 2, src.rows / 2));

        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 图像边缘检测
    string txbyjc_sobel(Mat src) {
        Mat dst;
        // 1 降噪 高斯
        Mat gaussian;
        GaussianBlur(src, gaussian, Size(3,3), 0);
        // 2 转灰度
        Mat gray;
        cvtColor(gaussian, gray, COLOR_BGR2GRAY);
        // 2. Sobel算子：
//        Mat kernel = (Mat_<char>(3,3)<< -1, 0, 1, -2, 0, 2,-1, 0, 1);//定义掩膜
//        filter2D(gray,dst,gray.depth(),kernel);
        // 3 Sobel 梯度
        // delta: 在计算结果的基础上再加上 delta
        // ddepth: -1 代表与 gray.depth() 相同,传bigray 的精度高的值
        // x,y 求梯度一般不用 sobel，用 Scharr 增强
        Mat sobel_x, sobel_y;
//        Sobel(gray, sobel_x, CV_32F, 1, 0, 3);
//        Sobel(gray, sobel_y, CV_32F, 0, 1, 3);
        Scharr(gray, sobel_x, CV_32F, 1, 0, 3);
        Scharr(gray, sobel_y, CV_32F, 0, 1, 3);

        // 取绝对值
        convertScaleAbs(sobel_x, sobel_x);
        convertScaleAbs(sobel_y, sobel_y);

        Mat sobel(gray.size(), gray.type());
        // 两张图像混合
//        addWeighted(sobel_x, 1, sobel_y, 1, 0, sobel);
        int width = gray.cols;
        int height = gray.rows;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int x_p = sobel_x.at<uchar>(i, j);
                int y_p = sobel_y.at<uchar>(i, j);
                int p = x_p + y_p;
                sobel.at<uchar>(i, j) = saturate_cast<uchar>(p);
            }
        }

        writeSd1(sobel, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 图像边缘检测
    string txbyjc_laplacian(Mat src) {
        Mat dst;
        // 1 降噪 高斯
        Mat gaussian;
        GaussianBlur(src, gaussian, Size(3,3), 0);
        // 2 转灰度
        Mat gray;
        cvtColor(gaussian, gray, COLOR_BGR2GRAY);
        // 3 lpls
        Mat lpls;
        Laplacian(gray, lpls, CV_16S, 5);
        // 4 求绝对值
        convertScaleAbs(lpls, lpls);

        // 5 二值化
        threshold(lpls, dst, 0, 255, THRESH_OTSU | cv::THRESH_BINARY);

        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 图像边缘检测
    string txbyjc_canny(Mat src) {
        Mat dst;

        // 原理
        // 1. 高斯去噪声
        // 2. 灰度转换
        // 3. 计算梯度 sobel / scharr
        // 4. 非最大信号抑制
        // 5. 高低阈值输出二值图像（0-255）threshold1 低阈值 threshold2 高阈值，
        // 在 threshold1，threshold2 之间，取最大值255，否则取 0，
        // 尽量 1：2 或 1：3 30-60、30-90 50-100 10-150

        // 图像边缘检测
        // L2gradient false L1gradient true:
        Canny(src, dst, 50, 150, 3, false);

        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 霍夫检测
    string txbyjc_hf_check(Mat src) {
        Mat dst;

        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);

        // 1. canny 边缘检测
        Mat cannyImg;
        Canny(gray, cannyImg, 140, 250, 3);

        // 2. HoughLines
        // lines 信息里面包含多条直线，每条直线是两个点
        // rho 像素扫描间隙
        // theta 每次增加的角度 CV_PI / 360 = 0.5 度
        // threshold 低阈值
        // minLineLength 最小线段长度
        // maxLineGap: 点之间的间隙
        vector<Vec4f> plines;
        //1. 直线检测
//        HoughLinesP(cannyImg, plines, 1, CV_PI / 360, 170, 30, 3);
//        for (int i = 0; i < plines.size(); ++i) {
//            Vec4f pline = plines[i];
//            // 划线
//            line(src, Point(pline[0], pline[1]),Point(pline[2], pline[3]), Scalar(0,0, 255), 4, LINE_AA);
//        }

        //2. 圆检测
        // minDist: 两个圆的最小距离
        // param1: 投票累加的结果超过多少才能算圆
        // param2: 低阈值
        // minRadius: 圆的最小半径
        // maxRadius：圆的最大半径
//        vector<Vec3f> circles;
//        HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 10, 100,30, 5, 50);
//        for (int i = 0; i < circles.size(); ++i) {
//            Vec3f cc = circles[i];
//            circle(src, Point(cc[0], cc[1]), cc[2], Scalar(0, 0, 255), 4, LINE_AA);
//        }

        // 3. 重映射 QQ上传空间照片，特效滤镜，镜像，万花筒效果
        // map1: x方向重映射规律
        // map2: y方向重映射规律
        // interpolation
        Mat map_x(src.size(), CV_32F);
        Mat map_y(src.size(), CV_32F);
        // 照片左右调换
        for (int row = 0; row < src.rows; ++row) {
            for (int col = 0; col < src.cols; ++col) {
                // 照片左右调换
//                map_x.at<float>(row,col) = src.cols - col - 1;
//                map_y.at<float>(row,col) = row;
                // 照片上下调换
//                map_x.at<float>(row,col) = col;
//                map_y.at<float>(row,col) = src.rows - row - 1;
                // 缩小2倍
                if(col>src.cols*0.25 && col<src.cols*0.75 && row>src.rows*0.25 && row<src.rows*0.75)
                {
                    map_x.at<float>(row,col)=static_cast<float>(2*(col-src.cols*0.25)+0.5);
                    map_y.at<float>(row,col)=static_cast<float>(2*(row-src.rows*0.25)+0.5);
                }
                else
                {
                    map_x.at<float>(row,col)=0;
                    map_y.at<float>(row,col)=0;

                }
            }
        }
        remap(src, dst, map_x, map_y, 1);

        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 直方图 - 均衡化
    string txbyjc_zft(Mat src) {
        Mat dst;
        Mat gray;
        cvtColor(src, gray, CV_BGR2GRAY);
        // 直方图均衡化
        equalizeHist(gray, dst);
        writeSd1(dst, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 直方图 - 数据直方图
    // 获取图片数据直方图，BGR 颜色直方图，任何数据直方图
    string txbyjc_data_zft(Mat src) {
        Mat dst;
        // 获取直方图 B G R 每个单独分离出来
        vector<Mat> bgr_s;
        split(src, bgr_s);

        // b 通道，但为什么不是 蓝色，而是一个灰度图？单通道
        //writeSd1(bgr_s[0], "mat_operate.jpg");

        // 计算获取直方图数据
        /*
         * calcHist( const Mat* images, int nimages,
                          const int* channels, InputArray mask,
                          OutputArray hist, int dims, const int* histSize,
                          const float** ranges, bool uniform = true, bool accumulate = false );
         */
        // images: 输入图像
        // nimages: 输入图像个数
        // channels：第几通道
        // mask: 掩膜
        // hist：输出
        // dims：需要统计的通道个数
        // histSize: 等级的个数 0 - 255
        // ranges: 数据的范围
        // uniform: true 是否对得到的图像进行归一化处理
        // accumulate: 在多个图像是是否累计计算像素值的个数
        int histSize = 256;
        float range[] = {0, 255};
        const float *ranges = {range};
        Mat hist_b, hist_g, hist_r;
        calcHist(&bgr_s[0], 1, 0, Mat(), hist_b, 1, &histSize, &ranges, true, false);
        calcHist(&bgr_s[1], 1, 0, Mat(), hist_g, 1, &histSize, &ranges, true, false);
        calcHist(&bgr_s[2], 1, 0, Mat(), hist_r, 1, &histSize, &ranges, true, false);
        // 画出来 hist_b 存的是什么？存的是各个灰度值的个数，hist_b 最小值 0，最大值 图片的宽 * 高

        // 归一化
        int hist_h = 400;
        int hist_w = 512; // 256 * 2
        int bin_w = hist_w / histSize; // 画笔的大小
        /*normalize( InputArray src, InputOutputArray dst, double alpha = 1, double beta = 0,
                             int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray())*/
        // alpha: 最小值
        // beta: 最大值
        // norm_type：NORM_MINMAX(缩放到一定区域)
        normalize(hist_b,hist_b, 0, hist_h, NORM_MINMAX, -1, Mat());
        normalize(hist_g,hist_g, 0, hist_h, NORM_MINMAX, -1, Mat());
        normalize(hist_r,hist_r, 0, hist_h, NORM_MINMAX, -1, Mat());
        // 画到一张图中
        Mat histImage(hist_h, hist_w, CV_8SC4, Scalar());
        for (int i = 1; i < histSize; ++i) {
            line(histImage, Point((i - 1)*bin_w, hist_h - hist_b.at<float>(i - 1)), Point((i)*bin_w, hist_h - hist_b.at<float>(i)), Scalar(255, 0, 0), bin_w, LINE_AA);
            line(histImage, Point((i - 1)*bin_w, hist_h - hist_g.at<float>(i - 1)), Point((i)*bin_w, hist_h - hist_g.at<float>(i)), Scalar(0, 255, 0), bin_w, LINE_AA);
            line(histImage, Point((i - 1)*bin_w, hist_h - hist_r.at<float>(i - 1)), Point((i)*bin_w, hist_h - hist_r.at<float>(i)), Scalar(0, 0, 255), bin_w, LINE_AA);
        }

        writeSd1(histImage, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 直方图比较
    string txbyjc_zft_bj(Mat src) {

        Mat src1 = (imread("/data/user/0/com.swan.study_opencv/files/yzm.png"));

        // RGB -> HSV 计算 HS直方图
        Mat hsv, hsv1;
        cvtColor(src, hsv, COLOR_RGB2HSV);
        cvtColor(src1, hsv1, COLOR_RGB2HSV);

        // 计算直方图
        // 复数形式
        MatND hist,hist1;
        int channels[] = {0, 1};
        int h_bins = 50;
        int s_bins = 50;
        int hist_Size[] = {h_bins, s_bins};
        float h_ranges[] = {0, 180}; // 0 - 360
        float s_ranges[] = {0, 255}; // 0 - 360
        const float *ranges[] = {h_ranges, s_ranges};
        calcHist(&hsv, 1, channels, Mat(), hist, 2, hist_Size, ranges);
        calcHist(&hsv1, 1, channels, Mat(), hist1, 2, hist_Size, ranges);

        // 归一化
        // alpha: 最小值
        // beta: 最大值
        // norm_type：NORM_MINMAX(缩放到一定区域)
        normalize(hist,hist, 0, 1, NORM_MINMAX);
        normalize(hist1,hist1, 0, 1, NORM_MINMAX);

        // 两个直方图，采用的方法
        // compareHist( InputArray H1, InputArray H2, int method );
        // 相关性比较
        //double hist_hist= compareHist(hist, hist, CV_COMP_CORREL); // 自己和自己比较  1 (最好)
        //double hist_hist1= compareHist(hist, hist1, CV_COMP_CORREL); // 自己和其他比较
        // 巴氏距离
        double hist_hist= compareHist(hist, hist, CV_COMP_BHATTACHARYYA); // 自己和自己比较  1 (最好)
        double hist_hist1= compareHist(hist, hist1, CV_COMP_BHATTACHARYYA); // 自己和其他比较

        LOGE("比较结果hist_hist：%f", hist_hist);
        LOGE("比较结果hist_hist1：%f", hist_hist1);

        writeSd1(src, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 直方图反向投射
    string txbyjc_zft_fxts(Mat src) {

        // RGB -> HSV 计算 HS直方图
        Mat hsv;
        cvtColor(src, hsv, COLOR_RGB2HSV);

        // 像素分离
        vector<Mat> hsv_s;
        split(hsv, hsv_s);

        Mat hueImage = hsv_s[0];

        // 计算直方图
        // 计算直方图
        // 复数形式
        Mat hist;
        int bins = 2;
        int hist_Size = MAX(bins, 2);

        float h_ranges[] = {0, 180}; // 0 - 360
        const float *ranges[] = {h_ranges};
        calcHist(&hueImage, 1, 0, Mat(), hist, 1, &hist_Size, ranges);

        // 归一化
        // alpha: 最小值
        // beta: 最大值
        // norm_type：NORM_MINMAX(缩放到一定区域)
        normalize(hist,hist, 0, 255, NORM_MINMAX);

        // 直方图反向投影到 mat
        /*calcBackProject(
         * const Mat* images,
         * int nimages,
         * const int* channels,
         * InputArray hist,
         * OutputArray backProject,
         * const float** ranges,
         * double scale = 1,
         * bool uniform = true
         * )*/
        // 反射投影的次数，并不是像素值
        Mat backProject;
        calcBackProject(&hueImage, 1, 0, hist, backProject, ranges);

        writeSd1(backProject, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 直方图 模版匹配
    string txbyjc_zft_mbpp(Mat src) {
        Mat src1 = imread("/data/user/0/com.swan.study_opencv/files/tem_src.png");
        Mat templ = imread("/data/user/0/com.swan.study_opencv/files/tem.png");
        Mat result(src.rows - templ.rows + 1, src.cols - templ.cols + 1, CV_32FC1);
        // 进行匹配和标准化
        /*matchTemplate( InputArray image, InputArray templ,
                                 OutputArray result, int method, InputArray mask = noArray() )*/
        // result : 匹配计算的结果
        // 平方差匹配 method=CV_TM_SQDIFF 这类方法利用平方差来进行匹配,最好匹配为0.匹配越差,匹配值越大.
        matchTemplate( src1, templ, result, CV_TM_SQDIFF);

        // 从结果里面去找最小值，for，找到那个最小值的点 Point
        // 传出参数
        double minVal = 0, maxVal = 0;
        Point minLoc;
        Point maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        LOGE("minVal = %f ", minVal);

        // 画出来
        rectangle(src1, minLoc, Point(minLoc.x + templ.cols,minLoc.y + templ.rows), Scalar(0,0,255), 4, LINE_AA);

        writeSd1(src1, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 银行卡轮廓查找
    string card_lunkuo(Mat src) {
        // 梯度和二值化
        Mat binary;
        Canny(src, binary,50, 150); // 边缘检测

        // 轮廓查找
        vector<vector<Point> > contours;
        /*findContours(
         * InputOutputArray image,
         * OutputArrayOfArrays contours,
         * int mode, RETR_LIST（在不建立任何层次关系的情况下检索所有轮廓）RETR_EXTERNAL(提取最外层轮廓)
         * int method,
         * Point offset = Point()
         * )*/
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        Mat contours_mat = Mat::zeros(src.size(), CV_8UC3);
        Rect card_rect;
        for (int i = 0; i < contours.size(); ++i) {
            // 画轮廓
            // 筛选轮廓
            Rect rect = boundingRect(contours[i]);
            if (rect.width > src.cols / 2 && rect.height > src.rows / 2){
                drawContours(contours_mat, contours, i, Scalar(0, 0, 255), 1);
                card_rect = rect;
                rectangle(contours_mat,card_rect, Scalar(255, 255, 255), 2);
                break;
            }
        }
        // 裁剪
        Mat card_mat(src, card_rect);

        writeSd1(contours_mat, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 图形矩-多边形测试
    // 作用：替换背景 背景虚化
    string card_txj(Mat src1) {
        /// 创建一个图形
        const int r = 100;
        Mat src = Mat::zeros( Size( 4*r, 4*r ), CV_8UC1 );

        /// 绘制一系列点创建一个轮廓:
        vector<Point2f> vert(6);

        vert[0] = Point( 1.5*r, 1.34*r );
        vert[1] = Point( 1*r, 2*r );
        vert[2] = Point( 1.5*r, 2.866*r );
        vert[3] = Point( 2.5*r, 2.866*r );
        vert[4] = Point( 3*r, 2*r );
        vert[5] = Point( 2.5*r, 1.34*r );

        /// 在src内部绘制
        for( int j = 0; j < 6; j++ ){
            line( src, vert[j],  vert[(j+1)%6], Scalar( 255 ), 3, 8 );
        }

        /// 得到轮廓
        vector<vector<Point> > contours; vector<Vec4i> hierarchy;
        Mat src_copy = src.clone();
        // 查找轮廓
        findContours( src_copy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        /// 计算到轮廓的距离
        Mat raw_dist( src.size(), CV_32FC1 );

        for( int j = 0; j < src.rows; j++ ){
            for( int i = 0; i < src.cols; i++ ){
                // 找到一个点在矩形的里面还是外面还是在上面
                raw_dist.at<float>(j,i) = pointPolygonTest( contours[0], Point2f(i,j), true );
            }
        }

        // 查找最大值 最小值
        double minVal; double maxVal;
        minMaxLoc( raw_dist, &minVal, &maxVal, 0, 0, Mat() );
        minVal = abs(minVal); maxVal = abs(maxVal);

        /// 图形化的显示距离
        Mat drawing = Mat::zeros( src.size(), CV_8UC3 );

        for( int j = 0; j < src.rows; j++ ){
            for( int i = 0; i < src.cols; i++ ){
                if( raw_dist.at<float>(j,i) < 0 ){ // 外面
                    drawing.at<Vec3b>(j,i)[0] = 255 - (int) abs(raw_dist.at<float>(j,i))*255/minVal;
                }else if(raw_dist.at<float>(j,i) > 0 ){ // 里面
                    drawing.at<Vec3b>(j,i)[2] = 255 - (int) raw_dist.at<float>(j,i)*255/maxVal;
                }else{ // 矩形上面
                    drawing.at<Vec3b>(j,i)[0] = 255;
                    drawing.at<Vec3b>(j,i)[1] = 255;
                    drawing.at<Vec3b>(j,i)[2] = 255;
                }
            }
        }


        writeSd1(drawing, "mat_operate.jpg");
        return "mat_operate.jpg";
    }

    // 图像风水岭切割
    string txfslqg(Mat src) {

        Mat markers;
        watershed(src, markers);

        writeSd1(markers, "mat_operate.jpg");
        return "mat_operate.jpg";
    }
}