package com.swan.study_opencv;

import android.graphics.Bitmap;

import com.swan.opencv2.CVType;
import com.swan.opencv2.Imgproc;
import com.swan.opencv2.Mat;
import com.swan.opencv2.Utils;

/**
 * @ClassName OpenCVUtils
 * @Description
 * @Author swan
 * @Date 2023/9/22 10:17
 **/
public class OpenCVUtils {
    // 图片旋转
    public static native Bitmap rotation(Bitmap bitmap);

    // 仿射变换
    public static native Bitmap warpAffine(Bitmap bitmap);

    // 图片缩放
    public static native Bitmap reSize(Bitmap bitmap, int width, int height);

    public static native Bitmap reMap(Bitmap bitmap);

    // 美容
    public static Bitmap mask(Bitmap bitmap){
        Mat kernel = new Mat(3, 3, CVType.CV_32FC1);
        // 添加参数
        kernel.put(0,0,0);
        kernel.put(0,1,-1);
        kernel.put(0,2,0);

        kernel.put(1,0,-1);
        kernel.put(1,1,5);
        kernel.put(1,2,-1);

        kernel.put(2,0,0);
        kernel.put(2,1,-1);
        kernel.put(2,2,0);
        // bitmap -> mat 对象
        Mat srcMat = new Mat();
        Utils.bitmap2mat(bitmap, srcMat);

        Mat dstMat = new Mat();
        Imgproc.filter2D(srcMat, dstMat, kernel);

        Utils.mat2bitmap(dstMat, bitmap);
        return bitmap;
    }

    // 模糊
    public static Bitmap mh(Bitmap bitmap){
        // 模糊效果
        int size = 35;
        Mat kernel = new Mat(size, size, CVType.CV_32FC1);
        float value = 1f/(size * size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                kernel.put(i, j, value);
            }
        }

        Mat srcMat = new Mat();
        Utils.bitmap2mat(bitmap, srcMat);

        Mat dstMat = new Mat();
        Imgproc.filter2D(srcMat, dstMat, kernel);

        Utils.mat2bitmap(dstMat, bitmap);
        return bitmap;
    }

    /**
     * 图片美容
     * @param bitmap
     * @return
     */
    public static native Bitmap imgFacial(Bitmap bitmap);

    // 微信公众号二维码检测与识别
    public static native Bitmap wx_qr_code(Bitmap src) ;
}
