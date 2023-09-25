package com.swan.opencv2;

/**
 * @ClassName Imgproc
 * @Description
 * @Author swan
 * @Date 2023/9/22 17:46
 **/
public class Imgproc {
    // 传递 Mat 对象
    public static void filter2D(Mat src, Mat dst, Mat kernel){
        nfilter2D(src.mNativePtr, dst.mNativePtr, kernel.mNativePtr);
    }

    private static native void nfilter2D(long srcPtr, long dstPtr, long kernelPtr);
}
