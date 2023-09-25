package com.swan.opencv2;

import android.graphics.Bitmap;

/**
 * @ClassName Utils
 * @Description
 * @Author swan
 * @Date 2023/9/22 17:54
 **/
public class Utils {

    public static void bitmap2mat(Bitmap bitmap, Mat mat) {
        nbitmap2mat(bitmap, mat.mNativePtr);
    }


    public static void mat2bitmap(Mat mat, Bitmap bitmap) {
        nmat2bitmap(mat.mNativePtr, bitmap);
    }

    private static native void nmat2bitmap(long matPtr, Bitmap bitmap);

    private static native void nbitmap2mat(Bitmap bitmap, long matPtr);
}
