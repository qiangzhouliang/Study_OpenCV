package com.swan.study_opencv;

import android.graphics.Bitmap;

/**
 * @ClassName Plugin
 * @Description
 * @Author swan
 * @Date 2023/9/12 17:25
 **/
public class Plugin {

    public static native String stringFromJNI();

    public static native String setImg(String imgPath);
    public static native Bitmap operateBitm(Bitmap srcB);
}
