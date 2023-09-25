package com.swan.study_opencv;

import android.graphics.Bitmap;

/**
 * @ClassName SdkUtils
 * @Description
 * @Author swan
 * @Date 2023/9/22 17:22
 **/
public class SdkUtils {

    // 美容：掩膜操作
    public static native Bitmap mark(Bitmap src);

    // 模糊操作
    public static native Bitmap blur(Bitmap bitmap);
}
