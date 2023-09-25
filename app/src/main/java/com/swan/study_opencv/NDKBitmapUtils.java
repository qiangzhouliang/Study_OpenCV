package com.swan.study_opencv;

import android.graphics.Bitmap;

/**
 * @ClassName NDKBitmapUtils
 * @Description
 * @Author swan
 * @Date 2023/9/21 15:40
 **/
public class NDKBitmapUtils {
    // 逆世界
    public native static Bitmap againstWorld(Bitmap src);

    // 浮雕效果
    public native static Bitmap anaglyph(Bitmap src);
    // 马赛克
    public native static Bitmap mosaic(Bitmap bitmap);

    // 毛玻璃
    public native static Bitmap groundGlass(Bitmap bitmap);
    // 油画效果
    public native static Bitmap oilPainting(Bitmap bitmap);
}
