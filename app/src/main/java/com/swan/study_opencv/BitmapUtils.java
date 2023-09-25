package com.swan.study_opencv;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;

/**
 * @ClassName BitmapUtils
 * @Description
 * @Author swan
 * @Date 2023/9/13 09:20
 **/
public class BitmapUtils {

    public static native int gary3(Bitmap src);
    // 转灰度图
    public static Bitmap gary(Bitmap src) {
        // 怎么变成灰度的？ 矩阵去操作
        Bitmap dst = Bitmap.createBitmap(src.getWidth(), src.getHeight(), src.getConfig());
        Canvas canvas = new Canvas(dst);

        Paint paint = new Paint();
        paint.setDither(true);
        paint.setAntiAlias(true);
        // 方法1 矩阵乘法 f = 0.11f * R + 0.59f * G + 0.30f * B
        //ColorMatrix colorMatrix = new ColorMatrix(new float[]{
        //    0.213f, 0.715f, 0.072f, 0, 0,
        //    0.213f, 0.715f, 0.072f, 0, 0,
        //    0.213f, 0.715f, 0.072f, 0, 0,
        //    0, 0, 0, 1, 0
        //});

        // 原图
        //ColorMatrix colorMatrix = new ColorMatrix(new float[]{
        //    1, 0, 0, 0, 0,
        //    0, 1, 0, 0, 0,
        //    0, 0, 1, 0, 0,
        //    0, 0, 0, 1, 0
        //});

        // 底片效果
        //ColorMatrix colorMatrix = new ColorMatrix(new float[]{
        //    -1, 0, 0, 0, 255,
        //    0, -1, 0, 0, 255,
        //    0, 0, -1, 0, 255,
        //    0, 0, 0, 1, 0
        //});

        // 提高饱和度
        ColorMatrix colorMatrix = new ColorMatrix(new float[]{
            1.2f, 0, 0, 0, 0,
            0, 1.2f, 0, 0, 0,
            0, 0, 1.2f, 0, 0,
            0, 0, 0, 1, 0
        });
        // 方法二： 饱和度设为 0
        //colorMatrix.setSaturation(0);

        paint.setColorFilter(new ColorMatrixColorFilter(colorMatrix));
        canvas.drawBitmap(src,0, 0, paint);
        return dst;
    }

    public static Bitmap gary2(Bitmap src) {
        // 怎么变成灰度的？ 矩阵去操作
        Bitmap dst = Bitmap.createBitmap(src.getWidth(), src.getHeight(), src.getConfig());
        // java 层像素操作
        int[] pixels = new int[src.getWidth() * src.getHeight()];
        src.getPixels(pixels, 0, src.getWidth(), 0, 0, src.getWidth(), src.getHeight());
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int a = pixel >> 24 & 0xff;
            int r = pixel >> 16 & 0xff;
            int g = pixel >> 8 & 0xff;
            int b = pixel & 0xff;

            // f = 0.213f * r + 0.715f * g + 0.072f * b
            int gray = (int) (0.213f * r + 0.715f * g + 0.072f * b);
            pixels[i] = (a << 24) | (gray << 16) | (gray << 8) | gray;

            // 黑白
            //int black_white = (a + r + b)/3 > 125 ? 255 : 0;
            //pixels[i] = (a << 24) | (black_white << 16) | (black_white << 8) | black_white;
        }

        dst.setPixels(pixels, 0, src.getWidth(), 0, 0, src.getWidth(), src.getHeight());
        return dst;
    }
}
