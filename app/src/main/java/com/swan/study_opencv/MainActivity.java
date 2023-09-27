package com.swan.study_opencv;

import static com.swan.study_opencv.Plugin.setImg;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Debug;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.swan.study_opencv.databinding.ActivityMainBinding;

import java.io.File;

public class MainActivity extends AppCompatActivity {
    // Used to load the 'study_opencv' library on application startup.
    static {
        System.loadLibrary("study_opencv");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        //Bitmap.createBitmap(1024, 1024 * 6000, Bitmap.Config.ARGB_8888);

        //testOpenCV();

        //testBitMap();
        //testBitMap1();

        // 手写 QQ 说说图片处理效果
        //testQQSayImage();
        // 图片集合变化
        //testTransform();
        // 封装sdk
        //testSdk();

        // 图片美容
        //testImgFacial();

        //微信公众号二维码检测与识别
        testwx_qr_code();
    }

    // 微信公众号二维码检测与识别
    private void testwx_qr_code() {
        Bitmap src = BitmapFactory.decodeResource(getResources(), R.drawable.qr_code);
        binding.image1.setImageBitmap(src);

        Bitmap dstB = OpenCVUtils.wx_qr_code(src);
        binding.image2.setImageBitmap(dstB);
    }

    private void testImgFacial() {
        Bitmap src = BitmapFactory.decodeResource(getResources(), R.drawable.test);
        binding.image1.setImageBitmap(src);

        Bitmap dstB = OpenCVUtils.imgFacial(src);
        binding.image2.setImageBitmap(dstB);
    }

    /**
     * 规则考虑周到（怎么才能考虑周到）？
     * 1. 细节拆分
     * 2. 尽量不要改native 层的代码，可以改Java层
     * 思考：时空算法复杂度？
     */
    private void testSdk() {
        // 需求假设
        // 1. 要对一张图片做美容，掩膜操作
        // 2. 又要一个模糊操作
        // 3. 又来一个关于 filter2D 方法的效果
        Bitmap src = BitmapFactory.decodeResource(getResources(), R.drawable.img);
        //binding.image1.setImageBitmap(src);
        
        binding.image2.setImageBitmap(OpenCVUtils.mh(src));
    }

    private void testTransform() {
        logMemory();
        Bitmap src = BitmapFactory.decodeResource(getResources(), R.drawable.yzm);
        binding.image1.setImageBitmap(src);

        // 图片旋转
        //Bitmap dstB = OpenCVUtils.rotation(src);
        // 仿射变换
        //Bitmap dstB = OpenCVUtils.warpAffine(src);
        // 图片缩放
        //Bitmap dstB = OpenCVUtils.reSize(src, src.getWidth()*2, src.getHeight()*2);
        // 重映射
        Bitmap dstB = OpenCVUtils.reMap(src);
        binding.image2.setImageBitmap(dstB);
        logMemory();
    }

    /**
     * 手写 QQ 说说图片处理效果
     */
    private void testQQSayImage() {
        logMemory();
        Bitmap src = BitmapFactory.decodeResource(getResources(), R.drawable.test_gl);
        binding.image1.setImageBitmap(src);

        // 测试 mat 和 Bitmap 之间的互转
        //Bitmap dstB = operateBitm(src);
        // 逆世界
        //Bitmap dstB = NDKBitmapUtils.againstWorld(src);
        // 浮雕效果
        //Bitmap dstB = anaglyph(src);
        // 马赛克
        //Bitmap dstB = NDKBitmapUtils.mosaic(src);
        // 毛玻璃
        //Bitmap dstB = NDKBitmapUtils.groundGlass(src);
        // 油画效果
        Bitmap dstB = NDKBitmapUtils.oilPainting(src);
        binding.image2.setImageBitmap(dstB);
        logMemory();
    }

    // bitmap 复用
    private void testBitMap1() {
        // 不复用的写法，消耗内存 20 M
        logMemory();
        Bitmap bitmap1 = BitmapFactory.decodeResource(getResources(), R.drawable.img);
        Bitmap bitmap2 = BitmapFactory.decodeResource(getResources(), R.drawable.img);
        logMemory();
        // 复用的写法，消耗内存 10 M
        //logMemory();
        //BitmapFactory.Options options = new BitmapFactory.Options();
        //options.inMutable = true;
        //Bitmap bitmap11 = BitmapFactory.decodeResource(getResources(), R.drawable.img, options);
        //options.inBitmap = bitmap11;
        //Bitmap bitmap22 = BitmapFactory.decodeResource(getResources(), R.drawable.img, options);
        //logMemory();
    }

    private void logMemory() {
        Runtime runtime = Runtime.getRuntime();
        Log.e("TAG", "logMemory: "+runtime.totalMemory() / 1024 / 1024);
        Log.e("TAG", "nativeMemory: "+ Debug.getNativeHeapAllocatedSize() / 1024 / 1024);
    }

    private void testBitMap() {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inMutable = true;
        // 如果设置为true，解码器将返回null（没有位图），但out。。。字段仍将被设置，允许调用者查询位图，而不必为其像素分配内存。
        // 可以获取图片真实的宽高
        //options.inJustDecodeBounds = true;
        // 有啥区别
        // ARGB_8888 32 位最好
        // ARGB_8888 -> RGB_565 RGB 5位R 6位G 5位B，总共 16位
        options.inPreferredConfig = Bitmap.Config.RGB_565;
        Bitmap src = BitmapFactory.decodeResource(getResources(), R.drawable.img, options);
        Log.e("TAG","Bitmap 宽=》"+src.getWidth());
        Log.e("TAG","Bitmap 高=》"+src.getHeight());
        Log.e("TAG","Bitmap 大小=》"+src.getByteCount());
        // 灰度图
        //Bitmap gary = BitmapUtils.gary2(src);
        //BitmapUtils.gary3(src);
        //binding.image2.setImageBitmap(src);
    }

    private void testOpenCV() {
        logMemory();
        Bitmap src = BitmapFactory.decodeResource(getResources(), R.drawable.card_n);
        binding.image1.setImageBitmap(src);
        //File mInFile = new File(getApplication().getFilesDir().getPath()+"/img.jpeg");
        File mInFile = new File(getApplication().getFilesDir().getPath()+"/card_n.png");
        String dstPath = setImg(mInFile.getPath());
        Bitmap dstBitmap = BitmapFactory.decodeFile(getApplication().getFilesDir().getPath()+"/" + dstPath);
        binding.image2.setImageBitmap(dstBitmap);
        logMemory();
    }


}