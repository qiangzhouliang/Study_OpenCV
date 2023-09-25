package com.swan.opencv2;

/**
 * @ClassName Mat
 * @Description Mat.java -> Mat.cpp
 * @Author swan
 * @Date 2023/9/22 17:52
 **/
public class Mat {
    private int rows;
    private int cols;
    private CVType type;
    public final long mNativePtr;

    public Mat(int rows, int cols, CVType typ) {
        this.rows = rows;
        this.cols = cols;
        this.type = typ;

        // 创建 Mat.cpp 对象，关键下次要能操作
        mNativePtr = nMatIII(rows, cols, type.value);
    }

    public Mat() {
        // 创建 Mat.cpp 对象，关键下次要能操作
        mNativePtr = nMat();
    }

    // 同一个方法参数不一样要区分
    private native long nMat();

    private native long nMatIII(int rows, int cols, int value);

    public void put(int row, int col, float value) {
        if (type != CVType.CV_32FC1){
            throw new UnsupportedOperationException("value 值不支持，请检查 CVType.value");
        }
        nputF(mNativePtr, row, col, value);
    }

    private native void nputF(long mNativePtr, int row, int col, float value) ;
}
