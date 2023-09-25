package com.swan.opencv2;

/**
 * @ClassName CVType
 * @Description type 类型，value 对应Mat.cpp 的type类型
 * @Author swan
 * @Date 2023/9/23 16:51
 **/
public enum CVType {
    CV_8UC1(0), CV_8UC2(8), CV_8UC4(24), CV_32FC1(5);

    final int value;
    CVType(int value){
        this.value = value;
    }
}
