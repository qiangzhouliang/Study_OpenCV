//
// Created by swan on 2023/9/21.
//
#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include "BitmapUtil.h"

/// 创建一个新的bitmap
jobject BitmapUtil::getBitMap(JNIEnv *env, int width, int height, int type) {
    // 根据 type 来获取 config
    char *config_name;
    if (type == CV_8UC4){ // ARGB_8888
        config_name = "ARGB_8888";
    }

    jstring configName = env->NewStringUTF(config_name);
    jclass bitmapConfigClass = env->FindClass("android/graphics/Bitmap$Config");
    jmethodID valueOfBitmapConfigFunction =
        env->GetStaticMethodID(bitmapConfigClass, "valueOf",
                                                                   "(Ljava/lang/Class;Ljava/lang/String;)Ljava/lang/Enum;");
    jobject bitmapConfig = env->CallStaticObjectMethod(bitmapConfigClass, valueOfBitmapConfigFunction,bitmapConfigClass, configName);

    // Bitmap newBitmap = Bitmap.createBitmap(int width,int height,Bitmap.Config config);
    jclass bitmap = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapFunction = env->GetStaticMethodID(bitmap, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jobject newBitmap = env->CallStaticObjectMethod(bitmap, createBitmapFunction, width, height, bitmapConfig);
    return newBitmap;
}
