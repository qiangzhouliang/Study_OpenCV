#include <android/log.h>

#define TAG "TAG"
// 方法宏定义 __VA_ARGS__：固定写法
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
