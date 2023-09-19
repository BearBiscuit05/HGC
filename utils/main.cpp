#include <iostream>
#include "OclUtils.h"
//point device and run kernel with some args
//example : OclUtils.device.runFunction(kernelName,args);
//
/*
 * 平台检测，设备检测
 * 选择平台，选择设备
 * 指明路径，运行程序
 *
 */
int main() {
    OclUtils utils;
    utils.ShowAllDevice();
    return 0;
}
