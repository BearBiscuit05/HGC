# HGC

图计算系统计算池实现(2022-2023初)

项目基于ICDE2022开源项目：https://github.com/thoh-testarossa/GX-Plug

此处为整理FPGA端的具体实现，分别在Xilinx和Intel的不同OpenCL环境下进行开发实现，GPU端和CPU端的实现因为当时git用的不熟练，导致目前也无法找到具体的位置。后续有时间，可能考虑将重新优化并实现此部分。

## 实现原理

手写makefile完成FPGA kernel的编译，生成二进制文件。

在主机端(cpp)，调用指定路径的二进制文件，来调用kernel实现。

