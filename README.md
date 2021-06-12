[TOC]

## 构建

> OS Platform：Ubuntu20.04

#### 物体渲染器 

> Nvidia Driver Version：440.44，CUDA Version：10.2，Optix Version：5.1

##### 1. 安装Nvidia显卡驱动和对应版本的CUDA

- 安装Nvidia显卡驱动（本项目必须使用**支持光线追踪**的Nvidia显卡）

- 下载对应版本的CUDA安装文件【[官网地址](https://developer.nvidia.com/cuda-downloads)】
  ```
  wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
  ```

- 运行安装文件，continue - accept - 取消勾选Driver - install

  ```
  sudo sh cuda_11.2.2_460.32.03_linux.run
  ```

  - 注意：对于cuda10.x来说，建议添加参数  `--librarypath`，否则可能无法顺利安装

    ```
    sudo ./cuda_10.2.89_440.33.01_linux.run --librarypath=/usr/local/cuda-10.2
    ```

- 安装完成后，打开文件  `~/.bashrc ` ，文件末尾添加下列语句（配置环境变量）

  ```
  export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  ```

- 运行  `source ~/.bashrc` 使更改生效，CUDA安装成功

##### 2. 安装Optix光线追踪引擎

- 下载对应版本的Optix安装文件【[官网地址](https://developer.nvidia.com/designworks/optix/download)】，运行安装

  ```
  sudo bash NVIDIA-OptiX-SDK-6.5.0-linux64.sh
  ```

##### 3. 编译OptixRenderer

> OptixRenderer **位于项目 `/render` 目录**，项目详细说明可见 `/render/README.md`
>
> OptixRenderer 【[源代码](https://github.com/lzqsd/OptixRenderer)】

- 安装依赖包（注意编译的时候选用gcc-7）

  ```
  sudo apt install libopencv-dev
  sudo apt install libdevil-dev
  sudo apt install cmake
  sudo apt install cmake-curses-gui
  ```

- 按照项目的说明（位于`/render/INSTALL-LINUX.txt`）进行编译，即可成功

#### 光照估计网络 

> 光照估计网络代码在项目中的位置：`/nets`
>
> InverseRenderingNet【[源代码](https://github.com/lzqsd/InverseRenderingOfIndoorScene)】

- 下载预训练模型参数【[地址](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip)】，解压到`/nets/models` 目录下，解压后文件结构如下

  ```
  ├── nets
  │   ├── models
  │   │   ├── checkBs_cascade0_w320_h240
  │   │   ├── checkBs_cascade1_w320_h240
  │   │   ├── check_cascade0_w320_h240
  │   │   ├── check_cascade1_w320_h240
  │   │   ├── check_cascadeIIW0
  │   │   ├── check_cascadeLight0_sg12_offset1
  │   │   ├── check_cascadeLight1_sg12_offset1
  │   │   ├── check_cascadeNYU0
  │   │   └── check_cascadeNYU1
  ```

#### 其他环境配置

- 环境以及package版本的要求见 `requirements.txt`

  ```
  pip install -r requirements.txt
  ```

------



## 运行

##### 1. 启动 Dash Server

```
python app.py
```

##### 2. 在浏览器访问对应端口

```
127.0.0.1:8050 
```

![image-20210612135145460](https://gitee.com/FujiW/pic-bed/raw/master/arlight-init-2021-6-12.png)

------



## 功能

#### 界面介绍

![image-20210612140126622](https://gitee.com/FujiW/pic-bed/raw/master/20210612140126.png)

![20210612160537](https://gitee.com/FujiW/pic-bed/raw/master/20210612161810.png)

#### 操作流程

> 演示视频1：https://bhpan.buaa.edu.cn:443/link/D4CA3F7525173DBAF1B321F0BCF7800C    访问密码：KItL
>
> 演示视频2：https://bhpan.buaa.edu.cn:443/link/12B52053FC8EE512FF8194FA47FE4623    访问密码：K4zP

##### 功能一 插入单个物体

1. 上传图像
2. 选择光照估计精度
3. 点击 “Start Estimate”，开始估计，估计完成后显示 “Estimate Done”
4. 通过各组件设置物体的形状、大小、颜色等参数
5. 选择 “Memory On” 或 “Memory Off” 模式
6. 鼠标点击图像中任一点作为插入位置
7. 点击 “Start Render”，开始渲染，渲染完成后显示 “Render Done”
8. 选择中间窗口中的 “Viewer of Image” 标签页可以观察输出结果
9. 选择中间窗口中的 “Viewer of Chart” 标签页可以查看应用运行过程中各函数的耗时情况

##### 功能二 插入多个物体

1. 上传图像
2. 选择光照估计精度
3. 点击 “Start Estimate”，开始估计，估计完成后显示 “Estimate Done”
4. 通过各组件设置物体的形状、大小、颜色等参数
5. 选择 “Multi Obj” 模式
6. 鼠标点击图像中任一点作为起始点
7. 点击 “Start Render”，显示 “Push Again”
8. 鼠标点击图像中任一点作为终点
9. 点击 “Start Render”，开始渲染，渲染完成后显示 “Render Done”
10. 选择中间窗口中的 “Viewer of Video” 标签页可以观察输出结果
11. 选择中间窗口中的 “Viewer of Chart” 标签页可以查看应用运行过程中各函数的耗时情况







