# ___2013 - 08 - 19 Android 底层驱动开发___
***

# 目录
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [___2013 - 08 - 19 Android 底层驱动开发___](#2013-08-19-android-底层驱动开发)
- [目录](#目录)
- [概述](#概述)
- [Android编译步骤](#android编译步骤)
- [JNI编程](#jni编程)

<!-- /TOC -->
***

# 概述
  - Android 应用开发：eclipse + ADT + JDK ---> APK + AVD ----> 运行
  - Android 低层：linux + gcc + JDK ----> 系统(u-boot / 内核/ 文件系统源码(Android版本))
  - binder(IPC) / 调试系统(log系统) / 电源管理(wakelock(在系统休眠时阻止获得该锁的进程被挂起)) / (low memory killer)----> android特有驱动
  - git是一个版本控制系统，以仓库的方式管理不同的版本
    ```shell
    git clone git://www.aleph1.co.uk/yaffs2 //从一个仓库中拷贝一份代码
    ```
***

# Android编译步骤
  ```shell
  source ./build/envsetup.sh
  choosecombo
  make -j2
  ```
***

# JNI编程
  - Java 类中任何方法和属性对 JNI 都是可见的，不管它是 public 的，还是private/protected
***
