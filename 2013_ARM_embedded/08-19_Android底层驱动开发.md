- 2013 - 08 - 19（Android底层驱动开发）
***

# <span style="color:#ff0000;">1). 概述： 
    ```python
    Android应用开发：eclipse + ADT + JDK ---> APK + AVD ----> 运行
    Android低层：linux + gcc + JDK ----> 系统(u-boot / 内核/ 文件系统源码(Android版本))
    binder(IPC) / 调试系统(log系统) / 电源管理(wakelock(在系统休眠时阻止获得该锁的进程被挂起)) / (low memory killer)----> android特有驱动
    git是一个版本控制系统，以仓库的方式管理不同的版本：
            eg: git clone git://www.aleph1.co.uk/yaffs2 //从一个仓库中拷贝一份代码
    ```
***

# <span style="color:#ff0000;">2). Android编译步骤： 
    ```python
    source ./build/envsetup.sh
    choosecombo
    make -j2
    ```
***

# <span style="color:#ff0000;">3). JNI编程： 
    ```python
    Java 类中任何方法和属性对 JNI 都是可见的，不管它是 public 的，还是private/protected
    ```
    ```python
    
