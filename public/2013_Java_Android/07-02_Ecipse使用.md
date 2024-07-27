# ___2013 - 07 - 02 Ecipse使用___
***

# Eclipse汉化
  - 从 [babel/downloads](http://www.eclipse.org/babel/downloads.php) 下载对应版本的汉化包
  - 选最新版本Babel Language Pack Zips and Update Sites - R0.9.1
  - 选择Language: Chinese (Simplified)下的Eclipse对应文件下载
  - 解压文件
    ```shell
    sudo unzip BabelLanguagePack-eclipse-zh_3.7.0.v20111128043401.zip
    sudo mv ./eclipse/plugins/* /usr/lib/eclipse/plugins/
    sudo mv ./eclipse/features/* /usr/lib/eclipse/features/
    ```
***

# Eclipse安装ADT
  - [Help] --> [Install New Software] --> [add]
  - Location: [http://dl-ssl.google.com/android/eclipse]
  - 选中下方 [Contact all update sites during install to find required software]
  - 重启Eclipse
  - [Window] --> [Preferences] --> [Android]
  - SDK Location: [SDK 路径]
***

# Eclipse连接Android虚拟机
  - 启动虚拟机后按Alt+F1打开终端(Alt+F7退出终端)并获取管理员权限
    ```shell
    netcfg  # 查看网络状态
    dhcpcd eth0 # 为eth0启动DHCP服务
    ifconfig eth0 xxx # 设置IP地址
    ```
  - Ubuntu中打开终端
    ```shell
    adb connect xxx # 连接Android虚拟机
    ```
  - Eclipse中选择【运行配置】【Target】选择合适选项并运行
***

# Eclipse配置交叉编译工具arm-linux-gcc
  - 新建工程后右键单击 → Properties → c/c++ Build → Settings → Manage Configurations
  - 新建ARM项并设置为活动，做如下更改
    ```shell
    GCC Compiler --> Command = /opt/arm-linux-4.4.3/bin/arm-linux-gcc
            All options = -O0 -g3 -Wall -c -fmessage-length=0
    GCC Linker --> Command = /opt/arm-linux-4.4.3/bin/arm-linux-gcc
    GCC Assembler --> Command = /opt/arm-linux-4.4.3/bin/arm-linux-as
    ```
***

# 常用快捷键
## 编程常用
  - 【Ctrl+Alt+/】:文字补全
  - 【Alt+/】:单词补全
  - 【Ctrl + 1】：快速修正功能
  - 【Ctrl + .】：定位下一个有问题的地方
  - 【Ctrl + ,】：定位上一个有问题的地方
  - 【Ctrl+I】:缩进行
  - 【Ctrl + Q】：定位最后编辑的地方
  - 【Ctrl+K】:查找下一个
  - 【Ctrl+Shift+K】:查找上一个
  - 【Ctrl+J】:增量查找
  - 【Ctrl+Shift+F】:格式化
  - 【Ctrl+Shift+M】：导入对应出错的类（一个）
  - 【Ctrl+Shift+M】：导入对应出错的类（全部）
  - 【Ctrl+Shift+P】:跳转至匹配的方括号
  - 【Ctrl+L】:转至行
## 视图切换
  - 【F2】：查看完整的函数帮助信息,并且可以复制粘贴文字
  - 【F3】：找到变量的定义
  - 【Ctrl+F3】:打开结构
  - 【F4】：找到接口方法的具体实现类
  - 【Ctrl+Shift+F6】:上一个编辑器
  - 【Ctrl+ F6】:下一个编辑器
  - 【Ctrl+ Shift+E】:切换至编辑器
  - 【Ctrl+F7】：切换到下一个视图
  - 【Ctrl+Shift+F7】：切换到上一个视图
  - 【Ctrl+F8】：切换到下一个透视图
  - 【Ctrl+Shift+F8】：切换到上一个透视图
  - 【Ctrl+F10】:显示标尺上下文菜单
## 移动选择
  - 【Ctrl+Home】:移至文本开头
  - 【Ctrl+End】:移至文本末尾
  - 【Shift+End】:选择至行末
  - 【Shift+Home】:选择至行首
  - 【Ctrl+左箭头】:上一个词语
  - 【Ctrl+右箭头】:下一个词语
  - 【Ctrl+上箭头】:向上滚动一行
  - 【Ctrl+下箭头】:向下滚动一行
  - 【Ctrl+Shift+左键头】:选择上一个词语
  - 【Ctrl+Shift+右键头】:选择下一个词语
  - 【Ctrl+Shift+向上键】:转到上一个成员
  - 【Ctrl+Shift+向下键】:转到下一个成员
  - 【Alt+向上键】:将行上移
  - 【Alt+向下键】:将行下移
  - 【Alt＋右键头】：切换到前进的下一个视图或者操作
  - 【Alt + 左键头】：切换到后退的下一个视图或者操作
  - 【Ctrl+Alt+向下键】:复制行
  - 【Ctrl+Alt+向上键】:重复行
  - 【Shift+Alt+左箭头】:选择上一个元素
  - 【Shift+Alt+右箭头】:选择下一个元素
  - 【Shift+Alt+向下键】:复原上一个选择
  - 【双击”{“或”}”】：选中对应”}”和”{“之间的代码块
## 编辑常用
  - 【Ctrl+Backspace】:删除上一个词语
  - 【Ctrl+Delete】:删除下一个词语
  - 【Ctrl+Shift+Delete】:删除至行末
  - 【Ctrl+D】:删除行
  - 【Ctrl+Shift+Enter】:在当前行上面插入行
  - 【Shift+Enter】:在当前行下面插入行
  - 【Ctrl+Shift+X】:更改为大写
  - 【Ctrl+Shift+Y】:更改为小写
## 注释
  - 【Ctrl+Shift+/】:添加块注释
  - 【Ctrl+Shift+\】:除去块注释
  - 【Ctrl+/】:添加行注释
  - 【Ctrl+\】:删除行注释
  - 【Alt+Shift+U】:除去出现注释
## 调试类快捷键
  - 【F5】：跟踪到方法中，当程序执行到某方法时，可以按【F5】键跟踪到方法中
  - 【Ctrl+F5】:单步跳入选择的内容
  - 【F6】：单步执行程序
  - 【F7】：执行完方法，返回到调用此方法的后一条语句
  - 【F8】：继续执行，到下一个断点或程序结束
  - 【F11】：调试最后一次执行的程序
  - 【Ctrl+F11】：运行最后一次执行的程序
  - 【Ctrl+R】:运行至行
  - 【Ctrl+Shift+B】：在当前行设置断点或取消设置的断点
  - 【shift+ctrl+i 】：查看变量的值
## 未分类
  - 【Alt+Shift+O】:切换标记出现
  - 【Ctrl+2,R】:快速辅助,在文件中重命名
  - 【Ctrl+2,F】:快速辅助,指定给字段
  - 【Ctrl+2,L】:快速辅助,指定给局部变量
  - 【Ctrl+O】：快速显示 OutLine
  - 【Ctrl+T】：快速显示当前类的继承结构
  - 【Ctrl+Shift+T】查找工作空间（Workspace）构建路径中的可找到Java类文件
  - 【Ctrl+Shift+R】查找Workspace中的所有文件（包括Java文件）
  - 【Ctrl+Shift+G】查找类、方法和属性的引用
  - 【ALT+Shift+W】查找当前文件所在项目中的路径
***
