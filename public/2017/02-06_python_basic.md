# ___2017 - 02 - 06 Python Basic___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2017 - 02 - 06 Python Basic___](#2017-02-06-python-basic)
  - [目录](#目录)
  - [Q / A](#q-a)
  - [Python2 to Python3](#python2-to-python3)
  - [python 程序的执行方式](#python-程序的执行方式)
  - [print 格式化输出](#print-格式化输出)
  - [数据类型与运算符](#数据类型与运算符)
  	- [数字常量](#数字常量)
  	- [布尔常量 True False](#布尔常量-true-false)
  	- [字符串常量](#字符串常量)
  	- [算数运算符](#算数运算符)
  	- [比较运算符与逻辑运算符](#比较运算符与逻辑运算符)
  - [if for while](#if-for-while)
  - [函数](#函数)
  	- [定义](#定义)
  	- [python 中函数的参数形式](#python-中函数的参数形式)
  	- [python 内建函数](#python-内建函数)
  	- [global](#global)
  	- [默认参数值](#默认参数值)
  	- [指定关键字参数](#指定关键字参数)
  	- [docstrings 文档字符串](#docstrings-文档字符串)
  	- [在函数中接收元组和列表](#在函数中接收元组和列表)
  	- [在函数中接收字典参数](#在函数中接收字典参数)
  	- [位置参数与关键字参数截断](#位置参数与关键字参数截断)
  	- [函数使用元组与字典返回多个参数](#函数使用元组与字典返回多个参数)
  	- [传递函数](#传递函数)
  	- [内嵌函数](#内嵌函数)
  - [模块](#模块)
  	- [sys 模块](#sys-模块)
  	- [字节编译的 .pyc 文件](#字节编译的-pyc-文件)
  	- [from MODULE import SUBMODULE](#from-module-import-submodule)
  	- [模块名称](#模块名称)
  	- [dir](#dir)
  - [序列](#序列)
  	- [Python 的内建数据结构](#python-的内建数据结构)
  	- [序列的切片与索引与 in 判断 (切片 / 索引 / in / not in / 转换工厂函数 / 内建函数)](#序列的切片与索引与-in-判断-切片-索引-in-not-in-转换工厂函数-内建函数)
  	- [序列的 in 判断与运算符](#序列的-in-判断与运算符)
  	- [序列的类型转换与内建函数](#序列的类型转换与内建函数)
  - [列表 list](#列表-list)
  	- [列表定义](#列表定义)
  	- [列表方法](#列表方法)
  	- [示例](#示例)
  	- [列表复制](#列表复制)
  	- [列表解析与列表生成器](#列表解析与列表生成器)
  	- [map 与 filter](#map-与-filter)
  	- [sorted 与 min 的 key 参数](#sorted-与-min-的-key-参数)
  - [元组 tuple](#元组-tuple)
  	- [元组定义](#元组定义)
  	- [元组操作](#元组操作)
  	- [元组的作用](#元组的作用)
  - [字典 dict](#字典-dict)
  	- [字典定义](#字典定义)
  	- [创建字典](#创建字典)
  	- [索引删除遍历](#索引删除遍历)
  	- [字典的格式化字符串](#字典的格式化字符串)
  	- [字典的方法](#字典的方法)
  	- [字典值排序](#字典值排序)
  - [集合 set](#集合-set)
  	- [集合定义](#集合定义)
  	- [集合比较](#集合比较)
  	- [集合关系运算](#集合关系运算)
  	- [集合的方法](#集合的方法)
  	- [集合方法用于字典计算](#集合方法用于字典计算)
  - [字符串](#字符串)
  	- [字符串对象方法](#字符串对象方法)
  	- [字符串方法](#字符串方法)
  	- [示例](#示例)
  	- [二进制与常规字符串转化](#二进制与常规字符串转化)
  	- [模块与字符串示例程序](#模块与字符串示例程序)
  - [正则表达式 re 模块](#正则表达式-re-模块)
  - [面向对象的编程](#面向对象的编程)
  	- [类](#类)
  	- [继承](#继承)
  	- [重载](#重载)
  	- [类中的 import](#类中的-import)
  	- [前置双下划线避免子类重写](#前置双下划线避免子类重写)
  - [文件](#文件)
  	- [open 与 close](#open-与-close)
  	- [读文件](#读文件)
  	- [写文件](#写文件)
  	- [二进制储存器](#二进制储存器)
  - [异常](#异常)
  - [Python 标准库](#python-标准库)
  	- [sys](#sys)
  	- [os](#os)
  	- [inspect](#inspect)
  - [其他语句](#其他语句)
  	- [lambda 匿名函数](#lambda-匿名函数)
  	- [exec 和 eval 执行语句或表达式](#exec-和-eval-执行语句或表达式)
  	- [assert 断言](#assert-断言)
  	- [repr 规范字符串表示](#repr-规范字符串表示)
  	- [range 列表生成器](#range-列表生成器)
  	- [enumerate 带指数的列表](#enumerate-带指数的列表)
  	- [format 格式化](#format-格式化)
  	- [Iterator 与 Generators 与 Yield](#iterator-与-generators-与-yield)
  - [Python 38](#python-38)
  - [Python 环境](#python-环境)
  	- [Virtualenv](#virtualenv)
  	- [Python 源码编译](#python-源码编译)
  	- [获取当前 python 的 site-packages](#获取当前-python-的-site-packages)

  <!-- /TOC -->
***

# Q / A
  - Python参考教程
    ```python
    (1) Magnus Lie Hetland,Beginning Python: from Novice to Professional, 2nd edition, Apress.（第二版中译版名为《Python基础教程》）
    (2) Wesley Chun, Core Python Applications Programming, Prentice Hall.（第二版中译版名为《Python核心编程》）
    (3) SciPy科学计算：http://www.scipy.org/
    (4) Wes McKinney, Python for Data Analysis. 东南大学出版社. （英文影印本，中译版名为《利用Python进行数据分析》）
    ```
  - 基本
    - $ python -V # 显示版本
    - $ python3
    - help(str) 显示str帮助信息
    - help('print') print帮助信息
    - print('Area is'， length * width) # 输出会在is后面自动添加空格 Area is 10
    - python中没有 ++ / -- 运算符
    - 获取变量类型： type() / isinstance(var, type)
    - python下载模块： easy_install-3.6 wx
  - Q： IndentationError: unexpected indent
    ```python
    $ python hello.py
    File "hello.py", line 18
     print('Value is', i) # Error! Notice a single space at the start of the line
    ^
    IndentationError: unexpected indent
    ```
    A： 检查行首是否有多余的空格，Python中行首的空白是重要的，在逻辑行首的空白(空格和制表符)用来决定逻辑行的缩进层次，从而用来决定语句的分组
  - Q: Encoding error while reading a file
    ```python
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xae in position 199: invalid start byte
    ```
    A: 非'utf-8'编码的字符报错，可以捕获 / 忽略该错误
    ```python
    f = open(path + 'foo', errors='ignore')
    ```
  - Q: 运行python文件，报错
    ```sh
    /usr/bin/env: "python\r": 没有那个文件或目录
    ```
    A: 可能创建在windows下，多了\r字符
    ```sh
    :set ff # 文件显示为dos格式
    :set ff=unix # 设置为unix格式
    :wq
    ```
  - Q: Windows 安装 anaconda 启动 ipython 报错
    ```md
    ImportError: DLL load failed while importing _sqlite3: 找不到指定的模块
    ```
    A: 首先安装 `pip install pysqlite3`，然后从 [SQLite Download Page](https://www.sqlite.org/download.html) 下载 `sqlite-dll-win64-x64-3390000.zip`，解压到 `{Anaconda 安装目录}/DLLS`
  - Q: `import numpy as np` 报错
    ```md
    Original error was: DLL load failed while importing _multiarray_umath: 找不到指定的模块。
    ```
    A: 添加环境变量
    ```
    export PATH="/d/ProgramData/Anaconda3:/d/ProgramData/Anaconda3/Scripts:/d/ProgramData/Anaconda3/Library/bin:$PATH"  # Window anaconda
    export PYTHONPATH="$PYTHONPATH:/d/ProgramData/Anaconda3/Lib:/d/ProgramData/Anaconda3/Library"  
    ```
  - Q: Python download SSL sertificate error
    ```sh
    [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1007)
    ```
    A: `request.get` 中添加 `verify=False`，或通过 ssl 关闭检验
    ```py
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    ```
***

# Python2 to Python3
  - Python3 中不再支持 **raw_input**，使用 **input**
  - Python2 中 **print** 不需要加()，python3 中需要使用print( ... )
    ```python
    print 'Hello World!'
         ^
    SyntaxError: Missing parentheses in call to 'print'
    ```
    python3 中使用
    ```python
    print('Hello World!')
    ```
  - Python3 print 语法 **end = ' '**
    ```python
    $ python2 hello.py
    File "hello.py", line 5
    print(5, end = ' ') # assign end character as ' ', instead of '\n'
        ^
    SyntaxError: invalid syntax
    ```
    Python2 不支持该语法，使用 Python3 运行
    ```python
    python3 hello.py
    ```
  - Python2 中的 **commands** 模块
    ```python
    ImportError: No module named 'commands'
    ```
    在 Python3 中不再使用 commands，而改用 subprocess 模块，使用 subprocess 替换 commands
  - Python2 中的 **file** 类
    ```python
    name 'file' is not defined    
    ```
    Python3 中不再使用 file 类，使用 open 打开文件
  - Python2 中 **dict** 的键值类型
    ```python
    TypeError: 'dict_keys' object does not support indexing
    ```
    Python3 中改变了dict.keys，返回的是 **dict_keys 对象**，支持 **iterable** 但不支持 **indexable**，可以将其明确的转化成 list
    ```python
    list(dict.keys)
    list(dict.values)
    ```
  - Python2 中的 **xrange**
    ```python
    NameError: name 'xrange' is not defined
    ```
    xrange() was renamed to **range()** in Python 3
***

# python 程序的执行方式
  - python能够轻松的集成C,C++,Fortran代码（Cython项目），可以同时用于研究和原型的构建以及生产系统的构建
  - 因为python是一种解释型语言，运行速度比编译型数据慢
  - 由于python有一个全局解释器锁（GIL）,防止解释器同时执行多条python字节码，所以python不适用于高并发、多线程的应用程序
  - Cython项目可以集成OpenMP（一个用于并行计算的C框架）以实现并行处理循环进而大幅度提高数值算法的速度
  - 交互式命令行
    ```python
    $ python3
    print("hello world")
    hello world
    ```
  - 文件
    ```python
    $ cat hello.py
    #!/usr/bin/python3
    #Filename: hello.py

    print('Hello World!')

    $ python3 hello.py
    Hello world

    $chmod a+x hello.py
    $ ./hello.py
    Hello world
    ```
  - 首行 #!/usr/bin/python3
    ```python
    Python至少应当有第一行那样的特殊形式的注释，它被称作组织行，源文件的头两个字符是#!，后面跟着一个程序
    这行告诉Linux/Unix系统当执行程序时，应该运行哪个解释器
    ```
***

# print 格式化输出
  - print 原型 print(* objects, sep=' ', end='\n', file=sys.stdout, flush=False)
    ```python
    sys.stdout.write('Hello World')        # 直接使用sys.stdout.write输出
    sys.stdout.flush()        # 刷新缓冲区
    ```
  - print格式化输出：
    ```python
    print('%s' %('example'))
    e = 'example'
    print('%(e)s' %vars())
    example

    strHello = "the length of (%s) is %d" %('Hello World',len('Hello World'))
    print(strHello)
    the length of (Hello World) is 11

    print(x, end="") # python3中结尾不使用换行符
    ```
  - 整数的进制：
    ```python
    %x：hex 十六进制
    %d：dec 十进制
    %o：oct 八进制

    nHex = 0x20
    print("nHex = %x, nDec = %d, nOct = %o" %(nHex, nHex, nHex))
    nHex = 20, nDec = 32, nOct = 40
    ```
  - 格式化输出浮点数
    ```python
    输出10位宽度，3位精度，左对齐
    import math
    print("PI = %-10.3f%-10.3f" %(math.pi, math.pi))
    PI = 3.142   3.142
    ```
  - **%f, %g, %G, %e, %E** 格式化输出，`%g / %G` 根据值的大小采用 `%e` 或 `%f`
    ```py
    print(["%f" % ii for ii in [1, 0.1, 0.2, 0.01, 0.0001, 0.000001]])
    # ['1.000000', '0.100000', '0.200000', '0.010000', '0.000100', '0.000001']
    print(["%g" % ii for ii in [1, 0.1, 0.2, 0.01, 0.0001, 0.000001]])
    # ['1', '0.1', '0.2', '0.01', '0.0001', '1e-06']
    print(["%G" % ii for ii in [1, 0.1, 0.2, 0.01, 0.0001, 0.000001]])
    # ['1', '0.1', '0.2', '0.01', '0.0001', '1E-06']
    print(["%e" % ii for ii in [1, 0.1, 0.2, 0.01, 0.0001, 0.000001]])
    # ['1.000000e+00', '1.000000e-01', '2.000000e-01', '1.000000e-02', '1.000000e-04', '1.000000e-06']
    print(["%.1E" % ii for ii in [1, 0.1, 0.2, 0.01, 0.0001, 0.000001]])
    # ['1.0E+00', '1.0E-01', '2.0E-01', '1.0E-02', '1.0E-04', '1.0E-06']
    ```
  - 格式化输出字符串
    ```python
    输出10位宽度，4位精度，左 / 右对齐
    print('%-10.4s%*.*s' % ('hello', 10, 4, 'world'))
    hell      worl
    ```
***

# 数据类型与运算符
  - del 删除 一个变量/名称
  - i = 5 # 不需要声明或定义数据类型
  - print(i+1)
## 数字常量
  - 整数： 2
    ```python
    type(3)
    <class 'int'>
    ```
    长整数： 大一些的整数，python3中不再有长整型
  - 浮点数： 3.23和52.3E-4，E标记表示10的幂，52.3E-4表示52.3 * 10 -4
    ```python
    type(1.)
    <class 'float'>
    ```
  - 复数： (-5+4j)和(2.3-4.6j)，real / imag 方法分别得到实部与虚部，conjugate得到共轭复数
    ```python
    type(-5+4j)
    <class 'complex'>
    x.real
    2.4
    x.imag
    5.6
    x.conjugate()
    (2.4-5.6j)
    ```
## 布尔常量 True False
  - 仅有2个值:True、False
    ```python
    type(True)
    <class 'bool'>
    x = True
    int(x)
    1
    ```
## 字符串常量
  - 可以使用 ' ' 或 " "
  - 使用 ''' ''' 或 """ """ ： 可以指定多行的字符串，或使用 '' / ""
  - 转义符 / 续行符： \
  - Python中没有专门的char数据类型
    ```python
    type("a")
    <class 'str'>
    type('a')
    <class 'str'>
    ```
  - 自然字符串： 字符串加上前缀r或R，不需要如转义符那样的特别处理的字符串，如路径名等
    ```python
    print(r"Newlines are indicated by \n")
    要用自然字符串处理正则表达式，否则会需要使用很多的反斜杠，如后向引用符可以写成 '\\1' 或 r'\1'
    ```
  - Unicode字符串： 前缀u或U
    ```python
    print(u"This is a Unicode string.")
    ```
  - 二进制编码：前缀b
## 算数运算符
  - **+** 加可用于字符串 / 列表 / 元组： 'a' + 'b' = 'ab'
  - **\*** 乘可用于字符串 / 列表 / 元组： 'a' *3 = 'aaa'
  - **\*\*** 幂运算： 3 ** 4 = 81，幂运算优先级高于负号-
    ```py
    -3 ** 2 = -9
    (-3) ** 2 = 9
    ```
  - **//** 商的整数部分： 4 // 3.0 = 1.0
## 比较运算符与逻辑运算符
  - 比较运算符可以连接
    ```python
    3 < 4 < 7 # same as (2 < 4) && (4 < 7)
    True
    3 < 4 < 2
    False
    ```
  - 逻辑运算符：
    ```python
    not / and / or / is / is not
    x, y = 1, 2
    not(x is y)
    True
    (x < 3) or (y > 3)
    True
    ```
***

# if for while
  - if示例：
    ```python
    #!/usr/bin/python
    # Filename: if.py
    number = 23
    guess = int(input('Enter an integer : '))        # 显示字符串并获得输入，然后转化为int型
    if guess == number:        # 不使用{}，缩进指定程序块
      print('Congratulations, you guessed it.')
      print("(but you do not win any prizes!)")
    elif guess < number:
      print('No, it is a little higher than that') # Another block
    else:
      print('No, it is a little lower than that')
    print('Done')
    ```
  - 三目： min = x if x<y else y
    ```python
    x, y = 1, 2
    min = x if x < y else y
    min
    1
    ```
    实现异或运算
    ```python
    xor = lambda a, b: False if (a and b) or (not a and not b) else True
    xor(1 ,2)
    Out[87]: False

    xor(0, 0)
    Out[88]: False

    xor(1, 0)
    Out[89]: True
    ```
  - while示例：
    ```python
    #!/usr/bin/python
    # Filename: while.py
    running = True                # True / False
    while running:
      ...
    else:        # while可以有一个else选项，正常结束循环执行else语句，break结束循环将不执行else语句
      ...
    ```
  - for示例：
    ```python
    可迭代对象，string / list / tuple / dictionary / file

    #!/usr/bin/python
    # Filename: for.py
    for i in range(1, 5):        # or for i in [1, 2, 3 ,4 ]
      print(i)
    else:
      print('The for loop is over')
    ```
    **单行操作** 此时循环内变量外部不可见
    ```py
    aa = [ii + 3 for ii in [1, 3, 4, 5]]
    print(aa)
    # [4, 6, 7, 8]

    print(ii)
    # NameError: name 'ii' is not defined
    ```
  - break / continue：
    ```python
    如果从 for 或 while 循环中终止，任何对应的循环 else 块将不执行
    ```
***

# 函数
## 定义
  - 关键字def，使用缩进指定代码块
    ```python
    def maximum(x, y):
      if x > y:
        return x
      else:
        return y

    print(maximum(2, 3))
    ```
## python 中函数的参数形式
  - 位置或关键字参数
  - 仅位置的参数
  - 可变长位置参数：列表list / 元组tuple，形参中使用*
  - 可变长关键字参数：字典dict，形参中使用**
  - (参数可以设定默认值)
## python 内建函数
  - 如abs / min / sum，不需要import：
    ```python
    dir(__builtins__) # 查看支持的内建函数
    ```
  - 其他数学math库的函数，如floor，需要import math：
    ```python
    round(4.5)
    4
    round(4.6)
    5
    floor(4.5)
    Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
    NameError: name 'floor' is not defined
    import math
    math.floor(4.6)
    4
    math.floor(-35.4)
    -36
    ```
## global
  - python 中的变量默认都是全局变量，可以在函数中直接使用外部变量
    ```py
    def func():
        print(a)

    a = 5
    func()  # 5
    ```
  - 但如果要修改全局变量，则会自动创建同名局部变量，如果此时没有初始化而直接，则会报错
    ```py
    ''' 修改时自动创建局部变量 '''
    def func():
        a = 4
        print(a)

    a = 5
    func()  # 4
    print(a)  # 5
    ```
    使用没有初始化的局部变量会报错 `UnboundLocalError: local variable 'a' referenced before assignment`
    ```py
    def func():
        a += 1
        print(a)

    a = 5
    # UnboundLocalError
    func()
    ```
  - **global** / **nonlocal** 关键字表明变量是在外面的块定义的
    ```python
    def func():
        global a
        print('func a is', a)
        a += 2
        print('func changes local a to', a)

    a = 5
    func()
    print('global a is', a)
    # func a is 5
    # func changes local a to 7
    # global a is 7
    ```
## 默认参数值
  - 有默认值的形参只能位于参数列表后面
    ```python
    def say(message, times = 1):        # 形参列表中使用= 指定默认值
      print(message * times)

    say('hello')
    say('world', 5)
    ```
## 指定关键字参数
  - 实参调用时，可以指定为某些参数赋值
  - 关键字参数后面只能是关键字参数
    ```python
    def func(a, b=5, c=10):
      print('a is', a, 'and b is', b, 'and c is', c)

    func(3, 7)
    func(25, c=24)
    func(c=50, a=100)
    ```
## docstrings 文档字符串
  - 函数的第一个逻辑行的字符串是这个函数的 文档字符串 ，DocStrings也适用于模块和类
  - 文档字符串的惯例是一个多行字符串，首行以大写字母开始，句号结尾，第二行是空行，从第三行开始是详细的描述
  - 使用__doc__调用函数的文档字符串属性
    ```python
    def func():
      '''Prints the maximum of two numbers.

      The two values must be integers.'''

    print(printMax.__doc__)
    ```
  - help()会抓取函数的__doc__属性，然后展示
  - 可以在程序中调用如： help(func)
## 在函数中接收元组和列表
  - * 前缀，所有多余的函数参数都会作为一个元组存储在args中
  - 用于函数获取可变数量的参数
    ```python
    def powersum(power, *args):
      total = 0
      for i in args:
          total += pow(i, power)
      return total

    powersum(2, 3, 4)
    25
    powersum(2, 10)
    100
    ```
## 在函数中接收字典参数
  - 可变长关键字参数
  - ** 前缀，多余的参数则会被认为是一个字典的键/值对
    ```python
    def func(args1, *argst, **argsd):
      print(args1)
      print(argst)
      print(argsd)

    func('Hello,', 'Wangdachui', 'Niuyun', 'Linling', a1=1, a2=2, a3=3)
    Hello,
    ('Wangdachui', 'Niuyun', 'Linling')
    {'a1': 1, 'a2': 2, 'a3': 3}
    ```
## 位置参数与关键字参数截断
  - `/` 指定在此之前的参数只接受位置参数
  - `*` 指定在此之后的参数只接受关键字参数
  - 在 `/` 与 `*` 之间的参数不受影响
  ```py
  def func(name="foo", **kwargs):
      print(name, kwargs)
  func(name='world')
  # world {}
  func('hello', name='world')
  # TypeError: func() got multiple values for argument 'name'

  """ / 截断位置参数 """
  def func(name="foo", /, **kwargs):
      print(name, kwargs)
  func('hello', name='world')
  # hello {'name': 'world'}
  ```
  **sorted 函数定义**：
  ```py
  sorted(iterable, /, *, key=None, reverse=False)
  ```
## 函数使用元组与字典返回多个参数
  - 返回元组
    ```python
    def func():
      return 1, 2, 3

    a, b, c = func()
    a, b, c
    Out[275]: (1, 2, 3)
    ```
  - 返回字典
    ```python
    def f():
      a = 5
      b = 6
      c = 7
      return {'a' : a, 'b' : b, 'c' : c}
    r = f()
    r
    Out[288]: {'a': 5, 'b': 6, 'c': 7}
    ```
  - **\*** 可用于将函数返回的元组值解包
    ```python
    fa = lambda : (1, 3)
    fb = lambda a, b : a + b
    fb(* fa())
    ```
## 传递函数
  - 传递函数，即函数作为参数传递
    ```python
    def addMe2Me(x):
      return (x+x)

    def self(f, y):
      print(f(y))

    self(addMe2Me, 2.2)
    4.4
    ```
## 内嵌函数
  - 内嵌函数
    ```python
    def FuncX(x):
      def FuncY(y):
          return x*y
      return FuncY # --> NOT FuncY()

    i = FuncX(3)
    i(5)
    # Out[278]: 15

    FuncX(3)(5)
    # Out[279]: 15
    ```
***

# 模块
  - 模块是一个包含函数和变量的文件，以.py为扩展名结尾
  - 当一个模块被第一次输入的时候,这个模块的主块将被运行
  - import sys 导入模块
  - import..as语法：取模块别名，可以使用更短的模块名称，还能够通过简单地改变一行就切换到另一个模块
    ```python
    import 1 as p
    # import 2 as p
    ```
## sys 模块
  - sys模块包含了与Python解释器和它的环境有关的函数
  - 当Python执行import sys语句的时候会在sys.path变量中所列目录中寻找sys.py模块，随后执行这个模块主块中的语句，然后这个模块将能够使用
  - 初始化过程仅在第一次 输入模块的时候进行
  - sys.argv变量是一个字符串的列表，包含了命令行参数
  - sys.path包含输入模块的目录名列表，第一个字符串是空的，表示当前目录也是sys.path的一部分
    ```python
    for i in sys.argv:
      print(i)
    print('\n\nPYTHONPATH = ', sys.path, '\n')
    ```
## 字节编译的 .pyc 文件
  - 导入模块时，pyc文件会快得多，因为一部分输入模块所需的处理已经完成了
  - 字节编译的文件是与平台无关的
## from MODULE import SUBMODULE
  - from sys import argv 直接输入argv变量到程序中(使用时可省略sys.)
  - from sys import * 输入所有sys模块使用的名字
  - 如使用了 import pandas as pd，再导入pandas.DataFrame时，应使用：
    ```python
    from pandas import DataFrame
    而不是
    from pd import DataFrame
    ```
  - 应该避免使用from..import而使用import语句，因为这样可以使程序更加易读，也可以避免名称的冲突
## 模块名称
  - 每个Python模块都有它的 `__name__`
  - 被用户单独运行的模块是 `__main__`
    ```python
    $ cat using_name.py
    #!/usr/bin/python
    #Filename: using_name.py

    if __name__ == '__main__':
      print('This program is running by itself, __name__ =', __name__)
    else:
      print('This program is imported from other module, __name__ =', __name__)
    ```
    执行结果
    ```py
    $ python3 using_name.py
    This program is running by itself, __name__ = __main__

    $ python
    Python 2.7.12 (default, Nov 19 2016, 06:48:10)
    [GCC 5.4.0 20160609] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    import using_name
    ('This program is imported from other module, __name__ =', 'using_name')
    ```
## dir
  - dir函数来列出模块定义的标识符，包括函数、类和变量
    ```python
    print(dir())        # 当前模块的符号列表
    print(dir(sys))        # sys模块的符号列表

    $ python3
    Python 3.5.2 (default, Nov 17 2016, 17:05:23)
    [GCC 5.4.0 20160609] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    dir()
    ['__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__']
    a=5
    dir()
    ['__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'a']
    del a
    dir()
    ['__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__']
    >>>
    ```
***

# 序列
## Python 的内建数据结构
  - Python 中内建的数据结构：列表 / 元组 / 字典 / 集合
    ```python
    type([1, 2])
    <class 'list'>

    type((1, 2))
    <class 'tuple'>

    type({1:1, 2:2})
    <class 'dict'>

    type({1, 2, 3, 4, 2, 3, 1})        # 集合中元素是惟一的
    <class 'set'>
    ```
## 序列的切片与索引与 in 判断 (切片 / 索引 / in / not in / 转换工厂函数 / 内建函数)
  - **序列** 列表 / 元组 / 字符串都是序列，序列的两个主要特点是 **索引操作符** 和 **切片操作符**
  - **索引** 从序列中抓取一个特定项目
    ```python
    shoplist = ['apple', 'mango', 'carrot', 'banana']

    print(shoplist[0])  # 第0个元素
    # apple
    print(shoplist[-1]) # 最后一个元素
    # banana
    print(shoplist[-2]) # 倒数第二的元素
    # carrot
    ```
  - **切片** 获取序列的一个切片，即一部分序列，格式 `[start : stop : step]`，包括开始的节点，不包括结束的节点
    ```python
    print(shoplist[1:3])  # 元素1, 2
    # ['mango', 'carrot']
    print(shoplist[2:]) # 编号2以后的元素，包括2
    # ['carrot', 'banana']
    print(shoplist[1:-1]) # 编号1到最后一个元素，包括1，不包括最后一个
    # ['mango', 'carrot']
    print(shoplist[:])  # 所有元素
    # ['apple', 'mango', 'carrot', 'banana']
    ```
  - **切片步长 step**
    ```python
    print(shoplist[::-1]) # 逆序
    # ['banana', 'carrot', 'mango', 'apple']
    print(shoplist[::2])  # 偶数元素，步长为 2，选取 0, 2, 4, ...
    # ['apple', 'carrot']
    print(shoplist[1::2])  # 奇数元素，步长为 2，选取 1, 3, 5, ...
    # ['mango', 'banana']
    print(shoplist[:-3:-1]) # 等价于 shoplist[::-1][:2]
    # ['banana', 'carrot']
    ```
  - **string 类型的切片操作**
    ```python
    name = 'swaroop'
    print('characters 1 to 3 is', name[1:3])
    # characters 1 to 3 is wa
    ```
  - **\*** 解包操作
    ```py
    aa = [1, 2, 3]
    bb = [4, * aa, 5]
    print(bb)
    # [4, 1, 2, 3, 5]
    ```
## 序列的 in 判断与运算符
  - **in / not in** 判断元素是否存在
    ```python
    'x' in 'xyz'  # 判断 s 是否包含 x
    # True

    'x' not in 'abc'
    # True
    ```
  - **运算符** `+` / `*`
    ```py
    print('xyz' * 3)
    # xyzxyzxyz
    print(3 * [1, 2, 3])
    # [1, 2, 3, 1, 2, 3, 1, 2, 3]

    print([1, 2, 3] + ['a', 'b', 'c'])
    # [1, 2, 3, 'a', 'b', 'c']
    ```
## 序列的类型转换与内建函数
  - **转换工厂函数**
    ```python
    list()
    str()
    basestring()
    tuple()
    ```
    ```python
    print(str(['hello', 'world']))
    # ['hello', 'world']
    print(list('hello, world'))
    # ['h', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd']
    print(tuple('hello, world'))
    # ('h', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd')
    ```
  - **内建函数**
    - **enumerate()** 返回一个元组 `(i, value)`，给参数的每一项加上索引
    - **len()** 序列成员数量，`len(member)`
    - **max()** / **min()** 返回序列中最大 / 最小值，`max(member)` / `min(member)`
    - **reversed()** 转置，返回一个列表生成器，`list(reversed(member))`
    - **sorted()** 排序，返回一个新的列表，不修改原序列 `sorted(member)`
    - **sum()** 返回 int 型序列成员总和，`sum(member)`
    - **zip()** 压缩组合，建立两个序列间的对应关系，可用于生成 dict，返回一个列表生成器 `a=[1, 2, 3, 4], b=[5, 6], list(zip(a, b)) --> [(1, 5), (2, 6)]`
***

# 列表 list
## 列表定义
  - 处理一组有序项目的数据结构，包括在方括号中
    ```python
    list = [1, 2, 3]
    dir(list)
    ```
    可以添加、删除或是搜索列表中的项目，是可变的数据类型，可以包含不同类型的数据
    ```python
    bList = [1,2,'a',3.5]
    pList = [(1, 'a'), (2, 'b'), [3, 4, 'c']]
    ```
    列表之间可以执行 = / > / < / == / + / *
## 列表方法
  - 头插入： insert(index, object) index之前插入元素
  - 尾插入： append(object)
  - 遍历： for str in list: ...
  - 索引：[n]，可使用如 -1 指定末尾的元素
  - 索引号：L.index(value, [start, [stop]]) -> integer -- return first index of value 从start开始第一个出现value的位置
  - 删除：L.remove(value) -> None -- remove first occurrence of value 删除第一个出现的value
  - 排序： L.sort(key=None, reverse=False) -> None 可指定排序使用的方法，如key=len，指定reverse = True降序
  - 转置：L.reverse() -- reverse *IN PLACE* 修改原列表
  - 成员计数：L.count(value) -> integer -- return number of occurrences of value 返回成员数量
  - 弹出：L.pop([index]) -> item -- remove and return item at index (default last) 返回并删除元素
  - 扩展：L.extend(iterable) -> None -- extend list by appending elements from the iterable 将容器中的元素添加到列表结尾，类似于使用 +=
## 示例
  - len / in / append / sort / del
    ```python
    #!/usr/bin/python

    shoplist = ['apple', 'mango', 'carrot', 'banana']
    print('len =', len(shoplist))
    print('list =', end=' ')
    for item in shoplist:
      print(item, end=' ')
    print()

    shoplist.append('rice')
    print("shoplist = ", shoplist)

    shoplist.sort()
    print("shoplist = ", shoplist)

    del shoplist[0]
    print("shoplist = ", shoplist)

    new_list = ['meat', shoplist]
    print("new_list = ", new_list, ", len = ", len(new_list))        # --> new_list = ['meat', ['banana', 'carrot', 'mango', 'rice']] , len = 2
    print(“new_list elements: ”new_list[1], new_list[1][2])        # --> ['banana', 'carrot', 'mango', 'rice'] mango
    ```
  - 基本运算
    ```python
    x = (1, 2, 3)
    y = (4, 5, 6)
    z = x + y
    z
    (1, 2, 3, 4, 5, 6)
    z = 3 * x
    z
    (1, 2, 3, 1, 2, 3, 1, 2, 3)
    ```
  - extend / append
    ```python
    week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekend = ['Saturday', 'Sunday']
    week.extend(weekend)        # extend将元素附加在列表后
    week
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    week.append(weekend)        # append将列表附加在列表后
    week
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', ['Saturday', 'Sunday']]
    ```
## 列表复制
  - 当创建一个对象并给它赋一个变量的时候，仅仅是创建一个原对象的引用
    ```python
    shoplist = ['apple', 'mango', 'carrot', 'banana']
    mylist = shoplist        # mylist is just another name pointing to the same object!
    del shoplist[0]
    shoplist
    ['mango', 'carrot', 'banana']
    mylist
    ['mango', 'carrot', 'banana']

    mylist = shoplist[:]        # make a copy by doing a full slice
    shoplist.pop(0)
    'mango'
    shoplist
    ['carrot', 'banana']
    mylist
    ['mango', 'carrot', 'banana']
    ```
## 列表解析与列表生成器
  - 动态创建列表，从一个已有的列表导出一个新的列表
  - 列表生成器只能遍历一次
    ```python
    [ expression for expr in sequence1
    for expr2 in sequence2 ...
    for exprN in sequenceN
    if condition ]
    ```
    对于符合条件codition的元素，使用expression计算后创建列表
    ```python
    listone = [2, 3, 4]
    listtwo = [2*i for i in listone if i > 2]        # 原列表中所有大于2的数都是原来的2倍
    ```
  - 列表解析，返回一个列表，使用[]：
    ```python
    [i+1 for i in range(10) if i%2 == 0]
    [1, 2, 5, 6, 9, 10]
    ```
  - 返回字典
    ```python
    data = [1, 2, 3, 4]
    {'%s' % ii: ii for ii in data}
    ```
  - 列表生成器，返回一个生成器，使用()：
    ```python
    (i+1 for i in range(10) if i % 2 == 0)
    <generator object <genexpr> at 0x7f173064f1a8>        # 返回一个列表生成器，python3中的range()函数的返回值也是一个生成器
    for i in (i+1 for i in range(10) if i % 2 == 0):
        print(i, end=' ')

    # [Out]
    1 2 5 6 9 10
    ```
## map 与 filter
  - map: 根据功能函数一一代入列表中参数
    ```python
    list(map(lambda x : x % 2, range(10)))
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    ```
  - filter: 过滤器，根据功能函数过滤列表
    ```python
    list(filter(lambda x : x % 2, range(10)))
    [1, 3, 5, 7, 9]
    ```
## sorted 与 min 的 key 参数
  - 用于指定比较的依据
  ```python
  l = [[2, 1], [1, 3], [3, 2]]
  sorted(l, key = lambda x : x[1])
  Out[164]: [[2, 1], [3, 2], [1, 3]]

  sorted(l, key = lambda x : x[1], reverse=True)  # 逆序
  Out[165]: [[1, 3], [3, 2], [2, 1]]

  min(l, key = lambda x : x[1])
  Out[166]: [2, 1]
  ```
***

# 元组 tuple
## 元组定义
  - 元组与列表类似，但元素不可修改
  - 通过圆括号中用逗号分割的项目定义，括号可以省略
    ```python
    t1 = (1, 2, 3, 4, 5)
    t1
    (1, 2, 3, 4, 5)
    t2 = 1 , 2, 3, 4, 5
    t2
    (1, 2, 3, 4, 5)
    ```
  - 空的元组： 由一对空的圆括号组成
    ```python
    myempty = ()
    ```
    单个元素的元组： 必须在第一个项目后跟一个逗号，Python才能区分元组和表达式中一个带圆括号的对象
    ```python
    singleton = (2 , )
    ```
## 元组操作
  - 可以使用 += / * = / 切片方法：
    ```python
    t1 = (1, 2, 3, 4, 5)
    t1 *= 2
    t1
    (1, 2, 3, 4, 5, 1, 2, 3, 4, 5)
    t1 += (6, )
    t1
    (1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6)
    ```
  - 元组的插入操作：
    ```python
    元组没有定义insert方法，使用：
    t1 = (1, 2, 3, 4, 5)
    t1 = t1[:2] + (6,) + t1[2:]
    t1
    (1, 2, 6, 3, 4, 5)
    ```
  - 元组的排序操作：
    ```python
    元组没有定义自己的sort()函数，对元组排序可使用：
    t = (1, 3, 4, 2, 5)
    sorted(t)
    [1, 2, 3, 4, 5]
    t
    (1, 3, 4, 2, 5)
    t = tuple(sorted(t))
    t
    (1, 2, 3, 4, 5) # 创建了新的元组
    ```
## 元组的作用
  - 在映射类型中作为键使用
  - 作为函数的可变长度形参
  - 作为很多内建函数的返回值
  - 元组与打印语句
    ```python
    print语句可以使用跟着 % 符号的项目元组的字符串，让输出满足某种特定的格式
    %s 表示字符串或 %d 表示整数，元组必须按照相同的顺序来对应

    age = 22
    name = 'Swoooop'

    print('%s is %d years old' % (name, age)) # 不使用逗号,分隔
    print('Ya! %s is playing python.' % name)
    ```
  - print 打印单个元组内容，应使用 ',' 分隔
    ```python
    tt = (1, 2, 3)
    print('tt = %s' % tt)
    # [Out]: TypeError: not all arguments converted during string formatting

    print('tt = %s' % (tt))
    # [Out]: TypeError: not all arguments converted during string formatting

    print('tt = %s' % (tt,))
    # [Out]: tt = (1, 2, 3)

    print('tt = %s' % str(tt))
    # [Out]: tt = (1, 2, 3)
    ```
  - 元组可以位于左值 (多元赋值)：
    ```python
    x = 1
    y = 2
    x, y = y, x
    x, y
    (2, 1)

    (x, y) = (3, 4)
    x
    3
    y
    4
    ```
  - 用于for循环中：
    ```python
    l = list("hello")
    for (i, j) in enumerate(l): # (i, j)的括号可以省略
    ...   print("%d: %s" %(i, j))
    ...
    0: h
    1: e
    2: l
    3: l
    4: o
    ```
***

# 字典 dict
## 字典定义
  - 键值对的方式存储数据，键必须是唯一的
  - 只能使用不可变的对象(比如字符串)来作为字典的键
  - 键/值对用冒号分割，而各个对用逗号分割，所有这些都包括在花括号中
  - 字典中的键/值对是没有顺序的
  - 字典是dict类的实例/对象
## 创建字典
  - 直接创建：
    ```python
    aInfo = {'Wangdachui': 3000, 'Niuyun':2000, 'Linling':4500, 'Tianqi':8000}
    aInfo
    # {'Wangdachui': 3000, 'Niuyun': 2000, 'Linling': 4500, 'Tianqi': 8000}

    from numpy.random import randn
    data = { i : randn() for i in range(1, 7) }        # range(1, 7) 作为index，值为randn()产生的随机数
    ```
  - 使用dict()方法：
    ```python
    info = [('Wangdachui',3000), ('Niuyun',2000), ('Linling',4500), ('Tianqi',8000)]
    bInfo = dict(info)
    bInfo
    # {'Wangdachui': 3000, 'Niuyun': 2000, 'Linling': 4500, 'Tianqi': 8000}

    cInfo = dict([('Wangdachui',3000), ('Niuyun',2000), ('Linling',4500), ('Tianqi',8000)])
    cInfo
    # {'Wangdachui': 3000, 'Niuyun': 2000, 'Linling': 4500, 'Tianqi': 8000}

    dInfo = dict(Wangdachui=3000, Niuyun=2000, Linling=4500, Tianqi=8000)
    dInfo
    # {'Wangdachui': 3000, 'Niuyun': 2000, 'Linling': 4500, 'Tianqi': 8000}
    ```
  - fromkeys()方法提供默认值，默认是None：
    ```python
    fromkeys(iterable, value=None, /)
    aDict = {}.fromkeys(('Wangdachui', 'Niuyun', 'Linling', 'Tianqi'),3000)
    aDict
    # {'Wangdachui': 3000, 'Niuyun': 3000, 'Linling': 3000, 'Tianqi': 3000}
    ```
  - 使用zip()：
    ```python
    a = tuple('hello')
    a
    # ('h', 'e', 'l', 'l', 'o')
    b = range(5)
    b
    # range(0, 5)

    dict(zip(a, b))
    # {'h': 0, 'e': 1, 'l': 3, 'o': 4}

    dict(zip(tuple('hello'), range(6)))
    # {'h': 0, 'e': 1, 'l': 3, 'o': 4}
    ```
  - for循环：
    ```python
    lf = [('AXP', 'American Express Company', '86.40'), ('BA', 'The Boeing Company', '122.64'),]
    d = {}
    for data in lf:
        d[data[0]] = data[2]

    d
    # {'AXP': '86.40', 'BA': '122.64'}
    ```
## 索引删除遍历
  - 示例：
    ```python
    #!/usr/bin/python

    ab = {
        'Swaroop' : 'swaroopch@byteofpython.info',
        'Larry' : 'larry@wall.org',
        'Matsumoto' : 'matz@ruby-lang.org',
        'Spammer' : 'spammer@hotmail.com'
    }
    print("Swaroop's address is %s" %ab['Swaroop'])

    ab['Guido'] = 'guido@python.org'
    del ab['Spammer']
    print('There are %d contacts in the address book' %len(ab))

    for name, address in ab.items():
      print('Contact %s at %s' %(name, address))

    # for eachkey in ab.keys():
    # for eachvalue in ab.values():
    # for each in ab.items():

    if 'Guido' in ab:        # python3中不能使用 has_key 方法
      print("Guido's address is %s" %ab['Guido'])
    ```
## 字典的格式化字符串
  - print中使用元组形式：
    ```python
    info = dict(zip(tuple('hello'), range(6)))
    info
    {'h': 0, 'e': 1, 'l': 3, 'o': 4}

    for key in info.keys():
    ...   print('char = %s, value = %d' %(key, info[key]))
    ...
    char = h, value = 0
    char = e, value = 1
    char = l, value = 3
    char = o, value = 4
    ```
  - %(key)格式说明符 % 字典对象名
    ```python
    'char h has value %(h)s.' %info
    'char h has value 0.'
    ```
  - 格式化字符串模板：
    ```python
    template = '''
    ... Welcome to python.
    ... h has value %(h)s.
    ... e has value %(e)s.
    ... '''
    print(template %info)

    Welcome to python.
    h has value 0.
    e has value 1.
    ```
## 字典的方法
  - items方法：返回一个元组的列表，其中每个元组都包含一对项目，即键与对应的值
  - fromkeys()方法提供默认值，默认是None：
    ```python
    fromkeys(iterable, value=None, /)
    ```
  - keys() 返回字典所有键
    ```python
    info.keys()
    dict_keys(['h', 'e', 'l', 'o'])
    list(info.keys())
    ['h', 'e', 'l', 'o']
    ```
  - values() 返回字典所有值
    ```python
    info.values()
    dict_values([0, 1, 3, 4])
    ```
  - get() 通过键查找值：
    ```python
    D.get(k[,d]) -> D[k] if k in D, else d. d defaults to None

    对于不存在的键，使用 info['a']将返回错误：
    info['a']
    Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
    KeyError: 'a'

    可使用get()方法，返回None：
    info.get('a')
    print(info.get('a'))
    None

    或指定不存在时get()的返回值：
    info.get('a', 4)
    4
    ```
  - update() 根据另一个列表更新列表，原先存在的键更新为新的值，不存在的键创建新的键值对：
    ```python
    info
    {'h': 0, 'e': 1, 'l': 3, 'o': 4}
    info2 = dict(zip(list('aloha'), range(5,10)))
    info2
    {'a': 9, 'l': 6, 'o': 7, 'h': 8}
    info.update(info2)
    info
    {'h': 8, 'e': 1, 'l': 6, 'o': 7, 'a': 9}
    ```
  - clear() 删除字典，将影响引用该字典的变量
    ```python
    info
    {'h': 8, 'e': 1, 'l': 6, 'o': 7, 'a': 9}
    info3 = info
    info3
    {'h': 8, 'e': 1, 'l': 6, 'o': 7, 'a': 9}
    info.clear()
    info3
    {}
    ```
  - copy() 复制字典，将生成一个新的字典
    ```python
    d = dict(zip(list('hello'), range(1, 6)))
    d1 = d.copy()
    id(d1)
    140077439899616
    id(d)
    140077439899904

    d
    {'h': 4, 'e': 2, 'l': 2, 'o': 3, 'a': 5}
    d1
    {'h': 4, 'e': 2, 'l': 2, 'o': 3, 'a': 5}
    d.popitem()
    ('a', 5)
    d
    {'h': 4, 'e': 2, 'l': 2, 'o': 3}
    d1
    {'h': 4, 'e': 2, 'l': 2, 'o': 3, 'a': 5}
    ```
  - setdefault()：D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D，键k存在时返回对应值，键k不存在时，添加并设置值为d
  - pop() 若键存在，返回对应值并删除，不存在抛出异常KeyError
  - popitem() 弹出键值对，[从末尾？？]字典为空时抛出异常KeyError
## 字典值排序
  ```python
  dd = {1:3, 4:4, 3:2}
  dd  # dict 默认会按照 key 排序
  Out[113]: {1: 3, 3: 2, 4: 4}
  [(ks, dd[ks]) for ks in sorted(dd, key=lambda k : dd[k])]
  Out[116]: [(3, 2), (1, 3), (4, 4)]

  min(dd, key=lambda x : x[1])
  Out[120]: (3, 2)
  ```
***

# 集合 set
## 集合定义
  - 集合中的元素不重复
    ```python
    s1 = set('hello')
    s1
    {'o', 'l', 'e', 'h'}

    l = [1, 2, 3, 4, 5, 5, 3, 1]
    l = list(set(l))
    l
    [1, 2, 3, 4, 5]
    ```
  - 可变集合(set) / 不可变集合(frozenset)
    ```python
    s2 = frozenset('hello')
    s2
    frozenset({'o', 'l', 'e', 'h'})
    type(s2)
    <class 'frozenset'>
    ```
## 集合比较
  - 比较运算符
    ```python
    in / not in：∈，集合是否包含元素
    == / !=：= / ≠，集合是否相同，frozenset / set只要元素相同则相等
    < / <=：⊂ / ⊆，集合是否是另一个集合的子集
    > / >=：⊃ / ⊇，集合是否包含另一个集合
    ```
    ```python
    s1 = set('sunrise')
    s2 = set('sunset')
    'u' in s2
    True
    s1 == s2
    False
    set('sun') < s2
    True
    ```
## 集合关系运算
  - 关系运算符
    ```python
    & / &=：∩，交集
    | / |=：∪，并集
    - / -=：差集
    ^ / ^=：Δ，对称差
    ```
    ```python
    s1 = set('sunrise')
    s2 = set('sunset')
    s1 &amp; s2
    {'s', 'u', 'e', 'n'}
    s1 | s2
    {'e', 'u', 'i', 'r', 'n', 's', 't'}
    s1 - s2
    {'r', 'i'}
    s1 ^ s2
    {'i', 'r', 't'}
    (s1 - (s1 &amp; s2)) | (s2 - (s1 &amp; s2))
    {'r', 'i', 't'}
    ```
## 集合的方法
  - 面向可变 / 不可变集合的方法
    ```python
    issubset(t)
    issuperset(t)
    union(t)
    intersection(t)
    difference(t)
    symmetric_difference(t)
    copy()
    ```
  - 面向可变集合的方法
    ```python
    update(t)
    intersection_update(t)
    difference_update(t)
    symmetric_difference_update(t)
    add(obj)
    remove(obj)        # 若元素不存在，抛出异常KeyError
    discard(obj)        # 若元素不存在，不做处理
    pop()
    clear()
    ```
## 集合方法用于字典计算
  ```py
  aa = dict(zip('abcd', [1, 2, 3, 4]))
  bb = dict(zip('cdef', [1, 2, 3, 4]))
  set(aa).difference(set(bb))
  # {'a', 'b'}

  {kk: aa[kk] for kk in set(aa).difference(set(bb))}
  # {'a': 1, 'b': 2}
  ```
***

# 字符串
  - 字符串是str类的对象
  - help(str)查看帮助信息
  - python3中不再使用cmp()函数
  - 可使用序列方法 / += / * =
## 字符串对象方法
  - lower / upper 转换为小写 / 大写
  - endswith / startswith 判断是否以某个后缀结尾 / 前缀开头
  - replace 用另一个子串替换指定子串
  - split拆分
    ```python
    val = 'a,b, guido'
    val.split(',')
    Out[463]: ['a', 'b', ' guido']
    ```
  - strip / rstrip / lstrip修剪空白符
    ```python
    pieces = [x.strip() for x in val.split(',')]
    pieces
    Out[468]: ['a', 'b', 'guido']
    ```
  - join连接
    ```python
    '::'.join(pieces)
    Out[469]: 'a::b::guido'
    ```
  - in / index / find / rfind查找
    ```python
    'guido' in val
    Out[470]: True

    # 如果找不到字符串，index将会引发一个异常
    val.index(',')
    Out[472]: 1

    val.find(':')
    Out[473]: -1
    ```
  - count返回指定子串的次数
    ```python
    val.count(',')
    Out[474]: 2
    ```
## 字符串方法
  ![](images/Selection_017.jpg)
## 示例
  - startswith / in / find / join / split
    ```python
    #!/usr/bin/python

    name = 'Swaroop'

    if name.startswith('Swa'):
      print('ya, the string starts with "Swa"')

    if 'a' in name:
      print('Yes, it contains the string "a"')

    if name.find('war') != -1:
      print('Yes, it contains the string "war"')

    delimiter = '_*_'
    mylist = ['Brazil', 'Russia', 'India', 'China']
    print(delimiter.join(mylist))        # --> Brazil_*_Russia_*_India_*_China

    str = 'acdhdca'
    if (str == ''.join(reversed(str))):        # 判断回文字符串
      print('Yes')
    else:
      print('No')

    aStr = 'What do you think of this saying "No pain, No gain"?'
    tempStr = aStr.split('\"')[1]        # split返回分割后的字符串列表，此时 tempStr = No pain, No gain
    ```
## 二进制与常规字符串转化
  - 将字符串转化为二进制编码
    ```python
    aa = "hello"
    aa.encode("ascii")
    # Out[29]: b'hello'
    ```
  - 将二进制编码转化为字符串
    ```python
    bb = b"hello"
    bb.decode("ascii")
    # Out[34]: 'hello'

    # str 转化
    str(bb)
    # Out[35]: "b'hello'"

    str(bb)[2:-1]
    Out[36]: 'hello'
    ```
## 模块与字符串示例程序
  - 将指定的文件备份到指定目录下，创建日期为名的文件夹，文件以时间+注释命名
    ```python
    #!/usr/bin/python
    #Filename: backup_ver1.py

    import os
    import time

    # Source file list to be backed up
    source = ['./using_*', './if.py']

    # Target directory stroring backup files
    target_dir = './'

    # backup directory name using date value
    today = target_dir + time.strftime('%Y%m%d')
    # backup file name using time value
    now = time.strftime('%H%M%S')

    # user comment used in file name
    comment = input('Enter a comment -->')

    if len(comment) == 0:
      target = today + os.sep + now + '.tar.gz' # os.sep representing '/' in Linux
    else:
      target = today + os.sep + now + \
          comment.replace(' ', '_') + '.tar.gz'

    if not os.path.exists(today):
      os.mkdir(today)

    tar_command = "tar -cvzf %s %s" %(target, " ".join(source))

    print("tar command = ", tar_command)

    if os.system(tar_command) == 0:
      print('Succesful backup to ', target)
    else:
      print('Failed to backup')
    ```
***

# 正则表达式 re 模块
  - Python内置的re模块负责对字符串应用正则表达式
    ```python
    import re
    ```
    re模块的函数可以分为三个大类：模式匹配、替换以及拆分
  - re.split 拆分
    ```python
    text = "foo  bar\t baz \tqux"
    re.split('\s+', text)
    Out[481]: ['foo', 'bar', 'baz', 'qux']
    ```
  - re.compile 编译regex
    ```python
    调用re方法时，正则表达式会先被编译，然后再在对象上调用其方法
    对许多字符串应用同一条正则表达式时，应先通过re.compile创建regex对象
    re.IGNORECASE的作用是使正则表达式对大小写不敏感
    text = 'Dave dave@google.com\nSteve steve@gmail.com\nRob rob@gmail.com\nRyan ryan@yahoo.com\n'        # 邮件列表
    pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'        # 匹配邮件地址的正则表达式
    regex = re.compile(pattern, flags=re.IGNORECASE)        # 编译正则表达式
    ```
  - findall / finditer 返回匹配regex的所有模式
    ```python
    regex.findall(text)
    Out[491]: ['dave@google.com', 'steve@gmail.com', 'rob@gmail.com', 'ryan@yahoo.com']
    ```
  - search 返回第一个匹配项
    ```python
    m = regex.search(text)
    m
    Out[493]: <_sre.SRE_Match object; span=(5, 20), match='dave@google.com'>

    text[m.start():m.end()]
    Out[497]: 'dave@google.com'
    ```
  - match 只从字符串首部开始查找，不匹配返回None
    ```python
    print(regex.match(text))
    None
    ```
  - sub / subn 将匹配到的模式替换为指定字符串，并返回所得到的新字符串
    ```python
    regex.sub('REDACTED', text)
    Out[505]: 'Dave REDACTED\nSteve REDACTED\nRob REDACTED\nRyan REDACTED\n'
    ```
  - match / groups / groupdict 分组
    ```python
    对于带有()分组的正则表达式，匹配后的字符串会带有分组信息
    pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
    regex = re.compile(pattern, flags=re.IGNORECASE)
    m = regex.match('wesm@bright.net')
    m.groups()
    Out[512]: ('wesm', 'bright', 'net')
    ```
  - 对于带有分组功能的模式，findall会返回一个元组列表
    ```python
    regex.findall(text)
    Out[516]:
    [('dave', 'google', 'com'),
     ('steve', 'gmail', 'com'),
     ('rob', 'gmail', 'com'),
     ('ryan', 'yahoo', 'com')]
    ```
  - sub可以通过诸如\1、\2之类的特殊符号访问各匹配项中的分组
    ```python
    # 此处:不是表示对应关系
    print(regex.sub(r'\tUsername: \1,\tDomain: \2,\tSuffix: \3', text))
    Dave         Username: dave,        Domain: google,        Suffix: com
    Steve         Username: steve,        Domain: gmail,        Suffix: com
    Rob         Username: rob,        Domain: gmail,        Suffix: com
    Ryan         Username: ryan,        Domain: yahoo,        Suffix: com
    ```
  - 更复杂的正则表达式可以实现更多功能
    ```python
    # 为各个匹配分组加上一个名称，得到一个带有分组名称的字典
    regex = re.compile(r"""(?P<username>[A-Z0-9._%+-]+)@(?P<domain>[A-Z0-9.-]+)\.(?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE|re.VERBOSE)
    m = regex.match('wesm@bright.net')
    m.groupdict()
    Out[528]: {'domain': 'bright', 'suffix': 'net', 'username': 'wesm'}
    ```
***

# 面向对象的编程
## 类
  - 定义
    ```python
    #!/usr/bin/python
    class Person:
      pass        # pass语句表示一个空白块
    p = Person()
    print p        # --> <__main__.Person object at 0x7feb4d4ebb70>
    ```
  - self
    ```python
    类的方法与普通的函数只有一个特别的区别，它们必须有一个额外的第一个参数名称，但是在调用这个方法时，不为这个参数赋值，Python会提供这个值
    这个特别的变量指对象本身，按照惯例它的名称是self

    对于类MyClass和实例MyObject，当调用这个对象的方法MyObject.method(arg1, arg2)时，会由Python自动转为MyClass.method(MyObject, arg1, arg2)
    ```
  - python一些特殊的方法
    ```python
    __init__(self,...) 这个方法在新建对象恰好要被返回使用之前被调用，用来做一些对象初始化
    __del__(self) 恰好在对象要被删除之前调用
    __str__(self) 在我们对对象使用print语句或是使用str()的时候调用
    __lt__(self,other) 当使用 小于 运算符(<)的时候调用。类似地,对于所有的运算符 (+,>等等)都有特殊的方法
    __eq__(self, other) 当使用 相等 运算符(==)的时候调用，如果其中一个是子类，默认使用子类的 __eq__ 方法
    __getitem__(self,key) 使用x[key]索引操作符的时候调用
    __len__(self) 对序列对象使用内建的len()函数的时候调用
    __slots__ 定义类只能包含指定的属性
    __add__(self, val2) 重载 + 运算
    ```
  - Python的访问控制符：
    ```python
    Python中所有的类成员(包括数据成员)都是public公共的，所有的方法都是有效的
    以双下划线前缀的数据成员名称，比如__privatevar，Python的名称管理体系会有效地把它作为私有变量
            __var属性会被__classname_var替换,将防止父类与子类中的同名冲突
    这样就有一个惯例，如果某个变量只想在类或对象中使用，就应该以单下划线前缀，而其他的名称都将作为公共的，可以被其他类/对象使用 (与双下划线前缀不同)
            在属性名前使用一个单下划线字符,防止模块的属性用“from mymodule import *”来加载
    ```
  - **定义示例**
    ```python
    class Person:
        '''Represents a person.'''
        population = 0
        def __init__(self, name):
            '''Initializes the person's data.'''
            self.name = name
            print('(Initializing %s)' % self.name)
            # When this person is created, he/she
            # adds to the population
            Person.population += 1
        def __del__(self):
            '''I am dying.'''
            print('%s says bye.' % self.name)
            Person.population -= 1
            if Person.population == 0:
                print('I am the last one.')
            else:
                print('There are still %d people left.' % Person.population)
        def sayHi(self):
            '''Greeting by the person.
            Really, that's all it does.'''
            print('Hi, my name is %s.' % self.name)
        def howMany(self):
            '''Prints the current population.'''
            if Person.population == 1:
                print('I am the only person here.')
            else:
                print('We have %d persons here.' % Person.population)
    ```
  - **__getitem__** 用于重载索引或切片
    ```py
    class Foo:
        def __init__(self, value="hello world"):
            self.value = value

        def __getitem__(self, expr):
            print(expr)
            if isinstance(expr, tuple):
                return [vv[cur_expr] for vv, cur_expr in zip(self.value.split(" "), expr)]
            if isinstance(expr, slice):
                return self.value[expr]
            else:
                return self.value[-expr]

    aa = Foo()
    print(aa[1])  # 1 'd'
    print(aa[:3])  # slice(None, 3, None) 'hel'
    print(aa[:, :2])  # (slice(None, None, None), slice(None, 2, None)) ['hello', 'wo']

    bb = slice(1, 3, 2)  #
    print(bb.start, bb.stop, bb.step)  # 1 3 2
    ```
    **slice** 有两种初始化形式 `slice(stop)` `slice(start, stop[, step])`
  - **__slots__ 定义类只能包含指定的属性**
    ```py
    class Foo:
        __slots__ = ("aa", "bb")
    foo = Foo()
    foo.__dict__ # AttributeError: 'Foo' object has no attribute '__dict__'
    foo.aa = 1
    foo.cc = 2  # AttributeError: 'Foo' object has no attribute 'cc'
    ```
  - **type 定义类** 使用 `type(classname, superclasses, attributedict)` 方式定义类
    ```py
    AA = type("Foo", (object,), {"aa": 11})

    print(AA.__name__)  # Foo
    aa = AA()
    print(aa.aa)  # 11

    # 等价于
    class Foo(object):
        aa = 11
    ```
## 继承
  - 基本类的名称作为一个元组跟在定义类时的类名称之后
  - Python总是首先查找对应类型的方法，然后到基本类中逐个查找
  - 多重继承：在继承元组中列了一个以上的类
  - 继承示例
    ```python
    #!/usr/bin/python
    class SchoolMember:
        '''Represents any school member.'''
        def __init__(self, name, age):
            self.name = name
            self.age = age
            print('Initialized SchoolMember: %s' % self.name)
        def tell(self):
            '''Tell my details.'''
            print('Name: %s, Age: %s' % (self.name, self.age))

    class Teacher(SchoolMember):
        '''Represents a teacher.'''
        def __init__(self, name, age, salary):
            SchoolMember.__init__(self, name, age)
            self.salary = salary
            print('Initialized Teacher: %s' % self.name)
        def tell(self):
            SchoolMember.tell(self)
            print('Salary: %d' % self.salary)

    class MathTeacher(Teacher):
        '''Represents a math teacher.'''
        def __init__(self, name, age, salary):
            super(MathTeacher, self).__init__(name, age, salary)
            self.area = 'Math'
            print('Initialized Math Teacher: %s' % self.name)
        def tell(self):
            SchoolMember.tell(self)
            print('Salary: %d, Area: %s' % (self.salary, self.area))
    ```
    **运行结果**
    ```py
    mm = MathTeacher('aa', 25, 10000)
    # Initialized SchoolMember: aa
    # Initialized Teacher: aa
    # Initialized Math Teacher: aa

    mm.tell()
    # Name: aa, Age: 25
    # Salary: 10000, Area: Math
    ```
## 重载
  ```py
  class Foo:
      def __init__(self, aa):
          self.aa = aa
      def __call__(self, bb):
          return self.test_func(self.aa, bb)
      def test_func(self, aa, bb):
          return aa + bb

  class Goo(Foo):
      def test_func(self, aa, bb):
          return(aa - bb)

  class Koo(Foo):
      def test_func(self, aa, bb):
          return(aa * bb)

  print(Foo(3)(4))  # 7
  print(Goo(3)(4))  # -1
  print(Koo(3)(4))  # 12
  ```
## 类中的 import
  - 使导入的模块仅在类中可用
  ```py
  class AA:
      from math import sqrt
      def __init__(self, aa):
          self.aa = aa
      def __call__(self):
          return self.sqrt(self.aa)
  ```
## 前置双下划线避免子类重写
  ```py
  class Foo:
      def __init__(self):
          self.aa = "foo"
          self._aa = "foo"
          self._aa_ = "foo"
          self.__aa = "foo"
          self.__aa__ = "foo"

  class Goo:
      def __init__(self):
          super().__init__()
          self.aa = "goo"
          self._aa = "goo"
          self._aa_ = "goo"
          self.__aa = "goo"
          self.__aa__ = "goo"

  foo = Foo()
  print(hasattr(foo, '__aa'))
  # False
  print(hasattr(foo, '_Foo__aa'))
  # true

  goo = Goo()
  print(hasattr(goo, '__aa'))
  # False
  print(hasattr(goo, '_Foo__aa'))
  # False
  print(hasattr(goo, '_Goo__aa'))
  # true
  ```
  **前置双下划线的方法**
  ```py
  class AA:
      def __check(self):
          print("AA")
      def show(self):
          self.__check()

  class BB(AA):
      def __check(self):
          print("BB")

  BB().show()  # AA
  ```
***

# 文件
  - 打开文件：python2使用file()，python3使用open()，python3中不再使用file类
  - 读写文件：read、readline或write方法，对文件的读写能力依赖于打开文件时指定的模式
  - 关闭文件：close方法来告诉Python完成了对文件的使用，文件写入后如果不关闭则文件内容会存在于缓冲区中

  - f.read() / f.write() / f.readline() / f.readlines() / f.writelines()        # readline()返回一行字符串，readlines()返回包含所有行的字符串列表
  - f.seek() / f.tell()        # seek(offset, whence=0)指定文件指针偏移
## open 与 close
  - **open** 打开文件
    ```py
    open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
    ```
    - **file** 文件名路径
    - **mode** 打开模式，'r' / 'w' / 'a' / 'b' / 'x'
    - **buffering** 缓存模式，0 表示不使用缓存，1 / 大于1 表示缓存一行或其他缓存区大小，默认为 -1
    - **encoding** 编码模式，[Standard Encodings](https://docs.python.org/3.6/library/codecs.html#standard-encodings)
    - **errors** 编码错误处理方式，默认为 `strict` 抛出异常，可以指定为 `ignore` 忽略错误
  - **使用**
    ```py
    f1 = open(r'~/foo')
    f2 = open(r'd:\outfile.txt', 'w')
    f3 = open('frecord.csv', 'ab', 0)

    with open('./foo', 'r') as ff:
        ...
    ```
## 读文件
  - list / in
    ```python
    f = open('Blowing_in_the_wind.lrc', 'r')
    >>> lines = list(f)
    for l in lines:
    ...   print(l)
    ...

    f = open('Blowing_in_the_wind.lrc', 'r')
    for l in f:
    ...   print(l)
    ...
    ```
  - read / tell
    ```python
    f = open('foo', 'r')
    p1 = f.read(5)        # read可以指定读取的字符串大小
    f.tell()
    5
    p2 = f.read()
    f.tell()
    13
    print(p1)
    hello
    print(p2)
    , world.
    f.close()
    ```
  - readlines
    ```python
    f = open('foo', 'r')
    print(f.readlines())
    ['hello, world.\n', 'hello, world.\n', 'hello, world.\n', 'hello, world.\n']
    f.seek(0)
    0
    print(f.readline())
    hello, world.
    f.close()
    ```
## 写文件
  - write
    ```python
    f = open(r'foo', 'w')
    f.write('hello, world.')
    13
    f.close()
    ```
  - 示例：
    ```python
    #!/usr/bin/python3

    poem = '''Programming is fun
    When the work is done
    if you wanna make your work also fun:
      use Python!
    '''

    f = open('foo', 'w') # open for writing
    f.write(poem)
    f.close()

    f = open('foo') # read mode is assummed bu default
    while True:
      line = f.readline()
      if len(line) == 0:
        break
      print(line, end = '')
    f.close()
    ```
## 二进制储存器
  - Python提供一个标准的模块，称为pickle，可以在一个文件中储存任何Python对象，之后可以把它完整无缺地取出来，称为持久地储存对象
  - python3不再使用cPickle模块
  - pickle存储方式默认是二进制方式，python3中与文件交互需要指定'wb' / 'rb'
    ```python
    #!/usr/bin/python3
    import pickle as p

    shoplistfile = 'foo'
    shoplist = ['apple', 'mango', 'carrot']

    f = open(shoplistfile, 'wb')
    p.dump(shoplist, f) # dump the object to a file
    f.close()

    del shoplist
    f = open(shoplistfile, 'rb')
    storedlist = p.load(f) # Read back from the storage
    print(storedlist)
    ```
***

# 异常
  - **try..except** 语句来处理异常，通常的语句放在 **try** 块中，错误处理语句放在 **except** 块中
  - **else** 没有异常发生的时候 else 块将被执行
  - **finally** 异常发生或不发生都会执行 finally 块
  - **raise** 抛出异常
  - **builtins** 更多系统预定义异常在模块 `builtins`，python 默认已导入 `from builtins import *`
    ```python
    # 查看其他 Error
    import buildins
    builtins.ImportError
    # Out[4]: ImportError

    builtins.IOError
    # Out[5]: OSError
    ```
  - **try / except / else**
    ```python
    #!/usr/bin/python3
    import sys

    try:
        s = input('Enter something --> ')
    except EOFError as e:
        print('\n WHy did you do an EOF on me? error = %s' % (e))
        sys.exit()
    except KeyboardInterrupt as e:
        print('\n Key board interrupt, error = %s' % (str(e)))
    except:
        print('\n Some error occured!')
    else:
        print('Done')
    ```
    ```shell
    $ python3 try_except.py
    Enter something --> <Ctrl + d>
     WHy did you do an EOF on me? error =

    $ python3 try_except.py
    Enter something --> hiya!
    Done
    ```
  - **finally** 异常触发程序退出时，finally 从句仍然被执行
    ```python
    #!/usr/bin/python3
    import time

    try:
      f = open('foo')
      while True:
        line = f.readline()
        if len(line) == 0:
          break
        time.sleep(2)
        print(line, end = '')
    finally:
      f.close()
      print('closed the file')
    ```
    ```shell
    $ python3 finally.py
    ^Cclosed the file
    Traceback (most recent call last):
     File "finally.py", line 11, in <module>
      time.sleep(2)
    KeyboardInterrupt
    ```
  - **自定义异常** 继承自 Error 或 Exception 类，在模块 `builtins` 中
    ```python
    #!/usr/bin/python3

    class ShortInputException(Exception):
        '''A user-defined exception class.'''
        def __init__(self, length, atleast):
            Exception.__init__(self)
            self.length = length
            self.atleast = atleast

    try:
        s= input('Enter somthing --> ')
        if len(s) < 3:
            raise ShortInputException(len(s), 3)
    except EOFError:
        print('\nWhy did you do an EOF on me?')
    except ShortInputException as x:        # python2中使用except ShortInputException, x:
        print('ShortInputException: The input was of length %d, \
          was expecting at least %d' %(x.length, x.atleast))
    else:
        print('No exception was raised.')
    ```
***

# Python 标准库
## sys
  - **sys** 模块包含系统对应的功能，如 `sys.argv` 列表包含命令行参数
    ```python
    import sys
    sys.version
    # '3.5.2 (default, Nov 17 2016, 17:05:23) \n[GCC 5.4.0 20160609]'
    sys.version_info
    # sys.version_info(major=3, minor=5, micro=2, releaselevel='final', serial=0)
    ```
## os
  - 包含普遍的操作系统功能，用于编写与平台无关的代码，如使用 `os.sep` 可以取代操作系统特定的路径分割符
  - **os.path.expanduser** 将用户路径解析成绝对路径
    ```py
    print(os.path.expanduser("~"))
    # /home/leondgarse
    ```
  - **os.environ** 查找系统环境变量
    ```py
    print(os.environ['HOME'])
    # /home/leondgarse
    print(os.getenv('HOME'))
    # /home/leondgarse
    ```
  - **os.path.split()** 返回一个路径的目录名和文件名
    ```py
    os.path.split('/home/leondgarse/workspace/foo.goo')
    ('/home/leondgarse/workspace', 'foo.goo')
    ```
  - **os.getcwd()** 得到当前工作目录，即当前 Python 脚本工作的目录路径
    ```py
    print(os.getcwd())
    # /home/leondgarse/workspace
    ```
    在脚本中执行时，如果是被导入的脚本，获取到的是执行脚本的目录，可以使用 `__file__` 或 `sys.argv[0]`
    ```py
    print('__file__ = %s' % os.path.abspath(__file__))
    # __file__ = /home/leondgarse/practice_code/python_basic/using_sys.py

    print(os.curdir, os.path.abspath('.'), os.getcwd(), sys.argv[0])
    # . /home/leondgarse/workspace /home/leondgarse/workspace /opt/anaconda3/bin/ipython
    ```
  - **os.name** 字符串，指示正在使用的平台，Windows 对应 'nt'，Linux / Unix 对应 'posix'
  - **os.linesep** 字符串，给出当前平台使用的行终止符，Windows 使用 '\r\n'，Linux 使用 '\n'，Mac 使用 '\r'
  - **os.getenv() / os.putenv()** 分别用来读取和设置环境变量
  - **os.listdir()** 返回指定目录下的所有文件和目录名
  - **os.path.isfile() / os.path.isdir()** 分别检验给出的路径是一个文件还是目录
  - **os.path.exists()** 检验给出的路径是否存在
  - **os.path.join()** 使用当前系统的分隔符，将两个字符串合并成一个路径
  - **os.path.abspath()** 返回参数的绝对路径，可以使用类似 './' / '../'
  - **os.rename(current_file_name, new_file_name)** 文件重命名，移动文件
  - **os.mkdir() / os.makedirs()** 创建目录
  - **os.chdir()** 改变目录
  - **os.remove()** 删除一个文件
  - **os.rmdir()** 删除目录
  - **os.system()** 运行 shell 命令
## inspect
  - 查看指定模块的源码文件
    ```py
    import inspect
    print(inspect.getsourcefile(inspect.getsourcefile))
    # /opt/anaconda3/lib/python3.6/inspect.py

    print(inspect.getsource(inspect.getsource))
    ```
  - 查找指定包中所有的函数与类
    ```py
    from inspect import getmembers, isfunction, isclass
    dict([o for o in getmembers(inspect) if isfunction(o[1]) or isclass(o[1])])
    ```
***

# 其他语句
## lambda 匿名函数
  - **lambda** 语句被用来创建新的函数对象，并且在运行时返回它们
  - **lambda** 需要一个参数，后面仅跟单个表达式作为函数体，而表达式的值被这个新建的函数返回
    ```python
    f = lambda x, y: x+y
    f(2, 3)
    # 5
    ```
  - lambda 作为函数返回值
    ```python
    def make_repeater(n):
        return lambda s: s*n

    twice = make_repeater(2)

    print(twice('word'))
    # wordword
    print(twice(5))
    # 10
    ```
## exec 和 eval 执行语句或表达式
  - **exec** 语句用来执行储存在字符串或文件中的 Python 语句
    ```python
    exec 'print "Hello World"'
    Hello World
    ```
  - **eval** 语句用来计算存储在字符串中的有效 Python 表达式
    ```python
    eval('2*3')
    ```
## assert 断言
  - **assert** 语句用来声明某个条件是真的，并且在它非真的时候引发一个错误 `AssertionError`
    ```python
    mylist=['item']
    assert len(mylist) >= 1

    mylist.pop()  # 'item'
    assert len(mylist) >= 1
    # Traceback (most recent call last):
    #  File "<stdin>", line 1, in <module>
    # AssertionError
    ```
## repr 规范字符串表示
  - **repr** 用来取得对象的规范字符串表示，python3 去除使用 `` ` ` ``，全部改用 `repr()`
  - 在大多数时候有 `eval(repr(object)) == object`
  - 用来获取对象的可打印的表示形式，可以通过定义类的 `__repr__` 方法来控制对象在被 `repr` 函数调用的时候返回的内容
    ```python
    i = []
    i.append('item')
    repr(i)
    # "['item']"

    i.append('item')
    repr(i)
    # "['item', 'item']"
    ```
## range 列表生成器
  - 包含起始，不包含结尾
  - python3中不再使用xrange
  - 返回一个列表生成器
    ```python
    range (start, end, step=1)
    range (start, end)
    range (end)        # 起始是0
    ```
    ```python
    for i in range(5, 30, 5):
        print(i, end=' ')

    # 5 10 15 20 25

    sum(range(5, 30, 5))
    # 75
    ```
## enumerate 带指数的列表
  - **enumerate** 为列表元素加上序号，返回一个元组，给参数的每一项加上指数 `(0, seq[0]), (1, seq[1]), (2, seq[2]), ...`
    ```py
    # iterator for index, value of iterable
    enumerate(iterable[, start])
    ```
    ```python
    aList = list('hello')
    for i, v in enumerate(aList):
        print(i, ' : ', v)

    # 0 : h
    # 1 : e
    # 2 : l
    # 3 : l
    # 4 : o
    ```
  - 生成字典
    ```python
    some_list = ['foo', 'bar', 'baz']
    print(dict((v, i) for i, v in enumerate(some_list)))
    # {'foo': 0, 'bar': 1, 'baz': 2}

    print({v: i for i, v in enumerate(some_list)})
    # {'foo': 0, 'bar': 1, 'baz': 2}
    ```
## format 格式化
  - 生成格式化字符串
    ```python
    format(value, format_spec='', /)
    ```
    ```py
    "{0} {1} love of a {2}".format("for", "the", "princess")
    # 'for the love of a princess'

    "{a} {b} love of a {c}".format(a="for", b="the", c="princess")
    # 'for the love of a princess'
    ```
  - 格式化输出
    ```py
    '{:.2f}'.format(3.1415)
    ```
## Iterator 与 Generators 与 Yield
  - **generator** 保存的是算法，真正需要计算出值的时候才会去往下计算出值，是一种惰性计算 lazy evaluation
  - **yield** 类似 return，在函数中使用 yield 关键字，函数就变成了一个 generator，执行到 yield 就会停住，当需要再往下算时才会再往下算
    ```py
    def foo(aa, loop):
        for _ in range(loop):
            aa += 1
            yield aa

    ff = foo(3, 4)
    print(next(ff), next(ff), next(ff), next(ff))
    # 4 5 6 7

    # Exception: StopIteration
    next(ff)

    print(list(foo(3, 4)))
    # [4, 5, 6, 7]
    ```
  - **next** 第二个参数为迭代结束时返回的默认值
    ```py
    aa = (ii for ii in range(2))
    for ii in range(4):
        print(next(aa, -1))
    # 0 1 -1 -1
    ```
  - 列表生成器使用 `()`，返回的是一个 generator
    ```py
    g = (x * x for x in range(10))
    type(g)
    # Out[3]: generator
    ```
  - **Iterable 可迭代对象** 可以直接作用于 for 循环的对象统称为可迭代对象
    ```py
    from collections import Iterable

    print(isinstance([], Iterable)) # True
    print(isinstance({}, Iterable)) # True
    print(isinstance('abc', Iterable))  # True
    print(isinstance((x for x in range(10)), Iterable)) # True
    print(isinstance(100, Iterable))  # False
    ```
  - **Iterator 迭代器** 表示的是一个数据流，Iterator 对象可以被 `next()` 函数调用并不断返回下一个数据，直到没有数据时抛出 `StopIteration` 异常
    - 生成器 generator 都是 Iterator 对象，但 list / dict / str 是 Iterable，不是 Iterator
    - **iter(iterable)** 把 list / dict / str 等 Iterable 变成 Iterator
    ```py
    print(isinstance([], Iterator)) # False
    print(isinstance((ii for ii in range(10)), Iterator)) # True
    print(isinstance(iter([]), Iterator)) # True
    ```
    - **iter(callable, sentinel)** 的另一种使用方式是调用一个函数，直到得到目标值 sentinel
    ```py
    aa = iter([1, 3, 4, 5, 2, 5, 6])
    def foo():
        return next(aa)
    list(iter(foo, 2))
    # [1, 3, 4, 5]
    ```
  - for 循环本质上就是通过不断调用 `next()` 函数实现的
    ```py
    for x in [1, 2, 3, 4, 5]:
        pass
    ```
    等价于
    ```py
    it = iter([1, 2, 3, 4, 5])
    while True:
        try:
            x = next(it)
        except StopIteration:
            break
    ```
  - **itertools 模块** 提供了用于操作迭代对象的函数
    ```py
    # 切片
    islice(iterable[, start], stop[, step])
    # 排列组合
    permutations(iterable[, r])
    ```
    ```py
    import itertools

    def foo(aa):
        while True:
            aa += 1
            yield aa

    ff = foo(3)
    print(list(itertools.islice(ff, 10)))
    # [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    print(list(itertools.islice(ff, 10)))
    # [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    ```
    ```py
    tt = itertools.permutations(range(3), 2)
    print(list(tt))
    # [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    ```
***

# Python 38
  - [Python 3.8 有什么新变化](https://docs.python.org/zh-cn/3.8/whatsnew/3.8.html)
  - **f 字符串支持 = 用于调试 `f'{expr=}'`**，自动记录表达式和调试文档
    ```py
    from datetime import date
    user = 'eric_idle'
    member_since = date(1975, 7, 31)
    print(f'{user = } {member_since = }')
    # user = 'eric_idle' member_since = datetime.date(1975, 7, 31)
    ```
    ```py
    theta = 30
    print(f'{theta=}  {cos(radians(theta))=:.3f}')
    theta=30  cos(radians(theta))=0.866
    ```
  - **`:=` 赋值表达式** 可在表达式内部为变量赋值，在判断的同时为表达式赋值，被称为 “海象运算符”
    ```py
    aa = [1, 2, 3, 4, 5] * 3
    if (n := len(aa)) > 10:
        print(f"List is too long ({n} elements, expected <= 10)")
    # List is too long (15 elements, expected <= 10)
    ```
    ```py
    import re
    discount = 0.0
    advertisement = "42% discount"
    if (mo := re.search(r'(\d+)% discount', advertisement)):
        discount = float(mo.group(1)) / 100.0
    print(f"{discount = }")
    # discount = 0.42
    ```
    ```py
    with open("foo", "r") as ff:
        while (line := ff.readline()) != "":
            print(line)
    ```
    ```py
    names = ["hello", "WORLD", "Abc", "dEf"]
    allowed_names = ["hello", "world"]
    [name_lower.title() for name in names if (name_lower := name.lower()) in allowed_names]
    # ['Hello', 'World']
    ```
  - **仅限位置形参 `/`** 限制某些函数形参必须使用仅限位置而非关键字参数的形式
    ```py
    def ppow(x, y, z=None, /):    
        r = x ** y
        if z is not None:
            r %= z
        return r
    print(f'{ppow(10, 2) = }, {ppow(10, 2, 3) = }')
    # ppow(10, 2) = 100, ppow(10, 2, 3) = 1

    ppow(10, 2, z=3)
    # TypeError: ppow() got some positional-only arguments passed as keyword arguments: 'z'
    ```
***

# Python 环境
## Virtualenv
  - 创建 `python3.6` 虚拟环境
    ```sh
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.6 python3.6-dev

    pip install virtualenv virtualenvwrapper
    source $(which virtualenvwrapper.sh)

    mkdir /opt/virtualenvs
    virtualenv -p /usr/bin/python3.6 /opt/virtualenvs/python36
    source /opt/virtualenvs/python36/bin/activate
    ...

    pip install mxnet==1.5.0 tensorflow==1.13.1
    pip install ipython jedi==0.17.2  # ipython may not compatible with jedi>=0.18
    pipi pandas scikit-image

    deactivate
    ```
## Python 源码编译
  ```sh
  sudo apt install libbz2-dev libssl-dev liblzma-dev libffi-dev libsqlite3-dev
  # sudo yum install bzip2-libs.aarch64 xz-devel.aarch64 openssl-libs.aarch64

  # https://www.python.org/downloads/ 下载特定版本的 Gzipped source tarball
  VERSION=3.10.19
  wget https://www.python.org/ftp/python/${VERSION}/Python-${VERSION}.tgz
  tar zxvf Python-${VERSION}.tgz
  cd Python-${VERSION}
  ./configure --prefix=$HOME/local_bin/python-${VERSION} --enable-shared --enable-loadable-sqlite-extensions --enable-optimizations
  make
  make install

  # 可以添加到环境变量
  # export PATH=$HOME/local_bin/python-3.10.5/bin:$PATH

  # Virtualenv 指定 python 版本
  virtualenv -p /local_bin/python-3.10.5/python3.10 /opt/virtualenvs/python310
  ```
  - 报错 `No module named '_bz2'`，需要先安装 `apt install libbz2-dev`，然后重新 `configure -> make -> make install`，在 `{安装目录}/lib-dynload/` 下生成 `_bz2.xxx.so`
  - 报错 `No module named '_ssl'`，需要先安装 `apt install libssl-dev`，然后重新 `configure -> make -> make install`，在 `{安装目录}/lib-dynload/` 下生成 `_ssl.xxx.so`
  - 报错 `No module named '_lzma'`，需要先安装 `apt install liblzma-dev`，然后重新 `configure -> make -> make install`，在 `{安装目录}/lib-dynload/` 下生成 `_ssl.xxx.so`
  - **configure** 参数
    - **--enable-shared** 指定编译 so 库, 而不是静态库, 针对某些需要动态库的包
    - **--enable-loadable-sqlite-extensions** 指定链接外部 sqlite3 库, 针对报错 `No module named _sqlite3`
## 获取当前 python 的 site-packages
  ```sh
  python -m site
  # sys.path = [
  #     '/home/leondgarse',
  #     '/d/ProgramData/Anaconda3/Lib',
  #     '/d/ProgramData/Anaconda3/Library',
  #     '/usr/lib/python310.zip',
  #     '/usr/lib/python3.10',
  #     '/usr/lib/python3.10/lib-dynload',
  #     '/home/leondgarse/.local/lib/python3.10/site-packages',
  #     '/usr/local/lib/python3.10/dist-packages',
  #     '/usr/lib/python3/dist-packages',
  # ]
  # USER_BASE: '/home/leondgarse/.local' (exists)
  # USER_SITE: '/home/leondgarse/.local/lib/python3.10/site-packages' (exists)
  # ENABLE_USER_SITE: True

  python -c 'import sys; print("\n".join(sys.path))'
  # ...
  # /usr/lib/python3.10
  # /usr/lib/python3.10/lib-dynload
  # ~/.local/lib/python3.10/site-packages
  # /usr/local/lib/python3.10/dist-packages
  # /usr/lib/python3/dist-packages

  python -c 'import sysconfig; print(sysconfig.get_paths())'
  # {'stdlib': '/usr/lib/python3.10',
  #  'platstdlib': '/usr/lib/python3.10',
  #  'purelib': '/usr/local/lib/python3.10/dist-packages',
  #  'platlib': '/usr/local/lib/python3.10/dist-packages',
  #  'include': '/usr/include/python3.10',
  #  'platinclude': '/usr/include/python3.10',
  #  'scripts': '/usr/local/bin',
  #  'data': '/usr/local',
  # }

  python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
  # /usr/local/lib/python3.10/dist-packages
  ```
***
