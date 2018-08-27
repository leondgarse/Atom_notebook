# ___2017 - 03 - 09 Python 数据分析 1___

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2017 - 03 - 09 Python 数据分析___](#2017-03-09-python-数据分析)
  - [目录](#目录)
  - [Q / A](#q-a)
  - [python扩展库 numpy / pandas / matplotlib / ipython / scipy / Biopython简介](#python扩展库-numpy-pandas-matplotlib-ipython-scipy-biopython简介)
  - [ipython](#ipython)
  	- [shell命令和别名](#shell命令和别名)
  	- [? 用于显示信息 / 搜索命名空间](#-用于显示信息-搜索命名空间)
  	- [魔术命令](#魔术命令)
  	- [%run命令](#run命令)
  	- [Qt的富GUI控制台 jupyter qtconsole](#qt的富gui控制台-jupyter-qtconsole)
  	- [matplotlib集成与pylab模式](#matplotlib集成与pylab模式)
  	- [输入和输出变量](#输入和输出变量)
  	- [记录输入和输出](#记录输入和输出)
  	- [调试器的其他使用场景](#调试器的其他使用场景)
  	- [%time / %timeit 测试代码的执行时间](#time-timeit-测试代码的执行时间)
  	- [基本性能分析：%prun和%run -p](#基本性能分析prun和run-p)
  	- [逐行分析函数性能 %lprun](#逐行分析函数性能-lprun)
  	- [ipython 键盘快捷键](#ipython-键盘快捷键)
  	- [IPython HTML Notebook](#ipython-html-notebook)
  	- [IPython 提高代码开发效率的几点提示](#ipython-提高代码开发效率的几点提示)
  	- [IPython 个性化和配置](#ipython-个性化和配置)
  - [NumPy 基本数据结构 (ndarray / random / 视图 / 索引 / take / put / reshape / ravel / flatten / resize / 轴变换transpose / swapaxes)](#numpy-基本数据结构-ndarray-random-视图-索引-take-put-reshape-ravel-flatten-resize-轴变换transpose-swapaxes)
  	- [ndarray：一种多维数组对象 (dtype / shape / array / zeros / ones / eye / diag / empty / arange / astype / linspace)](#ndarray一种多维数组对象-dtype-shape-array-zeros-ones-eye-diag-empty-arange-astype-linspace)
  	- [random模块 （random_sample / rand / randint / uniform / normal / randn / permutation / shuffle / choice)](#random模块-randomsample-rand-randint-uniform-normal-randn-permutation-shuffle-choice)
  	- [数组的视图](#数组的视图)
  	- [高维数组索引](#高维数组索引)
  	- [布尔型索引](#布尔型索引)
  	- [花式索引](#花式索引)
  	- [take / put 花式索引的等价函数](#take-put-花式索引的等价函数)
  	- [重新排列 (reshape / ravel / flatten)](#重新排列-reshape-ravel-flatten)
  	- [resize改变数组大小](#resize改变数组大小)
  	- [数组转置和轴对换 (transpose / T / dot计算内积)](#数组转置和轴对换-transpose-t-dot计算内积)
  	- [高维数组轴变换 (swapaxes)](#高维数组轴变换-swapaxes)
  - [Numpy 数组方法 (ufunc / meshgrid / where / 数学方法 / any / all / sort / unique / concatention / vstack / split / fromfunction / 文件 / 线性代数 / fft / 随机数)](#numpy-数组方法-ufunc-meshgrid-where-数学方法-any-all-sort-unique-concatention-vstack-split-fromfunction-文件-线性代数-fft-随机数)
  	- [通用函数 ufunc (一元 sqt / exp / modf / 二元 add / maximum)](#通用函数-ufunc-一元-sqt-exp-modf-二元-add-maximum)
  	- [np.meshgrid 函数](#npmeshgrid-函数)
  	- [将条件逻辑表述为数组运算 (where)](#将条件逻辑表述为数组运算-where)
  	- [数学和统计方法 (sum / mean / std / var / add / sin / power / sign / cumsum / cumprod / diff)](#数学和统计方法-sum-mean-std-var-add-sin-power-sign-cumsum-cumprod-diff)
  	- [用于布尔型数组的方法 (any / all / alltrue / allclose)](#用于布尔型数组的方法-any-all-alltrue-allclose)
  	- [排序 (sort / sort_values / sort_index )](#排序-sort-sortvalues-sortindex-)
  	- [唯一化以及其他的集合逻辑 (unique / in1d)](#唯一化以及其他的集合逻辑-unique-in1d)
  	- [数组合并与拆分concatentaion / vstack / row_stack / hstack / column_stack / split / hsplit / vsplit / dsplit](#数组合并与拆分concatentaion-vstack-rowstack-hstack-columnstack-split-hsplit-vsplit-dsplit)
  	- [fromfunction在每个坐标点上执行function](#fromfunction在每个坐标点上执行function)
  	- [save / load 存取二进制格式文件](#save-load-存取二进制格式文件)
  	- [savetxt / loadtxt 存取文本文件](#savetxt-loadtxt-存取文本文件)
  	- [线性代数 (dot / linalg / inv / qr / var / det / eig / svd)](#线性代数-dot-linalg-inv-qr-var-det-eig-svd)
  	- [傅立叶变换 fft](#傅立叶变换-fft)
  	- [示例：随机漫步 (randint / where / cumsum / abs / any / argmax)](#示例随机漫步-randint-where-cumsum-abs-any-argmax)
  - [pandas 一维数组对象Series](#pandas-一维数组对象series)
  	- [Series的创建](#series的创建)
  	- [Series的基本运算](#series的基本运算)
  	- [Series与Python字典](#series与python字典)
  	- [Series的数据对齐](#series的数据对齐)
  	- [Series的name属性](#series的name属性)
  - [pandas 表格数据结构DataFrame / Index对象 / 矢量化的字符串函数 / Panel](#pandas-表格数据结构dataframe-index对象-矢量化的字符串函数-panel)
  	- [创建 （zip / index / columns / 嵌套字典 / Series）](#创建-zip-index-columns-嵌套字典-series)
  	- [创建使用日期做index的DataFrame (pd.date_range)](#创建使用日期做index的dataframe-pddaterange)
  	- [数据查看 （index / columns / values / describe / ix / 切片）](#数据查看-index-columns-values-describe-ix-切片)
  	- [数据修改 (set_value)](#数据修改-setvalue)
  	- [数据删除 （del / drop）](#数据删除-del-drop)
  	- [DataFrame的 index / columns 的name属性](#dataframe的-index-columns-的name属性)
  	- [Index 索引对象](#index-索引对象)
  	- [pandas 矢量化的字符串函数 (contains / get / findall / extract)](#pandas-矢量化的字符串函数-contains-get-findall-extract)
  	- [面板数据Panel (三维版的DataFrame)](#面板数据panel-三维版的dataframe)
  - [Series / DataFrame 基本通用功能 (reindex / 索引 / ix / 算数运算 / 函数映射apply / 值计数 / 数据对齐 / 排序 / loc / at / iloc / iat)](#series-dataframe-基本通用功能-reindex-索引-ix-算数运算-函数映射apply-值计数-数据对齐-排序-loc-at-iloc-iat)
  	- [reindex 重新索引 (插值method / ix)](#reindex-重新索引-插值method-ix)
  	- [索引、选取和过滤 (切片 / head / tail / 索引ix / is_unique)](#索引选取和过滤-切片-head-tail-索引ix-isunique)
  	- [汇总和计算描述统计 (ufunc / sum / idmax / describe / 相关系数与协方差)](#汇总和计算描述统计-ufunc-sum-idmax-describe-相关系数与协方差)
  	- [函数映射 (apply / applymap)](#函数映射-apply-applymap)
  	- [唯一值、值计数以及成员资格 (unique / value_counts / isin / apply(pd.value_counts))](#唯一值值计数以及成员资格-unique-valuecounts-isin-applypdvaluecounts)
  	- [数据对齐和处理缺失数据 (isnull / notnull / dropna / fillna / DataFrame和Series之间的运算)](#数据对齐和处理缺失数据-isnull-notnull-dropna-fillna-dataframe和series之间的运算)
  	- [排序 (sort_index / order / by)](#排序-sortindex-order-by)
  	- [排名（ranking）](#排名ranking)
  	- [整数索引 与 loc / at / iloc / iat方法](#整数索引-与-loc-at-iloc-iat方法)
  - [Series / DataFrame 层次化索引 (MultiIndex / swaplevel / sortlevel / 根据级别汇总统计 / set_index / reset_index / stack / unstack / pivot)](#series-dataframe-层次化索引-multiindex-swaplevel-sortlevel-根据级别汇总统计-setindex-resetindex-stack-unstack-pivot)
  	- [层次化索引与MultiIndex](#层次化索引与multiindex)
  	- [重排分级顺序 swaplevel](#重排分级顺序-swaplevel)
  	- [数据排序 sortlevel](#数据排序-sortlevel)
  	- [根据级别汇总统计](#根据级别汇总统计)
  	- [使用DataFrame的列作为索引 (set_index / reset_index)](#使用dataframe的列作为索引-setindex-resetindex)
  	- [stack / unstack 旋转层次化索引的轴，转换Series / DataFrame](#stack-unstack-旋转层次化索引的轴转换series-dataframe)
  	- [stack / unstack操作中的缺失值](#stack-unstack操作中的缺失值)
  	- [DataFrame进行unstack操作](#dataframe进行unstack操作)
  	- [pivot转换方法，使用原有数据创建新的DataFrame](#pivot转换方法使用原有数据创建新的dataframe)
  - [数据存取 （文本文档 / 二进制文件 / HDF5 / Excel / SQL数据库 / MongoDB）](#数据存取-文本文档-二进制文件-hdf5-excel-sql数据库-mongodb)
  	- [pandas读写表格型文件 read_csv / read_table / 缺失值处理 / 逐块读取 / from_csv](#pandas读写表格型文件-readcsv-readtable-缺失值处理-逐块读取-fromcsv)
  	- [手工处理分隔符格式 csv.reader / csv.writer](#手工处理分隔符格式-csvreader-csvwriter)
  	- [二进制文本文件读写 (pickle / save / load)](#二进制文本文件读写-pickle-save-load)
  	- [HDF5格式 (HDFStore)](#hdf5格式-hdfstore)
  	- [读取 (read_excel / to_excel / ExcelFile类)](#读取-readexcel-toexcel-excelfile类)
  	- [SQL数据库 (sqlite3 / read_sql)](#sql数据库-sqlite3-readsql)
  	- [存取MongoDB中的数据](#存取mongodb中的数据)
  - [网络相关数据处理 (json / urllib / request / html / xml)](#网络相关数据处理-json-urllib-request-html-xml)
  	- [json库读取JSON数据](#json库读取json数据)
  	- [Python获取网络数据 (urllib / requests)](#python获取网络数据-urllib-requests)
  	- [lxml.html解析html文件 (Yahoo财经数据处理成DataFrame)](#lxmlhtml解析html文件-yahoo财经数据处理成dataframe)
  	- [lxml.objectify解析XML (地铁资料数据处理成DataFrame)](#lxmlobjectify解析xml-地铁资料数据处理成dataframe)
  - [数据合并 (merge / join / concat / combine_first)](#数据合并-merge-join-concat-combinefirst)
  	- [merge 根据指定的列名 / 索引合并DataFrame重名数据项](#merge-根据指定的列名-索引合并dataframe重名数据项)
  	- [join DataFrame索引上的合并](#join-dataframe索引上的合并)
  	- [concat Series / DataFrame 横轴或纵轴上的数据堆叠](#concat-series-dataframe-横轴或纵轴上的数据堆叠)
  	- [DataFrame分别使用 join / merge / concat](#dataframe分别使用-join-merge-concat)
  	- [combine_first 使用另一个数据集的数据，填补NA值](#combinefirst-使用另一个数据集的数据填补na值)
  - [数据整理 (duplicated / drop_duplicates / map / replace / rename / cut / qcut / 过滤异常值 / 随机采样 / get_dummies)](#数据整理-duplicated-dropduplicates-map-replace-rename-cut-qcut-过滤异常值-随机采样-getdummies)
  	- [duplicated / drop_duplicates 处理重复数据](#duplicated-dropduplicates-处理重复数据)
  	- [map映射](#map映射)
  	- [replace替换](#replace替换)
  	- [Index.map / rename重命名轴索引](#indexmap-rename重命名轴索引)
  	- [cut / qcut离散化和面元划分](#cut-qcut离散化和面元划分)
  	- [数组运算过滤 / 变换异常值](#数组运算过滤-变换异常值)
  	- [通过permutation / randint随机数排列和随机采样](#通过permutation-randint随机数排列和随机采样)
  	- [get_dummies 计算指标/哑变量](#getdummies-计算指标哑变量)
  	- [get_dummies 计算电影数据的各标签分布](#getdummies-计算电影数据的各标签分布)

  <!-- /TOC -->
***
nonzero / where / choose

# Q / A
  - GitHub上的数据文件及相关资料 http://github.com/pydata/pydata-book
  - JSON JavaScript Object Notation 一种常用的Web数据格式
  - XML（Extensible Markup Language） 一种常见的支持分层、嵌套数据以及元数据的结构化数据格式
  - append(other, ignore_index=False, verify_integrity=False) Append rows of `other` to the end of this frame, returning a new object. Columns not in this frame are added as new columns.
  - at_time(time, asof=False) Select values at particular time of day (e.g. 9:30AM)
  - between_time(start_time, end_time, include_start=True, include_end=True) Select values between particular times of the day (e.g., 9:00-9:30 AM)
  - asof(where, subset=None) The last row without any NaN is taken (or the last row without NaN considering only the subset of columns in the case of a DataFrame)
  - Q: TypeError: cannot use a string pattern on a bytes-like object
    ```python
    import urllib.request
    import re
    dStr = urllib.request.urlopen('https://hk.finance.yahoo.com/q/cp?s=%5EIXIC').read()
    m = re.findall('<tr><td class="yfnc_tabledata1"><b>(.*?)</b></td><td class="yfnc_tabledata1">(.*?)</td>.*?<b>(.*?)</b>.*?</tr>', dStr)

    return _compile(pattern, flags).findall(string)
    TypeError: cannot use a string pattern on a bytes-like object

    ```
    A: 传递给findall()之前使用 dStr = dStr.decode() 或 dStr = dStr.strip().decode('utf-8') 解码
    ```python
    在python3中urllib.read()返回bytes对象而非str，语句功能是将dStr转换成str
    对于二进制编码的字符串，输出时是以b开头的
    ```
  - Q: Basemap Error
    ```python
    x, y = m(cat_data.LONGITUDE, cat_data.LATITUDE)
    SystemError: <class 'RuntimeError'> returned a result with an error set
    ```
    A: 新版本中Basemap.quiver不再接受Series参数
    ```python
    type(cat_data.LONGITUDE)
    Out[6]: pandas.core.series.Series
    type(cat_data.LONGITUDE.values)
    Out[7]: numpy.ndarray<br />
    可使用x, y = m(cat_data.LONGITUDE.values, cat_data.LATITUDE.values)
    ```
  - Q: 命令行执行wx程序时，第二次运行报错
    ```python
    wx._core.PyNoAppError: The wx.App object must be created first!
    ```
    A: 运行前执行del app
  - Q：apply returns more result than expect
    ```
    eg: more labels when applying plot()
    ```
    A：当前apply的实现会在首行/列执行两次
    ```
    In the current implementation apply calls func twice on the first column/row
    to decide whether it can take a fast or slow code path.
    This can lead to unexpected behavior if func has side-effects,
    as they will take effect twice for the first column/row.
    ```
  - Q: 环境安装配置 yum
    ```shell
    sudo apt install python3-pip
    ```
    ```shell
    yum install python34.x86_64
    yum install python34-pip.noarch

    alias python='python3'
    alias pip='pip3'

    pip install --upgrade pip
    pip install tensorflow scikit-learn scikit-image ipython

    alias ipython='ipython3'
    alias ipy='ipython'

    yum install gcc.x86_64
    yum install python34-devel.x86_64 # Solve error: Python.h: No such file or directory
    pip install conda
    yum install anaconda.x86_64
    ```
***

# python扩展库 numpy / pandas / matplotlib / ipython / scipy / Biopython简介
  - 基于Python的软件生态圈，开源，主要为数学、科学和工程服务
  - 介绍
    ```python
    NumPy | numpy.org | N-dimensional array for numerical computation
    SciPy | scipy.org | Collection of numerical algorithms and toolboxes, including signal processing and optimization
    MatPlotLib | matplotlib.org | Plotting library for Python                 
    Pandas | pandas.pydata.org | Powerful Python data analysis toolkit        
    Seaborn | stanford.edu/~mwaskom/software/seaborn/ | Statistical data visualization
    Bokeh | bokeh.pydata.org | Interactive web visualization library
    SciKit-Learn | scikit-learn.org/stable | Python modules for machine learning and data mining
    NLTK | nltk.org | Natural language toolkit
    Notebook | jupyter.org | Web-based interactive computational environment combines code execution, rich text, mathematics, plots and rich media
    R essentials | conda.pydata.org/docs/r-with-conda.html | R with 80+ of the most used R packages for data science "conda install -c r r-essentials"
    scikit-learn        数据挖掘 / 机器学习，模型和参数估计的方法，cross-validate是及其学习里面估计参数的重要方法
    seaborn         画统计图
    TeansorFlow        Google出的deep learning的python包
    一般建模分析 / 数据准备 / 训练(参数估计) / 模型检验和比较
    pipeline        将python数据标准化以及训练和模型检验打包成一个流程
    ```
  - SciPy中的数据结构，Python原有数据结构的变化
    ```python
    ndarray(N维数组)
    Series(变长字典)
    DataFrame(数据框)
    ```
  - 命名惯例：
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pylab as pl
    import scipy as sp<br />
    from pandas import Series, DataFrame
    ```
  - NumPy
    ```python
    NumPy（Numerical Python的简称）是Python科学计算的基础包，为Python提供快速的数组处理能力，作为在算法之间传递数据的容器
            快速高效的多维数组对象ndarray
            用于对数组执行元素级计算以及直接对数组执行数学运算的ufunc函数
            用于读写硬盘上基于数组的数据集的工具
            线性代数运算、傅里叶变换，以及随机数生成
            用于将C、C++、Fortran代码集成到Python的工具
    对于数值型数据，NumPy数组在存储和处理数据时要比内置的Python数据结构高效得多
    由低级语言（比如C和Fortran）编写的库可以直接操作NumPy数组中的数据，无需进行任何数据复制工作
    ```
  - pandas
    ```python
    基于 SciPy 和 NumPy
    强大的可扩展数据操作与分析的Python库
    高效处理大数据集的切片等功能
    提供优化库功能读写多种文件格式,如CSV、HDF5

    pandas提供能够快速便捷地处理结构化数据的大量数据结构和函数
    用得最多的pandas对象是DataFrame，它是一个面向列（column-oriented）的二维表结构，且含有行标和列标
    ```
  - matplotlib
    ```python
    matplotlib是最流行的用于绘制数据图表的Python库
    基于NumPy
    二维绘图库,简单快速地生成曲线图、直方图和散点图等形式的图
    常用的pyplot是一个简单提供类似MATLAB接口的模块
    ```
  - IPython
    ```python
    IPython是Python科学计算标准工具集的组成部分，它将其他所有的东西联系到了一起
    是一个增强的Python shell，为交互式和探索式计算提供了一个强健而高效的环境，主要用于交互式数据处理和利用matplotlib对数据进行可视化处理
    ```
  - SciPy
    ```python
    Python中科学计算程序的核心包
    有效计算numpy矩阵,让NumPy和SciPy协同工作
    致力于科学计算中常见问题的各个工具箱,其不同子模块有不同的应用,如插值、积分、优化和图像处理等

    SciPy是一组专门解决科学计算中各种标准问题域的包的集合，主要包括：
    scipy.integrate：数值积分例程和微分方程求解器
    scipy.linalg：扩展了由numpy.linalg提供的线性代数例程和矩阵分解功能
    scipy.optimize：函数优化器（最小化器）以及根查找算法
    scipy.signal：信号处理工具
    scipy.sparse：稀疏矩阵和稀疏线性系统求解器
    scipy.special：SPECFUN（这是一个实现了许多常用数学函数（如伽玛函数）的Fortran库）的包装器
    scipy.stats：标准连续和离散概率分布（如密度函数、采样器、连续分布函数等）、各种统计检验方法，以及更好的描述统计法
    scipy.weave：利用内联C++代码加速数组计算的工具
    NumPy跟SciPy的有机结合完全可以替代MATLAB的计算功能（包括其插件工具箱）
    ```
  - 其他包
    - Biopython计划，一个使用Python开发计算分子生物学工具的国际社团，将生物信息学文件分析成Python可利用的数据结构，处理常用的在线生物信息学数据库代码，提供常用生物信息程序的界面
    - Beautiful Soup包 查找和解析HTML
    - Mrjob 用于在Amazon网络服务上启动MapReduce作业
    - Vote Smart 智能投票项目是一个美国政治数据的数据源，用户能通过REST API获取他们的数据，Sunlight实验室发布了一个资料齐全的Python接口来使用该API
    - Python-Twitter 是一个提供访问Twitter数据接口的模块
    - Universal Feed Parser 是Python中最常用的RSS程序库
    - Yahoo! PlaceFinder API 对给定的地址返回该地址对应的纬度与经度
***

# ipython
  - Tab键自动完成
  - 以下划线开头的方法和属性，需要先输入一个下划线，然后通过tab自动补全
  - %paste可以承载剪贴板中的一切文本，并在shell中以整体形式执行
  - %cpaste在最终执行之前可以先检查一遍
## shell命令和别名
  - IPython可以直接执行shell命令、更改目录、将命令的执行结果保存在Python对象（列表或字符串）中等
  - 以感叹号（!）开头的命令行表示其后的所有内容需要在系统shell中执行，或将shell命令的控制台输出存放到变量中
    ```python
    ip_info = !ifconfig | grep "inet addr"
    ip_info[0].strip()
    ```
    在使用！时，IPython还允许使用当前环境中定义的Python值，只需在变量名前面加上 $
    ```python
    In [12]: foo = 'test*'
    In [13]: !ls $foo
    ```
## ? 用于显示信息 / 搜索命名空间
  - 在变量的前面或后面加上一个问号（?），可以将有关该对象的一些通用信息显示出来
  - 函数或实例方法docstring会显示出来
  - 使用??将显示出该函数的源代码（如果可能的话）
  - ?还可以用于使用通配符搜索IPython命名空间
    ```python
    In [116]: b = []
    In [117]: b?
    Type:    list
    String form: []
    Length:   0
    Docstring:
    list() -> new empty list
    list(iterable) -> new list initialized from iterable's items
    In [130]: np.random.*rand*?
    np.random.mtrand
    np.random.rand
    np.random.randint
    np.random.randn
    np.random.random
    np.random.random_integers
    np.random.random_sample
    ```
## 魔术命令
  - 以百分号%为前缀，类似运行于IPython系统中的命令行程序
  - 大都有一些“命令行选项”，使用？即可查看其选项
    ```python
    %automagic： 魔术命令默认是可以不带百分号使用的，只要没有定义与其同名的变量即可，可以通过%automagic打开或关闭
    %quickref：显示python的快速参考
    %magic：查看IPython系统中所有特殊命令的文档
    %debug：从最新的异常跟踪的底部进入交互式调试器，执行u / d在frame之间切换
    %hist：命令的输入历史
    %pdb：在异常发生后自动进入调试器
    %paste：执行粘贴板中的oython命令
    %cpaste：手工粘贴python代码模式
    %reset：删除interactive命名空间中的全部变量/名称
    %page OBJECT：通过分页器打印输出OBJECT
    %run script.py：执行一个python脚本
    %prun statement：通过cProfile执行statement
    %time statement：报告statement执行时间
    %timeit statement：多次执行statement，并计算系统平均执行时间
    %who / %who_ls / %whos：显示interactive命名空间中定义的变量 / 信息级别 / 冗余度可变[ ??? ]
    %xdel variable：删除variable，并尝试清除其在ipython中对象上的一切引用
    %bookmark：创建目录书签，以在cd命令时使用，会自动持久化
            In [14]: %bookmark db /home/leondgarse/practice_code/python/data_analysis/pydata-book/
            In [15]: cd db
    %doctest_mode : 直接粘贴 python 控制台格式命令，不必去除提示符 >>>
    ```
## %run命令
  - 执行python脚本，之后该文件中所定义的全部变量（还有各种import、函数和全局变量）就可以在当前IPython shell中访问 In [139]:
    ```python
    %run 4.2.2_plot.py
    ```
  - -d 选项：执行代码文件之前打开调试器
    ```python
    b 设置断点
    s 单步执行
    c 继续执行到下一个断点
    n 执行下一行
    a(rgs) 显示当前函数参数
    l 显示代码
    w(here) 当前位置的完整栈跟踪
    ```
  - 如果%run某段脚本或执行某条语句时发生了异常，IPython默认会输出整个调用栈跟踪（traceback），其中还会附上调用栈各点附近的几行代码作为上下文参考
## Qt的富GUI控制台 jupyter qtconsole
  - $ jupyter qtconsole        # 在界面直接绘图的ipython终端
    ```python
    import matplotlib.pyplot as plt
    img = plt.imread('sweetslove_by_dezhimself.jpg')
    plt.imshow(img)
    plt.show()
    ```
## matplotlib集成与pylab模式
  - $ ipython --pylab
    ```python
    NumPy和matplotlib的大部分功能会被引入到最顶层的interactive命名空间以产生一个交互式的计算环境
    将IPython配置为使用所指定的matplotlib GUI后端（Tk、wxPython、PyQt、Mac OS X native、GTK）
    Pylab模式还会向IPython引入一大堆模块和函数以提供一种更接近于MATLAB的界面
    ```
    ```python
    img = plt.imread('sweetslove_by_dezhimself.jpg')
    plt.imshow(img)
    ```
## 输入和输出变量
  - IPython会将输入（输入的那些文本）和输出（返回的对象）的引用保存在一些特殊变量中
  - 最近的两个输出结果分别保存在_（一个下划线）和__（两个下划线）变量中：
    ```python
    In [10]: 2 ** 27
    Out[10]: 134217728

    In [11]: _
    Out[11]: 134217728
    ```
  - 输入的文本被保存在名为_iX的变量中，其中X是输入行的行号
  - 每个输入变量都有一个对应的输出变量_X，如在输入完第27行后，就会产生两个新变量_27（输出变量）和_i27（输入变量）
    ```python
    In [26]: foo = 'bar'

    In [27]: foo
    Out[27]: 'bar'

    In [28]: _i27
    Out[28]: u'foo'

    In [29]: _27
    Out[29]: 'bar'

    ```
    由于输入变量是字符串，因此可以用Python的exec关键字重新执行：
    ```python
    In [30]: exec(_i18)
    ```
## 记录输入和输出
  - %logstart： 记录IPython整个控制台会话，包括输入和输出
  - %logoff、%logon、%logstate以及%logstop
## 调试器的其他使用场景
  - 使用set_trace这个特别的函数（以pdb.set_trace命名），下面这两个方法可能会在你的日常工作中派上用场，也可以直接将其添加到IPython配置中：
    ```python
    def set_trace():
        from IPython.core.debugger import Pdb
        Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

    def debug(f, *args, ** kwargs):
        from IPython.core.debugger import Pdb
        pdb = Pdb(color_scheme='Linux')
        return pdb.runcall(f, *args, ** kwargs)
    ```
  - 第一个函数（set_trace）非常简单，可以将其放在代码中任何希望停下来查看一番的地方（比如发生异常的地方）：
    ```python
    In [7]: run ipython_bug.py
    > ipython_bug.py(16)calling_things()
         15       set_trace()
    ---> 16       throws_an_exception()
         17
    ```
    按下c（或continue）仍然会使代码恢复执行，不受任何影响
  - 另外那个debug函数能够直接在任意函数上使用调试器，如下函数：
    ```python
    def f(x, y, z=1):
        tmp = x + y
        return tmp / z
    ```
    现在想对其进行单步调试，f的正常使用方式应该类似于f(1,2,z=3)，为了能够单步进入f，将f作为第一个参数传给debug，后面按顺序再跟上各个需要传给f的关键字参数：
    ```python
    In [6]: debug(f, 1, 2, z=3)
    > <ipython-input>(2)f()
          1 def f(x, y, z):
    ----> 2     tmp = x + y
          3     return tmp / z
    ipdb>
    ```
## %time / %timeit 测试代码的执行时间
  - %time一次执行一条语句，然后报告总体执行时间(Wall time)
    ```python
    In [42]: strings = ['foo', 'foobar', 'baz', 'qux', 'python', 'Guido Van Rossum'] * 100000
    In [45]: %time method1 = [x for x in strings if x.startswith('foo')]
    CPU times: user 104 ms, sys: 0 ns, total: 104 ms
    Wall time: 101 ms

    In [46]: %time method2 = [x for x in strings if x[:3] == 'foo']
    CPU times: user 80 ms, sys: 0 ns, total: 80 ms
    Wall time: 82.2 ms
    ```
  - %timeit 自动多次执行一条语句，以产生一个平均执行时间
    ```python
    In [47]: %timeit method1 = [x for x in strings if x.startswith('foo')]
    10 loops, best of 3: 96.8 ms per loop

    In [48]: %timeit method2 = [x for x in strings if x[:3] == 'foo']
    10 loops, best of 3: 76.2 ms per loop
    ```
## 基本性能分析：%prun和%run -p
  - 主要的Python性能分析工具是cProfile模块，它不是专为IPython设计的，cProfile在执行一个程序或代码块时，会记录各函数所耗费的时间
    ```python
    $ python3 -m cProfile -s cumulative 4.1.7_merge.py        # -s 指定排序方式
         326332 function calls (319503 primitive calls) in 3.303 seconds
      Ordered by: cumulative time
      ncalls tottime percall cumtime percall filename:lineno(function)
      479/1  0.016  0.000  3.303  3.303 {built-in method builtins.exec}
        1  0.000  0.000  3.303  3.303 4.1.7_merge.py:3(<module>)

    cumtime列：各函数所耗费的总时间
    ```
  - %prun命令和带-p选项的%run
    ```python
    %prun：格式跟cProfile差不多，但分析的是Python语句
    In [53]: %prun -l 3 -s cumulative np.sin(np.arange(0, 2*np.pi, 0.0001))
         4 function calls in 0.002 seconds<br />
      Ordered by: cumulative time
      List reduced from 4 to 3 due to restriction <3><br />
      ncalls tottime percall cumtime percall filename:lineno(function)
        1  0.000  0.000  0.002  0.002 {built-in method builtins.exec}
        1  0.002  0.002  0.002  0.002 <string>:1(<module>)
        1  0.000  0.000  0.000  0.000 {built-in method numpy.core.multiarray.arange}<br />
    %run -p命令执行效果与在命令行中调用类似
    In [57]: %run -p -s cumulative 4.1.7_merge.py
    ```
## 逐行分析函数性能 %lprun
  - 可以使用一个叫做line_profiler的小型库（可以通过PyPI或随便一种包管理工具获取），其中有一个新的魔术函数%lprun，它可以对一个或多个函数进行逐行性能分析
  - 可以修改IPython配置（参考IPython文件）以启用这个扩展，代码如下所示：
    ```python
    # A list of dotted module names of IPython extensions to load.
    c.TerminalIPythonApp.extensions = ['line_profiler']
    ```
  - line_profiler可以通过编程的方式使用（请参阅完整文档），但其最强大的一面却是在IPython中的交互式使用
  - 假设你有一个prof_mod模块，其中有一些用于NumPy数组计算的代码，如下所示：
    ```python
    from numpy.random import randn

    def add_and_sum(x, y):
        added = x + y
        summed = added.sum(axis=1)
        return summed

    def call_function():
        x = randn(1000, 1000)
        y = randn(1000, 1000)
        return add_and_sum(x, y)
    ```
    如果我们想了解add_and_sum函数的性能，%prun会给出如下所示的结果：
    ```python
    In [569]: %run prof_mod

    In [570]: x = randn(3000, 3000)
    In [571]: y = randn(3000, 3000)
    In [572]: %prun add_and_sum(x, y)
            4 function calls in 0.049 seconds
    Ordered by: internal time
    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         1  0.036   0.036  0.046  0.046 prof_mod.py:3(add_and_sum)
         1  0.009   0.009  0.009  0.009 {method 'sum' of 'numpy.ndarray' objects}
         1  0.003   0.003  0.049  0.049 <string>:1(<module>)
         1  0.000   0.000  0.000  0.000 {method 'disable' of '_lsprof.Profiler' objects}
    ```
    这个结果并不能说明什么问题
  - 启用line_profiler这个IPython扩展之后，就会出现一个新的魔术命令%lprun，用法上唯一的区别就是：必须为%lprun指明想要测试哪个或哪些函数
  - %lprun的通用语法为：
    ```python
    %lprun -f func1 -f func2 statement_to_profile
    ```
    在本例中，我们想要测试的是add_and_sum，于是执行：
    ```python
    In [573]: %lprun -f add_and_sum add_and_sum(x, y)
    Timer unit: 1e-06 s
    File: book_scripts/prof_mod.py
    Function: add_and_sum at line 3
    Total time: 0.045936 s
    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
         3                                           def add_and_sum(x, y):
         4      1           36510  36510.0     79.5      added = x + y
         5      1            9425   9425.0     20.5      summed = added.sum(axis=1)
         6      1               1      1.0      0.0      return summed
    ```
    这个结果就容易理解多了
  - 通常，用%prun（cProfile）做“宏观的”性能分析，而用%lprun（line_profiler）做“微观的”性能分析
## ipython 键盘快捷键

  ![](images/ipython_short_key.png)
## IPython HTML Notebook
  - IPython Notebook有一种基于JSON的文档格式.ipynb，使你可以轻松分享代码、输出结果以及图片等内容
  - 目前在各种Python研讨会上，一种流行的演示手段就是使用IPython Notebook，然后再将.ipynb文件发布到网上以供所有人查阅
  - IPython Notebook应用程序是一个运行于命令行上的轻量级服务器进程。执行下面这条命令即可启动：
    ```python
    $ jupyter notebook
    [I 19:51:00.143 NotebookApp] Serving notebooks from local directory: /home/leondgarse/practice_code/python/data_processing
    [I 19:51:00.143 NotebookApp] 0 active kernels
    [I 19:51:00.143 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/?token=b556a416f4e6763e3acb3fdbf5a221ddeae925d71567e65b
    [I 19:51:00.143 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    ```
    在大多数平台上，Web浏览器会自动打开Notebook的仪表板（dashboard）
  - 由于我们是在一个Web浏览器中使用Notebook的，因此该服务器进程可以运行于任何地方。你甚至可以连接到那些运行在云服务（如Amazon EC2）上的Notebook
  - 直到写作本书时为止，一个新的名为NotebookCloud（http://notebookcloud.appspot.com）的项目已经诞生了，它可以轻松地在Amazon EC2上启动记事本
## IPython 提高代码开发效率的几点提示
  - 保留有意义的对象和数据，如果希望该模块是可引入的，也可以将这些代码放在if __name__=='__main__':块中
  - 扁平结构要比嵌套结构好，编写函数和类时应尽量注意低耦合和模块化，这样可以使它们更易于测试（如果你编写单元测试的话）、调试和交互式使用
  - 无惧大文件，维护更大的（具有高内聚度的）模块会更实用也更具有Python特点，在解决完问题之后，有时将大文件拆分成小文件会更好
  - 让自定义的类对IPython更加友好
    ```python
    IPython会获取__repr__方法返回的字符串（具体办法是output=repr(obj)），并将其显示到控制台上，因此可以为类添加一个简单的__repr__方法以得到一个更有意义的输出形式：
    In [61]: class Message:
      ...:   def __init__(self, msg):
      ...:     self.msg = msg
      ...:   def __repr__(self):
      ...:     return 'Message: %s' % self.msg
      ...:   <br />
    In [62]: x = Message("I have a secret")<br />
    In [63]: x
    Out[63]: Message: I have a secret
    ```
## 重新加载模块依赖项
  - 加载模块依赖项的问题
    - 在 Python 中，当输入 `import somelib` 时，some_lib 中的代码就会被执行，且其中所有的变量、函数和引入项都会被保存在一个新建的 some_lib 模块命名空间中
    - 下次再输入 `import some_lib` 时，就会得到这个模块命名空间的一个引用
    - 这对于 IPython 的交互式代码开发模式就会有一个问题，比如用 `%run` 执行的某段脚本中牵扯到了某个刚刚做了修改的模块
  - 假设有一个 **test_script.py** 文件，其中有下列代码
    ```python
    import some_lib

    x = 5
    y = [1, 2, 3, 4]
    result = some_lib.get_answer(x, y)
    ```
    如果在执行了 `%run test_script.py` 之后又对 some_lib.py 进行了修改，下次再执行 %run test_script.py 时将仍然会使用老版的 some_lib
  - Python2 使用 reload 函数
    ```python
    import some_lib
    reload(some_lib)

    x = 5
    y = [1, 2, 3, 4]
    result = some_lib.get_answer(x, y)
    ```
    当依赖变得更强时，就需要在很多地方插入很多的 reload
  - Python3.4 以上使用 **importlib.reload**
    ```python
    from importlib import reload
    reload(some_lib)
    ```
  - IPython 还提供了一个特殊的 **dreload** 函数（非魔术函数）来解决模块的“深度”（递归）重加载
    - 如果执行 import some_lib 之后再输入 dreload(some_lib)，则它会尝试重新加载 some_lib 及其所有的依赖项
    - 如果还是不行，重启 IPython
## IPython 个性化和配置
  - IPython shell在外观（如颜色、提示符、行间距等）和行为方面的大部分内容都是可以进行配置的
  - 配置选项定义在 ipython_config.py 文件中，默认IPython配置文件位于：
    ```python
    ~/.ipython/profile_default

    ```
    $ ipython profile list        # 列出当前所有的配置文件
  - $ ipython profile create secret_project         # 创建新的配置文件
  - $ $ ipython --profile=secret         # 使用新建的配置文件
  - 添加执行语句
    ```python
    ## lines of code to run at IPython startup.
    c.InteractiveShellApp.exec_lines = [
        'import numpy as np',
        'import pandas as pd',
        'import matplotlib.pyplot as plt',
        'import pylab as pl',

        'from pandas import Series, DataFrame'
    ]
    ```
***

# NumPy 基本数据结构 (ndarray / random / 视图 / 索引 / take / put / reshape / ravel / flatten / resize / 轴变换transpose / swapaxes)
## ndarray：一种多维数组对象 (dtype / shape / array / zeros / ones / eye / diag / empty / arange / astype / linspace)
  - NumPy最重要的一个特点就是其N维数组对象（即ndarray），该对象是一个快速而灵活的大数据集容器
  - 其中的所有元素必须是相同类型的
  - 矢量化（vectorization）：大小相等的数组之间的任何算术运算都会将运算应用到元素级，数组与标量的算术运算也会将那个标量值传播到各个元素
  - 每个数组都有一个shape（一个表示各维度大小的元组）和一个dtype（一个用于说明数组数据类型的对象）
  - dtype（数据类型）是一个特殊的对象，它含有ndarray将一块内存解释为特定数据类型所需的信息
  - dtype是NumPy如此强大和灵活的原因之一，多数情况下，它们直接映射到相应的机器表示，数值型dtype的命名方式相同：一个类型名（如float或int），后面跟一个用于表示各元素位长的数字
  - shape 用于返回数组的大小
  - array函数，它接受一切序列型的对象（包括其他数组），然后产生一个新的含有传入数据的NumPy数组
    ```python
    a = np.array(range(1, 10)).reshape(3, 3)        # 创建3*3的矩阵
    a.dtype
    Out[28]: dtype('int64')

    a.shape
    Out[29]: (3, 3)
    ```
    zeros和ones分别可以创建指定长度或形状的全0或全1数组
    ```python
    a = np.ones((3, 4))         # 创建 3x4 的矩阵，值初始化为1，zeros()创建初始值为0的数组
    ```
    eye / identity创建一个NxN单位矩阵，对角线为1，其余为0
    ```python
    np.eye(3)
    ```
    empty可以创建一个没有任何具体值的数组
    ```python
    np.empty((2, 3, 3))
    ```
    diag 创建一个对角矩阵，可以指定对角线上的值
    ```python
    np.diag((1, 2, 3))
    Out[39]:
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])
    ```
    arange是Python内置函数range的数组版
    ```python
    np.arange(0, 2*np.pi, 0.1)        # numpy的arange方法支持小数
    ```
  - astype方法转换dtype：
    ```python
    l = list('12345')
    a = np.array(l)
    a.dtype
    Out[42]: dtype('<U1')

    a.astype(np.int32)
    Out[43]: array([1, 2, 3, 4, 5], dtype=int32)

    a.astype(float).dtype
    Out[46]: dtype('float64')
    ```
    创建时指定使用的dtype：
    ```python
    b = np.array(l, dtype = np.float64)
    b.dtype
    Out[50]: dtype('float64')
    ```
  - help(numpy.linspace)
    ```python
    linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].<br />
    numpy.linspace(1, 49, 25, dtype=int)
    array([ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33,
        35, 37, 39, 41, 43, 45, 47, 49])
    ```
## random模块 （random_sample / rand / randint / uniform / normal / randn / permutation / shuffle / choice)
  - 对Python内置的random进行了补充，增加了一些用于高效生成多种概率分布的样本值的函数
  - random_sample / random / rand 返回[0.0, 1.0)区间内的随机数，可以指定形状
    ```
    random 是 random_sample 的别名
    rand 是 random_sample 的另一个版本，参数可以不使用元组
    ```
    ```python
    5 * np.random.random_sample((3, 2)) - 5
    Out[210]:
    array([[-4.20421995, -4.40482212],
           [-3.25372951, -1.78117074],
           [-4.1065046 , -2.47237069]])

    random.rand(2, 1)
    Out[213]:
    array([[ 0.54152849],
           [ 0.05451213]])
    ```
  - randint 返回随机整数值
    ```python
    randint(low, high=None, size=None, dtype='l')        # 范围(low, high]，大小为 size

    np.random.randint(2, size = 10)        # 当只有一个值时，作为最大值
    Out[355]: array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])

    np.random.randint(5, 10, size = (2, 4))        # size指定二维数组
    Out[360]:
    array([ [8, 6, 5, 7],
        [6, 7, 6, 5] ])
    ```
  - uniform 指定范围内的随机浮点数
    ```python
    np.random.uniform(-1, 0, [2, 3])
    Out[225]:
    array([[-0.92226819, -0.67579738, -0.15171841],
           [-0.83750923, -0.9686544 , -0.63621465]])
    ```
  - normal 得到一个标准正态(高斯)分布的样本数组：
    ```python
    samples = np.random.normal(size=(4, 4))
    ```
  - randn 返回一个符合正态分布(平均值0，标准差1)的随机取值列表
    ```python
    2.5 * np.random.randn(2, 4) + 3        # 2 *4的随机数列表
    ```
  - permutation 产生一组随机排列的序列
    ```python
    np.random.permutation(10)        # arange(x)内随机排序
    Out[343]: array([1, 3, 0, 7, 5, 6, 9, 8, 2, 4])

    np.random.permutation([1, 4, 9, 12, 15])        # 指定的序列随机排序
    Out[345]: array([ 1, 15, 4, 12, 9])

    arr = np.arange(9).reshape((3, 3))
    np.random.permutation(arr)        # 二维数组只在行上随机排序
    Out[351]:
    array([ [6, 7, 8],
        [0, 1, 2],
        [3, 4, 5] ])

    p.random.permutation(np.arange(9)).reshape(3, 3)
    Out[352]:
    array([ [1, 2, 0],
        [3, 7, 8],
        [5, 6, 4] ])
    ```
  - shuffle 在数组上随机排序
    ```python
    p = ["Python", "is", "powerful", "simple", "and so on..."]
    np.random.shuffle(p)
    p
    Out[213]: ['powerful', 'Python', 'simple', 'and so on...', 'is']
    ```
  - choice 随机选取序列中的一组值
    ```python
    choice(a, size=None, replace=True, p=None)
    size指定输出形状
    p指定每一个值的概率，总和需要等于1

    np.random.choice(5, (3, 3), p=[0.1, 0.2, 0.1, 0.1, 0.5])
    Out[59]:
    array([ [2, 3, 1],
        [4, 1, 1],
        [4, 4, 4] ])
    ```
## 数组的视图
  - 跟列表最重要的区别在于，数组切片是原始数组的视图，这意味着数据不会被复制，视图上的任何修改都会直接反映到源数组上
    ```python
    arr = np.arange(1, 10)
    arr_sl = arr[3:5]
    arr_sl[:] = 0
    arr
    Out[9]: array([1, 2, 3, 0, 0, 6, 7, 8, 9])

    这样可以避免大量数据的复制
    如果需要显示的复制，可以使用copy方法： arr_sl2 = arr[3:5].copy()
    ```
## 高维数组索引
  - 对于高维数组，各索引位置上的元素是一个低维数组，因此可以对各个元素递归访问：
    ```python
    arr2 = np.arange(1, 13).reshape((3, 4))
    arr2[0][2]
    Out[20]: 3

    arr2[0, 2]
    Out[21]: 3
    ```
  - 高维数组可以在一个或多个轴上进行切片 / 索引
    ```python
    arr3 = np.arange(1, 13).reshape((3, 4))
    arr3[:2, 1:]        # 先沿0轴切片，再沿1轴切片
    Out[26]:
    array([ [2, 3, 4],
        [6, 7, 8] ])

    arr3[1, :2]
    Out[38]: array([5, 6])

    arr3[:, :2]
    Out[39]:
    array([ [ 1, 2],
        [ 5, 6],
        [ 9, 10] ])
    ```
  - 二维数组生成列数组
    ```python
    # 一个数字索引产生的是一个一维数组
    arr3[:, 0]
    Out[189]: array([1, 5, 9])

    # 选取所有轴，列上使用切片索引
    arr3[:, :1]
    Out[190]:
    array([ [1],
        [5],
        [9] ])
    ```
## 布尔型索引
  - 使用布尔值组成的数组，选取其他数组中的行 / 列
  - 布尔型数组的长度必须跟被索引的轴长度一致
  - 可以将布尔型数组跟切片、整数（或整数序列）混合使用
    ```python
    In [40]: cha = np.array(list('hello'))
    In [43]: data = np.arange(1, 26).reshape(5, 5)

    In [45]: data[cha == 'l']
    Out[45]:
    array([ [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20] ])

    In [52]: data[:2, cha != 'l']
    Out[52]:
    array([ [ 1, 2, 5],
        [ 6, 7, 10] ])
    ```
  - 使用&（和）、|（或）之类的布尔算术运算符，组合应用多个布尔条件
    ```python
    In [53]: mask = (cha == 'h') | (cha == 'o')
    In [54]: data[mask]
    Out[54]:
    array([ [ 1, 2, 3, 4, 5],
        [21, 22, 23, 24, 25] ])
    ```
  - 通过布尔型索引选取数组中的数据，将总是创建数据的副本，即使返回一模一样的数组也是如此
    ```python
    In [33]: dt = data[cha == 'l']
    In [35]: dt[:] = 0
    In [36]: dt
    Out[36]:
    array([ [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0] ])

    In [37]: data
    Out[37]:
    array([ [ 1, 2, 3, 4, 5],
        [ 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25] ])
    ```
  - 通过布尔型数组设置值是一种经常用到的手段,将data中的所有负值都设置为0:
    ```python
    In [51]: data = np.random.randn(7, 4)
    In [54]: data[data < 0] = 0
    ```
## 花式索引
  - 指利用整数数组进行索引
  - 花式索引跟切片不一样，它总是将数据复制到新数组中
  - 传入一个用于指定顺序的整数列表或ndarray,以特定顺序选取行子集
    ```python
    In [74]: data = np.arange(25).reshape(5, 5)
    In [75]: data[ [0, 4, 2] ]
    Out[75]:
    array([ [ 0, 1, 2, 3, 4],
        [20, 21, 22, 23, 24],
        [10, 11, 12, 13, 14] ])
    ```
  - 使用负数索引将会从末尾开始选取行
    ```python
    In [78]: data[ [0, -5, 0] ]
    Out[78]:
    array([ [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4] ])
    ```
  - 一次传入多个索引数组,返回的是一个一维数组，其中的元素对应各个索引元组
    ```python
    In [80]: data[ [1, 4, 3], [0, 3, 2] ]
    Out[80]: array([ 5, 23, 17])
    最终选出的是元素(1, 0) (4, 3) (3, 2)
    ```
  - 矩形区域的形式,选取矩阵的行列子集
    ```python
    In [81]: data[ [1, 4, 3]][:, [0, 3, 2] ]
    Out[81]:
    array([ [ 5, 8, 7],
        [20, 23, 22],
        [15, 18, 17] ])
    ```
  - 另外一个办法是使用np.ix_函数，它可以将两个一维整数数组转换为一个用于选取方形区域的索引器
    ```python
    In [82]: data[np.ix_([1, 4, 3], [0, 3, 2])]
    Out[82]:
    array([ [ 5, 8, 7],
        [20, 23, 22],
        [15, 18, 17] ])
    ```
## take / put 花式索引的等价函数
  - 专门用于获取和设置单个轴向上的选区，直到编写本书时为止，take和put函数的性能通常要比花式索引好得多(已修正？)
    ```python
    arr = np.arange(10) * 100
    inds = [7, 1, 2, 6]

    arr.take(inds)
    Out[68]: array([700, 100, 200, 600])

    arr.put(inds, 42)
    arr[inds]
    Out[73]: array([42, 42, 42, 42])

    arr.put(inds, [40, 41, 42, 43])
    arr
    Out[75]: array([ 0, 41, 42, 300, 400, 500, 43, 40, 800, 900])
    ```
  - axis用于在其他轴上使用take
    ```python
    arrd = arr.reshape(2, 5)
    arrd
    Out[77]:
    array([ [ 0, 41, 42, 300, 400],
        [500, 43, 40, 800, 900] ])

    inds = [2, 0, 2, 1]
    arrd.take(inds)
    Out[79]: array([42, 0, 42, 41])

    arrd.take(inds, axis = 1)
    Out[80]:
    array([ [ 42,  0, 42, 41],
        [ 40, 500, 40, 43] ])
    ```
  - put不接受axis参数，它只会在数组的扁平化版本（一维，C顺序）上进行索引(这一点今后应该是会有所改善的)
## 重新排列 (reshape / ravel / flatten)
  - reshape重新排列
    ```python
    np.arange(8).reshape((2, 4))
    Out[474]:
    array([ [0, 1, 2, 3],
        [4, 5, 6, 7] ])
    ```
  - 作为参数的形状的其中一维可以是－1，表示该维度的大小由数据本身推断而来
    ```python
    np.random.randn(8, 2).reshape(4, -1)
    Out[478]:
    array([ [ 1.0145524 , 0.01896892, -2.1152739 , 1.30429964],
        [ 0.67861432, -0.47111913, 0.1382097 , 0.37572375],
        [ 0.96009437, 0.60651991, 0.48834527, 0.87227712],
        [ 0.5721235 , -0.64745798, 0.62888362, 0.58579363] ])
    ```
  - 散开raveling / 扁平化flattening
    ```python
    与reshape将一维数组转换为多维数组的运算过程相反的运算
    如果没有必要，ravel不会产生源数据的副本
    arr = np.arange(15).reshape((5, 3))
    arr.ravel()
    Out[480]: array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    flatten方法的行为类似于ravel，只不过它总是返回数据的副本
    arr.flatten()
    Out[481]: array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    ```
## resize改变数组大小
  - 将原数组重新排列成一维，可以扩大 / 缩小数组长度
  - resize不能用在由np.arange创建的数组上, refcheck=False指定不检查是否还有其他引用或视图
    ```python
    arr = np.array(np.arange(15).tolist())
    arr.resize(4, 5)
    arr
    Out[141]:
    array([ [ 0, 1, 2, 3, 4],
        [ 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [ 0, 0, 0, 0, 0] ])

    arr.resize(2, 3, refcheck=False)        # 第二次使用需要指定refcheck=False
    arr
    Out[149]:
    array([ [0, 1, 2],
        [3, 4, 5] ])

    arr = np.array(np.arange(15).reshape(3, 5).T.tolist())
    arr.resize(6, 3, refcheck=False)
    arr.T
    Out[187]:
    array([ [ 0, 1, 2, 3, 4, 0],
        [ 5, 6, 7, 8, 9, 0],
        [10, 11, 12, 13, 14, 0] ])
    ```
## 数组转置和轴对换 (transpose / T / dot计算内积)
  - 转置（transpose）是重塑的一种特殊形式，它返回的是源数据的视图（不会进行任何复制操作）
    ```python
    data = np.arange(12).reshape(3, 4)
    data.transpose()
    Out[90]:
    array([ [ 0, 4, 8],
        [ 1, 5, 9],
        [ 2, 6, 10],
        [ 3, 7, 11] ])

    In [92]: data.transpose()[:] = 0
    In [93]: data
    Out[93]:
    array([ [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0] ])
    ```
  - 数组不仅有transpose方法，还有一个特殊的T属性
    ```python
    In [91]: data.T
    Out[91]:
    array([ [ 0, 4, 8],
        [ 1, 5, 9],
        [ 2, 6, 10],
        [ 3, 7, 11] ])
    ```
  - np.dot计算矩阵内积XTX
    ```python
    In [95]: data = np.arange(12).reshape(3, 4)
    In [96]: np.dot(data, data.T)
    Out[96]:
    array([ [ 14, 38, 62],
        [ 38, 126, 214],
        [ 62, 214, 366] ])
    ```
## 高维数组轴变换 (swapaxes)
  - 对于高维数组，transpose需要得到一个由轴编号组成的元组才能对这些轴进行转置
    ```python
    In [100]: arr = np.arange(12).reshape(2, 2, 3)
    In [101]: arr
    Out[101]:
    array([ [ [ 0, 1, 2],
        [ 3, 4, 5] ],

        [ [ 6, 7, 8],
        [ 9, 10, 11] ] ])

    In [103]: arr.transpose(1, 0, 2)
    Out[103]:
    array([ [ [ 0, 1, 2],
        [ 6, 7, 8] ],

        [ [ 3, 4, 5],
        [ 9, 10, 11] ] ])
    ```
  - 简单的高维数组转置可以使用.T，它其实就是进行轴对换而已
    ```python
    In [102]: arr.T
    Out[102]:
    array([ [ [ 0, 6],
        [ 3, 9] ],

        [ [ 1, 7],
        [ 4, 10] ],

        [ [ 2, 8],
        [ 5, 11] ] ])
    ```
  - swapaxes方法，它需要接受一对轴编号
    ```python
    In [105]: arr.swapaxes(1, 2)
    Out[105]:
    array([ [ [ 0, 3],
        [ 1, 4],
        [ 2, 5] ],

        [ [ 6, 9],
        [ 7, 10],
        [ 8, 11] ] ])
    ```
***

# Numpy 数组方法 (ufunc / meshgrid / where / 数学方法 / any / all / sort / unique / concatention / vstack / split / fromfunction / 文件 / 线性代数 / fft / 随机数)
## 通用函数 ufunc (一元 sqt / exp / modf / 二元 add / maximum)
  - ufunc(universal function)是一种能对ndarray中的数据执行元素级运算的函数，可以看作简单函数的矢量化包装器
  - NumPy内置的许多ufunc函数都是在C语言级别实现的,计算速度非常快
  - 矢量化：NumPy数组可以将许多种数据处理任务表述为简洁的数组表达式（否则需要编写循环），用数组表达式代替循环的做法，通常被称为矢量化
  - 一般来说，矢量化数组运算要比等价的纯Python方式快上一两个数量级（甚至更多），尤其是各种数值计算
  - 查看帮助信息：
    ```python
    help(numpy.ufunc)
    np.info(numpy.sin)
    ```
  - 一元（unary）ufunc
    ```python
    arr = np.arange(10)
    np.sqrt(arr)
    Out[107]:
    array([ 0.    , 1.    , 1.41421356, 1.73205081, 2.    ,
        2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.    ])

    np.exp(arr)        # 自然对数
    Out[108]:
    array([ 1.00000000e+00,  2.71828183e+00,  7.38905610e+00,
         2.00855369e+01,  5.45981500e+01,  1.48413159e+02,
         4.03428793e+02,  1.09663316e+03,  2.98095799e+03,
         8.10308393e+03])
    ```
  - ufunc还可以返回多个数组，modf是Python内置函数divmod的矢量化版本，用于浮点数数组的小数和整数部分
    ```python
    arr = np.random.randn(8) * 5
    np.modf(arr)
    Out[118]:
    (array([-0.38118338, 0.39868922, 0.01417604, -0.75050765, 0.94605456,
         0.15511871, 0.96235234, -0.02172338]),
     array([-0., 6., 3., -1., 8., 8., 4., -5.]))
    ```
  - 二元（binary）ufunc
    ```python
    np.add(np.arange(1, 5), np.arange(2, 6))
    Out[13]: array([3, 5, 7, 9])

    x = np.random.randn(8)
    y = np.random.randn(8)
    np.maximum(x, y)
    Out[114]:
    array([ 0.35735776, 0.40592974, 1.41476104, -0.26213824, -0.22590334,
        0.4549781 , 1.82463044, -0.4841161 ])
    ```
## np.meshgrid 函数
  - np.meshgrid函数接受两个一维数组，并产生两个二维矩阵（对应于两个数组中所有的(x,y)对）
  - 在一组值（网格型）上计算函数sqrt(x^2+y^2)
    ```python
    import matplotlib.pyplot as plt
    points = np.arange(-5, 6)
    xs, ys = np.meshgrid(points, points)
    z = np.sqrt(xs ** 2 + ys ** 2)
    plt.imshow(z, cmap = plt.cm.Blues)
    plt.colorbar()
    plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')
    ```
## 将条件逻辑表述为数组运算 (where)
  - numpy.where函数是三元表达式x if condition else y的矢量化版本
  - 在数据分析工作中，where通常用于根据另一个数组而产生一个新的数组
  - 根据布尔数组 cond 中的值选取两个值数组 xarr 和 yarr 的值，当cond中的值为True时，选取xarr的值，否则从yarr中选取
    ```python
    纯python实现
    result = [(x if c else y)
            for x, y, c in zip(xarr, yarr, cond)]
    若使用np.where
    result = np.where(cond, xarr, yarr)
    ```
  - np.where的第二个和第三个参数不必是数组，它们都可以是标量值
  - 假设有一个由随机数据组成的矩阵，希望将所有正值替换为2，将所有负值替换为-2
    ```python
    arr = np.random.randn(4, 4)
    np.where(arr > 0, 2, -2)
    np.where(arr > 0, 2, arr) # 只将正值设置为2
    ```
  - 嵌套的where表达式：
    ```python
    np.where(cond1 & cond2, 0,        # if cond1 and cond2 --> 0
          np.where(cond1, 1,        # elif cond1 --> 1
               np.where(cond2, 2, 3)))        # elif cond2 --> 2, else --> 3
    ```
## 数学和统计方法 (sum / mean / std / var / add / sin / power / sign / cumsum / cumprod / diff)
  - 可以通过数组上的一组数学函数对整个数组或某个轴向的数据进行统计计算
  - 基本数学计算 + / * / >
    ```python
    a = numpy.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])        # 创建3*3的矩阵
    a * a        # 矩阵相乘
    array([ [ 1, 4, 9],
        [16, 25, 36],
        [49, 64, 81] ])
    a + 3        # 与数字相加
    array([ [ 4, 5, 6],
        [ 7, 8, 9],
        [10, 11, 12] ])
    a + a        # 矩阵相加
    array([ [ 2, 4, 6],
        [ 8, 10, 12],
        [14, 16, 18] ])
    a > 5        # 矩阵与数值比较
    array([ [False, False, False],
        [False, False, True],
        [ True, True, True] ], dtype=bool)
    ```
  - sum / mean / 标准差 std / 方差 var / 中位数 median 等聚合计算（aggregation，通常叫做约简（reduction））既可以当做数组的实例方法调用，也可以当做顶级NumPy函数使用
    ```python
    arr = np.random.randn(5, 4)
    arr.mean()
    np.mean(arr)
    ```
  - mean和sum这类的函数可以接受一个axis参数（用于计算该轴向上的统计值），最终结果是一个少一维的数组
    ```python
    arr.sum()        # 不指定坐标轴时，默认使用全部数据
    arr.sum(0)        # arr.sum(axis=0) 计算纵轴上的和
    arr.sum(1)        # arr.sum(axis=1) 计算横轴上的和
    arr.sum(-1)       # axis=-1 指定最后一个坐标轴
    ```
  - **median 中位数** 按照大小排序后，当数组长度为奇数时，中位数为 **处于中间位置的变量值**，当数组长度为偶数时，中位数为 **中间位置的2个变量值的平均数**
    ```python
    np.median([1, 3, 3, 6, 5, 6, 2])
    # Out[322]: 3.0

    np.median([1, 3, 3, 6, 5, 6])
    # Out[323]: 4.0
    ```
  - 其他方法 add / sin / power / sign
    ```python
    numpy.add(arr, arr)        # 两个矩阵相加
    numpy.sin(arr)        # 每个元素计算sin值
    numpy.power(arr, 2).sum()        # 计算每个元素的平方，然后求和
    np.sign(np.random.randn(5))        # Out[226]: array([ 1., 1., -1., -1., 1.])，提取序列中元素的符号
    ```
  - cumsum和cumprod之类的方法则不聚合，而是产生一个由中间结果组成的数组
    ```python
    cumsum累积和 / cumprod累积积
    arr = np.arange(9).reshape(3, 3)
    arr.cumsum(0)
    Out[31]:
    array([ [ 0, 1, 2],
        [ 3, 5, 7],
        [ 9, 12, 15] ])
    arr.cumprod(1)
    Out[32]:
    array([ [ 0,  0,  0],
        [ 3, 12, 60],
        [ 6, 42, 336] ])
    ```
  - diff 序列中的每一个元素与下一个元素相比的差值：
    ```python
    diff(a, n=1, axis=-1)
        Calculate the n-th discrete difference along given axis.
        n : The number of times values are differenced.

    x = np.array([1, 2, 3, 7, 0])
    np.diff(x)        # 计算一次
    array([ 1, 1, 4, -7])
    np.diff(x, n = 2)        # 计算两次
    array([ 0,  3, -11])
    np.diff(x, n = 3)        # 计算三次
    array([ 3, -14])

    x = np.array([ [1, 3, 6, 10], [0, 5, 6, 8] ])
    np.diff(x)
    array([ [2, 3, 4],
        [5, 1, 2] ])
    np.diff(x, axis=0)        # 二维指定坐标轴
    array([ [-1, 2, 0, -2] ])
    ```
## 用于布尔型数组的方法 (any / all / alltrue / allclose)
  - 布尔值会被强制转换为1（True）和0（False），因此sum可以被用来对布尔型数组中的True值计数
    ```python
    arr = np.random.randn(100)
    (arr > 0).sum()        # 正值的数量
    Out[35]: 55
    ```
  - any用于测试数组中是否存在一个或多个True
  - all则检查数组中所有值是否都是True
    ```python
    bools = np.array([False, False, True, False])
    bools.any()
    Out[37]: True

    bools.all()
    Out[38]: False
    ```
  - alltrue / allclose
    ```python
    alltrue 接受一个布尔型数组参数，判断是否全为True
    allclose 接受两个数组参数，判断两个数组在一定容忍度内是否相等
    np.allclose([1e10,1e-7], [1.00001e10,1e-8])
    Out[600]: False

    np.allclose([1e10,1e-8], [1.00001e10,1e-9])
    Out[601]: True

    np.allclose([1e10,1e-8], [1.0001e10,1e-9])
    Out[602]: False

    np.allclose([1.0, np.nan], [1.0, np.nan])
    Out[603]: False

    np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
    Out[604]: True
    ```
## 排序 (sort / sort_values / sort_index )
  - 顶级方法np.sort返回的是数组的已排序副本，而就地排序则会修改数组本身
  - 跟Python内置的列表类型一样，NumPy数组也可以通过sort方法就地排序
    ```python
    arr = np.random.randn(8)
    arr.sort()
    ```
  - 多维数组可以在任何一个轴向上进行排序，只需将轴编号传给sort即可
    ```python
    arr = np.random.randn(5, 3)
    arr.sort(0)
    ```
  - 计算数组分位数
    ```python
    对数组进行排序，然后选取特定位置的值
    large_arr = randn(1000)
    large_arr.sort()
    large_arr[int(0.05 * len(large_arr))]        # 5%分位数
    Out[54]: -1.5836855225757709
    ```
  - sort_values / sort_index 排序，sort方法不推荐使用
    ```python
    sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
      Sort by the values along either axis

    sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None)
      Sort object by labels (along an axis)

    ascending = False 使用降序排列
    by 参数指定用于排序的属性名
    sort_index 提供更多的选项，默认使用index排列
    ```
## 唯一化以及其他的集合逻辑 (unique / in1d)
  - NumPy提供了一些针对一维ndarray的基本集合运算
  - np.unique 用于找出数组中的唯一值并返回已排序的结果
    ```python
    arr = np.array(list('hello'))
    np.unique(arr)
    Out[56]:
    array(['e', 'h', 'l', 'o'],
       dtype='<U1')

    纯Python代码实现
    sorted(set(arr))
    Out[59]: ['e', 'h', 'l', 'o']
    ```
  - np.in1d用于测试一个数组中的值在另一个数组中的成员资格，返回一个布尔型数组
    ```python
    values = np.array([6, 0, 0, 3, 2, 5, 6])
    np.in1d(values, [2, 3, 6])
    Out[61]: array([ True, False, False, True, True, False, True], dtype=bool)
    ```
## 数组合并与拆分concatentaion / vstack / row_stack / hstack / column_stack / split / hsplit / vsplit / dsplit
  - numpy.concatenate可以按指定轴将一个由数组组成的序列连接到一起
    ```python
    arr1 = np.arange(1, 7).reshape(2, 3)
    arr2 = np.arange(7, 13).reshape(2, 3)
    np.concatenate([arr1, arr2], axis=1)
    Out[12]:
    array([ [ 1, 2, 3, 7, 8, 9],
        [ 4, 5, 6, 10, 11, 12] ])
    ```
  - vstack / row_stack / hstack / column_stack 面向行axis=0 / 列axis = 1的方式堆叠数组
    ```python
    np.vstack([arr1, arr2])        # np.row_stack([arr1, arr2])
    Out[14]:
    array([ [ 1, 2, 3],
        [ 4, 5, 6],
        [ 7, 8, 9],
        [10, 11, 12] ])

    np.hstack([arr1, arr2])        # np.column_stack([arr1, arr2])
    Out[15]:
    array([ [ 1, 2, 3, 7, 8, 9],
        [ 4, 5, 6, 10, 11, 12] ])
    a = np.array((1,2,3))
    b = np.array((2,3,4))
    np.column_stack((a,b))
      array([ [1, 2],
          [2, 3],
          [3, 4] ])
    ```
  - column_stack在转换一维数组时，会先将其转换为二维列向量
    ```python
    np.column_stack([np.array((1,2,3)), np.array((4, 5, 6))])
    Out[26]:
    array([ [1, 4],
        [2, 5],
        [3, 6] ])

    np.hstack([np.array((1,2,3)), np.array((4, 5, 6))])
    Out[27]: array([1, 2, 3, 4, 5, 6])

    这在增加一列标识列时很方便
    arr = np.arange(6).reshape(3, 2)
    arr2 = [1] * arr.shape[0]
    np.column_stack([arr, arr2])
    Out[32]:
    array([ [0, 1, 1],
        [2, 3, 1],
        [4, 5, 1] ])
    ```
  - split / hsplit / vsplit / dsplit 用于将一个数组沿指定轴拆分为多个数组
    ```python
    x = np.arange(9.0)
    np.split(x, 3)
    Out[38]: [array([ 0., 1., 2.]), array([ 3., 4., 5.]), array([ 6., 7., 8.])]

    x = np.arange(8.0)
    np.split(x, [3, 5, 10])
    Out[41]:
    [array([ 0., 1., 2.]),
     array([ 3., 4.]),
     array([ 5., 6., 7.]),
     array([], dtype=float64)]
    ```
## fromfunction在每个坐标点上执行function
  - fromfunction(function, shape, ** kwargs)
    ```python
    创建array，通过在每个坐标点上执行function
    ```
    参数function:
    ```python
    如果坐标形式是(x, y, z)，function的形式也需要是func(x, y, z)
    func得到的参数是一个array，每个参数代表一个坐标轴上坐标的递增变化
    如果shape = (2, 2)，则参数依次得到的值是(0, 0), (0, 1), (1, 0), (1, 1)
    ```
    参数shape :
    ```python
    int型的元组，定义输出array的形状，以及function得到的参数形式
    ```
    参数dtype :
    ```python
    数据类型，可选，默认是float
    ```
    ```python
    def func(x, y):
      print('x =\n', x)
      print('y =\n', y)
      return x + y

    numpy.fromfunction(func, (2, 2))
    x =
     [ [ 0. 0.]
     [ 1. 1.] ]
    y =
     [ [ 0. 1.]
     [ 0. 1.] ]
    Out[27]:
    array([ [ 0., 1.],
        [ 1., 2.] ])
    ```
    ```python
    numpy.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
    Out[28]:
    array([ [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4] ])
    ```
    ```python
    numpy.fromfunction(lambda i, j: (i+1) * (j+1), (9, 9), dtype = int)        # 创建乘法表
    Out[29]:
    array([ [ 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [ 2, 4, 6, 8, 10, 12, 14, 16, 18],
        [ 3, 6, 9, 12, 15, 18, 21, 24, 27],
        [ 4, 8, 12, 16, 20, 24, 28, 32, 36],
        [ 5, 10, 15, 20, 25, 30, 35, 40, 45],
        [ 6, 12, 18, 24, 30, 36, 42, 48, 54],
        [ 7, 14, 21, 28, 35, 42, 49, 56, 63],
        [ 8, 16, 24, 32, 40, 48, 56, 64, 72],
        [ 9, 18, 27, 36, 45, 54, 63, 72, 81] ])
    ```
## save / load 存取二进制格式文件
  - np.save和np.load是读写磁盘数组数据的两个主要函数
  - 默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中的
    ```python
    np.save('some_array', arr)
    np.load('some_array.npy')
    Out[69]:
    array(['e', 'h', 'l', 'o'],
       dtype='<U1')
    ```
  - np.savez可以将多个数组保存到一个压缩文件中，将数组以关键字参数的形式传入即可
    ```python
    np.savez('array_archive', a=arr, b=arr)
    ```
    加载.npz文件时，会得到一个类似字典的对象，该对象会对各个数组进行延迟加载
    ```python
    arch = np.load('array_archive.npz')
    arch['b']
    Out[77]:
    array(['e', 'h', 'l', 'o'],
       dtype='<U1')

    type(arch)
    Out[72]: numpy.lib.npyio.NpzFile

    arch.items()
    Out[75]:
    [('a', array(['e', 'h', 'l', 'o'],
        dtype='<U1')), ('b', array(['e', 'h', 'l', 'o'],
        dtype='<U1'))]

    arch.keys()
    Out[76]: ['a', 'b']
    ```
## savetxt / loadtxt 存取文本文件
  - np.loadtxt或更为专门化的np.genfromtxt将数据加载到普通的NumPy数组中
  - 这些函数都有许多选项可供使用：指定各种分隔符、针对特定列的转换器函数、需要跳过的行数等
  - <br />
  - 加载逗号分隔的csv文件，字段可以转化为数字：
    ```python
    arr = np.loadtxt('float.csv', delimiter=',')
    ```
    np.savetxt 将数组写到以某种分隔符隔开的文本文件中
  - genfromtxt跟loadtxt差不多，只不过它面向的是结构化数组和缺失数据处理
## 线性代数 (dot / linalg / inv / qr / var / det / eig / svd)
  - NumPy提供了一个用于矩阵乘法的dot函数（既是一个数组方法也是numpy命名空间中的一个函数）
    ```python
    x = np.arange(6).reshape(2, 3)
    y = np.arange(9).reshape(3, 3)
    np.dot(x, y)
    Out[89]:
    array([ [15, 18, 21],
        [42, 54, 66] ])
    ```
    高维数组
    ```python
    a = np.arange(3*4*5*6).reshape((3,4,5,6))
    b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    np.dot(a, b)[2,3,2,1,2,2]
    Out[117]: 499128

    sum(a[2,3,2,:] * b[1,2,:,2])
    Out[118]: 499128
    ```
  - **numpy.linalg** 中有一组标准的矩阵分解运算以及诸如求逆和行列式之类的东西，它们跟MATLAB和R等语言所使用的是相同的行业标准级Fortran库，如BLAS / LAPACK / Intel MKL
    ```python
    from numpy import linalg as LA
    ```
  - **inv** 计算方阵的逆，det 行列式的秩
    ```python
    x = randn(5, 5)
    xm = x.T.dot(x)
    LA.inv(xm)
    ```
  - **qr** 计算QR分解
    ```python
    xm.dot(inv(xm))
    q, r = LA.qr(xm)
    ```
  - **eig** 计算特征向量与特征值，对于矩阵A，满足Av = λv，其中λ是特征值，v是特征向量
    ```python
    d = np.diag((1, 2, 3))
    w, v = LA.eig(d)
    w
    Out[216]: array([ 1.,  2.,  3.])

    v
    Out[217]:
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    其中 w 是矩阵的特征值，v 是矩阵的特征向量，即满足 dot(d, v) == w * v
    (dot(d, v) == w * v).all()
    Out[219]: True

    对于单个值，满足
    [dot(d[:,:], v[:,i]) == w[i] * v[:,i] for i in range(d.shape[0])]
    ```
  - **svd** 计算矩阵的奇异值分解，将矩阵 Data 分解成三个矩阵 U、Σ 和 VT
    ```
    Data(m * n) = U(m * m) * Σ(m * n) * VT(n * n)

    svd(a, full_matrices=1, compute_uv=1)
    参数 full_matrices，bool, optional
        If True (default), u and v have the shapes M * M and N * N
        If False, the shapes are M * K and K * N, where K = min(M, N)
    参数 compute_uv : bool, optional，指定是否计算 u 和 v，默认True
    返回值 u, s, v，其中 s 表示奇异值，是一个只包含对角元素矩阵的行向量，降序排列
    分解结果 a == np.dot(u, np.dot(diag(s), v))
    ```
    ```python
    a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    U, s, V = np.linalg.svd(a, full_matrices=True)
    U.shape, V.shape, s.shape
    Out[585]: ((9, 9), (6, 6), (6,))

    S = np.zeros((9, 6), dtype=complex)
    S[:6, :6] = np.diag(s)
    np.allclose(a, np.dot(U, np.dot(S, V)))
    Out[586]: True
    # full_matrices=False
    U, s, V = np.linalg.svd(a, full_matrices=False)
    U.shape, V.shape, s.shape
    Out[587]: ((9, 6), (6, 6), (6,))

    S = np.diag(s)
    np.allclose(a, np.dot(U, np.dot(S, V)))
    Out[588]: True
    ```
  - scipy.linalg
    ```python
    扩展了由numpy.linalg提供的线性代数例程和矩阵分解功能

    from scipy import linalg
    arr = numpy.array(([1, 2], [3, 4]))        # 创建二维数组
    linalg.det(arr)        # Compute the determinant of a matrix 计算矩阵行列式，1*4 - 2*3 = -2
    -2.0
    ```
## 傅立叶变换 fft
  - numpy的fft模块，scipy也有类似方法
    ```python
    la = np.ones(500)
    la[100:300] = -1
    f = np.fft.fft(la)        # f = sp.fft(la)
    plt.plot(f)
    ```
## 示例：随机漫步 (randint / where / cumsum / abs / any / argmax)
  - 随机漫步：从0开始，步长1和－1出现的概率相等
  - 纯python实现
    ```python
    pos = 0
    walk = []
    walk.append(pos)
    steps = 1000

    for i in range(steps):
      step = 1 if randint(0, 2) else -1
      pos += step
      walk.append(pos)
    plt.plot(walk)
    ```
  - numpy实现
    ```python
    steps = 1000
    draws = np.random.randint(0, 2, size=steps)
    step = np.where(draws > 0, 1, -1)
    walk = step.cumsum()

    walk.max()        # 计算最大值
    (np.abs(walk) >= 10).argmax()        # 计算首次出现大于10的位置，argmax返回数组第一个最大值的索引 (True）
    ```
  - 一次模拟多个随机漫步
    ```python
    walks = 5000
    draws = np.random.randint(0, 2, size=(walks, steps))        # 生成二维数组
    steps = np.where(draws > 0, 1, -1)
    walks = steps.cumsum(1)
    walks.max()        # 计算极值
    walks.min()
    hints30 = (np.abs(walks) >= 30).any(1)        # 布尔数组筛选是否有大于30的值
    crossing_times = (np.abs(walks[hints30]) >= 30).argmax(1)        # 各一维数组中首次出现大于30的值的位置
    crossing_times.shape
    crossing_times.mean()
    ```
***

# pandas 一维数组对象Series
  - 类似一维数组的对象，由数据和索引组成，使用类似字典
  - Series的字符串表现形式为：索引在左边，值在右边
  - 可以通过Series 的values和index属性获取其数组表示形式和索引对象
## Series的创建
  - 自动创建索引，也可以指定index，index数量需要与值的数量相同
    ```python
    a = Series(arange(6))
    a.values
    Out[13]: array([0, 1, 2, 3, 4, 5])

    a.index
    Out[14]: RangeIndex(start=0, stop=6, step=1)
    ```
  - 创建带指定索引的Series
    ```python
    a = Series([1, 2.0, 'a'], index = [1, 2, 3])
    a
    Out[82]:
    1  1
    2  2
    3  a
    dtype: object
    ```
## Series的基本运算
  - 与普通NumPy数组相比，可以通过索引的方式选取Series中的单个或一组值
  - NumPy数组运算都会保留索引和值之间的链接
    ```python
    b = Series(arange(6), index=list('abcdef'))
    b['a']
    Out[33]: 0

    np.exp(b)['a']        # 计算元素的自然对数
    Out[36]: 1.0

    (b * 2)['b']
    Out[88]: 2
    ```
  - Series的索引可以通过赋值的方式就地修改
    ```python
    In [90]: b.index = list('hello ')
    ```
  - min / max / argmin / argmax 最小值 / 最大值 / 最小值的索引值 / 最大值的索引值
## Series与Python字典
  - 如果数据被存放在一个Python字典中，也可以直接通过这个字典来创建Series
  - 可以用在许多原本需要字典参数的函数中
    ```python
    data = dict(zip(list('abcdef'), range(6)))
    data
    Out[45]: {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}

    ds = Series(data)
    'f' in b
    Out[37]: True
    ```
  - 传入字典时，也可以指定新的索引，相匹配的值会被找出来并放到相应的位置上，没有对应的结果就为NaN
    ```python
    ind = list('data')
    ds = Series(data, index=ind)
    ds
    Out[52]:
    d  3.0
    a  0.0
    t  NaN
    a  0.0
    dtype: float64
    ```
  - pandas的isnull和notnull函数可用于检测缺失数据
    ```python
    pd.isnull(ds)
    pd.notnull(ds)
    ```
    Series也有类似的实例方法
    ```python
    ds.isnull()
    Out[56]:
    d  False
    a  False
    t   True
    a  False
    dtype: bool
    ```
## Series的数据对齐
  - 对于许多应用而言，Series最重要的一个功能是：它在算术运算中会自动对齐不同索引的数据
    ```python
    ds2 = Series({i : randn() for i in list('abcd')})
    ds2
    Out[102]:
    a  1.580103
    b  1.957518
    c  2.342100
    d  -1.303779
    dtype: float64

    ds + ds2        # 取两个Series相同index的值相加，不同index的值置为NaN
    Out[103]:
    a  1.580103
    a  1.580103
    b     NaN
    c     NaN
    d  1.696221
    t     NaN
    dtype: float64
    ```
## Series的name属性
  - Series对象本身及其索引均有一个name属性，该属性跟pandas其他的关键功能关系非常密切
    ```python
    ds.name = 'data'
    ds.index.name = 'volume'
    ds
    Out[61]:
    volume
    d  3.0
    a  0.0
    t  NaN
    a  0.0
    Name: data, dtype: float64
    ```
***

# pandas 表格数据结构DataFrame / Index对象 / 矢量化的字符串函数 / Panel
  - 一个表格型的数据结构，含有一组有序的列(类似于index)
  - DataFrame既有行索引也有列索引，可看成共享同一个index的Series集合
  - 层次化索引的表格型结构：虽然DataFrame是以二维结构保存数据的，但仍然可以将其表示为更高维度的数据
  - 对于DataFrame df，查看列数据可以使用df.name / df['name']，但创建新列 / 删除列只能使用 df['name']这种方式
## 创建 （zip / index / columns / 嵌套字典 / Series）
  - 使用zip创建DataFrame [ ??? ]
  - 使用字典数据创建
    ```python
    结果DataFrame会自动加上索引，且全部列会被有序排列
    data = {'name':['Wangdachui','Linling','Niuyun'], 'pay':[4000,5000,6000]}        # 用于创建DataFrame的列元素数目应相同
    pd.DataFrame(data)
    Out[109]:
         name  pay
    0 Wangdachui 4000
    1   Linling 5000
    2   Niuyun 6000
    ```
  - 可以通过index指定索引
  - 如果指定了列序列columns，则DataFrame的列就会按照指定顺序进行排列
  - 如果传入的列在数据中找不到，就会产生NA值
    ```python
    f = DataFrame(data, columns=['name', 'pay', 'year'])
    f
    Out[111]:
         name  pay year
    0 Wangdachui 4000 NaN
    1   Linling 5000 NaN
    2   Niuyun 6000 NaN
    ```
  - 嵌套字典
    ```python
    pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    将它传给DataFrame，它就会被解释为：外层字典的键作为列，内层键则作为行索引

    df2 = DataFrame(pop)
    df2
    Out[134]:
       Nevada Ohio
    2000   NaN  1.5
    2001   2.4  1.7
    2002   2.9  3.6

    对该结果进行转置：
    df2.T
    Out[135]:
        2000 2001 2002
    Nevada  NaN  2.4  2.9
    Ohio   1.5  1.7  3.6
    ```
  - 由Series组成的字典
    ```python
    pd = {'Ohio' : df2['Ohio'][:-1], 'Nevada' : df2['Nevada'][:2]}
    DataFrame(pd)
    Out[140]:
       Nevada Ohio
    2000   NaN  1.5
    2001   2.4  1.7
    ```
  - 结合matplotlab绘图：
    ```python
    x = np.arange(0, 2*np.pi, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    t = pd.DataFrame({'sin':y1, 'cos':y2, 'sin+cos':y1+y2}, index = x)
    t.plot()
    Out[168]: <matplotlib.axes._subplots.AxesSubplot at 0x7fa65dd2e5f8>        # pandas绘图
    plt.show()
    ```
## 创建使用日期做index的DataFrame (pd.date_range)
  - 使用日期作为index
    ```python
    dates = pd.date_range('20170508', periods=7)
    DataFrame(np.arange(1, 8), index=dates, columns=['value'])
    Out[372]:
          value
    2017-05-08   1
    2017-05-09   2
    2017-05-10   3
    2017-05-11   4
    2017-05-12   5
    2017-05-13   6
    2017-05-14   7
    ```
  - 创建两行四列的DataFrame
    ```python
    DataFrame(np.array([list(range(1, 8)), list('abcdefg')]), index = pd.date_range('20170303', periods = 2), columns = list('ABCDEFG'))
    Out[373]:
          A B C D E F G
    2017-03-03 1 2 3 4 5 6 7
    2017-03-04 a b c d e f g
    ```
  - 随机生成的二维数组，创建DataFrame
    ```python
    DataFrame(np.random.randn(7,3),index=dates,columns = list('ABC'))
    ```
## 数据查看 （index / columns / values / describe / ix / 切片）
  - index / columns / values / describe 属性
    ```python
    type(df)
    <class 'pandas.core.frame.DataFrame'>
    df.index
    df.columns
    df.values        # 以二维ndarray的形式返回DataFrame中的数据
    df.describe        # Generate various summary statistics, excluding NaN values.
    ```
  - 通过类似字典标记的方式或属性的方式，可以将DataFrame的列获取为一个Series
    ```python
    f['name']
    0  Wangdachui
    1    Linling
    2    Niuyun
    Name: name, dtype: object

    f.pay
    0  4000
    1  5000
    2  6000

    f.index
    RangeIndex(start=0, stop=3, step=1)
    list(f.index)
    [0, 1, 2]
    ```
  - 通过位置或名称的方式获取行，比如用索引字段ix
    ```python
    f.ix[2]
    name  Niuyun
    pay    6000
    Name: 2, dtype: object
    ```
  - 切片方式：
    ```python
    不支持选取不连续的两行
    不支持对于列的切片选取 df['open':'close']

    df[:5]
    df[-5:]
    df[3:5]
    df['open']
    df.open
    df[3:5]['open']
    df[u'2017-02-09':u'2017-02-16']
    df[df.index[1]:df.index[5]]        # 不同于df[1:5]，这种方法是包含尾节点的
    ```
## 数据修改 (set_value)
  - 列可以通过赋值的方式进行修改，可以赋上一个标量值
    ```python
    f.name = 'admin'
    f
    Out[207]:
      name  pay
    0 admin 4000
    1 admin 5000
    2 admin 6000
    ```
    或一组值
    ```python
    f['year'] = np.arange(3)
    f['high'] = f.pay >= 5000
    f
    Out[228]:
      name  pay year  high
    0 admin 4000   0 False
    1 admin 5000   1  True
    2 admin 6000   2  True
    ```
  - 如果赋值的是一个Series，就会精确匹配DataFrame的索引，所有的空位都将被填上缺失值
    ```python
    val = Series([-1.2, -1.5, -1.7], index = [1, 2, 3])
    f['base'] = val
    f
    Out[231]:
      name  pay year  high base
    0 admin 4000   0 False  NaN
    1 admin 5000   1  True -1.2
    2 admin 6000   2  True -1.5
    ```
  - Series方法set_value改变单个值：
    ```python
    f.name.set_value(2, 'root')        # 不能直接使用 f['name'][2] = 'root'
    f
    Out[209]:
      name  pay
    0 admin 4000
    1 admin 5000
    2  root 6000
    ```
  - 在空的DataFrame上添加新记录:
    ```python
    直接创建空的DataFrame添加会报错
    df = DataFrame(columns=list('abc'))
    df.ix[0] = [1, 2, 3]  # ValueError: cannot set by positional indexing with enlargement
    可以创建时添加一行数据
    df = DataFrame(np.ones([1, 3]), columns=list('abc'))
    df.ix[1] = [1, 2 ,3]
    ```
  - 创建新列时的问题 [ ??? ]：
    ```python
    data = {'name':['Wangdachui','Linling','Niuyun'], 'pay':[4000,5000,6000]}
    f = pandas.DataFrame(data)
    f.year = np.arange(3)
    f.year        # 使用 f['year'] 将报错
    Out[332]: array([0, 1, 2])

    f.columns        # f 中并没有year这一列
    Out[344]: Index(['name', 'pay'], dtype='object')

    f['year'] = np.arange(3)        # 只能使用索引方式
    f.columns
    Out[353]: Index(['name', 'pay', 'year'], dtype='object')

    f.year = 3        # 此时不能通过这种方法修改
    f['month'] = np.arange(4, 7)
    f.month = 7        # 没有通过.方法创建的列可以修改
    f
    Out[360]:
         name  pay year month
    0 Wangdachui 4000   0   7
    1   Linling 5000   1   7
    2   Niuyun 6000   2   7
    ```
## 数据删除 （del / drop）
  - 关键字del用于删除列
    ```python
    del f['year']        # 不能是 del f.year
    f.columns
    Out[131]: Index(['name', 'pay', 'base', 'high'], dtype='object')
    ```
  - drop方法
    ```python
    drop方法默认操作副本，返回的是一个在指定轴上删除了指定值的新对象
    默认删除index指定的行
    t = f.drop(1)
    t.index
    Out[405]: Int64Index([0, 2], dtype='int64')

    通过axis指定删除列：
    f['year'] = pd.date_range('20170508', periods=3)
    t = f.drop('pay', axis=1)
    t.columns
    Out[411]: Index(['name', 'year'], dtype='object')
    ```
  - 删除单个元素[ ??? ]：
    ```python
    f.pay.drop(1)        # 或使用del f.pay[1]
    Out[275]:
    0  4000
    2  6000
    Name: pay, dtype: int64

    f.pay        # drop函数默认操作副本，返回drop后的数据，原数据不变
    Out[276]:
    0  4000
    1  5000
    2  6000
    Name: pay, dtype: int64

    f.pay.drop(1, inplace=True)        # 通过指定inplace=True在原数据上操作
    f.pay
    Out[278]:
    0  4000
    2  6000
    Name: pay, dtype: int64

    f        # 但显示的DataFrame中的值没有更新 [ ??? ]
    Out[287]:
      name  pay  high base
    0 admin 4000 False  NaN
    1 admin 5000  True -1.2
    2 admin 6000  True -1.5

    f.pay = f.pay        # 可以将操作后的Series重新赋值给DataFrame
    f.pay = f.pay.drop(2)
    f
    Out[300]:
      name   pay  high base
    0 admin 4000.0 False  NaN
    1 admin   NaN  True -1.2
    2 admin   NaN  True -1.5
    ```
## DataFrame的 index / columns 的name属性
  - 指定索引 / 列名称
    ```python
    pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    df2 = DataFrame(pop)
    df2.index.name = 'year'
    df2.columns.name = 'state'
    In [145]: df2
    Out[145]:
    state Nevada Ohio
    year        
    2000   NaN  1.5
    2001   2.4  1.7
    2002   2.9  3.6
    ```
## Index 索引对象
  - pandas的索引对象负责管理轴标签和其他元数据，构建Series或DataFrame时，所用到的任何数组或其他序列的标签都会被转换成一个Index
  - Index对象是不可修改的（immutable），不可修改性非常重要，因为这样才能使Index对象在多个数据结构之间安全共享
    ```python
    ind = pd.Index(np.arange(3))
    df2.index = ind
    df2
    Out[153]:
    state Nevada Ohio
    0     NaN  1.5
    1     2.4  1.7
    2     2.9  3.6
    ```
  - 每个索引都有一些方法和属性，它们可用于设置逻辑并回答有关该索引所包含的数据的常见问题
    ```python
    df2.index is ind
    Out[154]: True

    3 in df2.index
    Out[163]: False

    df2.index.append(pd.Index(np.arange(2, 4)))
    Out[164]: Int64Index([0, 1, 2, 2, 3], dtype='int64')

    df2.index.append(pd.Index(np.arange(2, 4))).difference(ind)
    Out[165]: Int64Index([3], dtype='int64')
    ```
  - Index甚至可以被继承从而实现特别的轴索引功能
    ```python
    pandas库中内置的Index类
    Int64Index         针对整数的特殊Index
    MultiIndex         层次化索引对象，表示单个轴上的多层索引
    DatatimeIndex         存储纳秒级时间戳
    PeriodIndex        针对时间间隔Period数据的特殊Index
    ```
  - take / slice_indexer / slice_locs
## pandas 矢量化的字符串函数 (contains / get / findall / extract)
  - 通过Series.map，所有字符串和正则表达式方法都能被应用于（传入lambda表达式或其他函数）各个值，但是如果存在NA就会报错
  - Series的str属性有一些能够跳过NA值的字符串操作方法 help(Series.str)
  - str.contains 返回是否包含子串
    ```python
    data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
    data = Series(data)
    data.str.contains('gmail')
    Out[536]:
    Dave   False
    Rob    True
    Steve   True
    Wes    NaN
    dtype: object
    ```
  - str.get / 索引 获取元素
    ```python
    data.str.get(1)        # 参数只能取一个整数
    Out[551]:
    Dave    a
    Rob    o
    Steve   t
    Wes   NaN
    dtype: object

    data.str[:5]
    Out[554]:
    Dave   dave@
    Rob   rob@g
    Steve  steve
    Wes    NaN
    dtype: object
    ```
  - findall 查找匹配字符串
    ```python
    可以使用正则表达式，还可以加上任意re选项（如IGNORECASE）
    pattern = '([A-Z0-9._%+-]+)@([a-m0-9.-]+)\\.([A-Z]{2,4})'
    data.str.findall(pattern, flags=re.IGNORECASE)
    Out[565]:
    Dave            []
    Rob    [(rob, gmail, com)]
    Steve  [(steve, gmail, com)]
    Wes            NaN
    dtype: object
    ```
  - extract 根据分组结果生成DataFrame
    ```python
    expand : bool, 当前默认False，以后会改为True
        * If True, return DataFrame.
        * If False, return Series/Index/DataFrame

    data.str.extract(pattern, flags=re.IGNORECASE, expand=True)
    Out[548]:
          0    1  2
    Dave  dave google com
    Rob   rob  gmail com
    Steve steve  gmail com
    Wes   NaN   NaN NaN
    ```
## 面板数据Panel (三维版的DataFrame)
  - 三维版的DataFrame，Panel中的每一项（类似于DataFrame的列）都是一个DataFrame
  - 基于ix的标签索引被推广到了三个维度
  - to_frame / to_panel方法用于转换Panel / DataFrame
***

# Series / DataFrame 基本通用功能 (reindex / 索引 / ix / 算数运算 / 函数映射apply / 值计数 / 数据对齐 / 排序 / loc / at / iloc / iat)
## reindex 重新索引 (插值method / ix)
  - 其作用是根据新索引进行重排，并创建一个适应新索引的新对象
  - 如果某个索引值当前不存在，就引入缺失值
    ```python
    obj = Series(randn(4), index = list('dbac'))
    obj.reindex(list('abcde'))
    Out[382]:
    a  -0.446403
    b  0.040327
    c  -2.048808
    d  -0.144099
    e     NaN
    dtype: float64
    ```
  - 对于缺失的值，可以使用fill_value指定填充的值
    ```python
    obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
    ```
    或使用method选项
    ```python
    ffill / pad 前向填充值
    bfill / backfill 后向填充值
    obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
    obj3.reindex(range(6), method='ffill')
    Out[385]:
    0   blue
    1   blue
    2  purple
    3  purple
    4  yellow
    5  yellow
    dtype: object
    ```
  - 对于DataFrame，reindex可以修改（行）索引、列，或两个都修改
  - 如果仅传入一个序列，则会重新索引行，插值只能按行应用
    ```python
    frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
    frame.reindex(list('abcd'))
    Out[391]:
      Ohio Texas California
    a  0.0  1.0     2.0
    b  NaN  NaN     NaN
    c  3.0  4.0     5.0
    d  6.0  7.0     8.0

    ```
    使用columns关键字即可重新索引列
    ```python
    states = ['Texas', 'Utah', 'California']
    frame.reindex(index=list('abcd'), method='ffill', columns=states)
    Out[394]:
      Texas Utah California
    a   1  NaN      2
    b   1  NaN      2
    c   4  NaN      5
    d   7  NaN      8
    ```
  - 利用ix的标签索引功能，重新索引任务可以变得更简洁
    ```python
    frame.ix[list('acd'), states]
    Out[397]:
      Texas Utah California
    a   1  NaN      2
    c   4  NaN      5
    d   7  NaN      8
    ```
## 索引、选取和过滤 (切片 / head / tail / 索引ix / is_unique)
  - get_value / set_value 方法根据行列标签选取 / 设置值
  - Series索引（obj[...]）的工作方式类似于NumPy数组的索引，只不过Series的索引值可以是index，也可以是序号
    ```python
    obj = Series(np.arange(4.), index=list('abcd'))
    obj[ ['b', 'c'] ]
    Out[424]:
    b  1.0
    c  2.0
    dtype: float64

    obj[obj < 2]
    Out[425]:
    a  0.0
    b  1.0
    dtype: float64
    ```
  - 利用标签的切片运算与普通的Python切片运算不同，其末端是包含的
    ```python
    obj['b':'c']
    Out[418]:
    b  1.0
    c  2.0
    dtype: float64

    obj[1:3]
    Out[419]:
    b  1.0
    c  2.0
    dtype: float64
    ```
  - 对DataFrame进行索引 / 筛选
    ```python
    data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
    data
    Out[454]:
         one two three four
    Ohio    0  1   2   3
    Colorado  4  5   6   7
    Utah    8  9   10  11
    New York  12  13   14  15

    data[data['three'] > 5]
    Out[455]:
         one two three four
    Colorado  4  5   6   7
    Utah    8  9   10  11
    New York  12  13   14  15

    data[:2]
    Out[456]:
         one two three four
    Ohio    0  1   2   3
    Colorado  4  5   6   7

    data[data < 5] = 0
    data
    Out[458]:
         one two three four
    Ohio    0  0   0   0
    Colorado  0  5   6   7
    Utah    8  9   10  11
    New York  12  13   14  15
    ```
  - head / tail方法：
    ```python
    接受一个int参数
    df.head(5)
    df.tail(5)
    ```
  - 索引字段ix
    ```python
    可以通过NumPy式的标记法以及轴标签从DataFrame中选取行和列的子集
    obj.ix[val]        选取一行或多行
    obj.ix[:, val]        选取单列或多列
    obj.ix[val1, val2]        同时选取行列

    data.ix[['Ohio', 'Colorado'], :3]
    Out[460]:
         one two three
    Ohio    0  0   0
    Colorado  0  5   6
    ```
  - 带有重复值的轴索引
    ```python
    轴标签index并不要求唯一，索引的is_unique属性可以返回值是否唯一
    obj = Series(range(5), index=list('aabbc'))
    obj
    Out[67]:
    a  0
    a  1
    b  2
    b  3
    c  4
    dtype: int64

    obj.index.is_unique
    Out[69]: False
    ```
## 汇总和计算描述统计 (ufunc / sum / idmax / describe / 相关系数与协方差)
  - NumPy的ufuncs（元素级数组方法）可用于操作pandas对象
    ```python
    df = DataFrame([ [1.4, np.nan], [7.1, -4.5],
          [np.nan, np.nan], [0.75, -1.3] ],
          index=list('abcd'), columns=['one', 'two'])

    df
    Out[176]:
      one two
    a 1.40 NaN
    b 7.10 -4.5
    c  NaN NaN
    d 0.75 -1.3

    np.abs(df)
    ```
  - pandas对象拥有一组常用的数学和统计方法，大部分都属于约简和汇总统计，用于从Series中提取单个值（如sum或mean）或从DataFrame的行或列中提取一个Series
  - 跟对应的NumPy数组方法相比，都是基于没有缺失数据的假设而构建的
    ```python
    约简方法： df.sum(axis=1)
    间接统计方法： df.idxmax()         # idxmin / idxmax 返回达到最小值 / 最大值的索引
    累计型方法：df.cumsum()
    ```
  - 约简方法的常用选项
    ```python
    axis         约简的轴，DataFrame行用0，列用1
    skipna        排除缺失值，默认为True，自动排除NA值，除非整个切片（这里指的是行或列）都是NA
    level        如果轴是层次化索引的(MultiIndex)，则根据level分组约简
    ```
  - describe方法，用于一次性产生多个汇总统计
    ```python
    df.describe()

    对于非数值型数据，describe会产生另外一种汇总统计
    obj = Series(['a', 'a', 'b', 'c'] * 4)
    obj.describe()
    Out[76]:
    count   16
    unique   3
    top    a
    freq    8
    dtype: object
    ```
  - 相关系数与协方差
    ```python
    Series的corr方法用于计算两个Series中重叠的、非NA的、按索引对齐的值的相关系数，与此类似，cov用于计算协方差
    DataFrame的corr和cov方法将以DataFrame的形式返回完整的相关系数或协方差矩阵
    利用DataFrame的corrwith方法，可以计算其列或行跟另一个Series或DataFrame之间的相关系数
    传入一个Series将会返回一个相关系数值Series（针对各列进行计算）
    传入一个DataFrame则会计算按列名配对的相关系数
    ```
## 函数映射 (apply / applymap)
  - DataFrame的apply方法，用于将函数应用到由各列或行所形成的一维数组上
    ```python
    apply(self, func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)

    f = lambda x: x.max() - x.min()
    df.apply(f, axis=1)
    ```
    许多最为常见的数组统计功能都被实现成DataFrame的方法（如sum和mean），因此无需使用apply方法
    ```python
    df.sum(axis=0)
    ```
  - 除标量值外，传递给apply的函数还可以返回由多个值组成的Series
    ```python
    def f(x):
        return Series([x.min(), x.max()], index=['min', 'max'])

    df.apply(f)
    df.apply(f, axis=1).T
    Out[46]:
           a    b   c     d
    min  1.4 -4.5 NaN -1.30
    max  1.4  7.1 NaN  0.75
    ```
  - 元素级的Python函数也是可以用的
    ```python
    使用applymap得到frame中各个浮点值的格式化字符串
    format = lambda x: '%.2f' % x
    df.applymap(format)
    ```
  - 参数 raw : boolean, default False
    - False，将每一行 / 列转化为 Series
    - True，将每一行 / 列转化为 ndarray
  - 参数 args : tuple，指定传递给函数的其他参数，如果只有一个，指定args=(xx,)
    ```python
    f = lambda x, a : x + a
    df.apply(f, raw=True, args=(3,))
    Out[54]:
         one  two
    a   4.40  NaN
    b  10.10 -1.5
    c    NaN  NaN
    d   3.75  1.7
    ```
## 唯一值、值计数以及成员资格 (unique / value_counts / isin / apply(pd.value_counts))
  - unique 可以得到Series中的唯一值数组
    ```python
    obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
    obj.unique()
    Out[79]: array(['c', 'a', 'd', 'b'], dtype=object)
    ```
  - value_counts 用于计算一个Series中各值出现的频率
    ```python
    obj.value_counts()
    Out[80]:
    c  3
    a  3
    b  2
    d  1
    dtype: int64

    为了便于查看，结果Series是按值频率降序排列的
    value_counts还是一个顶级pandas方法，可用于任何数组或序列
    pd.value_counts(obj.values, sort=False)
    ```
  - isin 用于判断矢量化集合的成员资格，可用于选取Series中或DataFrame列中数据的子集
    ```python
    mask = obj.isin(['b', 'c'])
    obj[mask]
    Out[83]:
    0  c
    5  b
    6  b
    7  c
    8  c
    dtype: object
    ```
  - 将pandas.value_counts传给DataFrame的apply函数，得到DataFrame中多个相关列的一张柱状图
    ```python
    data = DataFrame({'Qu1': [1, 3, 4, 3, 4], 'Qu2': [2, 3, 1, 2, 3], 'Qu3': [1, 5, 2, 4, 4]})
    data
    Out[85]:
      Qu1 Qu2 Qu3
    0  1  2  1
    1  3  3  5
    2  4  1  2
    3  3  2  4
    4  4  3  4

    data.apply(pd.value_counts).fillna(0)
    Out[87]:
      Qu1 Qu2 Qu3
    1 1.0 1.0 1.0
    2 0.0 2.0 1.0
    3 2.0 2.0 0.0
    4 2.0 0.0 2.0
    5 0.0 0.0 1.0
    ```
## 数据对齐和处理缺失数据 (isnull / notnull / dropna / fillna / DataFrame和Series之间的运算)
  - pandas对象上的所有描述统计都排除了缺失数据
  - isnull / notnull 返回是否是空值
    ```python
    Python内置的None值也会被当做NA处理

    string_data = Series(['aardvark', 'artichoke', np.nan, None])
    string_data.isnull()
    Out[91]:
    0  False
    1  False
    2   True
    3   True
    dtype: bool
    ```
  - Series.dropna 滤除缺失数据
    ```python
    对于一个Series，dropna返回一个仅含非空数据和索引值的Series
    string_data.dropna()        # string_data[string_data.notnull()] 返回类似的结果
    Out[92]:
    0   aardvark
    1  artichoke
    dtype: object
    ```
  - DataFrame.dropna 滤除缺失数据
    ```python
    对于DataFrame对象，dropna默认丢弃任何含有缺失值的行
    dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

      how : {'any', 'all'} all将只丢弃全为NA的那些行，any丢弃任何含有NA的行
      axis : {0 or 'index', 1 or 'columns'} 指定丢弃行 / 列
      thresh : int, default None 保留包含n个非NA值的行
      subset : array-like
      inplace : boolean, default False，置为True时在原数据上修改，返回None
    ```
  - fillna填补缺失数据
    ```python
    fillna默认会返回新对象，但也可以指定inplace=True对现有对象进行就地修改

    可以指定替换为常数值
    data = Series([1., np.nan, 3.5, np.nan, 7])
    data.fillna(data.mean())        # 替换为平均值

    df = DataFrame(np.random.randn(6, 3))
    df.ix[2:, 1] = np.nan; df.ix[4:, 2] = np.nan
    df.fillna(0)

    若是通过一个字典调用fillna，就可以实现对不同的列填充不同的值
    df.fillna({1: 0.5, 3: -1})        # 列名1的替换为0.5，列名3的替换为-1

    对reindex有效的那些插值方法也可用于fillna
    df.fillna(method='ffill', limit=2)
    ```
  - 在算术方法中填充值
    ```python
    在对不同索引的对象进行算术运算时，当一个对象中某个轴标签在另一个对象中找不到时填充一个特殊值
    add / sub / mul / div 算数方法以及reindex重新索引时，都可以指定一个填充值

    df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
    df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
    df1.add(df2, fill_value=0)        # df1中不存在的值会填入0
    Out[7]:
       a   b   c   d   e
    0  0.0  2.0  4.0  6.0  4.0
    1  9.0 11.0 13.0 15.0  9.0
    2 18.0 20.0 22.0 24.0 14.0
    3 15.0 16.0 17.0 18.0 19.0
    ```
  - DataFrame和Series之间的运算
    ```python
    默认情况下，DataFrame和Series之间的算术运算会将Series的索引匹配到DataFrame的列，然后沿着行一直向下广播

    如果某个索引值在DataFrame的列或Series的索引中找不到，则参与运算的两个对象就会被重新索引以形成并集
    frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
    series2 = Series(range(3), index=['b', 'e', 'f'])
    frame + series2
    Out[16]:
         b  d   e  f
    Utah  0.0 NaN  3.0 NaN
    Ohio  3.0 NaN  6.0 NaN
    Texas  6.0 NaN  9.0 NaN
    Oregon 9.0 NaN 12.0 NaN

    如果希望匹配行且在列上广播，则必须使用算术运算方法，并指定轴号
    series3 = frame['d']
    frame + series3
    Out[20]:
        Ohio Oregon Texas Utah  b  d  e
    Utah   NaN   NaN  NaN  NaN NaN NaN NaN
    Ohio   NaN   NaN  NaN  NaN NaN NaN NaN
    Texas  NaN   NaN  NaN  NaN NaN NaN NaN
    Oregon  NaN   NaN  NaN  NaN NaN NaN NaN

    frame.sub(series3, axis=0)
    Out[22]:
         b  d  e
    Utah  -1.0 0.0 1.0
    Ohio  -1.0 0.0 1.0
    Texas -1.0 0.0 1.0
    Oregon -1.0 0.0 1.0
    ```
## 排序 (sort_index / order / by)
  - sort_index方法，对行或列索引进行排序，返回一个已排序的新对象
    ```python
    frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
    frame.sort_index()        # index排序
    ```
    数据默认是按升序排序的，但可以通过 ascending=False 降序排序
    ```python
    frame.sort_index(axis=1, ascending=False)        # column排序
    ```
  - Series上按值进行排序，可使用其order方法，任何缺失值默认都会被放到Series的末尾
    ```python
    obj = Series([4, np.nan, 7, np.nan, -3, 2])
    obj.order
    ```
    DataFrame上，通过by选项指定根据一个或多个列中的值进行排序
    ```python
    frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
    frame.sort_index(by=['a', 'b'])
    ```
## 排名（ranking）
  - 跟排序关系密切，且它会增设一个排名值，Series中排名是值在排序后的序号
  - 可以根据某种规则破坏平级关系，默认情况下，rank是通过“为各组分配一个平均排名”的方式破坏平级关系
    ```python
    obj = Series([7, -5, 7, 4, 2, 0, 4])
    obj.rank()
    Out[58]:
    0  6.5
    1  1.0
    2  6.5
    3  4.5
    4  3.0
    5  2.0
    6  4.5
    dtype: float64
    ```
  - 可以根据值在原数据中出现的顺序给出排名
    ```python
    obj.rank(method='first')
    ```
  - 可以按降序进行排名
    ```python
    obj.rank(ascending=False, method='max')
    Out[61]:
    0  2.0
    1  7.0
    2  2.0
    3  4.0
    4  5.0
    5  6.0
    6  4.0
    ```
  - DataFrame可以通过axis参数指定在行或列上计算排名
    ```python
    frame = DataFrame(np.arange(12).reshape(3, 4), columns=list('abcd'))
    frame.rank(axis=1)
    ```
## 整数索引 与 loc / at / iloc / iat方法
  - 为了保持良好的一致性，如果轴索引含有索引器，那么根据整数进行数据选取的操作将总是面向标签的，这包括用ix进行切片
    ```python
    ser = Series([0, 1, 1], index = [0, 1, 2])
    ser
    Out[160]:
    0  0
    1  1
    2  1
    dtype: int64

    ser[:1]
    Out[161]:
    0  0
    dtype: int64

    ser.ix[:1]        # 面向标签的索引包括尾节点
    Out[162]:
    0  0
    1  1
    dtype: int64
    ```
  - loc / at 方法：
    ```python
    loc方法使用字符串类型参数，选取 连续 / 不连续的 行/ 列
    at方法使用字符串类型参数，选取某一坐标点的数据
    loc选取的范围包括开始与结束的位置

    df.loc['2017-02-15':'2017-02-17',]
    df.loc[df.index[1],'high']
    df.loc[df.index[1]:df.index[5],'open':'high']
    df.loc[ [df.index[1],df.index[5]],['open', 'high'] ]        # 选取1，5两行中的'open', 'high'列
    df.loc[df.index[1]:df.index[5], :df.columns[2]]

    df.at[df.index[1], 'open']
    ```
  - iloc / iat方法：
    ```python
    loc / at的int方法
    iloc方法选取的范围包括开始的位置，不包括结尾的位置

    df.iloc[1:5, 0:2]
    df.iloc[1:5, [0, 2， 3]]
    df.iloc[ [1, 5], [0, 2] ]

    df.iat[1, 2]
    ```
***

# Series / DataFrame 层次化索引 (MultiIndex / swaplevel / sortlevel / 根据级别汇总统计 / set_index / reset_index / stack / unstack / pivot)
## 层次化索引与MultiIndex
  - 层次化索引在数据重塑和基于分组的操作（如透视表生成）中扮演着重要的角色
  - 层次化索引（hierarchical indexing）能在一个轴上拥有多个（两个以上）索引级别
    ```python
    dates = pd.date_range('20170508', periods=3)
    dates = dates.append(dates)
    items = list('abc') * 2
    items.sort()
    cont = np.arange(6)

    ds = Series(cont, index = [items, dates])
    ds.index.names = ['date', 'item']        # 层次化索引的MultiIndex类使用names
    ds
    Out[117]:
    date item
    a 2017-05-08  0
      2017-05-09  1
    b 2017-05-10  2
      2017-05-08  3
    c 2017-05-09  4
      2017-05-10  5
    dtype: int64

    ds.index
    Out[118]:
    MultiIndex(levels=[ ['a', 'b', 'c'], [2017-05-08 00:00:00, 2017-05-09 00:00:00, 2017-05-10 00:00:00] ],
          labels=[ [0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2] ])
    ```
  - 基本数据索引
    ```python
    data['b']
    data[ ['b', 'a'] ]
    data['b' : 'c']
    ds['b', '2017-05-10']
    ds[:, '2017-05-10']
    ```
  - DataFrame每条轴都可以有分层索引
    ```python
    frame = DataFrame(np.arange(12).reshape((4, 3)),
        index=[ ['a', 'a', 'b', 'b'], [1, 2, 1, 2] ],
        columns=[ ['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green'] ])
    ```
  - MultiIndex
    ```python
    可以单独创建MultiIndex然后复用
    MultiIndex的名称选项是names

    创建层次化索引的Series
    multi_index =pd.MultiIndex.from_arrays([dates, items], names=['date', 'item'])
    ds = Series(cont, index=multi_index)

    上面那个DataFrame中的（分级的）列可以这样创建：
    pd.MultiIndex.from_arrays([ ['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green'] ], names=['state', 'color'])
    ```
## 重排分级顺序 swaplevel
  - 接受两个级别编号或名称，并返回一个互换了级别的新对象
    ```python
    frame.swaplevel(0, 1)
    Out[141]:
       Ohio   Colorado
      Green Red  Green
    1 a   0  1    2
    2 a   3  4    5
    1 b   6  7    8
    2 b   9 10    11
    ```
## 数据排序 sortlevel
  - 根据单个级别中的值对数据进行排序（稳定的）
  - 交换级别swaplevel时，常常也会用到sortlevel，这样最终结果就是有序的了
    ```python
    frame.sortlevel(1)
    frame.swaplevel(0, 1).sortlevel(0)
    Out[145]:
       Ohio   Colorado
      Green Red  Green
    1 a   0  1    2
     b   6  7    8
    2 a   3  4    5
     b   9 10    11
    ```
## 根据级别汇总统计
  - 许多对DataFrame和Series的描述和汇总统计都有一个level选项，它用于指定在某条轴上求和的级别
    ```python
    frame.sum(level=1)
    Out[146]:
      Ohio   Colorado
     Green Red  Green
    1   6  8    10
    2  12 14    16

    这其实是利用了pandas的groupby功能
    ```
## 使用DataFrame的列作为索引 (set_index / reset_index)
  - DataFrame的set_index函数会将其一个或多个列转换为行索引，并创建一个新的DataFrame
    ```python
    frame = DataFrame({'a': range(7), 'b': range(7, 0, -1),
        'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
        'd': [0, 1, 2, 0, 1, 2, 3]})
    frame2 = frame.set_index(['c', 'd'])
    ```
  - 默认情况下，那些列会从DataFrame中移除，但也可以指定drop=False将其保留下来
    ```python
    frame.set_index(['c', 'd'], drop=False)
    ```
  - reset_index的功能跟set_index刚好相反，层次化索引的级别会被转移到列里面
    ```python
    frame2.reset_index()
    ```
## stack / unstack 旋转层次化索引的轴，转换Series / DataFrame
  - stack：将数据的列“旋转”为行，将DataFrame压缩成Series
  - unstack：将数据的行“旋转”为列，将层次化结构的Series扩展成DataFrame
  - stack方法 将DataFrame列转换为行，得到一个Series
    ```python
    data = DataFrame(np.arange(6).reshape((2, 3)), index=pd.Index(list('ab'), name='state'), columns=pd.Index(list('xyz'), name='number'))
    data
    Out[43]:
    number x y z
    state     
    a    0 1 2
    b    3 4 5

    data.stack()
    Out[44]:
    state number
    a   x     0
        y     1
        z     2
    b   x     3
        y     4
        z     5
    dtype: int64
    ```
  - unstack方法 将层次化索引的Series重排为一个DataFrame
    ```python
    r = data.stack()
    r.unstack()
    Out[11]:
    number x y z
    state     
    a    0 1 2
    b    3 4 5
    ```
  - stack / unstack 默认操作的是最内层，传入分层级别的编号或名称即可对其他级别进行unstack操作
    ```python
    r.unstack()        # --> r.unstack(1) / r.unstack('number')
    r.unstack(0)        # --> r.unstack('state')
    Out[12]:
    state  a b
    number   
    x    0 3
    y    1 4
    z    2 5
    ```
## stack / unstack操作中的缺失值
  - unstack操作，当不是所有的级别值都能在各分组中找到时，可能会引入缺失数据
    ```python
    s1 = Series([0, 1, 2, 3], index=list('abcd'))
    s2 = Series([4, 5, 6], index=list('cde'))
    data2 = pd.concat([s1, s2], keys=['one', 'two'])
    data2
    Out[18]:
    one a  0
       b  1
       c  2
       d  3
    two c  4
       d  5
       e  6
    dtype: int64

    data2.unstack()
    Out[19]:
        a  b  c  d  e
    one 0.0 1.0 2.0 3.0 NaN
    two NaN NaN 4.0 5.0 6.0
    ```
  - stack默认会滤除缺失数据，因此该运算是可逆的
    ```python
    data2.unstack().stack()
    Out[20]:
    one a  0.0
       b  1.0
       c  2.0
       d  3.0
    two c  4.0
       d  5.0
       e  6.0
    dtype: float64

    使用data2.unstack().stack(dropna=False)，将保留缺失值
    ```
## DataFrame进行unstack操作
  - 作为旋转轴的级别将会成为结果中的最低级别
  - 对于层次化索引的DataFrame，可以交换索引轴的位置
    ```python
    df = DataFrame({'left': r, 'right': r + 5}, columns=pd.Index(['left', 'right'], name='side'))
    df
    Out[24]:
    side     left right
    state number       
    a   x     0   5
       y     1   6
       z     2   7
    b   x     3   8
       y     4   9
       z     5   10

    df.unstack('state')        # 使用行索引的'state'作为旋转轴，将成为列名的内侧
    Out[25]:
    side  left  right  
    state   a b   a  b
    number         
    x     0 3   5  8
    y     1 4   6  9
    z     2 5   7 10

    df.unstack('state').stack('side')        # 将原先的列名旋转到行索引，并作为内侧索引
    Out[26]:
    state     a  b
    number side    
    x   left  0  3
        right 5  8
    y   left  1  4
        right 6  9
    z   left  2  5
        right 7 10
    ```
## pivot转换方法，使用原有数据创建新的DataFrame
  - pivot(self, index=None, columns=None, values=None)
    ```python
    前两个参数值分别用作行和列索引的列名，最后一个参数值则是用于填充DataFrame的数据列的列名

    dates = pd.date_range('20170508', periods=3)
    dates = dates.append(dates)
    items = list('abc') * 2
    items.sort()
    cont = np.arange(6)
    df = DataFrame({'date':dates, 'item':items, 'cont':cont, 'cont2':cont*2})
    df
    Out[122]:
      cont cont2    date item
    0   0   0 2017-05-08  a
    1   1   2 2017-05-09  a
    2   2   4 2017-05-10  a
    3   3   6 2017-05-08  b
    4   4   8 2017-05-09  b
    5   5   10 2017-05-10  b

    df.pivot('date', 'item', 'cont')        # date列做索引，item列做列名，cont列做填充值
    Out[123]:
    item     a  b  c
    date           
    2017-05-08 0.0 3.0 NaN
    2017-05-09 1.0 NaN 4.0
    2017-05-10 NaN 2.0 5.0

    df.pivot('date', 'item')        # 不指定value时，将使用所有值创建层次化索引
    Out[124]:
          cont  cont2  
    item     a b   a  b
    date            
    2017-05-08  0 3   0  6
    2017-05-09  1 4   2  8
    2017-05-10  2 5   4 10
    ```
  - set_index创建层次化索引，再用unstack重塑也可以实现
    ```python
    df.set_index(['date', 'item']).unstack('item')
    Out[126]:
          cont     
    item     a  b  c
    date           
    2017-05-08 0.0 3.0 NaN
    2017-05-09 1.0 NaN 4.0
    2017-05-10 NaN 2.0 5.0
    ```
***

# 数据存取 （文本文档 / 二进制文件 / HDF5 / Excel / SQL数据库 / MongoDB）
## pandas读写表格型文件 read_csv / read_table / 缺失值处理 / 逐块读取 / from_csv
  - csv 逗号分离值 comma separated values
  - pandas提供了一些用于将表格型数据读取为DataFrame对象的函数
    ```python
    read_csv        从文件 / URL / 文件型对象加载逗号分隔的csv文件
    read_table        从文件 / URL / 文件型对象加载制表符分隔的tsv文件
    read_fwf        读取没有分隔符的定宽列格式数据
    read_clipboard        读取剪贴板中的数据
    ```
  - 函数选项分类
    ```python
    索引：将一个或多个列当做返回的DataFrame处理，以及是否从文件、用户获取列名。
    类型推断和数据转换：包括用户定义值的转换、缺失值标记列表等。
    日期解析：包括组合功能，比如将分散在多个列中的日期时间信息组合成结果中的单个列。
    迭代：支持对大文件进行逐块迭代。
    不规整数据问题：跳过一些行、页脚、注释或其他一些不重要的东西
    ```
  - read_csv / read_tables函数常用参数
    ```python
    path        表示文件系统的位置 / URL / 文件型对象的字符串
    header        用作列名的行号，默认为0，没有标题行应设置为None
    sep / delimiter        指定作为分隔符的字符序列或正则表达式
    index_col        用作索引的列编号或列名
    names        用于结果的列名列表，结合header=None
    skiprows        需要忽略的行数或行号列表
    na_values        指定作为缺失值处理的字符串列表
    parse_dates        尝试将数据解析为日期
    nrows        需要读取的行数
    chunksize        用于迭代时文件块的大小
    ```
  - read_csv / read_table读取文本文件
    ```python
    df = pd.read_csv('practice_data/ex1.csv')        # 默认使用第一行作为标题行
    pd.read_table('practice_data/ex1.csv', sep=',')        # 指定分隔符
    pd.read_csv('practice_data/ex2.csv', header=None)        # 如果文件没有标题行，指定自动分配标题
    pd.read_csv('practice_data/ex2.csv', names=list('abcd')+['message'], index_col='message')        # 指定标题名，以及作为index使用的列
    pd.read_csv('practice_data/csv_mindex.csv', index_col=['key1', 'key2'])        # 指定两个列作为层次化索引
    pd.read_table('practice_data/ex3.txt', sep='\s+')        # 分隔符是数量不等的空格，使用正则表达式指定多个空格
    pd.read_csv('practice_data/ex4.csv', skiprows=[0, 2, 3])        # 指定跳过 0, 2, 3 行
    ```
  - na_values 缺失值处理：
    ```python
    缺失数据经常是要么没有（空字符串），要么用某个标记值表示
    默认情况下，pandas会用一组经常出现的标记值进行识别，如NA、-1.#IND以及NULL等
    pd.read_csv('practice_data/ex5.csv', na_values=['foo'])        # 指定其他作为缺失值处理的字符串
    sentinels = {'something':['two'], 'message':['foo']}
    pd.read_csv('practice_data/ex5.csv', na_values=sentinels)        # 使用字典，为不同的列指定不同的缺失值字符串
    ```
  - chunksize 逐块读取文本文件
    ```python
    pd.read_csv('practice_data/ex6.csv', nrows=5)        # 指定读取前5行

    # 要逐块读取文件，需要设置chunksize（行数）
    chunker = pd.read_csv('practice_data/ex6.csv', chunksize=1000)        # 根据chunksize对文件进行逐块迭代
    # 迭代处理ex6.csv，将值计数聚合到"key"列中
    tot = Series([])
    tot = Series([])
    for piece in chunker:
      tot = tot.add(piece['key'].value_counts(), fill_value=0)
    tot = tot.sort_values(ascending=False)

    TextParser还有一个get_chunk方法，可以读取任意大小的块
    ```
  - to_csv 将数据写出到文本格式
    ```python
    data.to_csv('practice_data/out.csv')        # 将DataFrame输出到csv文件中
    data.to_csv(sys.stdout, sep='|')        # 写入到stdout，使用|作为分隔符

    缺失值在输出结果中会被表示为空字符串
    data.to_csv(sys.stdout, na_rep='NULL')        # 指定缺失值标记
    data.to_csv(sys.stdout, index=False, header=False)        # 写入时禁用 行列标签
    data.to_csv(sys.stdout, index=False, cols=['a', 'b', 'c'])        # 指定写入的列
    ```
  - from_csv
    ```python
    pd.DataFrame.from_csv(path) 等同于 pd.read_csv(path, index_col=0, parse_dates=True)
    更适用于DataFrame / Series使用日期的数据

    dates = pd.date_range('1/1/2000', periods=7)
    ts = Series(np.arange(7), index=dates)
    ts.to_csv('practice_data/tseries.csv')
    Series.from_csv('practice_data/tseries.csv')
    ```
## 手工处理分隔符格式 csv.reader / csv.writer
  - 对于任何单字符分隔符文件，可以直接使用Python内置的csv模块，将任意已打开的文件或文件型的对象传给csv.reader
    ```python
    import csv
    f = open('practice_data/ex7.csv')
    reader = csv.reader(f)
    # 为了使数据格式合乎要求，需要对其做一些整理工作
    lines = list(csv.reader(open('practice_data/ex7.csv')))
    header, values = lines[0], lines[1:]
    data_dict = {h: v for h, v in zip(header, zip(*values))}
    ```
  - CSV文件的形式有很多，通过定义csv.Dialect的一个子类即可定义出新格式（如专门的分隔符、字符串引用约定、行结束符等）：
    ```python
    class my_dialect(csv.Dialect):
      lineterminator = '\n'
      delimiter = ';'
      quotechar = '"'

    reader = csv.reader(f, diaect=my_dialect)
    ```
  - csv.writer用于手工输出分隔符文件
    ```python
    接受一个已打开且可写的文件对象以及跟csv.reader相同的那些语支和格式化选项：
    with open('mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))
    ```
## 二进制文本文件读写 (pickle / save / load)
  - 实现数据的二进制格式存储最简单的办法之一是使用Python内置的pickle序列化
  - pickle存储方式默认是二进制方式，python3中与文件交互需要指定'wb' / 'rb'
  - pandas对象都有一个用于将数据以pickle形式保存到磁盘上的save方法
    ```python
    frame = pd.read_csv('practice_data/ex1.csv')
    frame.save('practice_data/frame_pickle')
    ```
  - pandas.load将数据读回到Python
    ```python
    pd.load('practice_data/frame_pickle')
    pickle仅建议用于短期存储格式。其原因是很难保证该格式永远是稳定的
    ```
## HDF5格式 (HDFStore)
  - HDF5（hierarchical data format） 层次型数据格式
    ```python
    很多工具都能实现高效读写磁盘上以二进制格式存储的科学数据
    HDF5是其中一个流行的工业级库，它是一个C库，带有许多语言的接口，如Java、Python和MATLAB等
    每个HDF5文件都含有一个文件系统式的节点结构，它能够存储多个数据集并支持元数据
    与其他简单格式相比，HDF5支持多种压缩器的即时压缩，还能更高效地存储重复模式数据
    对于那些非常大的无法直接放入内存的数据集，HDF5就是不错的选择，因为它可以高效地分块读写
    ```
  - 由于许多数据分析问题都是IO密集型（而不是CPU密集型），利用HDF5这样的工具能显著提升应用程序的效率
  - HDF5不是数据库，它最适合用作“一次写多次读”的数据集，虽然数据可以在任何时候被添加到文件中，但如果同时发生多个写操作，文件就可能会被破坏
  - Python中的HDF5库有两个接口 PyTables 和 h5py，各自采取了不同的问题解决方式
    ```python
    h5py提供了一种直接而高级的HDF5API访问接口
    PyTables则抽象了HDF5的许多细节以提供多种灵活的数据容器、表索引、查询功能以及对核外计算技术（out-of-core computation）的某些支持
    ```
  - pandas有一个最小化的类似于字典的HDFStore类，它通过PyTables存储pandas对象
    ```python
    store = pd.HDFStore('mydata.h5')
    store['obj1'] = frame
    store['obj1_col'] = frame['a']
    store
    Out[82]:
    <class 'pandas.io.pytables.HDFStore'>
    File path: mydata.h5
    /obj1        frame    (shape->[3,5])
    /obj1_col      series    (shape->[3])
    ```
  - HDF5文件中的对象可以通过与字典一样的方式进行获取
    ```python
    store['obj1']
    Out[83]:
      a  b  c  d message
    0 1  2  3  4  hello
    1 5  6  7  8  world
    2 9 10 11 12   foo
    ```
## 读取 (read_excel / to_excel / ExcelFile类)
  - read_excel / to_excel
    ```python
    d = pd.read_excel('foo.xls')
    d.to_excel('foo.xls')

    d = pd.read_excel('foo.xls')
    d
      number name python math
    0  1001  xm   77  87
    1  1002  xh   88  82
    2  1003  wh   99  91
    s = d.python + d.math
    d['sum_score'] = s
    d
      number name python math sum_score
    0  1001  xm   77  87    164
    1  1002  xh   88  82    170
    2  1003  wh   99  91    190
    d.to_excel('foo.xls')
    ```
  - pandas的ExcelFile类支持读取存储在Excel 2003（或更高版本）中的表格型数据
    ```python
    # 通过传入一个xls或xlsx文件的路径即可创建一个ExcelFile实例
    xls_file = pd.ExcelFile('practice_data/ex1.xlsx')
    xls_file.sheet_names
    Out[89]: ['Sheet1']

    # 存放在某个工作表中的数据可以通过parse读取到DataFrame中
    xls_file.parse('Sheet1')
    Out[92]:
      a  b  c  d message
    0 1  2  3  4  hello
    1 5  6  7  8  world
    2 9 10 11 12   foo
    ```
## SQL数据库 (sqlite3 / read_sql)
  - 将数据从SQL加载到DataFrame
    ```python
    # 通过sqlite3创建一个嵌入式的SQLite数据库
    import sqlite3
    query = """
      CREATE TABLE test
      (a VARCHAR(20), b VARCHAR(20),
      c REAL,        d INTEGER
      );"""
    con = sqlite3.connect(':memory:')
    con.execute(query)
    con.commit()

    # 插入几行数据
    data = [('Atlanta', 'Georgia', 1.25, 6),
      ('Tallahassee', 'Florida', 2.6, 3),
      ('Sacramento', 'California', 1.7, 5)]
      stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
    con.executemany(stmt, data)
    con.commit()

    # 从表中选取数据
    # 大部分Python SQL驱动器(PyODBC / psycopg2 / MySQLdb / pymssql等)都会返回一个元组列表
    cursor = con.execute('select * from test')
    rows = cursor.fetchall()
    rows
    Out[32]:
    [('Atlanta', 'Georgia', 1.25, 6),
     ('Tallahassee', 'Florida', 2.6, 3),
     ('Sacramento', 'California', 1.7, 5)]

    # 列名位于游标的description属性中
    cursor.description
    Out[33]:
    (('a', None, None, None, None, None, None),
     ('b', None, None, None, None, None, None),
     ('c', None, None, None, None, None, None),
     ('d', None, None, None, None, None, None))

    # 将这个元组列表传给DataFrame的构造器
    DataFrame(rows, columns=list(zip(*cursor.description))[0])        # python3中zip返回的对象不能直接使用索引
    ```
  - pandas有一个可以简化该过程的read_sql函数（位于pandas.io.sql模块），只需传入select语句和连接对象即可
    ```python
    import pandas.io.sql as sql
    sql.read_sql('select * from test', con)        # python3中使用read_sql替代read_frame
    ```
  - **连接 MySQL 数据库**
    ```shell
    conda install -c conda-forge MySQL-python
    ```
    ```python
    import MySQLdb

    # 连接，类似于 mysql -h 127.0.0.1 -P 3306 -D foo -u root -p123456
    conn = MySQLdb.connect(host='127.0.0.1',port=3306,user='root',passwd='123456',db='foo',charset='utf8', connect_timeout=5)
    ```
    **使用 `mysql-connector-python`**
    ```shell
    pip install mysql-connector-python
    ```
    ```python
    import mysql.connector
    conn = mysql.connector.connect(host='127.0.0.1',port=3306,user='root',passwd='123456',db='foo',charset='utf8', connect_timeout=5)
    ```
    **操作**
    ```python
    # 使用 cursor
    cur = conn.cursor()
    cur.execute("select * from websites")
    aa = cur.fetchall()
    DataFrame(list(aa), columns=list(zip(*cur.description))[0])
    cur.close()

    # 使用 read_sql
    sql.read_sql('select * from websites', conn)
    conn.close()
    ```
## 存取MongoDB中的数据
  - NoSQL数据库有许多不同的形式
    ```python
    有些是简单的字典式键值对存储（如BerkeleyDB和Tokyo Cabinet）
    另一些则是基于文档的（其中的基本单元是字典型的对象）
    ```
  - MongoDB（http://mongodb.org），可以先在自己的电脑上启动一个MongoDB实例，然后用pymongo（MongoDB的官方驱动器）通过默认端口进行连接：
    ```python
    import pymongo
    con = pymongo.Connection('localhost', port=27017)
    ```
    存储在MongoDB中的文档被组织在数据库的集合（collection）
  - MongoDB服务器的每个运行实例可以有多个数据库，而每个数据库又可以有多个集合
  - 假设要保存之前通过Twitter API获取的数据，首先，可以访问tweets集合（暂时还是空的）：
    ```python
    tweets = con.db.tweets
    ```
    然后，将那组tweet加载进来并通过tweets.save（用于将Python字典写入MongoDB）逐个存入集合中：
    ```python
    import requests, json
    url = 'http://search.twitter.com/search.json?q=python%20pandas'
    data = json.loads(requests.get(url).text)

    for tweet in data['results']:
      tweets.save(tweet)
    ```
    现在，如果想从该集合中取出自己发的tweet（如果有的话），可以用下面的代码对集合进行查询：
    ```python
    cursor = tweets.find({'from_user': 'wesmckinn'})
    ```
    返回的游标是一个迭代器，它可以为每个文档产生一个字典，可以将其转换为一个DataFrame，还可以只获取各tweet的部分字段：
    ```python
    tweet_fields = ['created_at', 'from_user', 'id', 'text']
    result = DataFrame(list(cursor), columns=tweet_fields)
    ```
***

# 网络相关数据处理 (json / urllib / request / html / xml)
## json库读取JSON数据
  - JSON（JavaScript Object Notation的简称）已经成为通过HTTP请求在Web浏览器和其他应用程序之间发送数据的标准格式之一，是一种比表格型文本格式（如CSV）灵活得多的数据格式
  - JSON非常接近于有效的Python代码，基本类型有对象（字典）、数组（列表）、字符串、数值、布尔值以及null，对象中所有的键都必须是字符串
  - 许多Python库都可以读写JSON数据
    ```python
    obj = """
    {"name": "Wes",
     "places_lived": ["United States", "Spain", "Germany"],
     "pet": null,
     "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
             {"name": "Katie", "age": 33, "pet": "Cisco"}]
    }
    """
    ```
  - 通过json.loads即可将JSON字符串转换成Python形式
    ```python
    import json
    result = json.loads(obj)
    json.dumps则将Python对象转换成JSON格式：
    asjson = json.dumps(result)
    ```
  - 将JSON对象转换为DataFrame或其他便于分析的数据结构
    ```python
    可以向DataFrame构造器传入一组JSON对象，并选取数据字段的子集
    siblings = DataFrame(result['siblings'], columns=['name', 'age'])
    siblings
    Out[63]:
      name age
    0 Scott  25
    1 Katie  33
    ```
## Python获取网络数据 (urllib / requests)
  - 抓取网页,解析网页内容
    ```python
    urllib / urllib2 (Python 3中被urllib.request代替)
    httplib / httplib2 (Python 3中被http.client代替)
    ```
  - python3中相关函数
    ```python
    打开 urllib.request.urlopen('URL')
    读取 r.read(), r.readline(), r.readlines()等方法
    关闭 r.close()

    import urllib.request
    r = urllib.request.urlopen('http://www.baidu.com/').read()        # python3中使用urllib.request.urlopen
    with urllib.request.urlopen('http://www.baidu.com') as url:        # 或使用 with 语句，使用完后自动关闭
    ...   r = url.read()
    ...
    r
    ```
  - requests包使用Web API
    ```python
    许多网站都有一些通过JSON或其他格式提供数据的公共API
    一个简单易用的办法（推荐）是requests包（http://docs.python-requests.org）

    发送一个HTTP GET请求，在Twitter上搜索"python pandas"
    import requests
    import json
    url = 'https://api.twitter.com/1.1/search/tweets.json?q=python%20pandas&result_type=recent'
    resp = requests.get(url)
    resp
    Out[947]: <Response [200]>        # 成功返回200，当前使用返回了400

    # Response对象的text属性含有GET请求的内容
    data = json.loads(resp.text)
    data.keys()        # Out[950]: [... u'results',...]

    # 响应结果中的results字段含有一组tweet，每条tweet被表示为一个Python字典
    # 用一个列表定义出感兴趣的tweet字段，然后将results列表传给DataFrame
    tweet_fields = ['created_at', 'from_user', 'id', 'text']
    tweets = DataFrame(data['results'], columns=tweet_fields)
    tweets.ix[7]
    ```
## lxml.html解析html文件 (Yahoo财经数据处理成DataFrame)
  - lxml (http://lxml.de)是一个读写HTML和XML格式数据的库，能够高效且可靠地解析大文件
  - 解析网络获取的yahoo财经数据
    ```python
    # 通过urllib.request.urlopen获取网络数据
    import pandas as pd
    from lxml.html import parse
    from urllib.request import urlopen

    parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
    doc = parsed.getroot()

    # HTML中的链接是a标签,通过该标签得到该文档中所有的URL链接
    links = doc.findall('.//a')
    links[15:20]

    # links中是表示HTML元素的对象，使用各对象的get方法得到URL和text_content方法得到显示文本
    lnk = links[28]
    lnk.get('href')        # Out[9]: '/quote/NFLX?p=NFLX'
    lnk.text_content()

    # 获取文档中的全部URL
    urls = [lnk.get('href') for lnk in doc.findall('.//a')]        # Out[10]: 'NFLX'
    urls[-10:]

    # 从文档中获取表格数据
    tables = doc.findall('.//table')

    calls = tables[1]        # 看涨数据表格
    puts = tables[2]        # 看跌数据表格

    # 每个表格都有一个标题行，然后才是数据行
    rows = calls.findall('.//tr')

    # 获取每个单元格内的文本
    def _unpack(row, kind='td'):
      elts = row.findall('.//%s' % kind)
      return [val.text_content() for val in elts]

    # th单元格对应标题行
    _unpack(rows[0], kind='th')
    Out[41]:
    ['Strike',
     'Contract Name',
     'Last Price',
     'Bid',
     'Ask',
     'Change',
     '% Change',
     'Volume',
     'Open Interest',
     'Implied Volatility']

    # td单元格对应数据行
    _unpack(rows[1], kind='td')
    Out[42]:
    ['2.50',
     'AAPL170519C00002500',
     '153.15',
     '151.95',
     '154.40',
     '8.45',
     '5.84%',
     '4',
     '0',
     '3,003.13%']

    # 由于数值型数据仍然是字符串格式，pandas的TextParser类可用于自动类型转换，将部分列转换为浮点数格式
    # read_csv和其他解析函数其实在内部都用到了它
    from pandas.io.parsers import TextParser

    def parse_options_data(table):
      rows = table.findall('.//tr')
      header = _unpack(rows[0], kind='th')
      data = [_unpack(r) for r in rows[1:]]
      return TextParser(data, names=header).get_chunk()

    # 得到最终的DataFrame
    call_data = parse_options_data(calls)
    put_data = parse_options_data(puts)
    call_data[:5]
    Out[48]:
      Strike    Contract Name Last Price   Bid   Ask Change % Change \
    0   2.5 AAPL170519C00002500   153.15 151.95 154.40  8.45  5.84%  
    1   5.0 AAPL170519C00005000   150.65 149.45 151.65  0.00  0.00%  
    2   7.5 AAPL170519C00007500   148.05 146.95 149.30  0.00  0.00%  
    3  10.0 AAPL170519C00010000   143.53 145.65 146.60  0.00  0.00%  
    4  20.0 AAPL170519C00020000   133.52 135.65 136.60  0.00  0.00%  

     Volume Open Interest Implied Volatility
    0   4       0     3,003.13%
    1   4       0     2,071.88%
    2   3       0     1,817.19%
    3   10       0     1,334.38%
    4   10       0      975.00%
    ```
## lxml.objectify解析XML (地铁资料数据处理成DataFrame)
  - XML（Extensible Markup Language）是一种常见的支持分层、嵌套数据以及元数据的结构化数据格式
  - 解析MNR地铁xml数据
    ```python
    # lxml.objectify解析XML文件
    from lxml import objectify
    path = 'practice_data/mta_perf/Performance_MNR.xml'

    parsed = objectify.parse(open(path))

    # 通过getroot得到该XML文件的根节点的引用
    root = parsed.getroot()

    # 排除的标签
    data = []
    skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ', 'DESIRED_CHANGE', 'DECIMAL_PLACES']

    # root.INDICATOR返回一个用于产生各个<INDICATOR>XML元素的生成器
    for elt in root.INDICATOR:
      el_data = {}
      for child in elt.getchildren():
        if child.tag in skip_fields:
          continue
        el_data[child.tag] = child.pyval
        data.append(el_data)

    # 将这组字典转换为一个DataFrame
    perf = DataFrame(data)
    ```
***

# 数据合并 (merge / join / concat / combine_first)
  - merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False) __
  - join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False)
  - concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
## merge 根据指定的列名 / 索引合并DataFrame重名数据项
  - merge做的是"inner"连接，结果中的键默认是交集inner，其他方式还有"left"、"right"以及"outer"
  - outer外连接求取的是键的并集，组合了左连接和右连接的效果
  - 在进行列－列连接时，DataFrame对象中的索引会被丢弃
  - merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False) __
    ```python
    left / right : 参与合并的左侧 / 右侧DataFrame
    how : {'left', 'right', 'outer', 'inner'}, default 'inner'
    on : 用于连接的列名，必须存在于左右两个DataFrame中，不指定时使用重叠的列名
    left_on / right_on : 左侧 / 右侧DataFrame用于连接的列
    left_index / right_index : boolean, 默认False，将左侧 / 右侧的index作为连接键
    sort : boolean, 默认False，根据连接键对合并后数据排序
    suffixes : 重复列名的后缀，默认是('_x', '_y')
    copy : boolean, 默认True，某些情况下置为False避免复制数据
    indicator : boolean or string, default False
    ```
  - 不指定合并的列名，merge会将重叠列的列名当做键
    ```python
    df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
    df2 = DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})
    pd.merge(df1, df2)
    Out[4]:
      data1 key data2
    0   0  b   1
    1   1  b   1
    2   6  b   1
    3   2  a   0
    4   4  a   0
    5   5  a   0
    ```
  - on 指定用来合并的列名
    ```python
    pd.merge(df1, df2, on='key')
    ```
  - left_on / right_on 指定不同的列名
    ```python
    df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
    df4 = DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
    pd.merge(df3, df4, left_on='lkey', right_on='rkey')
    Out[8]:
      data1 lkey data2 rkey
    0   0  b   1  b
    1   1  b   1  b
    2   6  b   1  b
    3   2  a   0  a
    4   4  a   0  a
    5   5  a   0  a
    ```
  - outer 外连接
    ```python
    pd.merge(df1, df2, how='outer')
    Out[9]:
      data1 key data2
    0  0.0  b  1.0
    1  1.0  b  1.0
    2  6.0  b  1.0
    3  2.0  a  0.0
    4  4.0  a  0.0
    5  5.0  a  0.0
    6  3.0  c  NaN
    7  NaN  d  2.0
    ```
  - 多对多的合并操作，产生的是行的笛卡尔积
    ```python
    df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
    df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'], 'data2': range(5)})
    pd.merge(df1, df2, on='key', how='left')
    Out[12]:
      data1 key data2
    0    0  b  1.0
    1    0  b  3.0
    2    1  b  1.0
    3    1  b  3.0
    4    2  a  0.0
    5    2  a  2.0
    6    3  c  NaN
    7    4  a  0.0
    8    4  a  2.0
    9    5  b  1.0
    10   5  b  3.0
    ```
  - on 指定一个由列名组成的列表，根据多个键进行合并
    ```python
    可以这样来理解：多个键形成一系列元组，并将其当做单个连接键
    left = DataFrame({'key1': ['foo', 'foo', 'bar'], 'key2': ['one', 'two', 'one'], 'lval': [1, 2, 3]})
    right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'], 'key2': ['one', 'one', 'one', 'two'], 'rval': [4, 5, 6, 7]})
    pd.merge(left, right, on=['key1', 'key2'], how='outer')
    Out[15]:
     key1 key2 lval rval
    0 foo one  1.0  4.0
    1 foo one  1.0  5.0
    2 foo two  2.0  NaN
    3 bar one  3.0  6.0
    4 bar two  NaN  7.0
    ```
  - suffixes选项处理重复列名
    ```python
    指定附加到左右两个DataFrame对象的重叠列名上的字符串
    pd.merge(left, right, on='key1').columns
    Out[17]: Index(['key1', 'key2_x', 'lval', 'key2_y', 'rval'], dtype='object')

    pd.merge(left, right, on='key1', suffixes=('_left', '_right')).columns
    Out[18]: Index(['key1', 'key2_left', 'lval', 'key2_right', 'rval'], dtype='object')
    ```
  - left_index / right_index 索引上的合并
    ```python
    left_index=True或right_index=True（或两个都传）以说明索引应该被用作连接键

    left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
    right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
    pd.merge(left1, right1, left_on='key', right_index=True, how='outer')
    Out[22]:
     key value group_val
    0  a   0    3.5
    2  a   2    3.5
    3  a   3    3.5
    1  b   1    7.0
    4  b   4    7.0
    5  c   5    NaN
    ```
  - 层次化索引的数据
    ```python
    以列表的形式指明用作合并键的多个列，注意对重复索引值的处理
    lefth = DataFrame({'key1': list('bbbaa'), 'key2': [2000, 2001, 2002, 2001, 2002], 'data': np.arange(5.)})
    ighth = DataFrame(np.arange(12).reshape(6, 2), index = [list('aabbbb'), [2001, 2000, 2000, 2000, 2001, 2002]], columns=['e1', 'e2'])
    pd.merge(lefth, righth, left_on = ['key1', 'key2'], right_index=True, how='outer')
    Out[26]:
      data key1 key2  e1  e2
    0  0.0  b 2000  4.0  5.0
    0  0.0  b 2000  6.0  7.0
    1  1.0  b 2001  8.0  9.0
    2  2.0  b 2002 10.0 11.0
    3  3.0  a 2001  0.0  1.0
    4  4.0  a 2002  NaN  NaN
    4  NaN  a 2000  2.0  3.0
    ```
## join DataFrame索引上的合并
  - 可用于合并多个带有相同或相似索引的DataFrame对象，而不管它们之间有没有重叠的列，DataFrame的join方法是在连接键上做左连接
  - join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False)
  - 使用索引合并
    ```python
    caller = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
    other = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'B': ['B0', 'B1', 'B2']})
    caller.join(other, lsuffix='_caller', rsuffix='_other')        # 列名有重复时，必须指定lsuffix / rsuffix
    Out[60]:
      A key_caller  B key_other
    0 A0     K0  B0    K0
    1 A1     K1  B1    K1
    2 A2     K2  B2    K2
    3 A3     K3 NaN    NaN
    4 A4     K4 NaN    NaN
    5 A5     K5 NaN    NaN
    ```
  - set_index 使用'key'列合并
    ```python
    other.set_index('key').join(caller.set_index('key'))
    Out[63]:
       B  A
    key    
    K0  B0 A0
    K1  B1 A1
    K2  B2 A2
    ```
  - on选项指定调用者用于合并的列
    ```python
    DataFrame.join合并时使用other的index，但on选项可以指定调用者用于合并的列
    caller.join(other.set_index('key'), on='key')
    Out[65]:
      A key  B
    0 A0 K0  B0
    1 A1 K1  B1
    2 A2 K2  B2
    3 A3 K3 NaN
    4 A4 K4 NaN
    5 A5 K5 NaN
    ```
  - 当other指定成一个DataFrame列表时，on / lsuffix / rsuffix不起作用
    ```python
    other2 = pd.DataFrame({'key2': ['K0', 'K1', 'K2'], 'B2': ['B0', 'B1', 'B2']})
    caller.columns=['A', 'key_caller']
    caller.join([other, other2])
    ```
## concat Series / DataFrame 横轴或纵轴上的数据堆叠
  - 这种数据合并运算也被称作连接concatenation \ 绑定binding \ 堆叠stacking
  - concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
  - 数据堆叠，不论有没有重复列
    ```python
    s1 = Series([10, 11], index=list('ab'))
    s2 = Series([22, 23, 24], index=list('bfg'))
    s3 = Series([35, 36], index=list('fg'))
    pd.concat([s1, s2, s3])
    Out[98]:
    a  10
    b  11
    b  22
    f  23
    g  24
    f  35
    g  36
    dtype: int64
    ```
  - axis 指定合并轴
    ```python
    pd.concat([s1, s2, s3], axis=1)
    Out[99]:
       0   1   2
    a 10.0  NaN  NaN
    b 11.0 22.0  NaN
    f  NaN 23.0 35.0
    g  NaN 24.0 36.0
    ```
  - join 指定合并方式，默认是outer
    ```python
    s4 = pd.concat([s1 * 4, s3 + 10])
    pd.concat([s1, s4], axis=1, join='inner')
    Out[107]:
      0  1
    a 10 40
    b 11 44
    ```
  - join_axes 指定要在其他轴上使用的索引
    ```python
    没有的索引值为NaN
    pd.concat([s1, s4], axis=1, join_axes=[ ['a', 'c', 'b', 'e'] ])
    Out[108]:
       0   1
    a 10.0 40.0
    c  NaN  NaN
    b 11.0 44.0
    e  NaN  NaN
    ```
  - ignore_index=True 忽略原有的索引
    ```python
    pd.concat([s1, s2, s3], ignore_index=True).index        # 丢弃原索引，重新改为0, 1, 2, ...
    Out[142]: RangeIndex(start=0, stop=7, step=1)

    pd.concat([s1, s2, s3], axis=1, ignore_index=True).index        # axis=1时，不再影响index，而影响columns
    Out[143]: Index(['a', 'b', 'f', 'g'], dtype='object')
    ```
  - keys 指定最外层的索引，创建层次化索引的结果
    ```python
    r = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
    r
    Out[110]:
    one  a  10
        b  11
    two  a  10
        b  11
    three f  35
        g  36
    dtype: int64

    r.unstack()        # 将Series扩展成DataFrame
    Out[153]:
         a   b   f   g
    one  10.0 11.0  NaN  NaN
    two  10.0 11.0  NaN  NaN
    three  NaN  NaN 35.0 36.0
    ```
  - 如果沿着axis=1对Series进行合并，则keys就会成为DataFrame的列头
    ```python
    pd.concat([s1, s1, s3], axis = 1, keys=['one', 'two', 'three'])
    Out[156]:
      one  two three
    a 10.0 10.0  NaN
    b 11.0 11.0  NaN
    f  NaN  NaN  35.0
    g  NaN  NaN  36.0
    ```
  - 使用字典参数，字典的键会被当做keys选项的值
    ```python
    pd.concat({'one':s1, 'two':s1, 'three':s3})
    pd.concat({'one':s1, 'two':s1, 'three':s3}, axis=1)
    ```
  - names 创建层次化索引时，各层索引的名称
    ```python
    pd.concat({'one':s1, 'two':s1, 'three':s3}, names=['k1', 'k2'])
    Out[176]:
    k1   k2
    one  a   10
        b   11
    three f   35
        g   36
    two  a   10
        b   11
    dtype: int64
    ```
## DataFrame分别使用 join / merge / concat
  - df1 / df2
    ```python
    df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'], columns=['one', 'two'])
    df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'], columns=['three', 'four'])

    df1
    Out[160]:
      one two
    a  0  1
    b  2  3
    c  4  5

    df2
    Out[161]:
      three four
    a   5   6
    c   7   8
    ```
  - join 索引上合并
    ```python
    df1.join(df2)
    Out[166]:
      one two three four
    a  0  1  5.0  6.0
    b  2  3  NaN  NaN
    c  4  5  7.0  8.0
    ```
  - merge 指定使用索引合并
    ```python
    pd.merge(df1, df2)        # MergeError: No common columns to perform merge on
    pd.merge(df1, df2, left_index=True, right_index=True, how='outer')        # 类似join索引上的合并
    Out[168]:
      one two three four
    a  0  1  5.0  6.0
    b  2  3  NaN  NaN
    c  4  5  7.0  8.0
    ```
  - concat
    ```python
    pd.concat([df1, df2])        # 堆叠，不论有没有重叠列
    Out[641]:
      four one three two
    a  NaN 0.0  NaN 1.0
    b  NaN 2.0  NaN 3.0
    c  NaN 4.0  NaN 5.0
    a  6.0 NaN  5.0 NaN
    c  8.0 NaN  7.0 NaN

    pd.concat([df1, df2], axis=1)        # 类似join索引上的合并
    Out[169]:
      one two three four
    a  0  1  5.0  6.0
    b  2  3  NaN  NaN
    c  4  5  7.0  8.0

    pd.concat({'level1': df1, 'level2': df2}, axis=1)        # 指定分层次列名
    Out[170]:
     level1   level2   
       one two three four
    a   0  1  5.0 6.0
    b   2  3  NaN NaN
    c   4  5  7.0 8.0

    pd.concat({'level1': df1, 'level2': df2}, axis=1, ignore_index=True)        # ignore_index = True将丢弃所有层次化索引
    Out[171]:
      0 1  2  3
    a 0 1 5.0 6.0
    b 2 3 NaN NaN
    c 4 5 7.0 8.0
    ```
## combine_first 使用另一个数据集的数据，填补NA值
  - 存在NA值的 a / b
    ```python
    a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=list('fedcba'))
    b = Series(np.arange(len(a), dtype=np.float64), index=list('fedcba'))
    b[-1] = np.nan

    a
    Out[197]:
    f  NaN
    e  2.5
    d  NaN
    c  3.5
    b  4.5
    a  NaN
    dtype: float64

    b
    Out[198]:
    f  0.0
    e  1.0
    d  2.0
    c  3.0
    b  4.0
    a  NaN
    dtype: float64
    ```
  - where / isnull / notnull方法填补NA值
    ```python
    np.where(pd.isnull(a), b, a)
    Out[199]: array([ 0. , 2.5, 2. , 3.5, 4.5, nan])

    a.where(pd.notnull(a), b)
    Out[206]:
    f  0.0
    e  2.5
    d  2.0
    c  3.5
    b  4.5
    a  NaN
    dtype: float64
    ```
  - combin_first方法使用b中的值填补a中的NA值
    ```python
    a.combine_first(b)
    Out[201]:
    f  0.0
    e  2.5
    d  2.0
    c  3.5
    b  4.5
    a  NaN
    dtype: float64

    b[:12].combine_first(a[2:])
    Out[202]:
    a  NaN
    b  4.0
    c  3.0
    d  2.0
    e  1.0
    f  0.0
    dtype: float64
    ```
***

# 数据整理 (duplicated / drop_duplicates / map / replace / rename / cut / qcut / 过滤异常值 / 随机采样 / get_dummies)
## duplicated / drop_duplicates 处理重复数据
  - DataFrame的duplicated方法返回一个布尔型Series，表示各行是否是重复行
    ```python
    data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [1, 1, 2, 3, 3, 4, 4]})
    data.duplicated().values
    Out[177]: array([False, True, False, False, True, False, True], dtype=bool)
    ```
  - duplicated 用于选取非重复行
    ```python
    data[data.duplicated() == False]
    Out[187]:
      k1 k2
    0 one  1
    2 one  2
    3 two  3
    5 two  4
    ```
  - drop_duplicates，用于返回一个移除了重复行的Data-Frame
    ```python
    data.drop_duplicates()
    Out[188]:
      k1 k2
    0 one  1
    2 one  2
    3 two  3
    5 two  4
    ```
  - 指定部分列进行重复项判断
    ```python
    data['v1'] = range(7)
    data.drop_duplicates('k1')        # 只根据 k1 列判断
    Out[191]:
      k1 k2 v1
    0 one  1  0
    3 two  3  3
    ```
  - keep选项指定保留方式
    ```python
    {'first', 'last', False} 默认'first'，分别指定保留第一个值 / 最后一个值 / 全部丢弃
    data.drop_duplicates(['k1', 'k2'], keep='last')
    Out[195]:
      k1 k2 v1
    1 one  1  1
    2 one  2  2
    4 two  3  4
    6 two  4  6
    ```
## map映射
  - map是一种实现元素级转换以及其他数据清理工作的便捷方式
  - Series的map方法可以接受一个函数或含有映射关系的字典型对象
    ```python
    data = DataFrame({'char': list('abcABC'), 'height':np.arange(-6, 0)})
    data
    Out[201]:
     char height
    0  a   -6
    1  b   -5
    2  c   -4
    3  A   -3
    4  B   -2
    5  C   -1

    data_map = dict(zip(list('abc'), [97, 98, 99]))
    data_map
    Out[203]: {'a': 97, 'b': 98, 'c': 99}

    data['weight'] = data['char'].map(str.lower).map(data_map)
    data
    Out[205]:
     char height weight
    0  a   -6   97
    1  b   -5   98
    2  c   -4   99
    3  A   -3   97
    4  B   -2   98
    5  C   -1   99
    ```
  - 使用lambda完成
    ```python
    data['weight'] = data['char'].map(lambda x : data_map[x.lower()])
    ```
## replace替换
  - 用其他数据替换原数据集中的值
    ```python
    data = Series([1., -999., 2., -999., -1000., 3.])
    data.replace(-999, np.nan)
    Out[223]:
    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4  -1000.0
    5    3.0
    dtype: float64
    ```
  - 将一组值替换为一个值
    ```python
    data.replace([-999, -1000], np.nan)
    ```
  - 将一组值替换为另一组值
    ```python
    data.replace([-999, -1000], [np.nan, 0])
    ```
  - 使用字典替换
    ```python
    data.replace({-999: np.nan, -1000: 0})
    ```
## Index.map / rename重命名轴索引
  - 轴标签也可以通过函数或映射进行转换，从而得到一个新对象
  - pd.Index.map方法修改轴索引
    ```python
    data = DataFrame(np.arange(12).reshape((3, 4)), index=['Ohio', 'Colorado', 'New York'], columns=list('abcd'))
    data.index.map(str.upper)
    Out[229]: array(['OHIO', 'COLORADO', 'NEW YORK'], dtype=object)
    data.index = data.index.map(str.upper)        # 将其赋值给index
    ```
  - rename 重命名轴标签
    ```python
    rename返回新数据集，inplace=True 指定修改原数据
    接受函数或字典，修改行 / 列标签
    data.rename(index=str.title, columns=str.upper)
    Out[234]:
         A B  C  D
    Ohio   0 1  2  3
    Colorado 4 5  6  7
    New York 8 9 10 11
    ```
  - 结合字典型对象实现对部分轴标签的更新
    ```python
    data.rename(index={'OHIO': 'INDIANA'}, columns=dict(zip(list('abc'), list('xyz'))))
    x y  z  d
    INDIANA  0 1  2  3
    COLORADO 4 5  6  7
    NEW YORK 8 9 10 11
    ```
  - Series可以单独修改name属性的值，并返回新对象
    ```python
    s = pd.Series([1, 2, 3], name='hh')
    s.rename('gg')
    Out[252]:
    0  1
    1  2
    2  3
    Name: gg, dtype: int64
    ```
  - Series的rename可以修改值
    ```python
    s.rename(lambda x: x ** 2)
    Out[256]:
    0  1
    1  2
    4  3
    Name: hh, dtype: int64

    s.rename({1: 3, 2: 5})
    Out[257]:
    0  1
    3  2
    5  3
    Name: hh, dtype: int64
    ```
## cut / qcut离散化和面元划分
  - 将连续数据离散化或拆分为“面元”(bin)
    ```python
    ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
    bins = [18, 25, 35, 60, 100]

    根据bIns将age拆分成四个区间组
    cats = pd.cut(ages, bins)
    cats
    Out[261]: '''
    [(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
    Length: 12
    Categories (4, object): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]'''
    ```
  - pandas返回的是一个特殊的Categorical对象
    ```python
    categories属性表示不同分类
    codes属性表示各数据的分组

    cats.categories
    Out[263]: Index(['(18, 25]', '(25, 35]', '(35, 60]', '(60, 100]'], dtype='object')

    cats.codes
    Out[265]: array([0, 0, 0, 1, 0, 0, 2, 1, 3, 2, 2, 1], dtype=int8)

    pd.value_counts(cats)
    Out[312]: '''
    (18, 25]   5
    (35, 60]   3
    (25, 35]   3
    (60, 100]  1
    dtype: int64'''

    cats[1]
    Out[262]: '(18, 25]'

    np.array(ages)[cats.codes == 2]
    Out[290]: array([37, 45, 41])
    ```
  - right = False指定区间包括左节点，不包括右节点
    ```python
    pd.cut(ages, [18, 26, 36, 61, 100], right=False)
    ```
  - labels选项设置分组名称
    ```python
    labels = False 重新编号0, 1, 2, ...

    group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
    pd.cut(ages, bins, labels=group_names)
    Out[295]:
    [Youth, Youth, Youth, YoungAdult, Youth, ..., YoungAdult, Senior, MiddleAged, MiddleAged, YoungAdult]
    Length: 12
    Categories (4, object): [Youth < YoungAdult < MiddleAged < Senior]
    ```
  - 可以指定划分的数量，而不是一组边界
    ```python
    data = np.random.rand(20)
    pd.cut(data, 4, precision=2).categories
    Out[300]: Index(['(0.027, 0.27]', '(0.27, 0.51]', '(0.51, 0.75]', '(0.75, 1]'], dtype='object')
    ```
  - qcut 基于样本分为点划分
    ```python
    pd.qcut(range(5), 4)        # 按4分位点切割
    Out[302]: '''
    [ [0, 1], [0, 1], (1, 2], (2, 3], (3, 4]]
    Categories (4, object): [[0, 1] < (1, 2] < (2, 3] < (3, 4] ]'''

    pd.qcut(range(5), 3, labels=["good","medium","bad"])
    Out[303]:
    [good, good, medium, bad, bad]
    Categories (3, object): [good < medium < bad]

    pd.qcut(range(5), 4, labels=False)
    Out[304]: array([0, 0, 1, 2, 3])
    ```
  - qcut自定义的分位数（0到1之间的数值，包含端点）
    ```python
    pd.qcut(range(11), [0, 0.1, 0.5, 0.9, 1.]).categories
    Out[322]: Index(['[0, 1]', '(1, 5]', '(5, 9]', '(9, 10]'], dtype='object')
    ```
  - 当 cut / qcut 指定划分的数量时
    ```python
    cut得到的分组区间长度基本相等
    qcut得到的分组各区间中的元素数量基本相等

    data = np.random.rand(20)
    pd.value_counts(pd.cut(data, 4, precision=2))
    Out[370]: '''
    (0.018, 0.26]  8
    (0.75, 0.99]   6
    (0.26, 0.51]   4
    (0.51, 0.75]   2
    dtype: int64'''

    pd.value_counts(pd.qcut(data, 4, precision=2))
    Out[369]: '''
    (0.76, 0.99]   5
    (0.35, 0.76]   5
    (0.11, 0.35]   5
    [0.019, 0.11]  5
    dtype: int64'''
    ```
## 数组运算过滤 / 变换异常值
  - 随机数据
    ```python
    np.random.seed(12345)
    data = DataFrame(np.random.randn(1000, 4))
    data.describe()
    Out[325]:
             0      1      2      3
    count 1000.000000 1000.000000 1000.000000 1000.000000
    mean   -0.067684   0.067924   0.025598  -0.002298
    std    0.998035   0.992106   1.006835   0.996794
    min   -3.428254  -3.548824  -3.184377  -3.745356
    25%   -0.774890  -0.591841  -0.641675  -0.644144
    50%   -0.116401   0.101143   0.002073  -0.013611
    75%    0.616366   0.780282   0.680391   0.654328
    max    3.366626   2.653656   3.260383   3.927528

    ```
    找出某列中绝对值大小超过3的值
    ```python
    col = data[3]
    col[np.abs(col) > 3]
    Out[327]:
    97   3.927528
    305  -3.399312
    400  -3.745356
    Name: 3, dtype: float64

    ```
    选出全部含有“超过3或－3的值”的行
    ```python
    data[(np.abs(data) > 3).any(1)]        # 生成沿行判断的Series，选取data值
    将超出(-3, 3)的值限制在区间内
    data[(np.abs(data) > 3).any(1)] = np.sign(data) * 3
    ```
## 通过permutation / randint随机数排列和随机采样
  - 通过需要排列的轴的长度调用permutation，可产生一个表示新顺序的整数数组
    ```python
    df = DataFrame(np.arange(5 * 4).reshape(5, 4), columns=list('abcd'))
    sampler = np.random.permutation(len(df))
    df.take(sampler[:3])
    Out[368]:
      a  b  c  d
    4 16 17 18 19
    0  0  1  2  3
    3 12 13 14 15
    ```
  - 通过np.random.randint得到一组随机整数
    ```python
    bag = Series([5, 7, -1, 6, 4], index = list('abcde'))
    sampler = np.random.randint(0, len(bag), size=10)
    bag.take(sampler)
    Out[377]:
    e  4
    a  5
    b  7
    d  6
    c  -1
    e  4
    c  -1
    c  -1
    b  7
    a  5
    dtype: int64
    ```
## get_dummies 计算指标/哑变量
  - 一种常用于统计建模或机器学习的转换方式：将分类变量(categorical variable)转换为“哑变量矩阵”(dummy matrix)或“指标矩阵”(indicator matrix)
  - get_dummies函数用于生成指标矩阵 (dummy/indicator variables)
    ```python
    get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
    Convert categorical variable into dummy/indicator variables

    s1 = ['a', 'b', np.nan]
    pd.get_dummies(s1)
    Out[388]:
      a b
    0 1 0
    1 0 1
    2 0 0
    ```
  - **np.where 生成 dummy 矩阵**
    ```python
    to_dummy = lambda a: np.array([ np.where(np.unique(a)==t, 1, 0) for t in a ])
    ss = np.array(list('abccba'))
    to_dummy(ss)
    Out[102]:
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [0, 0, 1],
           [0, 1, 0],
           [1, 0, 0]])
    ```
  - **tensorflow 生成 one_hot 矩阵**
    ```python
    import tensorflow as tf
    aa = np.array([1, 2, 3, 3])
    sess = tf.InteractiveSession()
    tf.one_hot(aa, 3).eval()
    Out[26]:
    array([[0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    ```
  - dummy_na指定保留NA值信息
    ```python
    pd.get_dummies(s1, dummy_na=True)
    Out[389]:
      a b NaN
    0 1 0  0
    1 0 1  0
    2 0 0  1
    ```
  - prefix 添加不同列的前缀
    ```python
    df = pd.DataFrame({'A': list('aba'), 'B': list('bac'), 'C': [1, 2, 3]})
    pd.get_dummies(df, prefix=['col1', 'col2'])
    Out[392]:
      C col1_a col1_b col2_a col2_b col2_c
    0 1    1    0    0    1    0
    1 2    0    1    1    0    0
    2 3    1    0    0    0    1
    ```
  - drop_first=True 丢弃第一个数据
    ```python
    pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)
    Out[395]:
      b c
    0 0 0
    1 1 0
    2 0 1
    3 0 0
    4 0 0
    ```
  - get_dummies与pd.qcut结合
    ```python
    pd.get_dummies(pd.qcut(range(11), [0, 0.1, 0.5, 0.9, 1.]))
    Out[454]: '''
      [0, 1] (1, 5] (5, 9] (9, 10]
    0    1    0    0    0
    1    1    0    0    0
    2    0    1    0    0
    3    0    1    0    0
    4    0    1    0    0
    5    0    1    0    0
    6    0    0    1    0
    7    0    0    1    0
    8    0    0    1    0
    9    0    0    1    0
    10    0    0    0    1'''
    ```
## get_dummies 计算电影数据的各标签分布
  ```python
  mnames = ['movie_id', 'title', 'genres']
  movies = pd.read_table('practice_data/movielens/movies.dat', sep='::', header=None, names=mnames)
  movies[:3]
  Out[400]:
    movie_id          title            genres
  0     1     Toy Story (1995)  Animation|Children's|Comedy
  1     2      Jumanji (1995) Adventure|Children's|Fantasy
  2     3 Grumpier Old Men (1995)        Comedy|Romance

  set(movies.genres[1].split('|'))        # set中的值是唯一的
  Out[407]: {'Adventure', "Children''s", 'Fantasy'}

  genre_iter = (set(x.split('|')) for x in movies.genres)
  genres = sorted(set.union(*genre_iter))
  genres[:3]
  Out[419]: ['Action', 'Adventure', 'Animation']

  # 创建一个全零DataFrame开始构建指标DataFrame
  dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)

  # 迭代每一部电影并将dummies各行的项设置为1
  for i, gen in enumerate(movies.genres):        # enumerate为每一项加上标号
     ...:   dummies.ix[i, gen.split('|')] = 1

  demo = dummies[:3]
  demo[demo.columns[(demo == 1).any()]]
  Out[449]:
    Adventure Animation Children's Comedy Fantasy Romance
  0    0.0    1.0     1.0   1.0   0.0   0.0
  1    1.0    0.0     1.0   0.0   1.0   0.0
  2    0.0    0.0     0.0   1.0   0.0   1.0

  # 将其与movies合并起来
  movies_windic = movies.join(dummies.add_prefix('Genre_'))
  movies_windic.ix[0]
  Out[451]:
  movie_id                    1
  title              Toy Story (1995)
  genres        Animation|Children's|Comedy
  Genre_Action                  0
  Genre_Adventure                0
  Genre_Animation                1
  Genre_Children's                1
  Genre_Comedy                  1
  Genre_Crime                  0
  Genre_Documentary               0
  Genre_Drama                  0
  Genre_Fantasy                 0
  Genre_Film-Noir                0
  Genre_Horror                  0
  Genre_Musical                 0
  Genre_Mystery                 0
  Genre_Romance                 0
  Genre_Sci-Fi                  0
  Genre_Thriller                 0
  Genre_War                   0
  Genre_Western                 0
  Name: 0, dtype: object
  ```
***
