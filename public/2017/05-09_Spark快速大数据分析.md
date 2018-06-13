# ___2017 - 05 - 09 Spark 快速大数据分析___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2017 - 04 - 09 Spark 快速大数据分析___](#2017-04-09-spark-快速大数据分析)
  - [目录](#目录)
  - [Q / A](#q-a)
  - [基本概念](#基本概念)
  - [使用spark运行python脚本](#使用spark运行python脚本)
  - [RDD编程基础 (创建 / 转化操作 / 行动操作 / 惰性求值 / 缓存persist)](#rdd编程基础-创建-转化操作-行动操作-惰性求值-缓存persist)
  	- [创建 RDD](#创建-rdd)
  	- [RDD的两种操作类型 / 惰性求值](#rdd的两种操作类型-惰性求值)
  	- [RDD 缓存persist()](#rdd-缓存persist)
  	- [基本RDD转化操作](#基本rdd转化操作)
  	- [基本RDD行动操作](#基本rdd行动操作)
  - [pair RDD 键值对操作 (创建 / 转化 / 行动)](#pair-rdd-键值对操作-创建-转化-行动)
  	- [创建 pair RDD](#创建-pair-rdd)
  	- [Pair RDD的转化操作](#pair-rdd的转化操作)
  		- [并行度调优](#并行度调优)
  		- [聚合操作](#聚合操作)
  		- [数据分组](#数据分组)
  		- [连接](#连接)
  		- [数据排序](#数据排序)
  	- [Pair RDD的行动操作](#pair-rdd的行动操作)
  - [数据分区](#数据分区)
  	- [自定义分区方式](#自定义分区方式)
  - [数据读取与保存 (文件格式 / 文件系统 / SQL / 数据库)](#数据读取与保存-文件格式-文件系统-sql-数据库)
  	- [文件格式](#文件格式)
  		- [Spark支持的一些常见格式](#spark支持的一些常见格式)
  		- [文本文件](#文本文件)
  		- [JSON](#json)
  		- [逗号分隔值 CSV 与 制表符分隔值TSV](#逗号分隔值-csv-与-制表符分隔值tsv)
  		- [SequenceFile](#sequencefile)
  		- [对象文件](#对象文件)
  		- [Hadoop输入输出格式](#hadoop输入输出格式)
  		- [protocol buffer](#protocol-buffer)
  		- [文件压缩](#文件压缩)
  	- [文件系统](#文件系统)
  		- [本地 / 常规文件系统](#本地-常规文件系统)
  		- [Amazon S3](#amazon-s3)
  		- [HDFS](#hdfs)
  	- [Spark SQL中的结构化数据](#spark-sql中的结构化数据)
  		- [Apache Hive](#apache-hive)
  		- [JSON](#json)
  	- [数据库](#数据库)
  		- [Cassandra](#cassandra)
  		- [HBase](#hbase)
  		- [Elasticsearch](#elasticsearch)
  - [共享变量 ( 累加器 / 广播变量 / 基于分区进行操作 / 与外部程序间的管道 / 数值RDD的操作)](#共享变量-累加器-广播变量-基于分区进行操作-与外部程序间的管道-数值rdd的操作)
  	- [累加器](#累加器)
  		- [累加器的用法](#累加器的用法)
  		- [行动操作 / 转化操作中的累加器容错性](#行动操作-转化操作中的累加器容错性)
  		- [自定义累加器](#自定义累加器)
  	- [广播变量](#广播变量)
  		- [广播变量的使用](#广播变量的使用)
  		- [广播的优化](#广播的优化)
  	- [基于分区进行操作](#基于分区进行操作)
  	- [与外部程序间的管道](#与外部程序间的管道)
  	- [数值RDD的操作](#数值rdd的操作)
  - [在集群上运行 Spark （运行时架构 / spark-submit / 集群管理器）](#在集群上运行-spark-运行时架构-spark-submit-集群管理器)
  	- [Spark运行时架构](#spark运行时架构)
  		- [分布式环境下Spark 集群的主 / 从结构](#分布式环境下spark-集群的主-从结构)
  		- [Spark驱动器程序](#spark驱动器程序)
  		- [执行器节点](#执行器节点)
  		- [集群管理器](#集群管理器)
  		- [启动一个程序](#启动一个程序)
  		- [在集群上运行 Spark 应用的详细过程](#在集群上运行-spark-应用的详细过程)
  	- [使用spark-submit部署应用](#使用spark-submit部署应用)
  	- [打包代码与依赖](#打包代码与依赖)
  	- [Spark应用内与应用间调度](#spark应用内与应用间调度)
  	- [独立集群管理器](#独立集群管理器)
  		- [启动集群管理器](#启动集群管理器)
  		- [提交应用](#提交应用)
  		- [配置资源用量](#配置资源用量)
  		- [高度可用性](#高度可用性)
  	- [集群管理器 Hadoop YARN](#集群管理器-hadoop-yarn)
  		- [在 Spark 里使用 YARN](#在-spark-里使用-yarn)
  		- [客户端模式 / 集群模式](#客户端模式-集群模式)
  		- [配置资源用量](#配置资源用量)
  	- [集群管理器 Apache Mesos](#集群管理器-apache-mesos)
  	- [集群管理器 Amazon EC2](#集群管理器-amazon-ec2)
  		- [启动集群](#启动集群)
  		- [登录集群](#登录集群)
  		- [销毁集群](#销毁集群)
  		- [暂停和重启集群](#暂停和重启集群)
  		- [集群存储](#集群存储)
  	- [选择合适的集群管理器](#选择合适的集群管理器)
  - [使用 SparkConf / spark-submit 配置Spark](#使用-sparkconf-spark-submit-配置spark)
  	- [SparkConf 类](#sparkconf-类)
  	- [spark-submit配置](#spark-submit配置)
  	- [Spark配置优先级](#spark配置优先级)
  	- [常用的Spark配置项的值](#常用的spark配置项的值)
  	- [SPARK_LOCAL_DIRS](#sparklocaldirs)
  - [Spark 性能调优 (作业 / 用户界面 / 日志 / 并行度 / 序列化格式 / 内存管理)](#spark-性能调优-作业-用户界面-日志-并行度-序列化格式-内存管理)
  	- [Spark执行的作业 / 任务 / 步骤](#spark执行的作业-任务-步骤)
  	- [Spark 内建的网页用户界面](#spark-内建的网页用户界面)
  	- [驱动器进程和执行器进程的日志](#驱动器进程和执行器进程的日志)
  	- [并行度](#并行度)
  	- [序列化格式](#序列化格式)
  		- [Kryo序列化工具](#kryo序列化工具)
  		- [NotSerializableException错误](#notserializableexception错误)
  	- [内存管理](#内存管理)
  		- [内存用途](#内存用途)
  		- [内存缓存策略优化](#内存缓存策略优化)
  - [Spark SQL (Spark 2.1.0有改动)](#spark-sql-spark-210有改动)
  - [Spark Streaming (Spark 1.2不支持python)](#spark-streaming-spark-12不支持python)
  - [基于 MLlib 的机器学习](#基于-mllib-的机器学习)

  <!-- /TOC -->
***

# Q / A
  - Q：安装
  - A：下载链接 http://spark.apache.org/downloads.html
        环境变量：
        export SPARK_HOME=/home/leondgarse/local_bin/spark-2.1.0-bin-hadoop2.7
        export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

        pip 安装 pip install pyspark
  - Q： spark打印log级别
  - A： $ cp conf/log4j.properties.template conf/log4j.properties
        $ vi conf/log4j.properties        # log设置
        log4j.rootCategory=WARN, console         # WARN级别
  - Q： pyspark警告： WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
  - A： 环境变量里添加 (HADOOP_HOME / JAVA_HOME 已正确设置)：
        export HADOOP_CONF_DIR=$HADOOP_HOME
        export SPARK_DAEMON_JAVA_OPTS=$JAVA_HOME
        export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native/:$LD_LIBRARY_PATH
  - Q： pyspark使用ipython
  - A： $ vi $SPARK_HOME/bin/pyspark # 配置文件
        PYSPARK_DRIVER_PYTHON="${PYSPARK_PYTHON:-"ipython"}"        # 指定使用的python shell (python3.4 / ipython)

        spark2.1.0不支持python3.6版本：
        conda create -n python3 python=3.5 anaconda
        source activate python3        # 可以添加到pyspark配置文件中

        删除可使用 anaconda-navigator 移除python3环境，然后删除不需要的包文件
        conda clean --all

        使用ipython notebook：
        export PYSPARK_DRIVER_PYTHON_OPTS="notebook"
  - Q： sc.textFile('README.md').count()错误：
        Py4JJavaError: An error occurred while calling z:org.apache.spark.api.python.PythonRDD.collectAndServe.
        : org.apache.hadoop.mapred.InvalidInputException: Input path does not exist: file:/home/leondgarse/RELEASE
  - A： ipython中虽然可以使用cd命令进入到文件所在位置，但sc.textFile()使用的路径依然是运行pyspark时的路径
        sc.textFile()中最好使用绝对路径
        tempdir = %pwd        # path to file
        path = os.path.join(tempdir, "README.md")
        sc.textFile(path).count()
***

# 基本概念
  - pyspark # python版本的spark
  - http://127.0.1.1:4040/jobs/ 访问 Spark 用户界面，查看关于任务和集群的各种信息，默认端口是4040，启动时可通过打开INFO级别log查看
  - Spark 本身是用 Scala 写的，运行在 Java 虚拟机（JVM）上
  - Spark 可以运行在许多种模式下，除了本地模式，还支持运行在 Mesos 或 YARN 上，也可以运行在 Spark 发行版自带的独立调度器上
  - Spark shell 可用来与分布式存储在许多机器的内存或者硬盘上的数据进行交互，并且处理过程的分发由 Spark 自动控制完成
  - Spark 能够在工作节点上把数据读取到内存中，所以许多分布式计算都可以在几秒钟之内完成
  - 弹性分布式数据集RDD（resilient distributed dataset）：
    ```python
    Spark 中通过对分布式数据集的操作来表达计算意图，这些计算会自动地在集群上并行进行，这样的数据集被称为RDD，是 Spark 对分布式数据和计算的基本抽象
    在 Spark 中，对数据的所有操作不外乎创建 RDD、转化已有 RDD 以及调用 RDD 操作进行求值，Spark 会自动将 RDD 中的数据分发到集群上，并将操作并行化执行
    ```
  - 驱动器程序（driver program）：
    ```python
    从上层来看，每个 Spark 应用都由一个驱动器程序来发起集群上的各种并行操作，包含应用的 main 函数，并且定义了集群上的分布式数据集，还对这些分布式数据集应用了相关操作
    实际的驱动器程序可以是 Spark shell 本身，只需要输入想要运行的操作就可以了
    ```
  - SparkContext 对象：
    ```python
    驱动器程序通过一个SparkContext 对象来访问 Spark，这个对象代表对计算集群的一个连接
    shell 启动时已经自动创建了一个 SparkContext 对象，是一个叫作 sc 的变量
            In [2]: sc.appName
            Out[2]: 'PySparkShell'
    一旦有了 SparkContext，就可以用它来创建 RDD，如调用 sc.textFile() 来创建一个代表文件中各行文本的 RDD，可以在这些行上进行各种操作，比如count()
            In [32]: lines = sc.textFile("README.md")        # 最好使用绝对路径
            In [33]: lines.count()
            Out[33]: 104
    ```
  - 执行器（executor）节点：
    ```python
    要执行操作，驱动器程序一般要管理多个执行器（executor）节点，如在集群上运行 count() 操作，那么不同的节点会统计文件的不同部分的行数
    在本地模式下运行 Spark shell时，所有的工作会在单个节点上执行，但可以将这个 shell 连接到集群上来进行并行的数据分析
    ```
    <br />
  - Spark API：
    ```python
    Spark有很多API用来传递函数 ，可以将对应操作运行在集群上
    像 filter 这样基于函数的操作也会在集群上并行执行，Spark 会自动将函数（比如 line.contains("Python")）发到各个执行器节点上
    因此可以在单一的驱动器程序中编程，并且让代码自动运行在多个节点上
            In [8]: python_lines = lines.filter(lambda line: "Python" in line)
            In [9]: python_lines.count()
            Out[9]: 3
            In [10]: python_lines.first()
            Out[10]: 'high-level APIs in Scala, Java, Python, and R, and an optimized engine that'
    ```
***

# 使用spark运行python脚本
  - 在脚本文件中先创建一个 SparkConf 对象来配置应用，然后基于这个 SparkConf 创建一个 SparkContext 对象
  - 创建 SparkContext 的最基本的方法，只需传递两个参数：
    ```python
    集群 URL：告诉 Spark 如何连接到集群上，本地的集群可以使用 local，这个特殊值可以让 Spark 运行在单机单线程上而无需连接到集群
    应用名：当连接到一个集群时，这个值可以帮助你在集群管理器的用户界面中找到你的应用（My App）
    ```
    执行： spark-submit my_script.py
    ```python
    引入 Python 程序的 Spark 依赖，为 Spark 的 PythonAPI 配置好运行环境
    ```
  - 示例：
    ```python
    $ cat my_script.py
    from pyspark import SparkConf, SparkContext
    conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(conf = conf)
    print("sc.appName = %s" % sc.appName)
    l = sc.parallelize(range(100),8).take(5)
    print("l = ", l)
    $ source activate python3
    $ spark-submit my_script.py
    sc.appName = my_script.py
    l = [0, 1, 2, 3, 4]
    ```
***

# RDD编程基础 (创建 / 转化操作 / 行动操作 / 惰性求值 / 缓存persist)
  - Spark 中的 RDD 就是一个不可变的分布式对象集合，每个 RDD 都被分为多个分区，这些分区运行在集群中的不同节点上
  - RDD 可以包含 Python、Java、Scala 中任意类型的对象，甚至可以包含用户自定义的对象
  - 每个 Spark 程序或 shell 会话都按如下方式工作：
    ```python
    (1) 从外部数据创建出输入 RDD。
    (2) 使用诸如 filter() 这样的转化操作对 RDD 进行转化，以定义新的 RDD。
    (3) 告诉 Spark 对需要被重用的中间结果 RDD 执行 persist() 操作。
    (4) 使用行动操作（例如 count() 和 first() 等）来触发一次并行计算，Spark 会对计算进行优化后再执行。
    ```
## 创建 RDD
  - 读取一个外部数据集
  - 或在驱动器程序里分发驱动器程序中的对象集合（比如 list 和 set）
    ```python
    l = sc.parallelize(range(100),8).take(5)        # 第二个参数8是分区数
    lines = sc.textFile("README.md")        # 最好使用绝对路径
    ```
  - parallelize(c, numSlices=None) method of pyspark.context.SparkContext instance
    ```python
    Distribute a local Python collection to form an RDD. Using xrange is recommended if the input represents a range for performance.

    sc.parallelize([0, 2, 3, 4, 6], 5).glom().collect()
    [ [0], [2], [3], [4], [6] ]
    sc.parallelize(xrange(0, 6, 2), 5).glom().collect()
    [ [], [0], [], [2], [4] ]
    ```
  - glom() method of pyspark.rdd.RDD instance
    ```python
    Return an RDD created by coalescing all elements within each partition into a list.

    rdd = sc.parallelize([1, 2, 3, 4], 2)
    sorted(rdd.glom().collect())
    [ [1, 2], [3, 4] ]
    ```
  - textFile(name, minPartitions=None, use_unicode=True) method of pyspark.context.SparkContext instance
    ```python
    Read a text file from HDFS, a local file system (available on all nodes), or any Hadoop-supported file system URI, and return it as an RDD of Strings.

    If use_unicode is False, the strings will be kept as `str` (encoding as `utf-8`), which is faster and smaller than unicode. (Added in Spark 1.2)

    path = os.path.join(tempdir, "sample-text.txt")
    with open(path, "w") as testFile:
    ...  _ = testFile.write("Hello world!")
    textFile = sc.textFile(path)
    textFile.collect()
    ['Hello world!']
    ```
## RDD的两种操作类型 / 惰性求值
  - 创建出来后，RDD 支持两种类型的操作：转化操作（transformation） 和行动操作（action）
    ```python
    转化操作会由一个 RDD 生成一个新的 RDD，根据谓词匹配情况筛选数据就是一个常见的转化操作
    行动操作会对 RDD 计算出一个结果，并把结果返回到驱动器程序中，或把结果存储到外部存储系统（如 HDFS）中，first() 就是一个行动操作
    ```
  - 转化操作：
    ```python
    RDD 的转化操作是返回新 RDD 的操作，转化出来的 RDD 是惰性求值的，只有在行动操作中用到这些 RDD 时才会被计算。
    通过转化操作，从已有的 RDD 中派生出新的 RDD，Spark 会使用谱系图（lineage graph）来记录这些不同 RDD 之间的依赖关系
    Spark 需要用这些信息来按需计算每个 RDD，也可以依靠谱系图在持久化的 RDD 丢失部分数据时恢复所丢失的数据
    ```
  - 行动操作：
    ```python
    由于行动操作需要生成实际的输出，它们会强制执行那些求值必须用到的 RDD 的转化操作
    需要注意的是，每当我们调用一个新的行动操作时，整个 RDD 都会从头开始计算。要避免这种低效的行为，用户可以将中间结果持久化
    ```
  - Spark 使用惰性求值，这样就可以把一些操作合并到一起来减少计算数据的步骤
  - 在类似 Hadoop MapReduce 的系统中，开发者常常花费大量时间考虑如何把操作组合到一起，以减少 MapReduce 的周期数
  - 在 Spark 中，写出一个非常复杂的映射并不见得能比使用很多简单的连续操作获得好很多的性能，因此，用户可以用更小的操作来组织他们的程序， 这样也使这些操作更容易管理
  - 如果对于一个特定的函数是属于转化操作还是行动操作感到困惑，可以看看它的返回值类型：转化操作返回的是 RDD，而行动操作返回的是其他的数据类型
## RDD 缓存persist()
  - 在实际操作中，会经常用 persist() 来把数据的一部分读取到内存中，并反复查询这部分数据
    ```python
    pythonLines.persist
    pythonLines.count()
    ```
  - persist() 调用本身不会触发强制求值
  - unpersist()方法可以手动把持久化的 RDD 从缓存中移除
## 基本RDD转化操作
  - map() 映射
    ```python
    接收一个函数，把这个函数用于 RDD 中的每个元素，将函数的返回结果作为结果 RDD 中对应元素的值
    nums = sc.parallelize([1, 2, 3, 4])
    squared = nums.map(lambda x:x * x)
    squared.collect()        // --> Out[126]: [1, 4, 9, 16]
    ```
  - filter() 筛选
    ```python
    接收一个函数，并将 RDD 中满足该函数的元素放入新的 RDD 中返回
    odd = nums.filter(lambda x : x % 2 == 1)
    odd.collect()        // --> Out[127]: [1, 3]
    ```
  - flatMap()
    ```python
    与map() 类似，函数被分别应用到了输入 RDD 的每个元素上，返回值是一个包含各个迭代器可访问的所有元素的 RDD，将返回值中的集合等拆开
    lines = sc.parallelize(['hello world', 'hi'])
    words = lines.flatMap(lambda l : l.split(" "))
    words.collect()        // --> Out[42]: ['hello', 'world', 'hi']
    line = lines.map(lambda l : l.split(" "))
    line.collect()        // --> Out[40]: [ ['hello', 'world'], ['hi'] ]
    ```
  - sample(withReplacement, fraction, seed=None) 随机采样
    ```python
    withReplacement: True / False 指定数据是否可以重复采样
    fraction: 期望的分片大小，实际的大小随机
      without replacement: probability that each element is chosen; fraction must be [0, 1]
      with replacement: expected number of times each element is chosen; fraction must be >= 0
    seed: 随机数种子
    rdd = sc.parallelize(range(100), 4)
    6 <= rdd.sample(False, 0.1, 81).count() <= 14        // --> Out[156]: True
    ```
  - distinct() 去重复
    ```python
    生成一个只包含不同元素的新 RDD
    需要注意，distinct() 操作的开销很大，因为它需要将所有数据通过网络进行混洗（shuffle），以确保每个元素都只有一份
    ```
    union(other) 并集
    ```python
    返回一个包含两个 RDD 中所有元素的 RDD，会包含这些重复数据
    nums1 = sc.parallelize([1, 2, 2, 3, 4])
    nums2 = sc.parallelize([2, 3, 5])
    nums1.union(nums2).collect()        // --> Out[50]: [1, 2, 2, 3, 4, 2, 3, 5]
    ```
    intersection(other) 交集
    ```python
    返回两个 RDD 中都有的元素 会去掉所有重复的元素 需要通过网络混洗数据来发现共有的元素
    nums1.intersection(nums2).collect()        // --> Out[51]: [2, 3]
    ```
    subtract(other) 补集
    ```python
    返回一个由只存在于第一个 RDD 中而不存在于第二个 RDD 中的所有元素组成的 RDD ，需要数据混洗
    nums1.subtract(nums2).collect()        // --> Out[52]: [1, 4]
    ```
    cartesian(other) 笛卡儿积
    ```python
    返回所有可能的 (a, b) 对，求大规模 RDD 的笛卡儿积开销巨大
    nums1.distinct().cartesian(nums2).collect()
    Out[53]:
    [(4, 2), (4, 3), (4, 5),
    ...
     (3, 2), (3, 3), (3, 5)]
    ```
## 基本RDD行动操作
  - foreach() 对 RDD 中的每个元素进行操作，而不需要把 RDD 发回本地
  - count() 返回元素的个数
  - countByValue() 返回一个从各值到值对应的计数的映射表
  - reduce() / fold() 接收一个函数作为参数，这个函数要操作两个 RDD 的元素类型的数据并返回一个同样类型的新元素
    ```python
    fold() 和 reduce() 都要求函数的返回值类型需要和我们所操作的 RDD 中的元素类型相同

    from operator import add
    nums1 = sc.parallelize(range(0, 10, 2))
    nums1.reduce(add)        // --> Out[119]: 20

    sc.parallelize(range(0, 10, 2)).fold(1, add)        // --> Out[138]: 25
    ```
  - collect() 将整个 RDD 的内容返回
    ```python
    collect() 通常在单元测试中使用，因为此时 RDD 的整个内容不会很大，可以放在内存中
    由于需要将数据复制到驱动器进程中，collect() 要求所有数据都必须能一同放入单台机器的内存中
    ```
    take(n) 返回 RDD 中的 n 个元素
    ```python
    尝试只访问尽量少的分区，因此该操作会得到一个不均衡的集合
    需要注意的是，这些操作返回元素的顺序与预期的可能不一样
    ```
  - aggregate(zeroValue, seqOp, combOp) 累加器
    ```python
    Aggregate the elements of each partition, and then the results for allthe partitions, using a given combine functions and a neutral "zero value."
    The functions C{op(t1, t2)} is allowed to modify C{t1} and return it as its result value to avoid object allocation; however, it should not modify C{t2}.
    The first function (seqOp) can return a different result type, U, than the type of this RDD.
    Thus, we need one operation for merging a T into an U and one operation for merging two U

    seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
    combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
    sc.parallelize([1, 2, 3, 4]).aggregate((0, 0), seqOp, combOp)
    Out[161]: (10, 4)

    sc.parallelize([1, 2, 3, 4]).aggregate((2, 1), seqOp, combOp)
    Out[163]: (20, 9)

    sc.parallelize([]).aggregate((0, 0), seqOp, combOp)
    Out[164]: (0, 0)
    ```
***

# pair RDD 键值对操作 (创建 / 转化 / 行动)
  - 键值对 RDD 通常用来进行聚合计算。我们一般要先通过一些初始 ETL（抽取、转化、装载）操作来将数据转化为键值对形式
  - 键值对 RDD 提供了一些新的操作接口（比如统计每个产品的评论，将数据中键相同的分为一组，将两个不同的 RDD 进行分组合并等）
  - 使用可控的分区方式把常被一起访问的数据放到同一个节点上，可以大大减少应用的通信开销
  - 为分布式数据集选择正确的分区方式和为本地数据集选择合适的数据结构很相似——在这两种情况下，数据的分布都会极其明显地影响程序的性能表现
  - 当需要把一个普通的 RDD 转为 pair RDD 时，可以调用 map() 函数来实现，传递的函数需要返回键值对
## 创建 pair RDD
  - 在 Python 中，为了让提取键之后的数据能够在函数中使用，需要返回一个由二元组组成的 RDD
    ```python
    lines = sc.textFile('README.md')
    pairs = lines.map(lambda x : (x.split(" ")[0], x))        # 使用第一个单词作为键创建出一个 pair RDD
    ```
  - 当用 Python 从一个内存中的数据集创建 pair RDD 时，只需要对这个由二元组组成的集合调用 SparkContext.parallelize() 方法
    ```python
    pnum = sc.parallelize({(1, 2), (3, 4), (3, 6)})
    ```
## Pair RDD的转化操作
  - Pair RDD 可以使用所有标准 RDD 上的可用的转化操作
  - 由于 pair RDD 中包含二元组，所以需要传递的函数应当操作二元组而不是独立的元素
    ```python
    results = pairs.filter(lambda keyValue : len(keyValue[1]) < 20)        # 筛选长度小于20的字符串
    ```
  - keys() 返回一个仅包含键的RDD
  - values() 返回一个仅包含值的RDD
  - 如果只想访问 pair RDD 的值部分，操作二元组很麻烦，因此 Spark 提供了 mapValues(func) 函数，功能类似于 map{case (x, y): (x, func(y))}
### 并行度调优
  - 每个 RDD 都有固定数目的分区，分区数决定了在 RDD 上执行操作时的并行度
  - 在执行聚合或分组操作时，可以要求 Spark 使用给定的分区数，Spark 始终尝试根据集群的大小推断出一个有意义的默认值，但是有时候可能要对并行度进行调优来获取更好的性能表现
    ```python
    data = [("a", 3), ("b", 4), ("a", 1)]
    sc.parallelize(data).reduceByKey(lambda x, y: x + y)   # 默认并行度
    sc.parallelize(data).reduceByKey(lambda x, y: x + y, 10) # 自定义并行度
    ```
  - Spark 提供了 repartition() 函数,它会把数据通过网络进行混洗，并创建出新的分区集合
  - 对数据进行重新分区是代价相对比较大的操作
  - coalesce()是Spark 中也有一个优化版的 repartition()，可以使用 Python 中的rdd.getNumPartitions 查看 RDD 的分区数，并确保调用 coalesce() 时将 RDD 合并到比现在的分区数更少的分区中
### 聚合操作
  - 当数据集以键值对形式组织的时候，聚合具有相同键的元素进行一些统计操作
  - reduceByKey()
    ```python
    与 reduce() 类似，接收一个函数，并使用该函数对值进行合并
    reduceByKey() 会为数据集中的每个键进行并行的归约操作，每个归约操作会将键相同的值合并起来
    使用 reduceByKey() 和 mapValues() 来计算每个键的对应值的均值：
            In [15]: pnum = sc.parallelize({(1, 0), (2, 3), (3, 3), (1, 1), (2, 4)})
            In [18]: pnum.mapValues(lambda x : (x, 1)).collect()
            Out[18]: [(1, (0, 1)), (1, (1, 1)), (2, (3, 1)), (2, (4, 1)), (3, (3, 1))]
            In [19]: pnum.mapValues(lambda x : (x, 1)).reduceByKey(lambda x, y : (x[0] + y[0], x[1] + y[1])).collect()
            Out[19]: [(1, (1, 2)), (2, (7, 2)), (3, (3, 1))]

    使用 reduceByKey() 对所有的单词进行计数：
            lines = sc.textFile('README.md')
            words = lines.flatMap(lambda x : x.split(' '))        # 使用 words.countByValue() 也可以完成单词计数，返回值是defaultdict
            results = words.map(lambda x : (x, 1)).reduceByKey(lambda x, y : x + y)
            results.top(5)
            Out[27]: [('your', 1), ('you', 4), ('with', 4), ('will', 1), ('which', 2)]
    ```
  - foldByKey()
    ```python
    与 fold() 类似，使用一个与 RDD 和合并函数中的数据类型相同的零值作为初始值
    与 fold() 一样，foldByKey() 操作所使用的合并函数对零值与另一个元素进行合并，结果仍为该元素
    ```
  - combineByKey()
    ```python
    更泛化的 接口，可以自定义合并的行为，大多数基于键聚合的函数都是用它实现的
    和 aggregate() 一样，combineByKey() 可以让用户返回与输入数据的类型不同的返回值
    combineByKey()会遍历分区中的所有元素
    如果这是一个新的元素，combineByKey() 会使用一个叫作 createCombiner() 的函数来创建那个键对应的累加器的初始值
    需要注意的是，这一过程会在每个分区中第一次出现各个键时发生，而不是在整个 RDD 中第一次出现一个键时发生
    如果这是一个在处理当前分区之前已经遇到的键，它会使用 mergeValue() 方法将该键的累加器对应的当前值与这个新的值进行合并

    由于每个分区都是独立处理的，因此对于同一个键可以有多个累加器
    如果有两个或者更多的分区都有对应同一个键的累加器，就需要使用用户提供的 mergeCombiners() 方法将各个分区的结果进行合并
    ```
  - Python 中使用 combineByKey() 求每个键对应的平均值
    ```python
    sumCount = nums.combineByKey((lambda x: (x,1)),
                   (lambda x, y: (x[0] + y, x[1] + 1)),
                   (lambda x, y: (x[0] + y[0], x[1] + y[1])))
    sumCount.map(lambda key, xy: (key, xy[0]/xy[1])).collectAsMap()
    ```
  - combineByKey(createCombiner, mergeValue, mergeCombiners, numPartitions=None, partitionFunc=<function portable_hash at 0x7f60d5150bf8>)
    ```python
    Generic function to combine the elements for each key using a custom set of aggregation functions.
    Turns an RDD[(K, V)] into a result of type RDD[(K, C)], for a "combined type" C.

    Users provide three functions:
        - C{createCombiner}, which turns a V into a C (e.g., creates a one-element list)
        - C{mergeValue}, to merge a V into a C (e.g., adds it to the end of a list)
        - C{mergeCombiners}, to combine two C's into a single one.

    In addition, users can control the partitioning of the output RDD.
      .. note:: V and C can be different -- for example, one might group an RDD of type (Int, Int) into an RDD of type (Int, List[Int]).

    x = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
    def add(a, b): return a + str(b)
    sorted(x.combineByKey(str, add, add).collect())
    [('a', '11'), ('b', '1')]
    ```
### 数据分组
  - 对于有键的数据，一个常见的用例是将数据根据键进行分组——比如查看一个顾客的所有订单
  - groupByKey()
    ```python
    对于已经以预期的方式提取了键的数据，使用 RDD 中的键来对数据进行分组
    对于一个由类型 K 的键和类型 V 的值组成的 RDD，所得到的结果 RDD 类型会是 [K, Iterable[V]]
    ```
  - groupBy()
    ```python
    可以用于未成对的数据上，也可以根据除键相同以外的条件进行分组
    它可以接收一个函数，对源 RDD 中的每个元素使用该函数，将返回结果作为键再进行分组
    ```
  - cogroup()
    ```python
    对多个共享同一个键的 RDD 进行分组
    对两个键的类型均为 K 而值的类型分别为 V 和 W 的 RDD 进行 cogroup() 时，得到的结果 RDD 类型为 [(K, (Iterable[V], Iterable[W]))]
    如果其中的一个 RDD 对于另一个 RDD 中存在的某个键没有对应的记录，那么对应的迭代器则为空
    ```
  - 如果先使用groupByKey() 然后再对值使用 reduce() 或者 fold() ，很有可能可以通过使用一种根据键进行聚合的函数来更高效地实现同样的效果
  - 对每个键归约数据，返回对应每个键的归约值的 RDD，而不是把 RDD 归约为一个内存中的值
  - 例如，rdd.reduceByKey(func) 与rdd.groupByKey().mapValues(value => value.reduce(func)) 等价，但是前者更为高效，因为它避免了为每个键创建存放值的列表的步骤
### 连接
  - 将有键的数据与另一组有键的数据一起使用是对键值对数据执行的最有用的操作之一
  - 连接方式多种多样：右外连接、左外连接、交叉连接以及内连接
  - join 操作符表示内连接
    ```python
    只有在两个 pair RDD 中都存在的键才会输出，当一个输入对应的某个键有多个值时，生成的 pair RDD 会包括来自两个输入 RDD 的每一组相对应的记录
    leftOuterJoin(other) 和rightOuterJoin(other) 都会根据键连接两个 RDD，但是允许结果中存在其中的一个 pair RDD 所缺失的键
    ```
    leftOuterJoin()
    ```python
    产生的 pair RDD 中，源 RDD 的每一个键都有对应的记录
    每个键相应的值是由一个源 RDD 中的值与一个包含第二个 RDD 的值的 Option（在 Java 中为 Optional）对象组成的二元组
    在 Python 中，如果一个值不存在，则使用 None 来表示；而数据存在时就用常规的值来表示，不使用任何封装
    和 join() 一样，每个键可以得到多条记录，当这种情况发生时，我们会得到两个 RDD 中对应同一个键的两组值的笛卡尔积
    ```
    rightOuterJoin()
    ```python
    几乎与 leftOuterJoin() 完全一样，只不过预期结果中的键必须出现在第二个 RDD 中，而二元组中的可缺失的部分则来自于源 RDD 而非第二个 RDD
    ```
    subtractByKey()
    ```python
    删掉 RDD 中键与 other RDD 中的键相同的元素
    ```
### 数据排序
  - 让数据排好序是很有用的，尤其是在生成下游输出时，如果键有已定义的顺序，就可以对这种键值对 RDD 进行排序
  - 当把数据排好序后，后续对数据进行 collect() 或 save() 等操作都会得到有序的数据
  - sortByKey() 函数接收一个叫作 ascending 的参数，表示我们是否想要让结果按升序排序（默认值为 true）
  - 要按完全不同的排序依据进行排序，可以提供自定义的比较函数
  - 在 Python 中以字符串顺序对整数进行自定义排序：
    ```python
    rdd.sortByKey(ascending=True, numPartitions=None, keyfunc = lambda x: str(x))
    ```
## Pair RDD的行动操作
  - 所有基础 RDD 支持的传统行动操作也都在 pair RDD 上可用
  - countByKey() 对每个键对应的元素分别计数
  - collectAsMap() 将结果以映射表的形式返回，以便查询
  - lookup(key) 回给定键对应的所有值
    ```python
    pnum = sc.parallelize({(1, 0), (2, 3), (3, 3), (1, 1), (2, 4)})
    pnum.countByKey()
    Out[47]: defaultdict(int, {1: 2, 2: 2, 3: 1})
    pnum.countByKey()
    Out[47]: defaultdict(int, {1: 2, 2: 2, 3: 1})
    pnum.lookup(2)
    Out[49]: [3, 4]
    ```
***

# 数据分区
  - 在分布式程序中，通信的代价是很大的，因此控制数据分布以获得最少的网络传输可以极大地提升整体性能
  - 如果给定 RDD 只需要被扫描一次，我们完全没有必要对其预先进行分区处理，只有当数据集多次在诸如连接这种基于键的操作中使用时，分区才会有帮助
  - Spark 中所有的键值对 RDD 都可以进行分区，系统会根据一个针对键的函数对元素进行分组，Spark 可以确保同一组的键出现在同一个节点上
  - 默认情况下，连接操作会将两个数据集中的所有键的哈希值都求出来，将该哈希值相同的记录通过网络传到同一台机器上，然后在那台机器上对所有键相同的记录进行连接操作
  - 对于将小数据合并到大数据集的情况，可以在程序开始时，使用 partitionBy() 转化操作，将原始数据表转为哈希分区
    ```python
    rdd.partitionBy(100)        # python中直接传递分区数，scala中需要传递一个spark.HashPartitioner 对象来实现该操作
    ```
    传给 partitionBy() 的 100 表示分区数目，它会控制之后对这个 RDD 进行进一步操作（比如连接操作）时有多少任务会并行执行，这个值至少应该和集群中的总核心数一样
  - 一旦创建就无法修改，因此应该对 partitionBy() 的结果进行持久化，不进行持久化会导致每次都会对整个 RDD 谱系图重新求值
  - Spark 的许多操作都引入了将数据根据键跨节点进行混洗的过程，所有这些操作都会从数据分区中获益，
  - 就 Spark 1.0 而言，能够从数据分区中获益的操作有cogroup()、groupWith()、join()、leftOuterJoin()、rightOuterJoin()、groupByKey()、reduceByKey()、combineByKey()以及 lookup()
  - 预先进行数据分区会导致其中至少一个 RDD（使用已知分区器的那个 RDD）不发生数据混洗
  - 对于二元操作，输出数据的分区方式取决于父 RDD 的分区方式
  - 许多Spark 操作会自动为结果RDD设定已知的分区方式信息，而且除 join() 外还有很多操作也会利用到已有的分区信息
  - 如，sortByKey() 和 groupByKey() 会分别生成范围分区的 RDD 和哈希分区的 RDD
  - 而另一方面，诸如 map() 这样的操作会导致新的 RDD 失去父 RDD 的分区信息，因为这样的操作理论上可能会修改每条记录的键
  - 不过，Spark 提供了另外两个操作 mapValues() 和 flatMapValues() 作为替代方法，它们可以保证每个二元组的键保持不变
  - 为了最大化分区相关优化的潜在作用，应该在无需改变元素的键时尽量使用 mapValues() 或 flatMapValues()
## 自定义分区方式
  - Spark 提供的 HashPartitioner 与 RangePartitioner 已经能够满足大多数用例，但 Spark 还允许通过提供一个自定义的 Partitioner 对象来控制 RDD 的分区方式
  - 在 Python 中，实现自定义分区器不需要扩展 Partitioner 类，而是把一个特定的哈希函数作为一个额外的参数传给RDD.partitionBy() 函数
    ```python
    import urlparse
    def hash_domain(url):
      return hash(urlparse.urlparse(url).netloc)        # 定义新的分区方式为依据url的域名分区

    rdd.partitionBy(20, hash_domain) # 创建20个分区
    ```
  - 所传过去的哈希函数会被用来与其他 RDD 的分区函数区分开来
  - 如果想要对多个 RDD 使用相同的分区方式，就应该使用同一个函数对象，比如一个全局函数，而不是为每个 RDD 创建一个新的函数对象
  - Spark 中有许多依赖于数据混洗的方法，比如 join() 和 groupByKey()，它们也可以接收一个可选的 Partitioner对象来控制输出数据的分区方式
***

# 数据读取与保存 (文件格式 / 文件系统 / SQL / 数据库)
  - Spark 支持很多种输入输出源，部分原因是 Spark 本身是基于 Hadoop 生态圈而构建
  - 特别是 Spark 可以通过 Hadoop MapReduce 所使用的 InputFormat 和 OutputFormat 接口访问数据，大部分常见的文件格式与存储系统（例如 S3、HDFS、Cassandra、HBase 等）都支持这种接口
  - 不过，基于这些原始接口构建出的高层 API 会更常用
## 文件格式
### Spark支持的一些常见格式
  - Spark 会根据文件扩展名选择对应的处理方式。这一过程是封装好的，对用户透明
  - 文本文件 非结构化 / 普通的文本文件，每行一条记录
  - JSON 半结构化 / 常见的基于文本的格式，半结构化；大多数库都要求每行一条记录
  - CSV 结构化 / 非常常见的基于文本的格式，通常在电子表格应用中使用
  - SequenceFiles 结构化 / 一种用于键值对数据的常见 Hadoop 文件格式
  - Protocol buffers 结构化 / 一种快速、节约空间的跨语言格式
  - 对象文件 结构化 / 用来将 Spark 作业中的数据存储下来以让共享的代码读取。改变类的时候它会失效，因为它依赖于 Java 序列化
### 文本文件
  - 当我们将一个文本文件读取为 RDD 时，输入的每一行都会成为 RDD 的一个元素
  - 将多个完整的文本文件一次性读取为一个 pair RDD，键是文件名，值是文件内容
  - 读取文本文件使用文件路径作为参数调用 SparkContext 中的 textFile() 函数，就可以读取一个文本文件，如果要控制分区数的话，可以指定 minPartitions
  - Spark 支持读取给定目录中的所有文件，以及在输入路径中使用通配字符（如 part-\*.txt），大规模数据集通常存放在多个文件中，因此这一特性很有用
  - 如果多个输入文件以一个包含数据所有部分的目录的形式出现，如果文件足够小，那么可以使用 SparkContext.wholeTextFiles() 方法
  - wholeTextFiles()方法会返回一个 pair RDD，其中键是输入文件的文件名，在每个文件表示一个特定时间段内的数据时非常有用
  - 保存文本文件
    ```python
    输出文本文件可以使用 saveAsTextFile() 方法接收一个路径，并将 RDD 中的内容都输入到路径对应的文件中
    Spark 将传入的路径作为目录对待，会在那个目录下输出多个文件，这样Spark 就可以从多个节点上并行输出了
    在这个方法中，我们不能控制数据的哪一部分输出到哪个文件中，不过有些输出格式支持控制
    在 Python 中将数据保存为文本文件
            result.saveAsTextFile(outputFile)
    ```
### JSON
  - JSON JavaScript Object Notation，一种常用的Web数据格式，是一种使用较广的半结构化数据格式
  - 读取 JSON 数据的最简单的方式是将数据作为文本文件读取，然后使用 JSON 解析器来对 RDD 中的值进行映射操作
  - 这种方法假设文件中的每一行都是一条 JSON 记录，如果有跨行的 JSON 数据，就只能读入整个文件，然后对每个文件进行解析
  - 在 Python 中读取非结构化的 JSON
    ```python
    import json
    data = input.map(lambda x: json.loads(x))
    ```
  - 保存JSON
    ```python
    写出 JSON 文件比读取它要简单得多，因为不需要考虑格式错误的数据，并且也知道要写出的数据的类型
    可以使用将字符串 RDD 转为解析好的 JSON 数据的库，将由结构化数据组成的 RDD 转为字符串 RDD，然后使用 Spark 的文本文件 API 写出去
     Python 保存为 JSON
            (data.filter(lambda x: x["lovesPandas"]).map(lambda x: json.dumps(x)).saveAsTextFile(outputFile))
    ```
### 逗号分隔值 CSV 与 制表符分隔值TSV
  - 逗号分隔值（CSV）文件每行都有固定数目的字段，字段间用逗号隔开
  - 记录通常是一行一条，不过也不总是这样，有时也可以跨行
  - 与 JSON 中的字段不一样的是，这里的每条记录都没有相关联的字段名，只能得到对应的序号，常规做法是使用第一行中每列的值作为字段名
  - 读取 CSV/TSV 数据和读取 JSON 数据相似，都需要先把文件当作普通文本文件来读取数据，再对数据进行处理
  - 如果在字段中嵌有换行符，就需要完整读入每个文件，然后解析各段
  - 如果每个文件都很大，读取和解析的过程可能会很不幸地成为性能瓶颈
  - 在 Python 中使用 textFile() 读取 CSV
    ```python
    import csv
    import StringIO
    ...
    def loadRecord(line):
      """解析一行CSV记录"""
      input = StringIO.StringIO(line)
      reader = csv.DictReader(input, fieldnames=["name", "favouriteAnimal"])
      return reader.next()
    input = sc.textFile(inputFile).map(loadRecord)
    ```
  - 在 Python 中完整读取 CSV
    ```python
    def loadRecords(fileNameContents):
      """读取给定文件中的所有记录"""
      input = StringIO.StringIO(fileNameContents[1])
      reader = csv.DictReader(input, fieldnames=["name", "favoriteAnimal"])
      return reader
    fullFileData = sc.wholeTextFiles(inputFile).flatMap(loadRecords)

    ```
    如果只有一小部分输入文件，需要使用wholeFile() 方法，可能还需要对输入数据进行重新分区使得 Spark 能够更高效地并行化执行后续操作
  - 保存CSV
    ```python
    和 JSON 数据一样，同样可以通过重用输出编码器来加速
    由于在 CSV 中我们不会在每条记录中输出字段名，因此为了使输出保持一致，需要创建一种映射关系
    一种简单做法是写一个函数，用于将各字段转为指定顺序的数组
    在 Python 中，如果输出字典，CSV 输出器会根据创建输出器时给定的 fieldnames 的顺序帮我们完成这一行为
    我们所使用的 CSV 库要输出到文件或者输出器，所以可以使用 StringWriter 或 StringIO 来将结果放到 RDD 中
    在 Python 中写 CSV
    def writeRecords(records):
      """写出一些CSV记录"""
      output = StringIO.StringIO()
      writer = csv.DictWriter(output, fieldnames=["name", "favoriteAnimal"])
      for record in records:
        writer.writerow(record)
      return [output.getvalue()]

    pandaLovers.mapPartitions(writeRecords).saveAsTextFile(outputFile)
    ```
### SequenceFile
  - SequenceFile 是由没有相对关系结构的键值对文件组成的常用 Hadoop 格式
  - SequenceFile 文件有同步标记，Spark可以用它来定位到文件中的某个点，然后再与记录的边界对齐，这可以让 Spark 使用多个节点高效地并行读取 SequenceFile 文件
  - SequenceFile 也是 Hadoop MapReduce 作业中常用的输入输出格式，所以如果你在使用一个已有的 Hadoop 系统，数据很有可能是以 SequenceFile 的格式供你使用的
  - 由于 Hadoop 使用了一套自定义的序列化框架，因此 SequenceFile 是由实现 Hadoop 的 Writable 接口的元素组成
  - 标准的经验法则是尝试在类名的后面加上 Writable 这个词，然后检查它是否是org.apache.hadoop.io.Writable已知的子类
  - 如果无法为要写出的数据找到对应的 Writable 类型（比如自定义的 case class），可以通过重载 org.apache.hadoop.io.Writable 中的 readfields 和 write 来实现自己的 Writable 类
  - Hadoop 的 RecordReader 会为每条记录重用同一个对象，因此直接调用 RDD 的 cache 会导致失败，只需要使用一个简单的 map() 操作然后将结果缓存即可
  - 许多 Hadoop Writable 类没有实现 java.io.Serializable 接口，因此为了让它们能在 RDD 中使用，还是要用 map()来转换它们
  - Spark 1.1 加入了在 Python 中读取和保存 SequenceFile 的功能，但还是需要使用 Java 或 Scala 来实现自定义 Writable 类
  - Spark 的 Python API 只能将 Hadoop 中存在的基本 Writable 类型转为 Python 类型，并尽量基于可用的 getter 方法处理别的类型
  - 读取SequenceFile
    ```python
    Spark 有专门用来读取 SequenceFile 的接口
    在 SparkContext 中，可以调用 sequenceFile(path, keyClass, valueClass, minPartitions)
    SequenceFile 使用 Writable 类，因此keyClass 和 valueClass 参数都必须使用正确的 Writable 类
    在 Python 读取 SequenceFile
            val data = sc.sequenceFile(inFile,
             "org.apache.hadoop.io.Text", "org.apache.hadoop.io.IntWritable")
    ```
### 对象文件
  - 对象文件看起来就像是对 SequenceFile 的简单封装，它允许存储只包含值的 RDD
  - 和 SequenceFile 不一样的是，对象文件是使用 Java 序列化写出的
  - 使用对象文件的主要原因是它们可以用来保存几乎任意对象而不需要额外的工作
  - 如果修改了类，比如增减了几个字段，已经生成的对象文件就不再可读了
  - 对象文件使用 Java 序列化，它对兼容同一个类的不同版本有一定程度的支持，但是需要程序员去实现
  - 首先，和普通的 SequenceFile 不同，对于同样的对象，对象文件的输出和 Hadoop 的输出不一样
  - 其次，与其他文件格式不同的是，对象文件通常用于 Spark 作业间的通信
  - 最后，Java 序列化有可能相当慢
  - 要保存对象文件，只需在 RDD 上调用 saveAsObjectFile 就行了
  - 读回对象文件也相当简单：用 SparkContext 中的 objectFile() 函数接收一个路径，返回对应的 RDD
  - 对象文件在 Python 中无法使用，不过 Python 中的 RDD 和 SparkContext 支持 saveAsPickleFile()和 pickleFile() 方法作为替代，这使用了 Python 的 pickle 序列化库
  - 对象文件的注意事项同样适用于 pickle 文件：pickle 库可能很慢，并且在修改类定义后，已经生产的数据文件可能无法再读出来
### Hadoop输入输出格式
  - 除了 Spark 封装的格式之外，也可以与任何 Hadoop 支持的格式交互
  - Hadoop 在演进过程中增加了一套新的 MapReduce API，不过有些库仍然使用旧的那套
  - Spark 支持新旧两套 Hadoop 文件 API，提供了很大的灵活性
  - 使用新版的 Hadoop API 读入一个文件，newAPIHadoopFile 接收一个路径以及三个类：
    ```python
    第一个类是“格式”类，代表输入格式，相似的函数 hadoopFile() 则用于使用旧的 API 实现的 Hadoop 输入格式
    第二个类是键的类
    最后一个类是值的类
    如果需要设定额外的 Hadoop 配置属性，也可以传入一个 conf 对象
    ```
    使用旧的 Hadoop API 读取文件在用法上几乎一样，但需要提供旧式 InputFormat 类，Spark 许多自带的封装好的函数（比如 sequenceFile()）都是使用旧式 Hadoop API 实现的
  - 保存Hadoop输出格式
    ```python
    saveAsHadoopFile / saveAsNewAPIHadoopFile
    hadoopDataset/saveAsHadoopDataSet 和 newAPIHadoopDataset/saveAsNewAPIHadoopDataset 来访问 Hadoop 所支持的非文件系统的存储格式
    hadoopDataset() 这一组函数只接收一个 Configuration 对象，这个对象用来设置访问数据源所必需的 Hadoop 属性
    需要使用与配置 Hadoop MapReduce 作业相同的方式来配置这个对象，所以应当按照在 MapReduce 中访问这些数据源的使用说明来配置，并把配置对象传给 Spark
    ```
### protocol buffer
  - Protocol buffer（简称 PB，https://github.com/google/protobuf）最早由 Google 开发，用于内部的远程过程调用（RPC），已经开源
  - PB 是结构化数据，它要求字段和类型都要明确定义，它们是经过优化的，编解码速度快，而且占用空间也很小
  - 比起 XML，PB 能在同样的空间内存储大约 3 到 10 倍的数据，同时编解码速度大约为 XML 的 20 至 100 倍
  - PB 采用一致化编码，因此有很多种创建一个包含多个 PB 消息的文件的方式
  - PB 使用领域专用语言来定义，PB 编译器可以生成各种语言的访问函数（包括 Spark 支持的那些语言）
  - 由于 PB 需要占用尽量少的空间，所以它不是“自描述”的，因为对数据描述的编码需要占用额外的空间
  - PB 由可选字段、必需字段、重复字段三种字段组成
  - 在解析时，可选字段的缺失不会导致解析失败，而必需字段的缺失则会导致数据解析失败
  - 在往 PB 定义中添加新字段时，最好将新字段设为可选字段，毕竟不是所有人都会同时更新到新版本（即使会这样做，还是有可能需要读取以前的旧数据）
  - PB 字段支持许多预定义类型，或者另一个 PB 消息，这些类型包括 string、int32、enum 等
  - Protocol Buffer 的网站（https://developers.google.com/protocol-buffers）了解更多细节
### 文件压缩
  - 在大数据工作中，我们经常需要对数据进行压缩以节省存储空间和网络传输开销
  - 对于大多数 Hadoop 输出格式来说，我们可以指定一种压缩编解码器来压缩数据
  - Spark 原生的输入方式（textFile 和 sequenceFile）可以自动处理一些类型的压缩，在读取压缩后的数据时，一些压缩编解码器可以推测压缩类型
  - 可分割：
    ```python
    选择一个输出压缩编解码器可能会对这些数据以后的用户产生巨大影响，对于像 Spark 这样的分布式系统，我们通常会尝试从多个不同机器上一起读入数据
    要实现这种情况，每个工作节点都必须能够找到一条新记录的开端
    有些压缩格式会使这变得不可能，而必须要单个节点来读入所有数据，这就很容易产生性能瓶颈，可以很容易地从多个节点上并行读取的格式被称为“可分割”的格式
    ```
## 文件系统
### 本地 / 常规文件系统
  - Spark 支持从本地文件系统中读取文件，不过它要求文件在集群中所有节点的相同路径下都可以找到
  - 一些像 NFS、AFS 以及 MapR 的 NFS layer 这样的网络文件系统会把文件以常规文件系统的形式暴露给用户
  - 如果数据已经在这些系统中，那么只需要指定输入为一个 file:// 路径；只要这个文件系统挂载在每个节点的同一个路径下，Spark 就会自动处理
  - 如果文件还没有放在集群中的所有节点上，可以在驱动器程序中从本地读取该文件而无需使用整个集群，然后再调用 parallelize 将内容分发给工作节点
  - 不过这种方式可能会比较慢，所以推荐的方法是将文件先放到像 HDFS、NFS、S3 等共享文件系统上
### Amazon S3
  - 用 Amazon S3 来存储大量数据正日益流行。当计算节点部署在 Amazon EC2 上的时候，使用 S3 作为存储尤其快，但是在需要通过公网访问数据时性能会差很多
  - 要在 Spark 中访问 S3 数据，应该首先把 S3 访问凭据设置为 AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY 环境变量，可以从 Amazon Web Service 控制台创建这些凭据
  - 接下来，将一个以 s3n:// 开头的路径以 s3n://bucket/path-within-bucket 的形式传给 Spark 的输入方法
  - 和其他所有文件系统一样，Spark 也能在 S3 路径中支持通配字符，例如 s3n://bucket/my-Files/\*.txt
  - 如果从 Amazon 那里得到 S3 访问权限错误，请确保你指定了访问密钥的账号对数据桶有“read”（读）和“list”（列表）的权限，Spark 需要列出桶内的内容，来找到想要读取的数据
### HDFS
  - Hadoop 分布式文件系统（HDFS）是一种广泛使用的文件系统，Spark 能够很好地使用它
  - HDFS 被设计为可以在廉价的硬件上工作，有弹性地应对节点失败，同时提供高吞吐量。
  - Spark 和 HDFS 可以部署在同一批机器上，这样 Spark 可以利用数据分布来尽量避免一些网络开销
  - 在 Spark 中使用 HDFS 只需要将输入输出路径指定为 hdfs://master:port/path 就够了
  - HDFS 协议随 Hadoop 版本改变而变化，因此如果使用的 Spark 是依赖于另一个版本的 Hadoop 编译的，那么读取会失败
  - 如果从源代码编译，可以在环境变量中指定 SPARK_HADOOP_VERSION= 来基于另一个版本的 Hadoop 进行编译，也可以直接下载预编译好的 Spark 版本
  - 可以根据运行 hadoop version 的结果来获得环境变量要设置的值
## Spark SQL中的结构化数据
  - Spark SQL 是在 Spark 1.0 中新加入 Spark 的组件
  - 结构化数据指的是有结构信息的数据，也就是所有的数据记录都具有一致字段结构的集合，Spark SQL 支持多种结构化数据源作为输入
  - 在各种情况下，我们把一条 SQL 查询给 Spark SQL，让它对一个数据源执行查询（选出一些字段或者对字段使用一些函数），然后得到由 Row 对象组成的 RDD，每个 Row 对象表示一条记录
  - 在 Python 中，可以使用 row[column_number] 以及 row.column_name 来访问元素
### Apache Hive
  - Hadoop 上的一种常见的结构化数据源
  - Hive 可以在 HDFS 内或者在其他存储系统上存储多种格式的表，这些格式从普通文本到列式存储格式，应有尽有，Spark SQL 可以读取 Hive 支持的任何表
  - 要把 Spark SQL 连接到已有的 Hive 上：
    ```python
    需要提供 Hive 的配置文件，将 hive-site.xml 文件复制到 Spark 的 conf/ 目录下
    再创建出 HiveContext 对象，也就是 Spark SQL 的入口
    然后就可以使用 Hive 查询语言（HQL）来对表进行查询，并以由行组成的 RDD 的形式拿到返回数据
    ```
  - 用 Python 创建 HiveContext 并查询数据
    ```python
    from pyspark.sql import HiveContext

    hiveCtx = HiveContext(sc)
    rows = hiveCtx.sql("SELECT name, age FROM users")
    firstRow = rows.first()
    print firstRow.name
    ```
### JSON
  - 对于记录间结构一致的 JSON 数据，Spark SQL 也可以自动推断出它们的结构信息，并将这些数据读取为记录
  - 要读取 JSON 数据，首先需要和使用 Hive 一样创建一个 HiveContext，不过在这种情况下我们不需要安装好 Hive，也就是说不需要 hive-site.xml 文件
  - 然后使用 HiveContext.jsonFile 方法来从整个文件中获取由 Row 对象组成的 RDD
  - 除了使用整个 Row 对象，也可以将 RDD 注册为一张表，然后从中选出特定的字段
  - 在 Python 中使用 Spark SQL 读取 JSON 数据
    ```python
    tweets = hiveCtx.jsonFile("tweets.json")
    tweets.registerTempTable("tweets")
    results = hiveCtx.sql("SELECT user.name, text FROM tweets")        # 选取 user.name text 字段
    ```
## 数据库
  - 通过数据库提供的 Hadoop 连接器或者自定义的 Spark 连接器，Spark 可以访问一些常用的数据库系统
### Cassandra
  - 随着 DataStax 开源其用于 Spark 的 Cassandra 连接器（https://github.com/datastax/spark-cassandra-connector），Spark 对 Cassandra 的支持大大提升
  - 这个连接器目前还不是 Spark 的一部分，因此需要添加一些额外的依赖到你的构建文件中才能使用它
  - Cassandra 还没有使用 Spark SQL，不过它会返回由 CassandraRow 对象组成的 RDD，这些对象有一部分方法与 Spark SQL 的 Row 对象的方法相同
  - Spark 的 Cassandra 连接器目前只能在 Java 和 Scala 中使用
  - Cassandra 连接器要读取一个作业属性来决定连接到哪个集群
  - 我们把spark.cassandra.connection.host 设置为指向 Cassandra 集群，如果有用户名和密码的话，则需要分别设置spark.cassandra.auth.username 和 spark.cassandra.auth.password
  - 假定只有一个 Cassandra 集群要连接，可以在创建 SparkContext 时就把这些都设好
  - Datastax 的 Cassandra 连接器使用 Scala 中的隐式转换来为 SparkContext 和 RDD 提供一些附加函数
  - 除了读取整张表，也可以查询数据集的子集，通过在 cassandraTable() 的调用中加上 where 子句，可以限制查询的数据，例如 sc.cassandraTable(...).where("key=?", "panda")
  - Cassandra 连接器支持把多种类型的RDD保存到 Cassandra 中，我们可以直接保存由 CassandraRow对象组成的 RDD，这对于在表之间复制数据比较有用
  - 通过指定列的映射关系，我们也可以存储不是行的形式而是元组和列表的形式的 RDD
### HBase
  - 由于 org.apache.hadoop.hbase.mapreduce.TableInputFormat 类的实现，Spark 可以通过 Hadoop 输入格式访问 HBase
  - 这个输入格式会返回键值对数据，其中键的类型为org.apache.hadoop.hbase.io.ImmutableBytesWritable，而值的类型为org.apache.hadoop.hbase.client.Result
  - Result 类包含多种根据列获取值的方法，在其 API 文档（https://hbase.apache.org/apidocs/org/apache/hadoop/hbase/client/Result.html）中有所描述
  - 要将 Spark 用于 HBase，需要使用正确的输入格式调用 SparkContext.newAPIHadoopRDD
  - TableInputFormat 包含多个可以用来优化对 HBase 的读取的设置项，比如将扫描限制到一部分列中，以及限制扫描的时间范围
  - 可以在 TableInputFormat 的 API 文档中找到这些选项，并在 HBaseConfiguration 中设置它们，然后再把它传给 Spark
### Elasticsearch
  - Spark 可以使用 Elasticsearch-Hadoop（https://github.com/elastic/elasticsearch-hadoop）从 Elasticsearch 中读写数据
  - Elasticsearch 是一个开源的、基于 Lucene 的搜索系统
  - Elasticsearch 连接器和我们研究过的其他连接器不大一样，它会忽略我们提供的路径信息，而依赖于在 SparkContext 中设置的配置项
  - Elasticsearch 的 OutputFormat 连接器也没有用到 Spark 所封装的类型，所以我们使用 saveAsHadoopDataSet 来代替，这意味着我们需要手动设置更多属性
  - 最新版的 Elasticsearch Spark 连接器用起来更简单，支持返回 Spark SQL 中的行对象，这个连接器仍然是隐藏的，因为行转换还不支持 Elasticsearch 中所有的原生类型
  - 就输出而言，Elasticsearch 可以进行映射推断，但是偶尔会推断出不正确的数据类型，因此如你要存储字符串以外的数据类型，最好明确指定类型映射（https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-put-mapping.html）
***

# 共享变量 ( 累加器 / 广播变量 / 基于分区进行操作 / 与外部程序间的管道 / 数值RDD的操作)
  - 对应代码 learning-spark/src/python/ChapterSixExample.py
  - 文本文件 learning-spark/files/callsigns
## 累加器
  - 累加器，提供了将工作节点中的值聚合到驱动器程序中的简单语法，累加器的一个常见用途是在调试时对作业执行过程中的事件进行计数
  - 工作节点上的任务不能访问累加器的值，从这些任务的角度来看，累加器是一个只写变量
  - 在这种模式下，累加器的实现可以更加高效，不需要对每次更新操作进行复杂的通信
  - 计数在很多时候都非常方便，比如有多个值需要跟踪时，或者当某个值需要在并行程序的多个地方增长时
### 累加器的用法
  - 通过在驱动器中调用 SparkContext.accumulator(initialValue) 方法，创建出存有初始值的累加器，返回值为 org.apache.spark.Accumulator[T] 对象，其中 T 是初始值 initialValue 的类型
  - Spark 闭包里的执行器代码可以使用累加器的 += 方法（在 Java 中是 add）增加累加器的值
  - 驱动器程序可以调用累加器的 value 属性（在 Java 中使用 value() 或 setValue()）来访问累加器的值
  - 在python中累加空行：
    ```python
    file = sc.textFile('inputFile')
    blankLines = sc.accumulator(0)
    def extractCallSigns(line):
      global blankLines
      if (line == ''):
        blankLines += 1
      return line.split(' ')
    callSigns = file.flatMap(extractCallSigns)
    callSigns.saveAsTextFile('./callSigns')
    print(blankLines.value)
    Out[90]: 2        # 两个空行

    注意，只有在运行saveAsTextFile() 行动操作后才能看到正确的计数，因为行动操作前的转化操作 flatMap() 是惰性的
    ```
### 行动操作 / 转化操作中的累加器容错性
  - Spark 会自动重新执行失败的或较慢的任务来应对有错误的或者比较慢的机器例如
  - 如果对某分区执行 map() 操作的节点失败了，Spark 会在另一个节点上重新运行该任务
  - 即使该节点没有崩溃，而只是处理速度比别的节点慢很多，Spark 也可以抢占式地在另一个节点上启动一个“投机”（speculative）型的任务副本
  - 如果该任务更早结束就可以直接获取结果
  - 即使没有节点失败，Spark 有时也需要重新运行任务来获取缓存中被移除出内存的数据
  - 因此最终结果就是同一个函数可能对同一个数据运行了多次，这取决于集群发生了什么
  - 对于要在行动操作中使用的累加器，Spark只会把每个任务对各累加器的修改应用一次
  - 因此，如果想要一个无论在失败还是重复计算时都绝对可靠的累加器，我们必须把它放在 foreach() 这样的行动操作中
  - 对于在 RDD 转化操作中使用的累加器，就不能保证有这种情况了，转化操作中累加器可能会发生不止一次更新，累加器通常只用于调试目的
  - python中使用foreach:
    ```python
    file = sc.textFile('/home/leondgarse/practice_code/spark/learning-spark/files/callsigns')
    count = sc.accumulator(0)
    def kCounter(line):
      global count
      if 'K' in line:
        count += 1
    file.foreach(kCounter)
    count.value
    Out[103]: 2

    file.foreach(kCounter)
    count.value
    Out[108]: 4
    ```
### 自定义累加器
  - Spark 还直接支持 Double、Long 和 Float 型的累加器
  - 除此以外，Spark 也引入了自定义累加器和聚合操作的 API（比如找到要累加的值中的最大值，而不是把这些值加起来）
  - 自定义累加器需要扩展 AccumulatorParam，这在 Spark API 文档（http://spark.apache.org/docs/latest/api/scala/index.html#package）中有所介绍
  - 只要该操作同时满足交换律和结合律，就可以使用任意操作来代替数值上的加法，如，sum 和 max 既满足交换律又满足结合律，是 Spark 累加器中的常用操作
## 广播变量
  - 可以让程序高效地向所有工作节点发送一个较大的只读值，以供一个或多个 Spark 操作使用，如，向所有节点发送一个较大的只读查询表，甚至是机器学习算法中的一个很大的特征向量
  - Spark 会自动把闭包中所有引用到的变量发送到工作节点上，虽然这很方便，但也很低效
  - 首先，默认的任务发射机制是专门为小任务进行优化的
  - 其次，事实上你可能会在多个并行操作中使用同一个变量，但是 Spark 会为每个操作分别发送
  - 从主节点为每个任务发送一个大的数组就会代价巨大，使用广播变量来解决这一问题
  - 广播变量其实就是类型为spark.broadcast.Broadcast[T] 的一个对象，其中存放着类型为 T 的值
  - 可以在任务中通过对Broadcast 对象调用 value 来获取该对象的值
  - 这个值只会被发送到各节点一次，使用的是一种高效的类似 BitTorrent 的通信机制
### 广播变量的使用
  - 通过对一个类型 T 的对象调用 SparkContext.broadcast 创建出一个 Broadcast[T] 对象，任何可序列化的类型都可以这么实现
  - 通过 value 属性访问该对象的值
  - 变量只会被发到各个节点一次，应作为只读值处理（修改这个值不会影响到别的节点）
  - python从文件中创建一个broadcast变量：
    ```python
    def loadCallSignTable():
      f = open('/home/leondgarse/practice_code/spark/learning-spark/files/callsign_tbl_sorted')
      return f.readlines()
    signPrefixes = sc.broadcast(loadCallSignTable)
    ```
### 广播的优化
  - 当广播一个比较大的值时，选择既快又好的序列化格式是很重要的
  - 因为如果序列化对象的时间很长或者传送花费的时间太久，这段时间很容易就成为性能瓶颈
  - 尤其是，Spark 的 Scala 和 Java API 中默认使用的序列化库为 Java 序列化库，因此它对于除基本类型的数组以外的任何对象都比较低效
  - 可以使用 spark.serializer 属性选择另一个序列化库来优化序列化过程（如使用 Kryo 这种更快的序列化库)
## 基于分区进行操作
  - 对于打开数据库连接或创建随机数生成器等操作，Spark 提供基于分区的 map 和 foreach，让你的部分代码只对 RDD 的每个分区运行一次，这样可以帮助降低这些操作的代价
  - 按分区执行的操作符
    ```python
    mapPartitions() 提供该分区中元素的迭代器，返回的元素的迭代器 f: (Iterator[T]) → Iterator[U]
    mapPartitionsWithIndex() 提供分区序号，以及每个分区中的元素的迭代器，返回的元素的迭代器，f: (Int, Iterator[T]) → Iterator[U]
    foreachPartitions() 提供元素迭代器，f: (Iterator[T]) → Unit
    ```
  - 在 Python 中使用共享连接池:
    ```python
    def processCallSigns(signs):
      """Lookup call signs using a connection pool"""
      # Create a connection pool
      http = urllib3.PoolManager()        # 在每个分区内共享一个数据库连接池，来避免建立太多连接
      # the URL associated with each call sign record
      urls = map(lambda x: "http://73s.com/qsos/%s.json" % x, signs)
      # create the requests (non-blocking)
      requests = map(lambda x: (x, http.request('GET', x)), urls)
      # fetch the results
      result = map(lambda x: (x[0], json.loads(x[1].data)), requests)
      # remove any empty results and return
      return filter(lambda x: x[1] is not None, result)
    ```
  - 在 Python 中使用 mapPartitions() 求平均值:
    ```python
    def partitionCtr(nums):
      """计算分区的sumCounter"""
      sumCount = [0, 0]
      for num in nums:
        sumCount[0] += num
        sumCount[1] += 1
      return [sumCount]

    def fastAvg(nums):
      """计算平均值"""
      sumCount = nums.mapPartitions(partitionCtr).reduce(combineCtrs)
      return sumCount[0] / float(sumCount[1])
    ```
## 与外部程序间的管道
  - Spark 的 pipe() 方法可以让我们使用任意一种语言实现 Spark 作业中的部分逻辑，只要它能读写 Unix 标准流就行
  - 通过 pipe()，可以将 RDD 中的各元素从标准输入流中以字符串形式读出，并对这些元素执行任何你需要的操作，然后把结果以字符串的形式写入标准输出，这个过程就是 RDD 的转化操作过程
  - 需要做的事情是让每个工作节点都能访问外部脚本，并调用这个脚本来对 RDD 进行实际的转化操作
  - 在 Python 中使用 pipe() 调用 finddistance.R 的驱动器程序
    ```python
    # Compute the distance of each call using an external R program
    distScript = os.getcwd()+"/src/R/finddistance.R"
    distScriptName = "finddistance.R"
    sc.addFile(distScript)

    def hasDistInfo(call):
      """Verify that a call has the fields required to compute the distance"""
      requiredFields = ["mylat", "mylong", "contactlat", "contactlong"]
      return all(map(lambda f: call[f], requiredFields))

    def formatCall(call):
      """Format a call so that it can be parsed by our R program"""
      return "{0},{1},{2},{3}".format(
        call["mylat"], call["mylong"],
        call["contactlat"], call["contactlong"])

    pipeInputs = contactsContactList.values().flatMap(
      lambda calls: map(formatCall, filter(hasDistInfo, calls)))
    distances = pipeInputs.pipe(SparkFiles.get(distScriptName))
    print distances.collect()
    ```
  - 通过 SparkContext.addFile(path)，可以构建一个文件列表，让每个工作节点在 Spark 作业中下载列表中的文件
  - 当作业中的行动操作被触发时，这些文件就会被各节点下载，然后就可以在工作节点上通过 SparkFiles.getRootDirectory 找到它们
  - 也可以使用 SparkFiles.get(Filename) 来定位单个文件
  - 也可以使用其他的远程复制工具来把脚本文件放到各节点可以找到的位置上
  - 如果需要的话，也可以通过 pipe() 指定命令行环境变量，只需要把环境变量到对应值的映射表作为 pipe() 的第二个参数传进去，Spark 就会设置好这些值
## 数值RDD的操作
  - Spark 的数值操作是通过流式算法实现的，允许以每次一个元素的方式构建出模型，这些统计数据都会在调用 stats() 时通过一次遍历数据计算出来，并以 StatsCounter 对象返回
  - StatsCounter中可用的汇总统计数据
    ```python
    count() RDD 中的元素个数
    mean() 元素的平均值
    sum() 总和
    max() 最大值
    min() 最小值
    variance() 元素的方差
    sampleVariance() 从采样中计算出的方差
    stdev() 标准差
    sampleStdev() 采样的标准差
    ```
    如果你只想计算这些统计数据中的一个，也可以直接对 RDD 调用对应的方法，比如 rdd.mean()或者 rdd.sum()
  - 用 Python 移除异常值
    ```python
    # Convert our RDD of strings to numeric data so we can compute stats and
    # remove the outliers.
    distanceNumerics = distances.map(lambda string: float(string))
    stats = distanceNumerics.stats()
    stddev = stats.stdev()
    mean = stats.mean()
    reasonableDistances = distanceNumerics.filter(
      lambda x: math.fabs(x - mean) < 3 * stddev)
    print reasonableDistances.collect()
    ```
***

# 在集群上运行 Spark （运行时架构 / spark-submit / 集群管理器）
  - Spark 的一大好处就是可以通过增加机器数量并使用集群模式运行，来扩展程序的计算能力
  - 可以在小数据集上利用本地模式快速开发并验证应用，然后无需修改代码就可以在大规模集群上运行
## Spark运行时架构
### 分布式环境下Spark 集群的主 / 从结构
  - Spark驱动器程序 --> 集群管理器Mesos / YARN 或独立集群管理器 --> 集群工作节点 (执行器进程) 1 / 2 / 3 / ...
  - 驱动器（Driver）节点：负责中央协调，调度各个分布式工作节点
  - 执行器（executor）节点：驱动器节点可以和大量的执行器节点进行通信，它们也都作为独立的 Java 进程运行
  - Spark 应用（application）：驱动器节点和所有的执行器节点一起被称为一个Spark 应用
  - 集群管理器（Cluster Manager）：在集群中的机器上启动Spark 应用的一个外部服务
  - Spark 自带的集群管理器被称为独立集群管理器，Spark 也能运行在 Hadoop YARN 和 Apache Mesos 这两大开源集群管理器上
  - 从上层来看，所有的 Spark 程序都遵循同样的结构：
    ```python
    程序从输入数据创建一系列 RDD
    再使用转化操作派生出新的 RDD
    最后使用行动操作收集或存储结果 RDD 中的数据
    ```
### Spark驱动器程序
  - Spark 驱动器是执行程序中的 main() 方法的进程，它执行用户编写的用来创建 SparkContext、创建 RDD，以及进行 RDD 的转化操作和行动操作的代码
  - 驱动器程序在 Spark 应用中有下述两个职责
  - 把用户程序转为任务
    ```python
    Spark 驱动器程序负责把用户程序转为多个物理执行的单元，这些单元也被称为任务（task）
    Spark 程序其实是隐式地创建出了一个由操作组成的逻辑上的有向无环图（Directed Acyclic Graph，简称 DAG）,当驱动器程序运行时，它会把这个逻辑图转为物理执行计划
    Spark 会对逻辑执行计划作一些优化，比如将连续的映射转为流水线化执行，将多个操作合并到一个步骤中等
    这样 Spark 就把逻辑计划转为一系列步骤（stage），而每个步骤又由多个任务组成，这些任务会被打包并送到集群中
    任务是 Spark 中最小的工作单元，用户程序通常要启动成百上千的独立任务
    ```
    为执行器节点调度任务
    ```python
    有了物理执行计划之后，Spark 驱动器程序必须在各执行器进程间协调任务的调度
    执行器进程启动后，会向驱动器进程注册自己，因此，驱动器进程始终对应用中所有的执行器节点有完整的记录
    每个执行器节点代表一个能够处理任务和存储 RDD 数据的进程
    Spark 驱动器程序会根据当前的执行器节点集合，尝试把所有任务基于数据所在位置分配给合适的执行器进程
    当任务执行时，执行器进程会把缓存数据存储起来，而驱动器进程同样会跟踪这些缓存数据的位置，并且利用这些位置信息来调度以后的任务，以尽量减少数据的网络传输
    驱动器程序会将一些 Spark 应用的运行时的信息通过网页界面呈现出来，默认在端口 4040 上，在本地模式下，访问 http://localhost:4040 可以看到这个网页
    ```
### 执行器节点
  - Spark 执行器节点是一种工作进程，负责在 Spark 作业中运行任务，任务间相互独立
  - Spark 应用启动时，执行器节点就被同时启动，并且始终伴随着整个 Spark 应用的生命周期而存在
  - 如果有执行器节点发生了异常或崩溃，Spark 应用也可以继续执行
  - 执行器进程有两大作用：
    ```python
    负责运行组成 Spark 应用的任务，并将结果返回给驱动器进程
    通过自身的块管理器（Block Manager）为用户程序中要求缓存的 RDD 提供内存式存储，RDD 是直接缓存在执行器进程内的，因此任务可以在运行时充分利用缓存数据加速运算
    ```
### 集群管理器
  - Spark 依赖于集群管理器来启动执行器节点，而在某些特殊情况下，也依赖集群管理器来启动驱动器节点
  - 集群管理器是 Spark 中的可插拔式组件，除了 Spark 自带的独立集群管理器，Spark 也可以运行在其他外部集群管理器上，比如 YARN 和 Mesos
  - Spark 文档中始终使用驱动器节点和执行器节点的概念来描述执行 Spark 应用的进程
  - 而主节点（master）和工作节点（worker）的概念则被用来分别表述集群管理器中的中心化的部分和分布式的部分
  - 在Hadoop YARN 会启动一个叫作资源管理器（Resource Manager）的主节点守护进程，以及一系列叫作节点管理器（Node Manager）的工作节点守护进程
  - 在 YARN 的工作节点上，Spark 不仅可以运行执行器进程，还可以运行驱动器进程
### 启动一个程序
  - 不论使用的是哪一种集群管理器，都可以使用 Spark 提供的统一脚本 spark-submit 将应用提交到那种集群管理器上
  - 通过不同的配置选项，spark-submit 可以连接到相应的集群管理器上，并控制应用所使用的资源数量
  - 在使用某些特定集群管理器时，spark-submit 也可以将驱动器节点运行在集群内部（比如一个 YARN 的工作节点），但对于其他的集群管理器，驱动器节点只能被运行在本地机器上
### 在集群上运行 Spark 应用的详细过程
  - (1) 用户通过 spark-submit 脚本提交应用。
  - (2) spark-submit 脚本启动驱动器程序，调用用户定义的 main() 方法。
  - (3) 驱动器程序与集群管理器通信，申请资源以启动执行器节点。
  - (4) 集群管理器为驱动器程序启动执行器节点。
  - (5) 驱动器进程执行用户应用中的操作。根据程序中所定义的对 RDD 的转化操作和行动操作，驱动器节点把工作以任务的形式发送到执行器进程。
  - (6) 任务在执行器程序中进行计算并保存结果。
  - (7) 如果驱动器程序的 main() 方法退出，或者调用了 SparkContext.stop()，驱动器程序会终止执行器进程，并且通过集群管理器释放资源
## 使用spark-submit部署应用
  - 如果在调用 spark-submit 时除了脚本或 JAR 包的名字之外没有别的参数，那么这个 Spark 程序只会在本地执行
  - 将应用提交到 Spark 独立集群上的时候，可以将独立集群的地址和希望启动的每个执行器进程的大小作为附加标记提供：
    ```python
    bin/spark-submit --master spark://host:7077 --executor-memory 10g my_script.py
    --master 标记指定要连接的集群 URL；在这个示例中，spark:// 表示集群使用独立模式
    ```
  - 除了集群 URL，spark-submit 还提供了各种选项，可以控制应用每次运行的各项细节
  - 这些选项主要分为两类，第一类是调度信息，比如你希望为作业申请的资源量，第二类是应用的运行时依赖，比如需要部署到所有工作节点上的库和文件
  - spark-submit 的一般格式
    ```python
    bin/spark-submit [options] <app jar | python file> [app options]
    [options] 是要传给 spark-submit 的标记列表，可以运行 spark-submit --help 列出所有可以接收的标记
    <app jar | python File> 表示包含应用入口的 JAR 包或 Python 脚本
    [app options] 是传给应用的选项，如果程序要处理传给 main() 方法的参数，它只会得到[app options] 对应的标记，不会得到 spark-submit 的标记
    ```
  - option一些常见的标记：
    ```python
    --master 表示要连接的集群管理器
    --deploy-mode 选择在本地（客户端“client”）启动驱动器程序，还是在集群中的一台工作节点机器（集群“cluster”）上启动
            在客户端模式下， spark-submit 会将驱动器程序运行在 spark-submit 被调用的这台机器上
            在集群模式下，驱动器程序会被传输并执行于集群的一个工作节点上
            默认是本地模式
    --class 运行 Java 或 Scala 程序时应用的主类
    --name 应用的显示名，会显示在 Spark 的网页用户界面中
    --jars 需要上传并放到应用的 CLASSPATH 中的 JAR 包的列表，如果应用依赖于少量第三方的 JAR 包，可以把它们放在这个参数里
    --files 需要放到应用工作目录中的文件的列表，这个参数一般用来放需要分发到各节点的数据文件
    --py-files 需要添加到 PYTHONPATH 中的文件的列表，其中可以包含 .py、.egg 以及 .zip 文件
    --executor-memory 执行器进程使用的内存量，以字节为单位，可以使用后缀指定更大的单位，比如“512m”（512 MB）或“15g”（15 GB）
    --driver-memory 驱动器进程使用的内存量，以字节为单位，可以使用后缀指定更大的单位，比如“512m”（512 MB）或“15g”（15 GB）
    ```
  - spark-submit的 --master 标记可以接收的值
    ```python
    spark://host:port 连接到指定端口的 Spark 独立集群上，默认情况下 Spark 独立主节点使用 7077 端口
    mesos://host:port 连接到指定端口的 Mesos 集群上，默认情况下 Mesos 主节点监听 5050 端口
     yarn 连接到一个 YARN 集群，当在 YARN 上运行时，需要设置环境变量 HADOOP_CONF_DIR 指向 Hadoop 配置目录，以获取集群信息
    local 运行本地模式，使用单核
    local[N] 运行本地模式，使用 N 个核心
    local[*] 运行本地模式，使用尽可能多的核心
    ```
  - spark-submit 还允许通过 --conf prop=value 标记设置任意的 SparkConf 配置选项，也可以使用 --properties-File 指定一个包含键值对的属性文件
  - 使用独立集群模式提交Java应用
    ```python
    $ ./bin/spark-submit \
     --master spark://hostname:7077 \
     --deploy-mode cluster \
     --class com.databricks.examples.SparkExample \
     --name "Example Program" \
     --jars dep1.jar,dep2.jar,dep3.jar \
     --total-executor-cores 300 \
     --executor-memory 10g \
     myApp.jar "options" "to your application" "go here"
    ```
  - 使用YARN客户端模式提交Python应用
    ```python
    $ export HADOP_CONF_DIR=/opt/hadoop/conf
    $ ./bin/spark-submit \
     --master yarn \
     --py-files somelib-1.2.egg,otherlib-4.4.zip,other-file.py \
     --deploy-mode client \
     --name "Example Program" \
     --queue exampleQueue \
     --num-executors 40 \
     --executor-memory 10g \
     my_script.py "options" "to your application" "go here"
    ```
## 打包代码与依赖
  - 如果程序引入了任何既不在 org.apache.spark 包内也不属于语言运行时的库的依赖，就需要确保所有的依赖在该 Spark 应用运行时都能被找到
  - 当提交应用时，绝不要把 Spark 本身放在提交的依赖中，spark-submit 会自动确保 Spark 在你的程序的运行路径中
  - Python 用户
    ```python
    可以通过标准的 Python 包管理器（比如 pip 和 easy_install）直接在集群中的所有机器上安装所依赖的库
    或者把依赖手动安装到 Python 安装目录下的 site-packages/ 目录中
    也可以使用 spark-submit 的 --py-Files 参数提交独立的库，这样它们也会被添加到 Python 解释器的路径中
    如果没有在集群上安装包的权限，可以手动添加依赖库，这也很方便，但是要防范与已经安装在集群上的那些包发生冲突
    ```
  - Java 和 Scala 用户
    ```python
    当只有一两个库的简单依赖时，可以通过 spark-submit 的 --jars 标记提交独立的 JAR 包依赖
    依赖库很多时，常规的做法是使用构建工具，生成单个大 JAR 包，包含应用的所有的传递依赖，这通常被称为超级（uber）JAR 或者组合（assembly）JAR
    ```
    Java 和 Scala 中使用最广泛的构建工具是 Maven 和 sbt，它们都可以用于这两种语言，不过 Maven 通常用于 Java 工程，而 sbt 则一般用于 Scala 工程
  - 依赖冲突
    ```python
    当用户应用与 Spark 本身依赖同一个库时可能会发生依赖冲突，导致程序崩溃，这种情况不是很常见
    通常，依赖冲突表现为 Spark 作业执行过程中抛出NoSuchMethodError、ClassNotFoundException，或其他与类加载相关的 JVM 异常
    对于这种问题，主要有两种解决方式：
    一是修改应用，使其使用的依赖库的版本与 Spark 所使用的相同
    二是使用通常被称为“shading”的方式打包你的应用
    shading 可以让你以另一个命名空间保留冲突的包，并自动重写应用的代码使得它们使用重命名后的版本
    ```
## Spark应用内与应用间调度
  - 许多集群是在多个用户间共享的，如果两个用户都启动了希望使用整个集群所有资源的应用，Spark 有一系列调度策略来保障资源不会被过度使用，还允许工作负载设置优先级
  - 在调度多用户集群时，Spark 主要依赖集群管理器来在 Spark 应用间共享资源
  - 当 Spark 应用向集群管理器申请执行器节点时，应用收到的执行器节点个数可能比它申请的更多或者更少，这取决于集群的可用性与争用
  - 许多集群管理器支持队列，可以为队列定义不同优先级或容量限制，这样 Spark 就可以把作业提交到相应的队列中，请查看所使用的集群管理器的文档获取详细信息
  - Spark 应用有一种特殊情况，就是那些长期运行（long lived）的应用，这意味着这些应用从不主动退出
  - Spark SQL 中的 JDBC 服务器就是一个长期运行的 Spark 应用，当 JDBC 服务器启动后，它会从集群管理器获得一系列执行器节点，然后就成为用户提交 SQL 查询的永久入口
  - 由于这个应用本身就是为多用户调度工作的，所以它需要一种细粒度的调度机制来强制共享资源
  - Spark 提供了一种用来配置应用内调度策略的机制
  - Spark 内部的公平调度器（Fair Scheduler）会让长期运行的应用定义调度任务的优先级队列，参考公平调度器的官方文档：（http://spark.apache.org/docs/latest/job-scheduling.html）
## 独立集群管理器
  - Spark 独立集群管理器提供在集群上运行应用的简单方法
  - 这种集群管理器由一个主节点和几个工作节点组成，各自都分配有一定量的内存和 CPU 核心，当提交应用时，可以配置执行器进程使用的内存量，以及所有执行器进程使用的 CPU 核心总数
  - 要启动独立集群管理器，既可以通过手动启动一个主节点和多个工作节点来实现，也可以使用 Spark 的 sbin 目录中的启动脚本来实现
  - 启动脚本使用最简单的配置选项，但是需要预先设置机器间的 SSH 无密码访问
### 启动集群管理器
  - 要使用集群启动脚本，请按如下步骤执行：
    ```python
    将编译好的 Spark 复制到所有机器的一个相同的目录下，比如 /home/yourname/spark
    设置好从主节点机器到其他机器的 SSH 无密码登陆
        需要在所有机器上有相同的用户账号，并在主节点上通过 ssh-keygen 生成 SSH 私钥，然后将这个私钥放到所有工作节点的 .ssh/authorized_keys 文件中
        # 在主节点上：运行ssh-keygen并接受默认选项
        $ ssh-keygen -t dsa
        # 在工作节点上：
        # 把主节点的~/.ssh/id_dsa.pub文件复制到工作节点上，然后使用：
        $ cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys
        $ chmod 644 ~/.ssh/authorized_keys

    编辑主节点的 conf/slaves 文件并填上所有工作节点的主机名
    在主节点上运行 sbin/start-all.sh（要在主节点上运行而不是在工作节点上）以启动集群
        如果全部启动成功，不会得到需要密码的提示符，而且可以在 http://masternode:8080 看到集群管理器的网页用户界面，上面显示着所有的工作节点
    要停止集群，在主节点上运行 bin/stop-all.sh
    ```
  - 手动启动集群：
    ```python
    使用 Spark 的 bin/ 目录下的 spark-class 脚本分别手动启动主节点和工作节点
    在主节点上，输入：
    bin/spark-class org.apache.spark.deploy.master.Master

    然后在工作节点上输入：
    bin/spark-class org.apache.spark.deploy.worker.Worker spark://masternode:7077
    其中 masternode 是主节点的主机名
    ```
  - 默认情况下，集群管理器会选择合适的默认值自动为所有工作节点分配 CPU 核心与内存
  - 配置独立集群管理器的更多细节请参考 Spark 的官方文档（http://spark.apache.org/docs/latest/spark-standalone.html）
### 提交应用
  - 要向独立集群管理器提交应用，需要把 spark://masternode:7077 作为主节点参数传给 spark-submit：
    ```python
    spark-submit --master spark://masternode:7077 yourapp
    ```
  - 这个集群的 URL 也显示在独立集群管理器的网页界面（位于 http://masternode:8080）上
  - 注意，提交时使用的主机名和端口号必须精确匹配用户界面中的 URL
  - 这有可能会使得使用 IP 地址而非主机名的用户遇到问题，即使 IP 地址绑定到同一台主机上，如果名字不是完全匹配的话，提交也会失败
  - 有些管理员可能会配置 Spark 不使用 7077 端口而使用别的端口，要确保主机名和端口号的一致性，一个简单的方法是从主节点的用户界面中直接复制粘贴 URL
  - 可以使用 --master 参数以同样的方式启动 spark-shell 或 pyspark，来连接到该集群上：
    ```python
    spark-shell --master spark://masternode:7077
    pyspark --master spark://masternode:7077
    ```
  - 要检查应用或者 shell 是否正在运行，需要查看集群管理器的网页用户界面 http://masternode:8080 并确保：
    ```python
    应用连接上了（即出现在了 Running Applications 中）
    列出的所使用的核心和内存均大于 0
    ```
    阻碍应用运行的一个常见陷阱是为执行器进程申请的内存（spark-submit 的 --executor-memory 标记传递的值）超过了集群所能提供的内存总量
  - 在这种情况下，独立集群管理器始终无法为应用分配执行器节点，请确保应用申请的值能够被集群接受
  - 最后，独立集群管理器支持两种部署模式，在这两种模式中，应用的驱动器程序运行在不同的地方
  - 在客户端模式中（默认情况），驱动器程序会运行在执行 spark-submit 的机器上，是 spark-submit 命令的一部分
    ```python
    意味着可以直接看到驱动器程序的输出，也可以直接输入数据进去（通过交互式 shell）
    但是这要求提交应用的机器与工作节点间有很快的网络速度，并且在程序运行的过程中始终可用
    ```
    在集群模式下，驱动器程序会作为某个工作节点上一个独立的进程运行在独立集群管理器内部，它也会连接主节点来申请执行器节点
    ```python
    在这种模式下，spark-submit 是“一劳永逸”型，可以在应用运行时关掉电脑，还可以通过集群管理器的网页用户界面访问应用的日志
    向 spark-submit 传递 --deploy-mode cluster 参数可以切换到集群模式
    ```
### 配置资源用量
  - 如果在多应用间共享 Spark 集群，需要决定如何在执行器进程间分配资源
  - 独立集群管理器使用基础的调度策略，这种策略允许限制各个应用的用量来让多个应用并发执行
  - Apache Mesos 支持应用运行时的更动态的资源共享
  - YARN 则有分级队列的概念，可以限制不同类别的应用的用量
  - 在独立集群管理器中，资源分配靠下面两个设置来控制
  - 执行器进程内存
    ```python
    可以通过 spark-submit 的 --executor-memory 参数来配置此项
    每个应用在每个工作节点上最多拥有一个执行器进程 ，但是一台机器上可以运行多个从节点
    因此这个设置项能够控制执行器节点占用工作节点的多少内存，此设置项的默认值是 1 GB，在大多数服务器上，可能需要提高这个值来充分利用机器
    ```
    占用核心总数的最大值
    ```python
    可以通过 spark-submit 的 --total-executorcores 参数设置这个值，或者在你的 Spark 配置文件中设置 spark.cores.max 的值
    这是一个应用中所有执行器进程所占用的核心总数，此项的默认值是无限，也就是说，应用可以在集群所有可用的节点上启动执行器进程
    对于多用户的工作负载来说，应该要求用户限制他们的用量
    ```
    要验证这些设定，可以从独立集群管理器的网页用户界面（http://masternode:8080）中查看当前的资源分配情况
  - 独立集群管理器在默认情况下会为每个应用使用尽可能分散的执行器进程
    ```python
    假设有一个 20 个物理节点的集群，每个物理节点是一个四核的机器，然后使用 --executor-memory 1G 和 --total-executor-cores 8 提交应用
    Spark 就会在不同机器上启动 8 个执行器进程，每个 1 GB 内存
    Spark 默认这样做，以尽量实现对于运行在相同机器上的分布式文件系统（比如 HDFS）的数据本地化，因为这些文件系统通常也把数据分散到所有物理节点上
    可以通过设置配置属性 spark.deploy.spreadOut 为 false 来要求 Spark 把执行器进程合并到尽量少的工作节点中
    在这样的情况下，这个应用就只会得到两个执行器节点，每个有 1 GB 内存和 4 个核心
    这一设定会影响运行在独立模式集群上的所有应用，并且必须在启动独立集群管理器之前设置好
    ```
### 高度可用性
  - 当在生产环境中运行时，会希望独立模式集群始终能够接收新的应用，哪怕当前集群中所有的节点都崩溃了
  - 其实，独立模式能够很好地支持工作节点的故障
  - 如果想让集群的主节点也拥有高度可用性，Spark 还支持使用 Apache ZooKeeper（一个分布式协调系统）来维护多个备用的主节点，并在一个主节点失败时切换到新的主节点上
  - 为独立集群配置 ZooKeeper在 Spark 官方文档（https://spark.apache.org/docs/latest/spark-standalone.html#high-availability）中有所描述
## 集群管理器 Hadoop YARN
  - YARN 是在 Hadoop 2.0 中引入的集群管理器
  - 它可以让多种数据处理框架运行在一个共享的资源池上，并且通常安装在与 Hadoop 文件系统（简称 HDFS）相同的物理节点上
  - 在这样配置的 YARN 集群上运行 Spark 是很有意义的，它可以让 Spark 在存储数据的物理节点上运行，以快速访问 HDFS 中的数据
### 在 Spark 里使用 YARN
  - 只需要设置指向 Hadoop 配置目录的环境变量，然后使用 spark-submit 向一个特殊的主节点 URL 提交作业即可
  - 找到 Hadoop 的配置目录，并把它设为环境变量 HADOOP_CONF_DIR：
    ```python
    这个目录包含 yarn-site.xml 和其他配置文件
    如果 Hadoop 装到 HADOOP_HOME 中，那么这个目录通常位于 HADOOP_HOME/conf 中，否则可能位于系统目录 /etc/hadoop/conf 中
    ```
    然后用如下方式提交应用：
    ```python
    export HADOOP_CONF_DIR="..."
    spark-submit --master yarn yourapp
    ```
### 客户端模式 / 集群模式
  - 和独立集群管理器一样，有两种将应用连接到集群的模式：客户端模式以及集群模式
    ```python
    在客户端模式下应用的驱动器程序运行在提交应用的机器上
    在集群模式下，驱动器程序也运行在一个 YARN 容器内部
    ```
    可以通过 spark-submit 的 --deploy-mode 参数设置不同的模式 (yarn-client / yarn-cluster)
  - Spark 的交互式 shell 以及 pyspark 也都可以运行在 YARN 上,只要设置好 HADOOP_CONF_ DIR 并对这些应用使用 --master yarn 参数即可
  - 由于这些应用需要从用户处获取输 入，所以只能运行于客户端模式下
### 配置资源用量
  - 当在 YARN 上运行时，根据在 spark-submit 或 spark-shell 等脚本的 --num-executors 标记中设置的值，Spark 应用会使用固定数量的执行器节点
  - 默认情况下，这个值仅为 2，所以可能需要提高它
  - 也可以设置通过 --executor-memory 设置每个执行器的内存用量，通过 --executor-cores 设置每个执行器进程从 YARN 中占用的核心数目
  - 对于给定的硬件资源，Spark 通常在用量较大而总数较少的执行器组合（使用多核与更多内存）上表现得更好，因为这样 Spark 可以优化各执行器进程间的通信
  - 然而，需要注意的是，一些集群限制了每个执行器进程的最大内存（默认为 8 GB），不允许使用更大的执行器进程
  - 出于资源管理的目的，某些 YARN 集群被设置为将应用调度到多个队列中，使用 --queue 选项来选择你的队列的名字
  - 要了解关于 YARN 的更多配置选项的相关信息，可以查阅 Spark 官方文档（http://spark.apache.org/docs/latest/submitting-applications.html）
## 集群管理器 Apache Mesos
  - Apache Mesos 是一个通用集群管理器，既可以运行分析型工作负载又可以运行长期运行的服务（比如网页服务或者键值对存储）
  - 要在 Mesos 上使用 Spark，需要把一个 mesos:// 的 URI 传给 spark-submit：
    ```python
    spark-submit --master mesos://masternode:5050 yourapp
    ```
  - 在运行多个主节点时，可以使用 ZooKeeper 来为 Mesos 集群选出一个主节点
  - 在这种情况下，应该使用 mesos://zk:// 的 URI 来指向一个 ZooKeeper 节点列表
    ```python
    三个 ZooKeeper 节点（node1、node2 和 node3），并且 ZooKeeper 分别运行在三台机器的 2181 端口上时
    mesos://zk://node1:2181/mesos,node2:2181/mesos,node3:2181/mesos
    ```
  - Mesos调度模式
    ```python
    和别的集群管理器不同，Mesos 提供了两种模式来在一个集群内的执行器进程间共享资源
    “细粒度”模式（默认）中，执行器进程占用的 CPU 核心数会在它们执行任务时动态变化，因此一台运行了多个执行器进程的机器可以动态共享 CPU 资源
    “粗粒度”模式中，Spark 提前为每个执行器进程分配固定数量的 CPU 数目，并且在应用结束前绝不释放这些资源，哪怕执行器进程当前不在运行任务
    可以通过向 spark-submit 传递 --conf spark.mesos.coarse=true 来打开粗粒度模式

    当多用户共享的集群运行 shell 这样的交互式的工作负载时，由于应用会在它们不工作时降低它们所占用的核心数，细粒度模式显得非常合适
    然而，在细粒度模式下调度任务会带来更多的延迟，一些像 Spark Streaming 这样需要低延迟的应用就会表现很差
    可以在一个 Mesos 集群中使用混合的调度模式，将一部分 Spark 应用的 spark.mesos.coarse 设置为 true，而另一部分不这么设置
    ```
  - 客户端 / 集群模式
    ```python
    就 Spark 1.2 而言，在 Mesos 上 Spark 只支持以客户端的部署模式运行应用，驱动器程序必须运行在提交应用的那台机器上
    如果还是希望在 Mesos 集群中运行驱动器节点，可以使用其他框架将任意脚本提交到 Mesos 上执行，并监控它们的运行
            Aurora（<a href="http://aurora.apache.org/">http://aurora.apache.org/</a>）或 Chronos（http://airbnb.io/chronos）
    ```
  - 配置资源用量
    ```python
    可以通过 spark-submit 的两个参数 --executor-memory 和 --total-executor-cores 来控制运行在 Mesos 上的资源用量
    前者用来设置每个执行器进程的内存
    后者则用来设置应用占用的核心数（所有执行器节点占用的总数）的最大值
    默认情况下，Spark 会使用尽可能多的核心启动各个执行器节点，来将应用合并到尽量少的执行器实例中，并为应用分配所需要的核心数
    如果不设置 --total-executor-cores 参数，Mesos 会尝试使用集群中所有可用的核心
    ```
## 集群管理器 Amazon EC2
  - Spark 自带一个可以在 Amazon EC2 上启动集群的脚本
  - 这个脚本会启动一些节点，并且在它们上面安装独立集群管理器
  - 一旦集群启动起来，就可以根据独立模式使用方法来使用这个集群
  - 除此以外，EC2 脚本还会安装好其他相关的服务，比如 HDFS、Tachyon 还有用来监控集群的 Ganglia
  - Spark 的 EC2 脚本叫作 spark-ec2，位于 Spark 安装目录下的 ec2 文件夹中
  - 它需要 Python 2.6 或更高版本的运行环境，可以在下载 Spark 后直接运行 EC2 脚本而无需预先编译 Spark
  - EC2 脚本可以管理多个已命名的集群（cluster），并且使用 EC2 安全组来区分它们
  - 对于每个集群，脚本都会为主节点创建一个叫作 clustername-master 的安全组，而为工作节点创建一个叫作 clustername-slaves 的安全组
### 启动集群
  - 要启动一个集群，应该先创建一个 Amazon 网络服务（AWS）账号，并且获取访问键 ID 和访问键密码，然后把它们设在环境变量中：
    ```python
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    ```
  - 然后再创建出 EC2 的 SSH 密钥对，然后下载私钥文件（通常叫作 keypair.pem），这样就可以 SSH 到机器上
  - 接下来，运行 spark-ec2 脚本的 launch 命令，提供密钥对的名字、私钥文件和集群的名字
  - 默认情况下，这条命令会使用 m1.xlarge 类型的 EC2 实例，启动一个有一个主节点和一个工作节点的集群：
    ```python
    cd /path/to/spark/ec2
    ./spark-ec2 -k mykeypair -i mykeypair.pem launch mycluste
    ```
  - 也可以使用 spark-ec2 的参数选项配置实例的类型、工作节点个数、EC2 地区，以及其他一些要素：
    ```python
    # 启动包含5个m3.xlarge类型的工作节点的集群
    ./spark-ec2 -k mykeypair -i mykeypair.pem -s 5 -t m3.xlarge launch mycluster
    ```
  - 要获得参数选项的完整列表，运行 spark-ec2 --help
  - spark-ec2的常见选项
    ```python
    -k KEYPAIR 要使用的 keypair 的名字 
    -i IDENTITY_FiLE 私钥文件（以 .pem 结尾） 
    -s NUM_SLAVES 工作节点数量 
    -t INSTANCE_TYPE 使用的实例类型 
    -r REGION 使用 Amazon 实例所在的区域（例如  us-west-1）
    -z ZONE 使用的地带（例如  us-west-1b）
    --spot-price=PRICE 在给定的出价使用 spot 实例（单位为美元） 
    ```
  - 从启动脚本开始，通常需要五分钟左右来完成启动机器、登录到机器上并配置 Spark 的全部过程。
### 登录集群
  - 可以使用存有私钥的 .pem 文件通过 SSH 登录到集群的主节点上，spark-ec2 提供了登录命令：
    ```python
    ./spark-ec2 -k mykeypair -i mykeypair.pem login mycluster

    ```
    可以通过运行下面的命令获得主节点的主机名：
    ```python
    ./spark-ec2 get-master mycluster
    ```
  - 然后自行使用 ssh -i keypair.pem root@masternode 命令 SSH 到主节点上
  - 进入集群以后，就可以使用 /root/spark 中的 Spark 环境来运行程序了
  - 这是一个独立模式的集群环境，主节点 URL 是 spark://masternode:7077
  - 当使用 spark-submit 启动应用时，Spark 会自动配置为将应用提交到这个独立集群上，可以从 http://masternode:8080 访问集群的网页用户界面
  - 注意，只有从集群中的机器上启动的程序可以使用 spark-submit 把作业提交上去，出于安全目的，防火墙规则会阻止外部主机提交作业
  - 要在集群上运行一个预先打包的应用，需要先把程序包通过 SCP 复制到集群上：
    ```python
    scp -i mykeypair.pem app.jar root@masternode:~
    ```
### 销毁集群
  - 要销毁 spark-ec2 所启动的集群，运行：
    ```python
    ./spark-ec2 destroy mycluster
    ```
    这条命令会终止集群中的所有的实例，包括 mycluster-master 和 mycluster-slaves 两个安全组中的所有实例
### 暂停和重启集群
  - 除了将集群彻底销毁，spark-ec2 还可以中止运行集群的 Amazon 实例，并且可以稍后重启这些实例
  - 停止这些实例会将它们关机，并丢失“临时”盘上的所有数据，“临时”盘上还配有一个给 spark-ec2 使用的 HDFS 环境
  - 不过，中止的实例会 保留 root 目录下的所有数据（如上传到那里的所有文件），这样就可以快速恢复自己的工作
  - 要中止一个集群，运行：
    ```python
    ./spark-ec2 stop mycluster
    ```
    然后，过一会儿，再次启动集群：
    ```python
    ./spark-ec2 -k mykeypair -i mykeypair.pem start mycluster
    ```
  - Spark EC2 的脚本并没有提供调整集群大小的命令，但可以通过增减 mycluster-slaves 安全组中的机器来实现对集群大小的控制
  - 增加机器
    ```python
    首先应当中止集群
    然后使用 AWS 管理控制台，右击一台工作节点并选择“Launch more like this（启动更多像这个实例一样的实例）”，以在同一个安全组中创建出更多实例
    然后使用 spark-ec2 start 启动集群
    ```
    移除机器，只需在 AWS 控制台上终止这一实例即可，不过要小心，这也会破坏集群中 HDFS 上的数据
### 集群存储
  - Spark EC2 集群已经配置好了两套 Hadoop 文件系统以供存储临时数据，可以很方便地将数据集存储在访问速度比 Amazon S3 更快的媒介中 
  - “临时”HDFS
    ```python
    使用节点上的临时盘
    大多数类型的 Amazon 实例都在“临时”盘上带有大容量的本地空间，这些空间会在实例关机时消失
    这种文件系统安装在节点的 /root/ephemeral-hdfs 目录中，可以使用 bin/hdfs 命令来访问并列出文件
    可以访问这种 HDFS 的网页用户界面，其 URL 地址位于 http://masternode:50070
    ```
    “永久”HDFS
    ```python
    使用节点的 root 分卷
    这种 HDFS 在集群重启时仍然保留数据，不过一般空间较小，访问起来也比临时的那种慢
    这种 HDFS 适合存放不想多次下载的中等大小的数据集中
    它安装于 /root/persistent-hdfs 目录，网页用户界面地址是 http://masternode:60070
    ```
    除了这些以外，最有可能访问的就是 Amazon S3 中的数据了，可以在 Spark 中使用 s3n:// 的 URI 结构来访问其中的数据
## 选择合适的集群管理器
  - 如果是从零开始，可以先选择独立集群管理器，独立模式安装起来最简单，而且如果只是使用 Spark 的话，独立集群管理器提供与其他集群管理器完全一样的全部功能
  - 如果要在使用 Spark 的同时使用其他应用，或者是要用到更丰富的资源调度功能（例如队列），那么 YARN 和 Mesos 都能满足需求
  - 对于大多数 Hadoop 发行版来说，一般 YARN 已经预装好了
  - Mesos 相对于 YARN 和独立模式的一大优点在于其细粒度共享的选项
  - 该选项可以将类似 Spark shell 这样的交互式应用中的不同命令分配到不同的 CPU 上，因此这对于多用户同时运行交互式 shell 的用例更有用处
  - 在任何时候，最好把 Spark 运行在运行 HDFS 的节点上，这样能快速访问存储
  - 可以自行在同样的节点上安装 Mesos 或独立集群管理器
  - 如果使用 YARN 的话，大多数发行版已经把 YARN 和 HDFS 安装在了一起
***

# 使用 SparkConf / spark-submit 配置Spark
## SparkConf 类
  - 对 Spark 进行性能调优，通常就是修改 Spark 应用的运行时配置选项
  - Spark 中最主要的配置机制是通过 SparkConf 类对 Spark 进行配置
  - 当创建出一个 SparkContext 时，就需要创建出一个 SparkConf 的实例
  - 一旦传给了 SparkContext 的构造方法，应用所绑定的 SparkConf 就不可变了。这意味着所有的配置项都必须在 SparkContext 实例化出来之前定下来
  - 在 Python 中使用 SparkConf 创建一个应用
    ```python
    # 创建一个conf对象
    conf = new SparkConf()
    conf.set("spark.app.name", "My Spark App")
    conf.set("spark.master", "local[4]")
    conf.set("spark.ui.port", "36000") # 重载默认端口配置

    # 使用这个配置对象创建一个SparkContext
    sc = SparkContext(conf)
    ```
  - SparkConf 实例包含用户要重载的配置选项的键值对
  - 要使用创建出来的 SparkConf 对象，可以调用 set() 方法来添加配置项的设置，然后把这个对象传给 SparkContext 的构造方法
  - 除了 set() 之外，SparkConf 类也包含了一小部分工具方法，可以很方便地设置部分常用参数
  - 如 setAppName() 和 setMaster() 分别设置 spark.app.name 和 spark.master 的配置值
## spark-submit配置
  - 当应用被 spark-submit 脚本启动时，脚本会把这些配置项设置到运行环境中
  - 当一个新的 SparkConf 被创建出来时，这些环境变量会被检测出来并且自动配好
  - 在使用 spark-submit 时，用户应用中只要创建一个“空”的 SparkConf，并直接传给 SparkContext 的构造方法就行了
  - spark-submit 工具为常用的 Spark 配置项参数提供了专用的标记，还有一个通用标记 --conf 来接收任意 Spark 配置项的值
  - 在运行时使用标记设置配置项的值
    ```python
    $ bin/spark-submit \
      --class com.example.MyApp \
      --master local[4] \
      --name "My Spark App" \
      --conf spark.ui.port=36000 \
      myApp.jar
    ```
  - spark-submit 也支持从文件中读取配置项的值，这对于设置一些与环境相关的配置项比较有用，方便不同用户共享这些配置
  - 默认情况下，spark-submit 脚本会在 Spark 安装目录中找到 conf/spark-defaults.conf 文件，尝试读取该文件中以空格隔开的键值对数据
  - 也可以通过 spark-submit 的 --properties-File 标记，自定义该文件的路径
  - 运行时使用默认文件设置配置项的值
    ```python
    $ bin/spark-submit \
      --class com.example.MyApp \
      --properties-file my-config.conf \
      myApp.jar

    ## Contents of my-config.conf ##
    spark.master    local[4]
    spark.app.name  "My Spark App"
    spark.ui.port   36000
    ```
## Spark配置优先级
  - 有时，同一个配置项可能在多个地方被设置了，如用户可能在程序代码中直接调用了 setAppName() 方法，同时也通过 spark-submit 的 --name 标记设置了这个值
  - Spark 有特定的优先级顺序来选择实际配置：
    ```python
    优先级最高的是在用户代码中显式调用 set() 方法设置的选项
    其次是通过 spark-submit 传递的参数，再次是写在配置文件中的值
    最后是系统的默认值
    ```
    可以在应用的网页用户界面中查看应用中实际生效的配置
## 常用的Spark配置项的值
  - 完整的配置项列表，请参考 Spark 文档（http://spark.apache.org/docs/latest/configuration.html）
  - spark.executor.memory(--executor-memory)
    ```python
    默认值 512m
    为每个执行器进程分配的内存，格式与 JVM 内存字符串格式一样（例如 512m，2g）。关于本配置项的更多细节，请参阅 8.4.4 节 
    ```
  - spark.executor.cores (--executor-cores)spark.cores.max(--total-executor-cores)
    ```python
    限制应用使用的核心个数的配置项
    在YARN 模式下， spark.executor.cores 会为每个任务分配指定数目的核心
    在独立模式和Mesos 模式下，spark.core.max 设置了所有执行器进程使用的核心总数的上限
    ```
  - spark.speculation
    ```python
    默认值 false
    设为  true 时开启任务预测执行机制
    当出现比较慢的任务时，这种机制会在另外的节点上也尝试执行该任务的一个副本，打开此选项会帮助减少大规模集群中个别较慢的任务带来的影响
    ```
  - spark.storage.blockManagerTimeoutIntervalMs
    ```python
    默认值 45000
    内部用来通过超时机制追踪执行器进程是否存活的阈值
    对于会引发长时间垃圾回收（GC）暂停的作业，需要把这个值调到 100 秒（对应值为 100000）以上来防止失败
    在 Spark 将来的版本中，这个配置项可能会被一个统一的超时设置所取代，所以请注意检索最新文档 
    ```
  - spark.executor.extraJavaOptions / spark.executor.extraClassPath / spark.executor.extraLibraryPath
    ```python
    这三个参数用来自定义如何启动执行器进程的 JVM
    分别用来添加额外的 Java 参数、classpath 以及程序库路径
    使用字符串来设置这些参数，如  spark.executor.extraJavaOptions="- XX:+PrintGCDetails-XX:+PrintGCTimeStamps"
    请注意，虽然这些参数可以自行添加执行器程序的 classpath，我们还是推荐使用 spark-submit 的 --jars 标记来添加依赖而不是使用这几个选项
    ```
  - spark.serializer
    ```python
    org.apache.spark.serializer.JavaSerializer
    指定用来进行序列化的类库，包括通过网络传输数据或缓存数据时的序列化
    默认的 Java 序列化对于任何可以被序列化的 Java 对象都适用，但是速度很慢
    我们推荐在追求速度时使用  org.apache.spark.serializer.KryoSerializer 并且对 Kryo 进行适当的调优
    该项可以配置为任何 org.apache.spark.Serializer 的子类
    ```
  - spark.[X].port
    ```python
    用来设置运行Spark 应用时用到的各个端口
    这些参数对于运行在可靠网络上的集群是很有用的
    有效的 X 包括  driver、fileserver、broadcast、replClassServer、blockManager，以及 executor
    ```
  - spark.eventLog.enabled
    ```python
    默认值 false
    设为  true 时，开启事件日志机制，这样已完成的 Spark 作业就可以通过历史服务器（history server）查看
    关于历史服务器的更多信息，请参考官方文档
    ```
  - spark.eventLog.dir
    ```python
    默认值 file:///tmp/spark-events
    指开启事件日志机制时，事件日志文件的存储位置
    这个值指向的路径需要设置到一个全局可见的文件系统中，比如 HDFS 
    ```
## SPARK_LOCAL_DIRS
  - 几乎所有的 Spark 配置都发生在 SparkConf 的创建过程中，但有一个重要的选项是个例外
  - 需要在 conf/spark-env.sh 中将环境变量 SPARK_LOCAL_DIRS 设置为用逗号隔开的存储位置列表，来指定 Spark 用来混洗数据的本地存储路径，这需要在独立模式和 Mesos 模式下设置
  - 这个配置项之所以和其他的 Spark 配置项不一样，是因为它的值在不同的物理主机上可能会有区别
***

# Spark 性能调优 (作业 / 用户界面 / 日志 / 并行度 / 序列化格式 / 内存管理)
## Spark执行的作业 / 任务 / 步骤
  - Spark 执行时有下面所列的这些流程：
    ```python
    用户代码定义RDD的有向无环图
    行动操作把有向无环图强制转译为执行计划
    任务于集群中调度并执行
    ```
    在一个给定的 Spark 应用中，由于需要创建一系列新的 RDD，因此上述阶段会连续发生很多次
  - Spark 调度器会创建出用于计算行动操作的 RDD 物理执行计划
  - Spark 调度器从最终被调用行动操作的 RDD出发，向上回溯所有必须计算的 RDD，调度器会访问 RDD 的父节点，递归向上生成计算所有必要的祖先 RDD 的物理计划
  - 调度器为有向图中的每个 RDD 输出计算步骤，步骤中包括 RDD 上需要应用于每个分区的任务，然后以相反的顺序执行这些步骤，计算得出最终所求的 RDD
  - 特定的行动操作所生成的步骤的集合被称为一个作业
  - Spark 提供了 toDebugString() 方法来查看 RDD 的谱系
    ```python
    lines = sc.textFile('README.md')
    lines.toDebugString()
    Out[3]: b'(2) README.md MapPartitionsRDD[1] at textFile at NativeMethodAccessorImpl.java:0 []
     | README.md HadoopRDD[0] at textFile at NativeMethodAccessorImpl.java:0 []'
    ```
  - 当一个 RDD 已经缓存在集群内存或磁盘上时，Spark 的内部调度器也会自动截短 RDD 谱系图，Spark 会“短路”求值，直接基于缓存下来的 RDD 进行计算
  - 还有一种截短 RDD 谱系图的情况发生在当 RDD 已经在之前的数据混洗中作为副产品物化出来时，哪怕该 RDD 并没有被显式调用 persist() 方法
  - 这种内部优化是基于 Spark 数据混洗操作的输出均被写入磁盘的特性，同时也充分利用了 RDD 图的某些部分会被多次计算的事实
  - 一个物理步骤会启动很多任务，每个任务都是在不同的数据分区上做同样的事情，任务内部的流程是一样的：
    ```python
    从数据存储（如果该 RDD 是一个输入 RDD）或已有 RDD（如果该步骤是基于已经缓存的数据）或数据混洗的输出中获取输入数据
    执行必要的操作来计算出这些操作所代表的 RDD，例如，对输入数据执行 filter() 和 map()函数，或者进行分组或归约操作
    把输出写到一个数据混洗文件中，写入外部存储，或者是发回驱动器程序（如果最终 RDD 调用的是类似 count() 这样的行动操作）
    ```
    Spark 的大部分日志信息和工具都是以步骤、任务或数据混洗为单位的
## Spark 内建的网页用户界面
  - 默认情况下，它在驱动器程序所在机器的 4040 端口上
  - 对于 YARN 集群模式来说，应用的驱动器程序会运行在集群内部，应该通过 YARN 的资源管理器来访问用户界面，YARN 的资源管理器会把请求直接转发给驱动器程序
  - 作业页面：
    ```python
    步骤与任务的进度和指标，包含正在进行的或刚完成不久的 Spark 作业的详细执行情况
    其 中一个很重要的信息是正在运行的作业、步骤以及任务的进度情况，针对每个步骤，这个 页面提供了一些帮助理解物理执行过程的指标

    本页面经常用来评估一个作业的性能表现，可以着眼于组成作业的所有步骤，看看是不是有一些步骤特别慢，或是在多次运行同一个作业时响应时间差距很大
    如果遇到了格外慢的步骤，可以点击进去来更好地理解该步骤对应的是哪段用户代码
    确定了需要着重关注的步骤之后，可以再借助骤页面来定位性能问题
    在 Spark 这样的并行数据系统中，数据倾斜是导致性能问题的常见原因之一
    当看到少量的任务相对于其他任务需要花费大量时间的时候，一般就是发生了数据倾斜
    步骤页面可以帮助我们发现数据倾斜，只需要查看所有任务各项指标的分布情况
        是不是有一部分任务花的时间比别的任务多得多？
        是不是有一小部分任务读取或者输出了比别的任务多得多的数据？
        是不是运行在某些特定节点上的任务特别慢？
    ```
  - 存储页面：
    ```python
    存储页面包含了缓存下来的 RDD 的信息
    当有人在一个 RDD 上调用了 persist() 方法，并且在某个作业中计算了该 RDD 时，这个 RDD 就会被缓存下来
    有时，如果缓存了许多 RDD，比较老的 RDD 就会从内存中移出来，把空间留给新缓存的 RDD
    这个页面可以告诉我们到底各个 RDD 的哪些部分被缓存了，以及在各种不同的存储媒介（磁盘、内存等）中所缓存的数据量
    ```
  - 执行器页面：
    ```python
    本页面列出了应用中申请到的执行器实例，以及各执行器进程在数据处理和存储方面的一些指标
    本页面的用处之一在于确认应用可以使用所预期使用的全部资源量
    调试问题时也最好先浏览这个页面，因为错误的配置可能会导致启动的执行器进程数量少于我们所预期的，显然也就会影响实际性能
    这个页面对于查找行为异常的执行器节点也很有帮助，比如某个执行器节点可能有很高的任务失败率，只要把这台主机从集群中移除，就可以提高性能表现

    执行器页面的另一个功能是使用线程转存（Thread Dump）按钮收集执行器进程的栈跟踪信息（ Spark 1.2 中引入）
    可视化呈现执行器进程的线程调用栈可以精确地即时显示出当前执行的代码
    在短时间内使用该功能对一个执行器进程进行多次采样，就可以发现“热点”，也就是用户代码中消耗代价比较大的代码段，这种信息分析通常可以检测出低效的用户代码
    ```
  - 环境页面：
    ```python
    用来调试Spark配置项
    本页面枚举了 Spark 应用所运行的环境中实际生效的配置项集合，这里显示的配置项代表应用实际的配置情况
    当检查哪些配置标记生效时，这个页面很有用，尤其是当同时使用了多种配置机制时
    这个页面也会列出添加到应用路径中的所有 JAR 包和文件，在追踪类似依赖缺失的问题时可以用到
    ```
## 驱动器进程和执行器进程的日志
  - 日志会更详细地记录各种异常事件，例如内部的警告以及用户代码输出的详细异常信息
  - Spark 日志文件的具体位置取决于以下部署模式
    ```python
    在 Spark 独立模式下，所有日志会在独立模式主节点的网页用户界面中直接显示，这些日志默认存储于各个工作节点的 Spark 目录下的 work/ 目录中
    在 Mesos 模式下，日志存储在 Mesos 从节点的 work/ 目录中，可以通过 Mesos 主节点用户界面访问
    在 YARN 模式下，最简单的收集日志的方法是使用 YARN 的日志收集工具（运行 yarn logs -applicationId <app ID>）来生成一个包含应用日志的报告
        这种方法只有在应用已经完全完成之后才能使用，因为 YARN 必须先把这些日志聚合到一起
        要查看当前运行在 YARN 上的应用的日志，可以从资源管理器的用户界面点击进入节点（Nodes）页面，然后浏览特定的节点，再从那里找到特定的容器
        YARN 会提供对应容器中 Spark 输出的内容以及相关日志
    ```
  - 自定义日志行为：
    ```python
    在默认情况下，Spark 输出的日志包含的信息量比较合适，我们也可以自定义日志行为，改变日志的默认等级或者默认存放位置
    Spark 的日志系统是基于广泛使用的 Java 日志库 log4j 实现的，使用 log4j 的配置方式进行配置
    log4j 配置的示例文件已经打包在 Spark 中，具体位置是 conf/log4j.properties.template
    要自定义 Spark 的日志，首先把这个示例文件复制为 log4j.properties，然后就可以修改日志行为，比如修改根日志等级（即日志输出的级别门槛），默认值为 INFO
    如果想要更少的日志输出，可以把日志等级设为 WARN 或者 ERROR
    当设置了满意的日志等级或格式之后，可以通过 spark-submit 的 --Files 标记添加 log4j.properties 文件
    如果在设置日志级别时遇到了困难，请首先确保没有在应用中引入任何自身包含 log4j.properties 文件的 JAR 包
    Log4j 会扫描整个 classpath，以其找到的第一个配置文件为准，因此如果在别处先找到该文件，它就会忽略自定义的文件
    ```
## 并行度
  - 当 Spark 调度并运行任务时，Spark 会为每个分区中的数据创建出一个任务，该任务在默认情况下会需要集群中的一个计算核心来执行
  - Spark 也会针对 RDD 直接自动推断出合适的并行度，这对于大多数用例来说已经足够了
  - 输入 RDD 一般会根据其底层的存储系统选择并行度，例如，从 HDFS 上读数据的输入 RDD 会为数据在 HDFS 上的每个文件区块创建一个分区
  - 从数据混洗后的 RDD 派生下来的 RDD 则会采用与其父 RDD 相同的并行度
  - Spark 提供了两种方法来对操作的并行度进行调优
    ```python
    第一种方法是在数据混洗操作时，使用参数的方式为混洗后的 RDD 指定并行度
    第二种方法是对于任何已有的 RDD，可以进行重新分区来获取更多或者更少的分区数
    ```
  - 重新分区操作
    ```python
    通过 repartition() 实现，该操作会把 RDD 随机打乱并分成设定的分区数目
    如果确定要减少 RDD 分区，可以使用 coalesce() 操作,由于没有打乱数据，该操作比 repartition() 更为高效
    ```
  - filter后减少分区
    ```python
    假设我们从 S3 上读取了大量数据，然后马上进行 filter() 操作筛选掉数据集中的绝大部分数据
    默认情况下，filter() 返回的 RDD 的分区数和其父节点一样，这样可能会产生很多空的分区或者只有很少数据的分区

    在这样的情况下，可以通过合并得到分区更少的 RDD 来提高应用性能
    在 PySpark shell 中合并分区过多的 RDD
    # 以可以匹配数千个文件的通配字符串作为输入
    input = sc.textFile("s3n://log-files/2014/*.log")
    input.getNumPartitions()
    35154
    # 排除掉大部分数据的筛选方法
    lines = input.filter(lambda line: line.startswith("2014-10-17"))
    lines.getNumPartitions()
    35154
    # 在缓存lines之前先对其进行合并操作
    lines = lines.coalesce(5).cache()
    lines.getNumPartitions()
    4
    # 可以在合并之后的RDD上进行后续分析
    lines.count()
    ```
## 序列化格式
  - 当 Spark 需要通过网络传输数据，或是将数据溢写到磁盘上时，Spark 需要把数据序列化为二进制格式
  - 序列化会在数据进行混洗操作时发生，此时有可能需要通过网络传输大量数据
  - 默认情况下，Spark 会使用 Java 内建的序列化库，Spark 也支持使用第三方序列化库
### Kryo序列化工具
  - Kryo（https://github.com/EsotericSoftware/kryo），可以提供比 Java 的序列化工具更短的序列化时间和更高压缩比的二进制表示，但不能直接序列化全部类型的对象
  - 几乎所有的应用都在迁移到 Kryo 后获得了更好的性能
  - 要使用 Kryo 序列化工具，需要设置 spark.serializer 为org.apache.spark.serializer.KryoSerializer
  - 为了获得最佳性能，还应该向 Kryo 注册你想要序列化的类，注册类可以让 Kryo 避免把每个对象的完整的类名写下来，成千上万条记录累计节省的空间相当可观
  - 如果想强制要求这种注册，可以把spark.kryo.registrationRequired 设置为 true，这样 Kryo 会在遇到未注册的类时抛出错误
  - 使用 Kryo 序列化工具并注册所需类
    ```python
    val conf = new SparkConf()
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    // 严格要求注册类
    conf.set("spark.kryo.registrationRequired", "true")
    conf.registerKryoClasses(Array(classOf[MyClass], classOf[MyOtherClass]))
    ```
### NotSerializableException错误
  - 不论是选用 Kryo 还是 Java 序列化，如果代码中引用到了一个没有扩展 Java 的 Serializable 接口的类，都会遇到 NotSerializableException
  - 在这种情况下，要查出引发问题的类是比较困难的，因为用户代码会引用到许许多多不同的类
  - 很多 JVM 都支持通过一个特别的选项来帮助调试这一情况："-Dsun.io.serialization.extended DebugInfo=true”
  - 可以通过设置 spark-submit 的 --driver-java-options 和 --executor-java-options 标记来打开这个选项
  - 一旦找到了有问题的类，最简单的解决方法就是把这个类改为实现了 Serializable 接口的形式
  - 如果没有办法修改这个产生问题的类，就需要采用一些高级的变通策略，比如为这个类创建一个子类并实现 Java 的 Externalizable 接口
  - 参考（https://docs.oracle.com/javase/7/docs/api/java/io/Externalizable.html）
  - 或者自定义 Kryo 的序列化行为
## 内存管理
### 内存用途
  - RDD存储
    ```python
    当调用 RDD 的 persist() 或 cache() 方法时，这个 RDD 的分区会被存储到缓存区中
    Spark 会根据 spark.storage.memoryFraction 限制用来缓存的内存占整个 JVM 堆空间的比例大小
    如果超出限制，旧的分区数据会被移出内存
    ```
  - 数据混洗与聚合的缓存区
    ```python
    当进行数据混洗操作时，Spark 会创建出一些中间缓存区来存储数据混洗的输出数据
    这些缓存区用来存储聚合操作的中间结果，以及数据混洗操作中直接输出的部分缓存数据
    Spark 会尝试根据 spark.shuffle.memoryFraction 限定这种缓存区内存占总内存的比例
    ```
  - 用户代码
    ```python
    Spark 可以执行任意的用户代码，所以用户的函数可以自行申请大量内存
    如果一个用户应用分配了巨大的数组或者其他对象，那这些都会占用总的内存
    用户代码可以访问 JVM 堆空间中除分配给 RDD 存储和数据混洗存储以外的全部剩余空间
    ```
### 内存缓存策略优化
  - 调整内存比例
    ```python
    默认情况下，Spark 会使用 60％的空间来存储 RDD，20% 存储数据混洗操作产生的数据，剩下的 20% 留给用户程序
    如果用户代码中分配了大量的对象，那么降低 RDD 存储和数据混洗存储所占用的空间可以有效避免程序内存不足的情况
    ```
  - 改进缓存行为的存储等级
    ```python
    Spark 默认的cache() 操作会以 MEMORY_ONLY 的存储等级持久化数据
    这意味着如果缓存新的 nRDD 分区时空间不够，旧的分区就会直接被删除,当用到这些分区数据时，再进行重算
    所以有时以MEMORY_AND_DISK 的存储等级调用 persist() 方法会获得更好的效果
    因为在这种存储等级下，内存中放不下的旧分区会被写入磁盘，当再次需要用到的时候再从磁盘上读取回来
    这样的代价有可能比重算各分区要低很多，也可以带来更稳定的性能表现
    当 RDD 分区的重算代价很大（比如从数据库中读取数据）时，这种设置尤其有用
    ```
  - 缓存序列化后的对象而非直接缓存
    ```python
    可以通过MEMORY_ONLY_SER 或者 MEMORY_AND_DISK_SER 的存储等级来实现这一点
    缓存序列化后的对象会使缓存过程变慢，因为序列化对象也会消耗一些代价
    不过这可以显著减少 JVM 的垃圾回收时间，因为很多独立的记录现在可以作为单个序列化的缓存而存储
    垃圾回收的代价与堆里的对象数目相关，而不是和数据的字节数相关
    这种缓存方式会把大量对象序列化为一个巨大的缓存区对象
    如果需要以对象的形式缓存大量数据（比如数 GB 的数据），或者是注意到了长时间的垃圾回收暂停，可以考虑配置这个选项
    这些暂停时间可以在应用用户界面中显示的每个任务的垃圾回收时间那一栏看到
    ```
***

# Spark SQL (Spark 2.1.0有改动)
***

# Spark Streaming (Spark 1.2不支持python)
***

# 基于 MLlib 的机器学习
***
