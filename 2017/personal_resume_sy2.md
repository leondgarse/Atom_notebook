# ___个人简历 - 王少颖___
***

# 个人信息
  - **姓名**：王少颖
  - **性别**： 女
  - **出生年月**：1985.09
  - **工作经验**： 5年
  - **电话**：18678969015
  - **籍贯**：山东青岛
  - **邮箱**：18678969015@163.com
***

# 个人技能
  - 熟练掌握 **C / C++ / Python** 语言，以及 Linux 环境下的应用开发，具有良好的编程风格
  - 掌握 **Linux** 下常用命令，以及 shell 脚本编写
  - 掌握网络相关技术，**SIP** 通信开发等
  - 熟练使用 **git / hg** 等主流版本控制工具的使用
  - 熟悉 **sql** 语言，数据库相关知识
  - 具有很好的 **英语** 读写能力
  - 熟练使用统计类软件 **SPSS / R / SAS / MATLAB** 仿真工具等
  - 了解大数据 **MapReduce / Spark** 等框架
***

# 教育经历
  - `2009.09 - 2012.04` - `北京理工大学` - `概率论与数理统计专业` - `理学硕士`
***

# 工作经历　          　
## 2015.01 --- 2017.05 - 青岛百灵信息科技有限公司 - 软件开发工程师
  - **北京爱立信5G Radio AAS 项目**
    - LO本振频率模块
    - Jesdlink Supervision模块Rx部分
    - Equipment Led模块
    - Over Temperature Backoff模块
    - 参与模块设计文档编写，模块功能实现，code编写，UT测试，以及相关lab环境搭建，CT10及Radio test board维护
  - **上海贝尔 FN ONT 光纤接入设备研发，Feature开发和FR debug**
    - Feature开发：Local announcements
    - Feature开发：Call Return - Call to last caller set up using 302
    - Feature开发：TR-104 management
  - **阿尔卡特-朗讯 CTS SCG产品 鉴权中心 (Authentication center)**
    - 在IMS-HSS, HLR-HSS,LTE-HSS环境下对鉴权中心进行测试
    - 测试内容包括Milenage算法 、SHA1 算法等，鉴权三元组、鉴权五元组等
    - 测试的消息有MAR、MAA， AIR、AIA，等
    - 涉及到的3GPP标准有3GPP TS 29.109 3GPP TS 35.205 3GPP TS 35.206
    - SCG 2.0 Feature 99731测试
    - Heat based Cloud System of LCP
## 2014.04 --- 2015.01 - 青岛海信电器股份有限公司 - 软件研发工程师
  - **项目名称**：海外出口机型Whale项目
  - **项目介绍**：德国机型Loewe及北美US机型设计研发
  - **工作内容**：
    - 负责电视音视频模块的需求文档分析及模块设计
    - 负责电视音视频模块code的编写和单元测试
    - 负责项目开发过程中音视频相关bug的分析及解决
    - 参与论证阶段相关软件模块PRD等文档的评审
    - 与硬件工程师共同排查与分析某些功能缺陷问题
## 2012.04 --- 2014.04 - 青岛海信电器股份有限公司 - 算法设计师
  - **项目名称**：高清数字电视SOC芯片
  - **项目介绍**：
    - 开发高清数字电视SOC芯片，支持全球全制式模拟电视信号和高清数字电视传输流（TS）信号的解码以及音视频后处理功能
    - 芯片集成（PAL/NTSC/SECAM）CVBS信号模拟前端接收和解码器，多格式高清MPEG2/AVS/H.264等视频解码器，音频（AAC/AC-3/MP3）解码器
    - 集成32位 RISC处理器、数字电视中间件、和视频后处理（基于边缘融合的缩放技术，基于逐像素运动估算的120Hz帧率提升技术，超分辨率提升技术，画质提升技术、智能彩色管理等）
    - 可以广泛应用于各类中高端高清液晶电视，等离子电视等平板电视市场
  - **工作内容**：
    - 主要负责电视视频后处理芯片的算法研发工作以及国家核高基项目
    - **视频压缩算法**
      - 对图像的亮度色度数据进行压缩，以节省DDR的带宽，此算法是后续 **4K * 2K 高清分辨率电视芯片的基础和核心算法**
      - **主要负责** 数据分块压缩打包，编码，解压缩的算法设计，将压缩比控制在50%左右
      - 此 **算法的原理** 是区间估计及频数累积的简单概率统计方法，使用了极其简单的数学编码，但是达到了很好的实现效果，是此算法的最大亮点
    - **SCALER 缩放算法**
      - **图像缩放技术** 可以满足将视频信号在任意大小的屏上显示的要求
      - 信号源分辨率的多样性，以及数字电视信号终端显示设备显示比例的不同，为了使视频在水平和垂直方向被非线性拉伸使得其接近目标宽高比，数字图像缩放技术显得越来越重要
      - **主要负责** 针对振铃效应和锯齿效应是影响算法效果的主要问题，应用横向纵向去振铃，去锯齿算法，使用块平均滤波进行横向纵向缩放，最大可支持64倍缩放
    - **DI 去隔行算法**
      - **去隔行技术**，即模拟电视的隔行扫描到逐行扫描的转换，以适应现代显示设备进入逐行扫描的高清时代，算法具有很强的实用价值
      - **主要负责** 去隔行算法的研发及实现工作，通过对经过待插值像素点的各方向的可靠性检测，挑选可靠度最高的方向，综合了垂直方向插值和基于边界方向的插值，插值结果准确，获得了较好的去隔行效果
    - **FRC 帧率转换算法**
      - 不同视频标准扫描率之间的相互转换
      - **主要负责** MEMC运动补偿和MVR运动矢量提取模块的算法设计工作
***

# ___English Version___
***

# Personal Information
  - **Name**: Wang ShaoYing
  - **Tel**: (86) 18678969015
  - **Email**: 18678969015@163.com
***

# Computer Abilities
  - Skilled in **C / C++ / Python**, and Linux application development, with advanced coding style
  - Skilled in **Linux** commands and shell script
  - Familiar with network related technology and **SIP** protocol related development
  - Skilled in using of **git / hg** and other distributed revision control system
  - Familiar with **SQL**, and other database related knowledge
  - Good **English** reading and writing ability, and daily communication is well
  - Mastered in statistical analysis, and using of SPSS, R, SAS and MATLAB emulation
***

# Education background
  - `2009.09 - 2012.04` - `Mathematical statistics, Beijing institute of technology` - `M.E`
***

# Work & Practice experience
## 2015.01 --- 2017.05 - Qing Dao Centling Information and Technology CO. - Software R&D Engineer
  - **Beijing Ericsson 5G Radio AAS project**
    - LO Local Oscillator Module
    - Jesdlink Supervision Module Rx
    - Equipment Led Module
    - Over Temperature Backoff Module
    - Participate in composing module design documents, module function realization, coding and UT
    - Managing lab D&T environment, CT10 and Radio test board
  - **NSB FN AONT related feature and FR fix**
    - Feature: Local announcements - Realizing different actions and announcements in specific customer scenario
    - Feature: Call Return - Call to last caller set up using 302
    - Feature: TR-104 management – Providing flexible configure method
  - **Alcatel-Lucent CTS SCG product feature: Authentication center**
    - Authentication center test under IMS-HSS, HLR-HSS, LTE-HSS environment
    - Including Milenage, SHA1 algorithm, triple authentication
    - Tested message includes MAR, MAA, AIR, AIA
    - 3GPP related protocol includes 3GPP TS 29.109 3GPP TS 35.205 3GPP TS 35.206
    - SCG 2.0 Feature 99731 test
    - Heat based Cloud System of LCP
## 2012.04 --- 2015.01 - HISENSE ELECTRIC CO.,LTD - Algorithm designer & Software R&D Engineer
  - Mainly engaged in R&D video post-processing algorithm and has been responsible for DI, SCALER, FRC, video compression algorithms, etc.
  - **Video compression algorithm**
    - In order to save the bandwidth of the DDR, we compress the brightness and chroma data of images
    - This algorithm is the core algorithm of the 4 k * 2 k hd resolution TV chip
    - I mainly responsible for packaging data, coding, and decompression, and control the compression ratio to 50% probably
  - **SCALER**
    - SCALER can satisfy the video signal in any size shown on the screen
    - Using the methods of de-jag and de-ring to optimization algorithm.
  - **DI**
    - DI is the conversion of interlaced scanning to progressive in analog TV.
  - **FRC**
    - Frame Rate Conversion
***
