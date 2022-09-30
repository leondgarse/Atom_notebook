# ___2017 - 01 - 17 shell 流处理命令___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2017 - 01 - 17 shell 流处理命令___](#2017-01-17-shell-流处理命令)
  - [目录](#目录)
  - [Three ways finding a string in a file](#three-ways-finding-a-string-in-a-file)
  - [find](#find)
  - [grep](#grep)
  - [awk](#awk)
  	- [awk工作流程](#awk工作流程)
  	- [示例](#示例)
  	- [awk内置变量](#awk内置变量)
  	- [awk自定义变量](#awk自定义变量)
  	- [awk 正则表达式匹配](#awk-正则表达式匹配)
  	- [awk 使用 if / for 语句](#awk-使用-if-for-语句)
  	- [awk 变量](#awk-变量)
  	- [awk 其他选项](#awk-其他选项)
  	- [awk编程](#awk编程)
  	- [重定向和管道](#重定向和管道)
  - [awk内置函数](#awk内置函数)
  	- [算术函数](#算术函数)
  	- [字符串函数](#字符串函数)
  	- [格式化字符串输出（sprintf使用）](#格式化字符串输出sprintf使用)
  	- [一般函数](#一般函数)
  	- [时间函数](#时间函数)
  - [sed](#sed)
  	- [Q / A](#q-a)
  	- [参数](#参数)
  	- [sed 只在匹配的第一行后添加一行](#sed-只在匹配的第一行后添加一行)
  	- [每一行前面添加字符串](#每一行前面添加字符串)
  - [cut](#cut)
  - [wc](#wc)
  - [diff](#diff)
  - [rename](#rename)
  - [sort 排序](#sort-排序)
  	- [参数](#参数)
  	- [指定 分隔符 域 的排序](#指定-分隔符-域-的排序)
  	- [k 选项的语法格式](#k-选项的语法格式)
  	- [t 选项分割](#t-选项分割)
  - [expect](#expect)
  	- [命令](#命令)
  	- [参数](#参数)
  	- [Tcl 函数](#tcl-函数)
  	- [if 判断 for 循环](#if-判断-for-循环)
  	- [Example expect file(exp_timeout.sh)](#example-expect-fileexptimeoutsh)
  - [xargs](#xargs)
  	- [FOO](#foo)

  <!-- /TOC -->
***

# Q / A
## 其他命令
  - 比较两个排序后的文件内容 **comm**
  - **dirname** 获取文件夹名，**basename** 获取文件名，**pwd** 获取当前文件夹名
  - **set -o vi** 将整个命令状态设置为 VI 编辑器
  - **rsync** [rsync详解之exclude排除文件](http://wanwentao.blog.51cto.com/2406488/582432/)
    ```shell
    rsync -arv --exclude "/home/ben/.ccache:/home/ben/build" /home/ben /media/ben/thumbdrive/
    rsync -arv --exclude=.ccache --exclude=build /home/ben /media/ben/thumbdrive/
    rsync -arv --exclude={.ccache,build} /home/ben /media/ben/thumbdrive/
    ```
## grep sed awk 查找字符串
  ```shell
  $ grep -n root /etc/passwd
  $ sed -n "/root/=;/root/p" /etc/passwd
  $ cat -n /etc/passwd | sed -n '/root/p'
  $ awk '/root/{print NR": "$0}' /etc/passwd
  ```
## Shell 执行 python 语句
  ```shell
  # 执行 python 语句，将 bb 下与 aa 中文件同名的文件，复制到 cc
  $ python -c "
  import os
  import shutil
  for tt in os.listdir('./aa'):
      if os.path.exists(os.path.join('./bb', tt)):
          shutil.copyfile(os.path.join('./bb', tt), os.path.join('./cc', tt))
  "
  ```
***

# find
  - 查找当前文件夹下所有文件
    ```shell
    # f regular file
    find ./* -maxdepth 0 -type f
    # d directory
    find ./* -maxdepth 0 -type d
    # l symbolic link
    find ./* -maxdepth 0 -type l
    ```
  - find exec 后面需要跟上 `\;` 表示命令的分割符
    ```shell
    find ./* -type f -exec echo {} \;
    ```
    exec 后面的命令实际是展开成多条命令，需要 `;` 分割
    ```shell
    $ find ./* -type f -exec echo {}
    find: missing argument to -exec
    ```
  - 流式编辑器sed命令修改当前文件夹下所有Makefile
    ```shell
    find ./ -name 'Makefile' -exec sed -i "s/CC = gcc/CC = arm-linux-gcc/g" {} \;
    ```
  - insmod所有的.ko文件
    ```shell
    find ./ -name '*.ko' -exec insmod {} \;
    ```
  - 以下命令开机加载所有的模块(/etc/init.d/rcS)
    ```shell
    find /ehome/modules/ -name *.ko -exec insmod {} \;
    ```
  - find & rm
    ```shell
    find / -type f -name *fetion* -exec rm {} \;
    find / -type f -name "*fetion*" | xargs rm -rf {}\;
    ```
  - find & grep
    ```shell
    find ./ -type f -iname "makefile" -exec grep -Hn "VOICE_GMI" {} \;
    find ./* -type f -iname *.c -o -iname *.cc -o -iname *.cpp | xargs grep -Hisn "" {} \;
    find ./* -type f -iname *.h -o -iname *.hh -o -iname *.hpp | xargs grep -Hisn "" {} \;
    ```
  - 当前文件夹中所有目录压缩为单独文件
    ```shell
    find ./* -maxdepth 0 -type d -exec zip -r {}.zip {} \;
    ```
  - Q: find 在目标路径下存在匹配的文件时
    ```shell
    find: paths must precede expression
    ```
    A: shell的通配符扩展 wildcard expansion 会将 * 这些通配符扩展成完整名称，因此在执行
    ```shell
    find . -name *test.c
    ```
    会用实际文件的名称扩展成
    ```shell
    find . -name bobtest.c cattest.c snowtest.c
    ```
    此时应使用引号
    ```shell
    find . -name '*test.c'
    ```
  - 查找 `/tmp` 下没有正在使用的文件
    ```sh
    find /tmp -mtime +7 -and -not -exec fuser -s {} ';' -and -exec echo {} ';'
    # -mtime +7: 大于 7 天
    # fuser -s: silent 执行
    ```
***

# grep
  - -s 不输出错误信息
  - -l 只输出匹配的文件名
  - -H 显示文件名，用于与find命令组合
  - 查找文件2中文件1没有的行
    ```
    grep -vFf 1.txt 2.txt
    ```
  - 忽略tags文件与二进制文件
    ```
    grep --exclude=tags -rinI "xp_fn_fillerrtone"
    ```
  - A / B / C
    ```
    -B, --before-context=NUM  print NUM lines of leading context
    -A, --after-context=NUM   print NUM lines of trailing context
    -C, --context=NUM         print NUM lines of output context
    -NUM                      same as --context=NUM
    ```
    ```bash
    grep -sI -A 2 strcpy ./*  # 后2行
    grep -sI -B 2 strcpy ./*  # 前2行
    grep -sI -C 2 strcpy ./*  # 前后2行
    grep -sI -2 strcpy ./*  # 前后2行
    ```
  - 其他
    ```
    grep -Ern '^(PRIVATE|PUBLIC) *S16\s*\S* *$|^(PRIVATE|PUBLIC) *S16\s*\S* *\(' zc_bdy4.c
    grep -Ern '^(PRIVATE|PUBLIC) *S16\s*'$function' *$|^(PRIVATE|PUBLIC) *S16\s*'$function' *\(|^'$function' *\(' zc_bdy4.c
    ```
***

# awk
## awk工作流程
  - awk '{pattern + action}' {filenames}
  - awk [-F field-separator] 'commands' input-file(s)
  - awk -f awk-script-file input-file(s)
  - 先执行BEGING
  - 然后读取文件，读入有/n换行符分割的一条记录
  - 然后将记录按指定的域分隔符划分域，填充域，$0则表示所有域,$1表示第一个域,$n表示第n个域
  - 随后开始执行模式所对应的动作action
  - 接着开始读入第二条记录，直到所有的记录都读完
  - 最后执行END操作。
## 示例
  ```sh
  awk -F ':' '{print $1}' /etc/passwd
  awk -F ':' '{print $1"\t"$7}' /etc/passwd
  awk -F ':' 'BEGIN {print "name,shell"} {print $1","$7} END {print "blue,/bin/nosh"}' /etc/passwd
  ```
  ```sh
  awk -F: '/root/' /etc/passwd        # 查找root，并打印整行
  awk -F: '/root/{print $7}' /etc/passwd        # 查找root，并打印第7个分割的字符串
  ```
  ```sh
  echo "this is a test" | awk '$0~/test/{print $0}'        # 匹配正则表达式则打印

  awk -F : '$0~/leondgarse/{print $1}' /etc/group         # 查找用户属于的组

  awk -F: '$3~/[0-9][0-9]$/{print $1"\t"$3}' /etc/passwd # 第三个域以两个数字结束就打印这个记录
  awk -F: '$3>200 || $4==65534' /etc/passwd # 第三个域大于200或者第四个域等于65534，则打印该行
  ```
  **在 shell 脚本中，$ 需要转义**
  ```sh
  DD="fff ggg  hhh"
  for (( i=1; ; i=$i+1 ))
  do
      FDD=$(echo $DD | awk -F ' ' "{print \$$i}")
      if [ -z $FDD ]; then
          break
      fi
      echo "FDD = $FDD"
  done
  ```
  **计算**
  ```sh
  # 浮点数除法
  awk -v aa=1 -v bb=30 'BEGIN {printf "%0.2f\n", aa / bb}'
  awk -v aa=1 -v bb=30 'BEGIN {printf "%0.2f%%\n", aa / bb * 100}'

  # 计算器
  calc() { awk "BEGIN {print $*}"; }
  calc '1 + 2 * 3 / 4'  # 2.5
  ```
  **统计 `top` 占用 CPU 最高的 30 个进程的总量**
  ```py
  top -b -n 1 | head -n 37 | tail -n 30 | awk -F' ' 'BEGIN {total=0} {size=size+$9} END {print "Sum 30: " size "%"}'
  ```
## awk内置变量
  | 变量名   |                                            |
  | -------- | ------------------------------------------ |
  | ARGC     | 命令行参数个数                             |
  | ARGV     | 命令行参数排列                             |
  | ENVIRON  | 支持队列中系统环境变量的使用               |
  | FILENAME | awk浏览的文件名                            |
  | FNR      | 浏览文件的记录数                           |
  | FS       | 设置输入域分隔符，等价于命令行 -F选项      |
  | NF       | 浏览记录的域的个数，$NF 可以表示最后一个域 |
  | NR       | 已读的记录数                               |
  | OFS      | 输出域分隔符                               |
  | ORS      | 输出记录分隔符                             |
  | RS       | 控制记录分隔符                             |

  ```sh
  awk -F ':' '{print "filename:" FILENAME ",linenumber:" NR ",columns:" NF ",linecontent:"$0}' /etc/passwd

  # printf version:
  awk -F ':' '{printf("filename: %-15s linenumber: %-3s columns: %-3s linecontent: %s\n",FILENAME,NR,NF,$0)}' /etc/passwd

  # 打印第 1 行的第 5 个元素
  awk -F ':' 'NR==1 {print $5}' /etc/passwd
  ```
## awk自定义变量
  ```sh
  # 统计/etc/passwd的账户人数
  awk 'BEGIN {count=0;print "[start]user count is ", count} {count=count+1;print $0;} END{print "[end]user count is ", count}' /etc/passwd

  # 统计某个文件夹下的文件占用的字节数
  ls -l | awk 'BEGIN {size=0;} {size=size+$5;} END{print "[end]size is ", size}'
  ```
## awk 正则表达式匹配
  ```shell
  awk 'BEGIN {info="this is a test"; if(info ~ /test/){print "ok"}}'        # 包含字符串test
  awk 'BEGIN {info="hi i a"; if(info !~ "[test]"){print "ok"}}'        # 包含test中的任意一个字符
  awk 'BEGIN {info="hi i a"; if(info !~ /[test]/){print "ok"}}'        
  awk -F '[ ,]*' '{printf("This is a test for %s\n", $2)}' foo > bar        # 正则表达式用于分隔符

  awk -F ']' '{if($1 !~ "\\[SSTK") {print $0}}' cvp.log.sstk > foo        # 特殊字符匹配转义使用 \\
  ```
## awk 使用 if / for 语句
  - **if 语句**
    ```sh
    # 打印第四个字段长度大于 3 的行
    awk -F: '{if (length($4)>3) print $0}' /etc/passwd

    # 统计某个文件夹下的文件占用的字节数，过滤4096大小的文件(一般都是文件夹)
    ls -l | awk 'BEGIN {size=0;print "[start]size is ", size} {if($5!=4096){size=size+$5;}} END{print "[end]size is ", size/1024/1024,"M"}'

    # 查找当前文件夹下大小为 0 的文件
    find ./* -type f -exec ls -l {} \;  | awk '{if ($5 == 0) print $0}'

    # 解析 tsv 文件，NR > 1 跳过表头，NF>4 表示一行中元素数量大于 4，$1 ~/^[0-9]/ 表示第一个数据是数字开头的
    awk -F '\t' 'NR > 1 {if (NF>4 && $1 ~/^[0-9]/) print $2" "$3" "$4" "$5}' SSS.tsv
    ```
  - **while 循环 / for 循环 / special for 循环**
    ```sh
    $ awk '{ i = 1; while ( i <= NF ) { print NF,$i; i++}}' test 变量的初始值为1，若i小于等于NF(域个数),则执行打印语句，且i增加1。直到i的值大于NF.
    $ awk '{for (i = 1; i<NF; i++) print NF,$i}' test 作用同上
    ```
    循环语句，显示 /etc/passwd 的账户
    ```sh
    awk -F ':' 'BEGIN {count=0;} {name[count] = $1;count++;} END{for (i = 0; i < NR; i++) print i, name[i]}' /etc/passwd
    ```
  - **breadk 与 continue语句** break 用于在满足条件的情况下跳出循环，continue 用于在满足条件的情况下忽略后面的语句，直接返回循环的顶端
    ```py
    {for ( x=3; x<=NF; x++)
            if ($x<0){print "Bottomed out!"; break}}
    {for ( x=3; x<=NF; x++)
            if ($x==0){print "Get next item"; continue}}
    ```
  - **next 语句** 从输入文件中读取一行，然后从头开始执行awk脚本
    ```py
    { if ($1 ~/test/){ next }
            else { print } }
    ```
## awk 变量
  - $ awk '$1 ~/test/{count = $2 + $3; print count}' test
    ```
    上式的作用是,awk先扫描第一个域，一旦test匹配，就把第二个域的值加上第三个域的值，并把结果赋值给变量count，最后打印出来
    ```
  - awk可以在命令行中给变量赋值，然后将这个变量传输给awk脚本
    ```
    如$ awk -F: -f awkscript month=4 year=2004 test，上式的month和year都是自定义变量，分别被赋值为4和2004
    在awk脚本中，这些变量使用起来就象是在脚本中建立的一样
    注意，如 果参数前面出现test，那么在BEGIN语句中的变量就不能被使用
    ```
  - 域变量也可被赋值和修改, 如$ awk '{$2 = 100 + $1; print }' test
    ```
    上式表示,如果第二个域不存在，awk将计算表达式100加$1的值，并将其赋值给$2
    如果第二个域存在，则用表达式的值覆盖$2原来的值
    再例如：$ awk '$1 == "root"{$1 ="test";print}' test，如果第一个域的值是“root”，则把它赋值为“test”
    注意，字符串一定要用双引号
    ```
  - 内建变量的使用
    ```
    变量列表在前面已列出，$ awk -F: '{IGNORECASE=1; $1 == "MARY"{print NR,$1,$2,$NF}'test
    把IGNORECASE设为1代表忽略大小写，打印第一个域是mary的记录数、第一个域、第二个域和最后一个域
    ```
## awk 其他选项
  - -v 定义变量值
    ```
    查找长度为3个字母的字符串：
    printf "%s\n%s\n%s\n" aaaa adf bbb | awk -F '' -vs=3 '{for(i=1; i<=NF;i++)if($i~/[a-z]/)num++; if(num==s)print $0; num=0}'
    ```
## awk编程
  - 用户tty号
    ```
    who am i | awk '{print $2}'
    ```
  - 查找文件中的字符串，第一项由后两项加中间数字组成
    ```
    printf "%s %s %s\n%s %s %s\n%s %s %s\n%s %s %s\n%s %s %s\n" \
            abc234o abc o \
            1abc234 abc q \
            kjl333 jjj d \
            xyz33z  xyz z \
            xyz33  xyz z |
    awk '{var="^"$2"[0-9]+"$3"$"}$1~var'
    输出：abc234o abc o
          xyz33z  xyz z
    ```
  - 打印文件中有且只有三个字母连续出现的行
    ```
    awk -vn=3 '{s=$0}gsub(/[a-z]/,"",s)==n' foo # fail
    awk -vn=3 'gsub(/[a-z]+/,"&")==1&&gsub(/[a-z]/,"&")==n' foo # done
    awk -F "" -vs=3 '{for(i=1;i<=NF;i++)if($i~/:alpha:/)num++;if(num==3)print $0;num=0}' foo # fail
    awk --posix '{if($0~/^[^[:alpha:]]*:alpha:{3}[^[:alpha:]]*$/)print $0}' foo # done
    ```
## 重定向和管道
  - $ awk 'BEGIN{ "date" | getline d; print d}'
    ```
    执行linux的date命令，并通过管道输出给getline，然后再把输出赋值给自定义变量d，并打印它
    ```
  - $ awk 'BEGIN{"date" | getline d; split(d,mon); print mon[2]}'
    ```
    执行shell的date命令，并通过管道输出给getline，然后getline从管道中读取并将输入赋值给d
    split函数把变量d转化成数组mon，然后打印数组mon的第二个元素
    ```
  - $ awk 'BEGIN{while( "ls" | getline) print}'
    ```
    命令ls的输出传递给geline作为输入，循环使getline从ls的输出中读取一行，并把它打印到屏幕
    这里没有输入文件，因为BEGIN块在打开输入文件前执行，所以可以忽略输入文件
    ```
  - $ awk 'BEGIN{printf "What is your name?"; getline name < "/dev/tty" } $1 ~name {print "Found " name " on line :" NR "."} END{print "See you," name ": "$0"."}' /etc/passwd
    ```
    在屏幕上打印”What is your name?",并等待用户应答
    当一行输入完毕后，getline函数从终端接收该行输入，并把它储存在自定义变量name中
    如果第一个域匹配变量name的值，print函数就被执行，END块打印See you和name的值
    ```
  - $ awk 'BEGIN{while (getline < "/etc/passwd" > 0) lc++; print lc}'
    ```
    awk将逐行读取文件/etc/passwd的内容，在到达文件末尾前，计数器lc一直增加，当到末尾时，打印lc的值
    注意，如果文件不存在，getline返回-1，如果到达文件的末尾就返回0，如果读到一行，就返回1
    所以命令 while (getline < "/etc/passwd")在文件不存在的情况下将陷入无限循环，因为返回-1表示逻辑真
    ```
  - 可以在awk中打开一个管道，且同一时刻只能有一个管道存在，通过close()可关闭管道，如：
    ```
    $ awk '{print $1, $2 | "sort" }' test END {close("sort")}
    awd把print语句的输出通过管道作为linux命令sort的输入,END块执行关闭管道操作
    ```
  - system函数可以在awk中执行linux的命令。如：
    ```
    $ awk 'BEGIN{system("clear")}'
    ```
***

# awk内置函数
## 算术函数
  - 以下算术函数执行与 C 语言中名称相同的子例程相同的操作

    | 函数名 | 说明 |
    |-------|-----|
    | cos( x ) | 返回 x 的余弦；x 是弧度 |
    | sin( x ) | 返回 x 的正弦；x 是弧度 |
    | exp( x ) | 返回 x 幂函数 |
    | log( x ) | 返回 x 的自然对数 |
    | sqrt( x ) | 返回 x 平方根
    | int( x ) | 返回 x 的截断至整数的值 |
    | rand( ) | 返回任意数字 n，其中 0 <= n < 1 |
    | srand( [Expr] ) | 将 rand 函数的种子值设置为 Expr 参数的值，或如果省略 Expr 参数则使用某天的时间。返回先前的种子值 |
  - 举例说明：
    ```
    $ awk 'BEGIN{OFMT="%.3f";fs=sin(1);fe=exp(10);fl=log(10);fi=int(3.1415);print fs,fe,fl,fi;}'
    0.841 22026.466 2.303 3
    OFMT 设置输出数据格式是保留3位小数
    ```
    获得随机数：
    ```
    $ awk 'BEGIN{srand();fr=int(100*rand());print fr;}'
    78
    $ awk 'BEGIN{srand();fr=int(100*rand());print fr;}'
    31
    $ awk 'BEGIN{srand();fr=int(100*rand());print fr;}'
    41
    ```
## 字符串函数
  - Ere都可以是正则表达式

    | 函数 | 说明 |
    |-------|-----|
    | gsub( Ere, Repl, [ In ] )  | 除了正则表达式所有具体值被替代这点，它和 sub 函数完全一样地执行，|
    | sub( Ere, Repl, [ In ] )  | 用 Repl 参数指定的字符串替换 In 参数指定的字符串中的由 Ere 参数指定的扩展正则表达式的第一个具体值。sub 函数返回替换的数量。出现在 Repl 参数指定的字符串中的 &（和符号）由 In 参数指定的与 Ere 参数的指定的扩展正则表达式匹配的字符串替换。如果未指定 In 参数，缺省值是整个记录（$0 记录变量）|
    | index( String1, String2 )  | 在由 String1 参数指定的字符串（其中有出现 String2 指定的参数）中，返回位置，从 1 开始编号。如果 String2 参数不在 String1 参数中出现，则返回 0（零）|
    | length [(String)]  | 返回 String 参数指定的字符串的长度（字符形式）。如果未给出 String 参数，则返回整个记录的长度（$0 记录变量）|
    | blength [(String)]  | 返回 String 参数指定的字符串的长度（以字节为单位）。如果未给出 String 参数，则返回整个记录的长度（$0 记录变量）|
    | substr( String, M, [ N ] )  | 返回具有 N 参数指定的字符数量子串。子串从 String 参数指定的字符串取得，其字符以 M 参数指定的位置开始。M 参数指定为将 String 参数中的第一个字符作为编号 1。如果未指定 N 参数，则子串的长度将是 M 参数指定的位置到 String 参数的末尾 的长度 |
    | match( String, Ere )  | 在 String 参数指定的字符串（Ere 参数指定的扩展正则表达式出现在其中）中返回位置（字符形式），从 1 开始编号，或如果 Ere 参数不出现，则返回 0（零）。RSTART 特殊变量设置为返回值。RLENGTH 特殊变量设置为匹配的字符串的长度，或如果未找到任何匹配，则设置为 -1（负一）|
    | split( String, A, [Ere] )  | 将 String 参数指定的参数分割为数组元素 A[1], A[2], . . ., A[n]，并返回 n 变量的值。此分隔可以通过 Ere 参数指定的扩展正则表达式进行，或用当前字段分隔符（FS 特殊变量）来进行（如果没有给出 Ere 参数）。除非上下文指明特定的元素还应具有一个数字值，否则 A 数组中的元素用字符串值来创建 |
    | tolower( String )  | 返回 String 参数指定的字符串，字符串中每个大写字符将更改为小写。大写和小写的映射由当前语言环境的 LC_CTYPE 范畴定义 |
    | toupper( String )  | 返回 String 参数指定的字符串，字符串中每个小写字符将更改为大写。大写和小写的映射由当前语言环境的 LC_CTYPE 范畴定义 |
    | sprintf(Format, Expr, Expr, . . . )  | 根据 Format 参数指定的 printf 子例程格式字符串来格式化 Expr 参数指定的表达式并返回最后生成的字符串 |
  - gsub,sub使用
    ```
    $ awk 'BEGIN{info="this is a test2010test!";gsub(/[0-9]+/,"!",info);print info}'  
    this is a test!test!
    在 info中查找满足正则表达式，/[0-9]+/ 用””替换，并且替换后的值，赋值给info 未给info值，默认是$0
    ```
  - 查找字符串（index使用）
    ```
    $ awk 'BEGIN{info="this is a test2010test!";print index(info,"test")?"ok":"no found";}'   
    ok
    未找到，返回0
    ```
  - 正则表达式匹配查找(match使用）
    ```
    $ awk 'BEGIN{info="this is a test2010test!";print match(info,/[0-9]+/)?"ok":"no found";}'          
    ok
    ```
  - 截取字符串(substr使用）
    ```
    $ awk 'BEGIN{info="this is a test2010test!";print substr(info,4,10);}'                        
    s is a tes
    从第 4个 字符开始，截取10个长度字符串
    ```
  - 字符串分割（split使用）
    ```
    $ awk 'BEGIN{info="this is a test";split(info,tA," ");print length(tA);for(k in tA){print k,tA[k];}}'
    4
    4 test
    1 this
    2 is
    3 a
    分割info,动态创建数组tA,这里比较有意思，awk for …in 循环，是一个无序的循环。 并不是从数组下标1…n ，因此使用时候需要注意。
    ```
## 格式化字符串输出（sprintf使用）
  - 格式化字符串格式：
    ```
    其中格式化字符串包括两部分内容: 一部分是正常字符, 这些字符将按原样输出; 另一部分是格式化规定字符, 以"%"开始, 后跟一个或几个规定字符,用来确定输出内容格式。
    ```
    | 格式符  | 说明 |
    |-------|-----|
    | %d  | 十进制有符号整数 |
    | %u  | 十进制无符号整数 |
    | %f  | 浮点数 |
    | %s  | 字符串 |
    | %c  | 单个字符 |
    | %p  | 指针的值 |
    | %e  | 指数形式的浮点数 |
    | %x  | %X 无符号以十六进制表示的整数 |
    | %o  | 无符号以八进制表示的整数 |
    # | %g  | 自动选择合适的表示法 |
  - 举例说明：
    ```sh
    $ awk 'BEGIN{n1=124.113;n2=-1.224;n3=1.2345; printf("%.2f,%.2u,%.2g,%X,%o\n",n1,n2,n3,n1,n1);}'
    124.11,18446744073709551615,1.2,7C,174
    ```
## 一般函数
  | 函数  | 说明 |
  |-------|-----|
  | close( Expression )  | 用同一个带字符串值的 Expression 参数来关闭由 print 或 printf 语句打开的或调用 getline 函数打开的文件或管道。如果文件或管道成功关闭，则返回 0；其它情况下返回非零值。如果打算写一个文件，并稍后在同一个程序中读取文件，则 close 语句是必需的 |
  | system(Command )  | 执行 Command 参数指定的命令，并返回退出状态。等同于 system 子例程 |
  | Expression | getline [ Variable ]  | 从来自 Expression 参数指定的命令的输出中通过管道传送的流中读取一个输入记录，并将该记录的值指定给 Variable 参数指定的变量。如果当前未打开将 Expression 参数的值作为其命令名称的流，则创建流。创建的流等同于调用 popen 子例程，此时 Command 参数取 Expression 参数的值且 Mode 参数设置为一个是 r 的值。只要流保留打开且 Expression 参数求得同一个字符串，则对 getline 函数的每次后续调用读取另一个记录。如果未指定 Variable 参数，则 $0 记录变量和 NF 特殊变量设置为从流读取的记录 |
  | getline [ Variable ] < Expression  | 从 Expression 参数指定的文件读取输入的下一个记录，并将 Variable 参数指定的变量设置为该记录的值。只要流保留打开且 Expression 参数对同一个字符串求值，则对 getline 函数的每次后续调用读取另一个记录。如果未指定 Variable 参数，则 $0 记录变量和 NF 特殊变量设置为从流读取的记录 |
  | getline [ Variable ]  | 将 Variable 参数指定的变量设置为从当前输入文件读取的下一个输入记录。如果未指定 Variable 参数，则 $0 记录变量设置为该记录的值，还将设置 NF、NR 和 FNR 特殊变量 |
  - 打开外部文件（close用法）
    ```sh
    $ awk 'BEGIN{while("cat /etc/passwd"|getline){print $0;};close("/etc/passwd");}'
    root:x:0:0:root:/root:/bin/bash
    bin:x:1:1:bin:/bin:/sbin/nologin
    daemon:x:2:2:daemon:/sbin:/sbin/nologin
    ```
  - 逐行读取外部文件(getline使用方法）
    ```sh
    $ awk 'BEGIN{while(getline < "/etc/passwd"){print $0;};close("/etc/passwd");}'
    root:x:0:0:root:/root:/bin/bash
    bin:x:1:1:bin:/bin:/sbin/nologin
    daemon:x:2:2:daemon:/sbin:/sbin/nologin

    $ awk 'BEGIN{print "Enter your name:";getline name;print name;}'
    Enter your name:
    ```
  - 调用外部应用程序(system使用方法）
    ```sh
    $ awk 'BEGIN{b=system("ls -al");print b;}'
    b返回值，是执行结果
    ```
## 时间函数
  | 函数名  | 说明 |
  | -------|-----|
  | mktime( YYYY MM DD HH MM SS[ DST])  | 生成时间格式 |
  | strftime([format [, timestamp]])  | 格式化时间输出，将时间戳转为时间字符串, 具体格式，见下表. |
  | systime()  | 得到时间戳,返回从1970年1月1日开始到当前时间(不计闰年)的整秒数 |
  - 创建指定时间(mktime使用）
    ```sh
    $ awk 'BEGIN{tstamp=mktime("2001 01 01 12 12 12");print strftime("%c",tstamp);}'
    2001年01月01日 星期一 12时12分12秒 
    $ awk 'BEGIN{tstamp1=mktime("2001 01 01 12 12 12");tstamp2=mktime("2001 02 01 0 0 0");print tstamp2-tstamp1;}'
    2634468
    求2个时间段中间时间差,介绍了strftime使用方法 
    $ awk 'BEGIN{tstamp1=mktime("2001 01 01 12 12 12");tstamp2=systime();print tstamp2-tstamp1;}'
    308201392 
    ```
  - strftime日期和时间格式说明符

    | 格式 | 描述 |
    | -------|-----|
    | %a  | 星期几的缩写(Sun) |
    | %A  | 星期几的完整写法(Sunday) |
    | %b  | 月名的缩写(Oct) |
    | %B  | 月名的完整写法(October) |
    | %c  | 本地日期和时间 |
    | %d  | 十进制日期 |
    | %D  | 日期 08/20/99 |
    | %e  | 日期，如果只有一位会补上一个空格 |
    | %H  | 用十进制表示24小时格式的小时 |
    | %I  | 用十进制表示12小时格式的小时 |
    | %j  | 从1月1日起一年中的第几天 |
    | %m  | 十进制表示的月份 |
    | %M  | 十进制表示的分钟 |
    | %p  | 12小时表示法(AM/PM) |
    | %S  | 十进制表示的秒 |
    | %U  | 十进制表示的一年中的第几个星期(星期天作为一个星期的开始) |
    | %w  | 十进制表示的星期几(星期天是0) |
    | %W  | 十进制表示的一年中的第几个星期(星期一作为一个星期的开始) |
    | %x  | 重新设置本地日期(08/20/99) |
    | %X  | 重新设置本地时间(12：00：00) |
    | %y  | 两位数字表示的年(99) |
    | %Y  | 当前月份 |
    | %Z  | 时区(PDT) |
    | %%  | 百分号(%) |
***

# sed
## Q / A
  - **sed** stream editor for filtering and transforming text
  - 将文本处理命令应用到每一行
  - 若使用shell变量，应使用""，而不是''，并做转义：
    ```shell
    /bin/sed -i "\$a\\Match User $newUserName" /etc/ssh/sshd_config
    ```
    应用到每个文件
    ```shell
    find ./* -type f -exec sed -i 's:/home/leondgarse:~:g' {} \;        # 使用xargs会报错
    ```
  - 在既使用正则表达式又使用 shell 变量时应将 sed 命令字符串分割开
    ```shell
    HOST_NAME=`cat /etc/hostname`
    NEW_HOST_NAME=$HOST_NAME"_NEW"
    sed 's/^127.0.1.1\s*'$HOST_NAME'/127.0.1.1\t'$NEW_HOST_NAME'/' /etc/hosts    
    ```
## 参数
  - -e 表示使用sed脚本，后面紧跟sed脚本，可使用 ; 链接多个命令，或者使用多个 -e
    ```shell
    sed -n -e '=' -e 'p' foo
    ```
    -n 输出时指定为不打印原文
  - -i.bak 表示将原文件加后缀.bak作为备份，改动将保存在原文件中，若i后为空，则不做备份
  - -f 指定使用脚本文件
  - -r 使用扩展的正则表达式
  - s/// 替换命令，可以使用不同的分隔符s:::：
    ```shell
    # 替换每一行的第一个foo为bar
    sed -ie 's/foo/bar/' foo
    # 替换全部foo为bar
    sed -ie 's:foo:bar:g' foo

    # 1-10行删除<>包含的语句，会查找最长的匹配，类似
    # <b>This</b> is what <b>I</b> meant 将变成 meant
    sed -ie '1,10s/<.*>//g' foo
    # 查找以<开头，不包含>，并以>结尾的字符串删除，类似：
    # <b>This</b> is what <b>I</b> meant 将变成 This is what I meant
    sed -ie 's/<[^>]*>//g' foo

    # 从空行到以 END 开头的一行，替换 foo 为 bar，指定的查找区间不止一个时，替换所有符合条件的区间
    sed -ie '/^$/,/^END/s:foo:bar:g' foo

    # 删除行首的空格与制表符
    echo "    foo goo  " | sed 's/^[ \t]*//'
    ```
    **'&' 表示插入整个匹配的规则表达式**
    ```shell
    # foo 在每一行的前面加上foo:
    sed -ie 's/.*/foo: &/'
    # foo 在所有bar前添加foo，并重复bar三次
    sed -ie 's/bar/foo&&&/'
    ```
    **使用 () 捕获组**
    ```shell
    # 使用捕获组将每组2个的单词，重新组合成A: B:的形式
    sed -i.bak -e 's/\([^ ]*\) \([^ ]*\) /A: \1 B: \2\n/g' foo

    # 将 ![](images/foo.png) 替换为 ![foo](images/foo.png) 的格式
    printf 'foo\n![image](images/foo.jpg)\n![](images/goo.png)\n' | sed 's#!\[\](\(.*\)\/\(.*\)\.\(.*\))#!\[\2\](\1\/\2\.\3)#'
    ```
  - d 删除：
    ```shell
    sed '1d' foo 删除第一行
    sed '2,$d' foo 删除第二行到最后一行
    ```
  - p 打印，通常与 -n一起使用，=打印行号：
    ```shell
    sed -n '1p' foo 显示第一行
    sed -n '/bash/p' foo 显示包含bash的行
    sed -n '/sed/=;/sed/p' foo 打印包含sed的行号与行
    sed -n -e 's/bash/zsh/g;/zsh/p' foo 将bash替换为zsh，并打印这些行
    sed -n '/bash/p' foo | sed -e 's/bash/zsh/g' 只提取包含bash的行，将其中的bash替换为zsh
    ```
  - i 当前行之前插入一行：
    ```shell
    sed -i.bak -e 'i\ FOO: ' foo 每一行前面插入新行FOO:
    sed -i.bak -n -e 'i\ FOO: ' -e 'p' foo 使用p打印
    ```
  - a 当前行之后插入一行：
    ```shell
    sed -i.bak -e '1a\Insert this line after each line. Thanks!' foo 第一行后插入新行
    ```
  - c 替换当前行：
    ```shell
    sed -i.bak -e 'c\neeeew!' foo
    ```
  - n 读取下一行
    ```shell
    sed -n '1,$n;p' foo 显示偶数行
    sed -n '1,$p;n' foo 显示奇数行
    ```
  - addr1,+N Will match addr1 and the N lines following addr1.
  - addr1,~N Will match addr1 and the lines following addr1 until the next line whose input line number is a multiple of N.
    ```shell
    sed '2~2d' file 显示奇数行
    sed -n '1~2p' file 显示奇数行
    sed '1~2d' file 显示偶数行
    sed -n '2~2p' file 显示偶数行
    ```
  - l List out the current line in a ''visually unambiguous'' form.
    ```shell
    sed -n l foo Printing '\t' for TAB
    ```
  - y 替换单个字符，不能使用正则表达式，一一对应地替换
    ```shell
    echo 'hello' | sed 'y/abcdefghigklmn/ABCDEFGHIJKLMN/'
    # [Out]: HELLo
    ```
## sed 只在匹配的第一行后添加一行
  ```shell
  printf "foo\ntest\ntest\ntest\nfoo\n" | sed '0,/test/s/.*test.*/&\n\tnothing/'
  [Out]
  foo
  test
  	nothing
  test
  test
  foo
  ```
## 每一行前面添加字符串
  - sed
    ```shell
    echo 'test' | sed 's/\(.*\)/foo \1/'
    # foo test

    echo 'test' | sed 's/.*/foo &/'
    # foo test
    ```
    只在匹配的第一行前面添加字符串
    ```sh
    # 查找第一行后面不以 # 开头的一行，添加一个 #，然后删除第一行前面添加的 #
    printf "foo\n#test\ntest\ntest\nfoo\n" | sed '1,/^[^#]/s/^[^#].*/# &/' | sed '1s/^# \(.*\)/\1/'
    foo
    #test
    # test
    test
    foo
    ```
  - awk
    ```shell
    echo 'test' | awk '{print "foo "$0}'
    # foo test
    ```
  - python 中使用 os.system 时，需要将 ``\1`` 转义
    ```python
    import os

    os.system("echo 'test' | sed 's/^\(.*\)$/foo \1/'")
    # 输出乱码 foo   [ ??? ]

    os.system("echo 'test' | sed 's/^\(.*\)$/foo \\1/'")
    # foo test
    ```
    在 shell 中直接调用 python 还需要再转义
    ```shell
    python -c "import os; os.system(\"echo 'test' | sed 's/^\(.*\)$/foo \\1/'\")"
    # foo 

    python -c "import os; os.system(\"echo 'test' | sed 's/^\(.*\)$/foo \\\1/'\")"
    # foo test
    ```
***

# cut
  - cut 命令从文件的每一行剪切字节、字符和字段并将这些字节、字符和字段写至标准输出
    ```shell
    cut  [-bn] [file]
    cut [-c] [file]
    cut [-df] [file]
    ```
  - **参数** 必须指定 -b / -c / -f 标志之一
    - **-b** 以字节为单位进行分割。这些字节位置将忽略多字节字符边界，除非也指定了 -n 标志
    - **-c** 以字符为单位进行分割
    - **-d** 自定义分隔符，默认为制表符
    - **-f** 与-d一起使用，指定显示哪个区域
    - **-s** --only-delimited, do not print lines not containing delimiters
  - **-b** 以字节为单位
    ```shell
    $ who|cut -b 3 # extracting 3rd character
    $ who|cut -b 3-5,8 # extracting 3,4,5,8characters
    $ who|cut -b -3 # extracting first 3 characters
    $ who|cut -b 3- # extracting characters except first 3
    ```
  - **-c** 以字符为单位
    ```shell
    echo "星期一" | cut -c 2 # [working?]
    ```
  - **-f** 域分割
    ```shell
    $ cut -d: -f1 /etc/passwd
    $ cut -d: -f1-3,5 /etc/passwd
    ```
  - **-s** only-delimited
    ```shell
    $ cut -d1 -f1 /etc/passwd # output lines dont contain '1'
    $ cut -d1 -sf1 /etc/passwd # ignore lines dont contain '1'
    ```
  - cut 只擅长处理 **以一个字符间隔** 的文本内容，如果文件里面的某些域是由若干个空格来间隔的，那么用cut就有点麻烦了
    ```shell
    echo "   fff  ggg  hhh" > foo
    cat foo #    fff  ggg  hhh
    # cut 只能切割单个字符，分隔符有多个连续时，使用 awk
    cut -d' ' -f 4 foo  # fff
    awk -F' ' '{print $1}' foo  # fff
    ```
***

# wc
  - 统计给定文件中的字节数、字数、行数。如果没有给出文件名，则从标准输入读取
  - c 统计字节数
  - l 统计行数
  - w 统计字数
  - 这些选项可以组合使用, 输出列的顺序和数目不受选项的顺序和数目的影响, 总是按下述顺序显示并且每项最多一列
    ```shell
    行数 字数 字节数 文件名
    ```
    ```shell
    wc -lcw foo
    4 4 40 foo
    ```
  - 如果命令行中没有文件名，则输出中不出现文件名
  - 省略选项wc命令的执行 -lcw
  - 统计文件夹下的文件数量
    ```shell
    # HOME 目录下隐藏文件的数量
    ls -d ~/.* | wc -w  # 88
    # 当前文件夹下包括子目录下的文件夹数量
    ls -lR | grep "^d" | wc -l
    ```
***

# diff
  - diff(differential) 功能说明：比较文件的差异
  - 语 法
    ```
    diff [-abBcdefHilnNpPqrstTuvwy] [-<行数>][-C <行数>][-D <巨集名称>][-I <字符或字符串>][-S < 文件>][-W <宽度>][-x <文件或目录>][-X <文件>][--help][--left- column][--suppress-common-line][文件或目录1][文件或目录2]
    ```
  - 补充说明
    ```
    diff以逐行的方式，比较文本文件的异同处。所是指定要比较目录，则diff会比较目录中相同文件名的文件，但不会比较其中子目录
    ```
  - 参 数：
    ```
    -<行数> 指定要显示多少行的文本。此参数必须与-c或-u参数一并使用。
    -a或--text diff预设只会逐行比较文本文件。
    -b或--ignore-space-change 不检查空格字符的不同。
    -B或--ignore-blank-lines 不检查空白行。
    -c 显示全部内文，并标出不同之处。
    -C<行数>或--context<行数> 与执行"-c-<行数>"指令相同。
    -d或--minimal 使用不同的演算法，以较小的单位来做比较。
    -D<巨集名称>或ifdef<巨集名称> 此参数的输出格式可用于前置处理器巨集。
    -e或--ed 此参数的输出格式可用于ed的script文件。
    -f或-forward-ed 输出的格式类似ed的script文件，但按照原来文件的顺序来显示不同处。
    -H或--speed-large-files 比较大文件时，可加快速度。
    -l<字符或字符串>或--ignore-matching-lines<字符或字符串> 若两个文件在某几行有所不同，而这几行同时都包含了选项中指定的字符或字符串，则不显示这两个文件的差异。
    -i或--ignore-case 不检查大小写的不同。
    -l或--paginate 将结果交由pr程序来分页。
    -n或--rcs 将比较结果以RCS的格式来显示。
    -N或--new-file 在比较目录时，若文件A仅出现在某个目录中，预设会显示：
    Only in目录：文件A若使用-N参数，则diff会将文件A与一个空白的文件比较。
    -p 若比较的文件为C语言的程序码文件时，显示差异所在的函数名称。
    -P或--unidirect        ional-new-file 与-N类似，但只有当第二个目录包含了一个第一个目录所没有的文件时，才会将这个文件与空白的文件做比较。
    -q或--brief 仅显示有无差异，不显示详细的信息。
    -r或--recursive 比较子目录中的文件。
    -s或--report-identical-files 若没有发现任何差异，仍然显示信息。
    -S<文件>或--starting-file<文件> 在比较目录时，从指定的文件开始比较。
    -t或--expand-tabs 在输出时，将tab字符展开。
    -T或--initial-tab 在每行前面加上tab字符以便对齐。
    -u,-U<列数>或--unified=<列数> 以合并的方式来显示文件内容的不同。
    -v或--version 显示版本信息。
    -w或--ignore-all-space 忽略全部的空格字符。
    -W<宽度>或--width<宽度> 在使用-y参数时，指定栏宽。
    -x<文件名或目录>或--exclude<文件名或目录> 不比较选项中所指定的文件或目录。
    -X<文件>或--exclude-from<文件> 您可以将文件或目录类型存成文本文件，然后在=<文件>中指定此文本文件。
    -y或--side-by-side 以并列的方式显示文件的异同之处。
    --help 显示帮助。
    --left-column 在使用-y参数时，若两个文件某一行内容相同，则仅在左侧的栏位显示该行内容。
    --suppress-common-lines 在使用-y参数时，仅显示不同之处。
    ```
***

# rename
  - 用字符串替换的方式批量改变文件名
  - 原字符串：将文件名需要替换的字符串
  - 目标字符串：将文件名中含有的原字符替换成目标字符串
  - 文件：指定要改变文件名的文件列表
  - 将main1.c重命名为main.c
    ```shell
    $ rename main1.c main.c main1.c
    ```
  - ? 可替代单个字符
  - * 可替代多个字符
  - [charset] 可替代charset集中的任意单个字符
  - 用例
    ```shell
    # 把foo1到foo9的文件重命名为foo01到foo09，重命名的文件只是有4个字符长度名称的文件，文件名中的foo被替换为foo0
    $ rename foo foo0 foo?

    # foo01到foo99的所有文件都被重命名为foo001到foo099，只重命名5个字符长度名称的文件，文件名中的foo被替换为foo0
    $ rename foo foo0 foo??

    # foo001到foo278的所有文件都被重命名为foo0001到foo0278，所有以foo开头的文件都被重命名
    $ rename foo foo0 foo*

    # 从foo0200到foo0278的所有文件都被重命名为foo200到foo278，文件名中的foo0被替换为foo
    $ rename foo0 foo foo0[2]*
    ```
  - rename支持正则表达式:
    ```shell
    rename "s/AA/aa/" *        # 把文件名中的AA替换成aa
    rename "s//.html//.php/" * # 把.html 后缀的改成 .php后缀
    rename "s/$//.txt/" *      # 把所有的文件名都以txt结尾
    rename "s//.txt//" *       # 把所有以.txt结尾的文件名的.txt删掉
    ```
  - Ubuntu命令 rename foo foo0 foo? 这样使用不对，报错：
    ```md
    Bareword "foo" not allowed while "strict subs" in use at (eval 1) line 1.
    ```
    经过Google之后发现有这样的说法：
    ```shell
    On Debian-based distros it takes a perl expression and a list of files. you need to would need to use:
    rename 's/foo/foo0/' foo?
    ```
***

# sort 排序
## 参数
  - **-b** 忽略每行前面开始出的空格字符
  - **-c** 检查文件是否已经按照顺序排序
  - **-d** 排序时，处理英文字母、数字及空格字符外，忽略其他的字符
  - **-f** 排序时，将小写字母视为大写字母
  - **-i** 排序时，除了040至176之间的ASCII字符外，忽略其他的字符
  - **-k** 选择以哪个区间进行排序
  - **-m** 将几个排序好的文件进行合并
  - **-M** 将前面3个字母依照月份的缩写进行排序
  - **-n** 依照数值的大小排序
  - **-h** 按照可读方式的数字排序，如 K / M / G 大小排序
  - **-o<输出文件>** 将排序后的结果存入指定的文件
  - **-r** 以相反的顺序来排序
  - **-t<分隔字符>** 指定排序时所用的栏位分隔字符
  - **-u**  输出行中去除重复行
  - **+<起始栏位>-<结束栏位>** 以指定的栏位来排序，范围由 **起始栏位** 到 **结束栏位的前一栏位**
  - **--debug** 显示用于排序的域
  - 示例
    ```shell
    sort foo
    sort -u foo # 去除重复行
    sort -r foo # 逆序排序
    sort -n foo # 按照文件中的数字排序
    du -hd1 | sort -h # 按照可读方式的数字排序
    ```
## 指定 分隔符 域 的排序
  - sort.txt
    ```shell
    $ cat sort.txt
    AAA:BB:CC
    aaa:30:1.6
    ccc:50:3.3
    ddd:20:4.2
    bbb:10:2.5
    eee:40:5.4
    eee:60:5.1
    ```
  - 将 BB 列按照数字从小到大顺序排列
    ```shell
    $ sort -n -k 2 -t: sort.txt
    ```
  - 将 CC 列数字从大到小顺序排列
    ```shell
    $ sort -nr k 3 -t: sort.txt
    ```
## k 选项的语法格式
  - **-k 选项** 的语法格式 `F[.C][OPTS][,F[.C][OPTS]]`
    ```shell
    FStart.CStart Options,       FEnd.CEnd Options
    -------Start--------,        -------End--------
    FStart.CStart 选项 ,          FEnd.CEnd 选项
    ```
    - 语法格式按照逗号分为两大部分，**Start 部分** 和 **End 部分**
    - 其中的 **Options 部分** 是类似 n 和 r 的选项部分，表示单独指定该域使用的排序方式
    - **Start 部分 FStart.CStart** 其中 **FStart** 表示使用的域，**CStart** 表示在 FStart 域中从第几个字符开始算 **排序首字符**
    - **.CStart** 可以省略，表示从本域的开头部分开始
    - **End 部分 FEnd.CEnd**，表示域结束的字符位置，如果不指定 End，则到整个字符串的结尾
    - **.CEnd** 也可以省略，表示结尾到 **域尾**，即本域的最后一个字符，设定为 **0**，也是表示 **结尾到域尾**
  - **示例**
    ```shell
    # 第一个域的第二个字符 到 本域的最后一个字符 排序
    sort -t ' ' -k 1.2,1 foo

    # 第一个域开头 到 第一个域第二个字符 排序
    sort -t ' ' -k 1,1.2 foo

    # 只对第一个域的第二个字母进行排序，如果相同的按照第三个域进行排序
    $ sort -t ' ' -k 1.2,1.2 -nrk 3,3
    ```
## t 选项分割
  - **-t** 分割字符，默认按照一个或多个空格 / 制表符 分割，但划分出的字符串会包含空格
  - 自定义分隔符时，只能指定一个，如指定空格时，如果字符串包含多个连续空格，将按照单个空格划分成多个空白部分 [ ??? ]
    ```shell
    $ printf "foo  bar\ngoo car\n" | sort -t " " -k2,2 --debug
    sort: using ‘en_US.UTF-8’ sorting rules
    foo  bar
        ^ no match for key
    ________
    goo car
        ___
    _______
    ```
  - `du` 自定义指定按照 `<tab>` 分割
    ```shell
    du -d1 | sort -t $'\t' -k 1,1n
    ```
  - `ls -l` 只按照时间排序
    ```shell
    $ ls -l | grep -i shell
    -rwxr-xr-x 1 leondgarse leondgarse  17660 六月 28 13:24 01-04_Shell_script.md
    -rwxr-xr-x 1 leondgarse leondgarse  51414 八月  3 15:57 01-17_shell流处理命令.md
    ```
    使用默认的分隔符，会使划分出的字符串包含空格
    ```shell
    $ ls -l | grep -i shell | sort -k 8,8.2 -k 8.4,8 --debug
    -rwxr-xr-x 1 leondgarse leondgarse  17660 六月 28 13:24 01-04_Shell_script.md
                                                     __
                                                        ___
    _____________________________________________________________________________
    -rwxr-xr-x 1 leondgarse leondgarse  51460 八月  3 15:58 01-17_shell流处理命令.md
                                                     __
                                                        ___
    ________________________________________________________________________________
    ```
    可以指定 `-b` 参数，去除空格
    ```shell
    $ ls -l | grep -i shell | sort -b -k 8,8.2 -k 8.4,8 --debug
    $ ls -l | grep -i shell | sort -k 8b,8.2bn -k 8.4b,8n --debug
    ```
    使用自定义的分隔符时，可以先用 `sed` 替换
    ```shell
    $ ls -l | grep -i shell | sed 's/ \+/ /g' | sort -t " " -k 8,8.2n -k 8.4,8n --debug
    -rwxr-xr-x 1 leondgarse leondgarse 17660 六月 28 13:24 01-04_Shell_script.md
                                                     __
                                                        __
    ____________________________________________________________________________
    -rwxr-xr-x 1 leondgarse leondgarse 52009 八月 3 16:02 01-17_shell流处理命令.md
                                                    __
                                                       __
    ______________________________________________________________________________

    ```
## ls 的排序功能
  ```shell
  ls -lh | sort -k 5,5 -h # 按照可读方式的数字排序

  ls -lh | sort -k 5,5 -hr  # 等价于 ls -lhS
  ls -lh | sort -k 5,5 -h  # 等价于 ls -lhSr

  ls -lh --time-style=iso | sort -b -k6,6.3nr -k 6.5,6nr -k 7,7.3nr -k 7.5,7nr  # 等价于 ls -lht --time-style=iso
  ls -lh --time-style=iso | sort -b -k6,6.3n -k 6.5,6n -k 7,7.3n -k 7.5,7n  # 等价于 ls -lhtr --time-style=iso
  ```
***

# expect
## 命令
  - 实现自动和交互式任务进行通信
  - **ssh alias**
    ```sh
    alias sshtest="expect -c 'spawn ssh root@192.168.1.1; expect \"assword:\" { send \"root\n\" }; interact'"
    ```
  - expect需要Tcl编程语言的支持，要在系统上运行Expect必须首先安装Tcl
  - 第一行使用 `#!/usr/bin/expect`
  - set, 设置变量
  - set timeout, 设置后面所有的expect命令的等待响应的超时时间, -1参数用来关闭任何超时设置
  - spawn, expect内部命令, 开始一个expect交互进程，只有spawn执行的命令结果才会被expect捕捉到
    ```python
    spawn ssh -l username 192.168.1.1
    ```
  - send, 发送命令,命令字符串“\r”结尾
  - send_tty
  - send_user, 显示提示信息到父进程(一般为用户的shell)的标准输出，类似与echo
  - expect, 等待字符串
    ```python
    expect "]#" { send "touch a\n" }
    expect {
            "A"        { do a }
            "B"        { do b }
            timeout { do timeout }
    }
    ```
    exp_continue, 继续执行下面的匹配
    ```python
    expect {
      "yes/no" { send "yes\r"; exp_continue}
      "Password" {send "$passwd\r"; exp_continue}
    }
    ```
  - interact, 执行完成后保持交互状态，把控制权交给控制台，可以手工操作, 如果没有登录完成后会退出，而不是留在远程终端上
  - $argc 参数个数
  - $argv 参数数组, 从0开始，分别表示第一个,第二个,第三个....参数
    ```python
    set user [lindex $argv 0]
    ```
    $argv0 脚本名字
  - 设置数组型变量
    ```python
    #!/usr/bin/expect -f
    set val_list {"foo1" "foo2" "foo3"}
    if {$argc>0} {
        set val_ind "[lindex $argv 0]"
        set val_val [lindex $val_list $val_ind]
    } else {
        set val_val "foo"
    }

    send_user "val_val = $val_val\n"
    ```
  - 用于 minicom 的示例：
    ```python
    set DEV "/dev/ttyUSB0"
    if {$argc>0} {
      set DEV "/dev/ttyUSB[lindex $argv 0]"
    }

    send_user "DEV = $DEV"
    send_user "\n"

    spawn minicom -D $DEV
    ```
  - expect 用于 smbpasswd 命令时，需要加延时
    ```shell
    expect -c "spawn smbpasswd -a test; set timeout 2; expect \"New SMB password:\" {send \"123456\r\n\"; sleep 1}; expect \"Retype new SMB password:\" {send \"123456\r\n\"; sleep 1}; send_user \"\n\""
    ```
## 参数
  - **-c** 从命令行执行expect脚本
    ```shell
    expect -c 'expect "\n" {send "pressed enter\n"}'
    ```
  - **-r** 表示指定的的字符串是一个正则表达式
    ```shell
    # capturing [xxx, and print xxx.
    expect -c 'expect -re "\\\[(.*)" {puts "$expect_out(1,string)"}'
    ```
    在一个正则表达时中，可以在 `()` 中包含若干个部分并通过 expect_out 数组访问它们
  - **-i** 可以通过来自于标准输入的读命令来交互地执行 expect 脚本
    ```shell
    expect -i arg1 arg2 arg3
    ```
  - **-d** 当执行 expect 脚本的时候，输出调试信息
    ```shell
    expect -d sample.sh
    ```
  - **-D** 启动expect调试器
    ```shell
    expect -c 'set timeout 10' -D 1 -c 'set a 1'
    ```
    - 选项左边的选项会在调试器启动以前被处理
    - 然后，在调试器启动以后，剩下的命令才会被执行
    - 接受一个布尔值的参数，表示提示器必须马上启动，还是只是初始化调试器，以后再使用它
  - **-b** 逐行地执行expect脚本
    ```shell
    expect -b sample.sh
    ```
## Tcl 函数
  - **lindex** Tcl 函数，从列表 / 数组得到一个特定的元素
    ```python
    set user [lindex $argv 0]
    ```
    `[]` 用来实现将函数 lindex 的返回值作为 set 命令的参数
  - **isfile** Tcl 函数, 验证参数指定的文件存在
    ```python
    if {[file isfile $file]!=1}
    ```
  - **lappend** append string to a string array
    ```python
    lappend parametre "[lindex $argv $i]"
    ```
  - **trimright**
    ```python
    set response [string trimright "$raw" " "]
    ```
## if 判断 for 循环
  - if判断
    ```python
    if {[file isfile $file]!=1}

    if { {${ops}} == {telnet} } {
      spawn telnet -E ${ip}
    } elseif { {${ops}} == {ssh} } {
      spawn ssh -l ${usrs} ${ip}
    } else {
      puts {ERROR: Invalid Login Operation: ${ops}}
      exit
    }
    ```
  - for循环
    ```python
    for {set i 0} {$i<$argc} {incr i} { puts "arg$i = [lindex $argv $i]" }

    set parametre {}
    for {set i 0} {$i<$argc} {incr i} {
      lappend parametre "[lindex $argv $i]"
    }

    set i 0
    foreach arg $parametre {
      puts "arg$i = $arg"
      incr i
    }
    ```
## Example expect file(exp_timeout.sh)
  ```python
  #!/usr/bin/expect
  # Prompt function with timeout and default.
  # user input a string within a time limit(argv 2), or output default string(argv 1)

  if {$argc<2} {
    send_user "usage: $argv0 prompt default timeout\n"
    exit
  }

  set prompt [lindex $argv 0]
  set def [lindex $argv 1]
  set response $def
  set tout [lindex $argv 2]

  send_tty "$prompt: "
  set timeout $tout

  expect "\n" {
    set raw $expect_out(buffer)
    # remove final carriage return
    set response [string trimright "$raw" " "]
  }

  if {"$response" == ""} {set response $def}
  send "$response\n"
  ```
  Executing:
  ```shell
  $ ./exp_timeout.sh "who's there: " "memememe!" 1
  $ expect exp_timeout.sh "who's there: " "me!" 5
  $ ANSWER=`expect exp_timeout.sh "who's there: " "memememe!" 6`
  $ echo $ANSWER
  ```
***

# xargs
  - xargs 与 awk
    ```python
    ls -1 | awk '{print "echo "$1; print "touch "$1}' | sh  
    ls -1 | xargs -i sh -c 'echo {}; touch {}'
    ```
  - 删除不同文件夹中的同名文件
    ```shell
    ls foo/ | sed "s/ /\\\ /g" | sed "s/'/\\\'/g" | xargs -I '{}' rm -f bar/{}
    ```
  - 将 home 下所有用户配置到 smb.conf 中相应的路径
    ```shell
    ls /home | grep -v '[+.]' | xargs -I '{}' sed -i '$a\[{}]\npath = /opt/{}\navailable = yes\nbrowseable = yes\npublic = yes\nwritable = yes\n' /etc/samba/smb.conf
    ```
  - 将当前文件夹下所有文件夹压缩到单独的压缩文件
    ```shell
    ls -1 | xargs -I {} zip -r {}.zip {}

    ls -1 *.zip | sed "s/'/\\\'/" | xargs -I "{}" unzip "{}"
    rm *.zip
    ls -1 | sed "s/'/\\\'/" | xargs -I {} sh -c "zip -r '{}'.zip {} rm {} -r"
    ```
  - 将当前文件夹下的所有文件重命名成 `pic_` + `递增数字` + `后缀类型名` 的格式
    ```shell
    # 其中 sed 替换中的 '\' 要用  '\\\\'，否则 sh -c 执行时拿不到
    # awk 中 $NF 表示最后一个域，NR 表示行数
    ls | sed 's/ /\\\\ /; s/(/\\\\(/; s/)/\\\\)/' | awk -F '.' 'BEGIN {IND=0;} {name[IND]=$0; type[IND]=$NF; IND++} END {for (i = 0; i < NR; i++) print name[i] " pic_" i "." type[i]}' | xargs -I {} sh -c 'mv {}'
    ```
  - 将当前文件夹下同名的 `jpg` / `png` 移动到同一个文件夹下
    ```shell
    ls *.jpg *.png | sed 's/.jpg//' | sed 's/.png//' | xargs -i sh -c "mkdir -p '{}' && mv -f '{}.jpg' '{}.png' '{}'"
    ```
  - 移除当前路径下的所有文件夹，文件夹下的文件放到当前路径
    ```shell
    find ./* -type f -exec mv {} ./ \; && find ./* -type d | xargs -i rm -rf {}
    ```
  - 复制当前文件夹的目录结构到另一个文件夹
    ```shell
    find ./ -type d | sed "s/ /\\\ /g" | xargs -I "{}" mkdir ~/foo/{} -p
    ```
  - mp3info 查找音频文件，并删除比特率大于 320 的
    ```shell
    mp3info -x -p "%r#%f\n" *.mp3 | grep 320 | cut -d '#' -f 2- | sed 's/ /\\ /g' | xargs rm {} \;
    ```
  - 更新当前文件夹下所有 `git` 库
    ```sh
    ls -1 | xargs -I {} sh -c "cd {}; echo '>>>> $PWD/{}'; git remote -v; git pull; cd -; echo ''"
    ```
  - 随机选取当前文件夹下的 100 个文件夹中的随机 5 个文件，并输出路径
    ```sh
    ls -1 | sort -R | head -n 100 | xargs -I {} sh -c 'ls {}/*.jpg | sort -R | head -n 5' | xargs -I {} sh -c "echo $PWD/{}" > ~/foo
    ```
  - **md5**
    ```sh
    ls -1 ./*.h5 | xargs -I {} md5sum {}
    ```
  - **code format**
    ```sh
    find ./* -name "*.py" | grep -v __init__ | grep -v setup.py | xargs -I {} black -l 160 {} --diff
    ```
  - **sed replace**
    ```sh
    find ./* -name "*.md" -or -name "*.py" | xargs -I {} sed -i 's/pretraind/pretrained/' {}
    ```
***

# bc 计算
  - **bc** 是一种任意精度的计算语言，提供了一些语法结构，比如条件判断、循环等，支持三角函数 / 指数 / 对数等运算
  - **参数**
    - **-i** 强制交互模式
    - **-l** 使用bc的内置库，bc 里有一些数学库，对三角计算等非常实用
    - **-q** 进入 bc 交互模式时不再输出版本等多余的信息
    - **ibase** / **obase** 用于进制转换，ibase 是输入的进制，obase 是输出的进制，默认是十进制
    - **scale** 小数保留位数，默认保留0位，`-l` 下默认保留 21 位
  - **命令行计算**
    ```sh
    echo "scale=5; 1 / 30" | bc  # .03333
    echo "scale=5; sqrt(15)" | bc  # 3.87298
    echo "scale=5; 9 + 8 - 7 * 6 / 5 ^ 2" | bc # 15.32000

    # 进制转换
    echo "ibase=16; obase=2; ABC" | bc # 101010111100

    # 计算 sin(30°)，其中 4 * arctan(1) == π
    echo "4 * a (1)" | bc -l # 3.14159265358979323844
    echo "scale=5; s(4 * a (1) / 6)" | bc -l  # .49999
    ```
  - **交互式计算**
    ```sh
    bc -l -q

    4 / 3
    1.33333333333333333333

    scale = 5
    4 / 3
    1.33333

    ibase = 2; 1010101
    85

    quit
    ```
***
