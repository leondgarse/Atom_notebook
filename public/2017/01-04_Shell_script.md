# ___2017 - 01 - 04 Shell script___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2017 - 01 - 04 Shell script___](#2017-01-04-shell-script)
  - [目录](#目录)
  - [Shell 基础](#shell-基础)
  - [变量](#变量)
  - [命令行参数](#命令行参数)
  - [计算 expr](#计算-expr)
  - [字符串的长度 index 抽取 替换](#字符串的长度-index-抽取-替换)
  - [if 判断条件语法](#if-判断条件语法)
  - [case 选择语句](#case-选择语句)
  - [for while until 循环](#for-while-until-循环)
  - [function 函数](#function-函数)
  - [getopts 命令行参数处理](#getopts-命令行参数处理)
  - [read 用户输入](#read-用户输入)
  	- [示例](#示例)
  	- [expr 计算浮点数除法](#expr-计算浮点数除法)
  	- [shell 下载迅雷链接](#shell-下载迅雷链接)

  <!-- /TOC -->
***

# Shell 基础
  - shell 中的注释会注视调用“#”字符以后的所有内容,直到这一行结束
    ```shell
    #!/bin/sh
    #!/usr/bin/env bash
    ```
  - 当前使用的bash：
    ```shell
    $ echo $$ # 30044
    $ ps | grep $$  # 30044 pts/0    00:00:00 bash

    $ echo $SHELL
    $ which bash
    ```
  - $? - The exit status of the last command executed.
  - $$ - The process ID of the current shell. For shell scripts, this is the process ID under which they are executing.
  - $! - The process number of the last background command.
  - exit(1)表示发生错误后退出程序， exit(0)表示正常退出
***

# 变量
  - 双引号，避免shell对字符串做拆分处理，可引用除了字符 $、反引号（\`）、\ 外的任意字符或字符串
    ```shell
    echo *        # 输出当前路径下所有文件名
    echo "*"        # 输出*

    echo `ls -l`        # 空格/换行符等被解释为字符串拆分标志，输出时作为拆分后的字符串数组输出
    echo "`ls -l`"      # 保留输出时的所有格式信息
    ```
  - 单引号与双引号类似，不同的是shell会忽略任何引用值，即屏蔽单引号内的特殊字符的原本含义。
    ```shell
    CAL="cal"
    echo $CAL        # 输出 cal
    echo '$CAL'        # 输出$CAL
    ```
  - 反引号（`）用于设置系统命令输出到变量，shell认为反引号中的内容是一个系统命令
    ```shell
    echo '$CAL'        # 输出cal命令的输出，但会在同一行中输出
    echo "`$CAL`"        # 可以输出换行符
    ```
  - $ shell 脚本中的一个保留字，表示变量的开始，初始化变量时等号两边不能有空格
    ```shell
    $string="hello world"
    ```
  - unset 命令清除一个变量的值
    ```shell
    unset $string
    ```
  - {} 使用大括号引用变量，用于跟后面字符区分
    ```shell
    echo "${var}foo"
    ```
  - "" 引号里面的变量输出时会保留所有空格
    ```shell
    greeting='Hello    world!'
    echo $greeting" now with spaces: $greeting"
    # >output: Hello world! now with spaces: Hello    world!
    ```
  - \`\`或$() \`在~下面，将变量值替换为命令输出内容
    ```shell
    FILELIST=`ls`
    echo "$FILELIST"
    # >outout: prog.sh

    FileWithTimeStamp=/tmp/my-dir/file_$(/bin/date +%Y-%m-%d).txt
    echo "$FileWithTimeStamp"
    # >outout: /tmp/my-dir/file_2017-01-04.txt
    ```
  - alias中每次使用命令更新为不同值
    ```shell
    alias UPDATEWORKSPACE='export __WORKSPACE="`hg root 2>/dev/null`"'
    ```
    将每次执行""中的命令，更新为不同的值
    ```shell
    alias UPDATEWORKSPACE=“export __WORKSPACE=`hg root 2>/dev/null`”
    ```
    只会在创建alias时执行一次，命令执行的结果作为固定的字符串
  - () 创建字符数组
    ```shell
    my_array=(apple banana "Fruit Basket" orange)
    new_array[2]=apricot

    echo ${my_array[3]}        # 需要使用{} --> orange
    ```
  - ($(EXP)) 初始化为命令输出的数组
    ```shell
    FILELIST=($(ls))
    echo "${FILELIST[3]}"        # 输出第3个元素值
    echo "${FILELIST}"        # 输出第0个元素值
    ```
  - ${VAR[@]} ${VAR[\*]} 输出全部元素，将数组值转换为一个字符串
    ```shell
    echo "${FILELIST[@]}"
    ```
  - ${#VAR[@]} ${#VAR[\*]} 元素数目
    ```shell
    echo ${#my_array[@]}        # 输出元素数目 --> 4

    # adding another array element
    my_array[4]="carrot"        # value assignment without a $ and curly brackets
    echo ${#my_array[@]}        # --> 5
    echo ${my_array[${#my_array[@]}-1]}        # --> carrot
    ```
  - declare -i var  # 使用 declare 命令,明确指定变量 b 是一个整型变量
  - ${var:-defaultvalue}  # 引用该变量时，设置变量默认值
***

# 命令行参数
  - **$0** 脚本名称
  - **$n** 传递给脚本或函数的第 n 个参数
  - **$#** 传递给脚本 / 函数的变量数目
  - **$@ / $*** 传递给脚本 / 函数的所有参数
  - **shift** 所有的位置参数都向左移动一个位置，但$0不变
  - **$10** 不能获取第十个参数，获取第十个参数需要 ${10}，当 n>=10 时，需要使用 ${n} 来获取参数
  - 在使用双引号 "" 时，**$@** / **$*** 行为不同：
    ```shell
    #!/bin/bash
    function func {
      echo "--> \$*"
      for ARG in "$*"; do
        echo $ARG
      done

      echo "--> \$@"
      for ARG in "$@"; do
        echo $ARG
      done
    }
    func We are argument

    # output:
    --> $*
    We are argument
    --> $@
    We
    are
    argument
    ```
***

# 计算 expr
  - **expr** 用于对一个表达式求值，只适用于整数运算，乘法符号必须被转义
    ```sh
    ''' 数学运算 + / - / * / / / % '''
    expr 3 \* 2 # 6

    ''' 关系运算 | / & '''
    expr 3 \| 2 # 3
    expr 0 \| 2 # 2

    ''' 逻辑运算 < / <= / > / >= / = / !='''
    expr 0 \< 2 # 1
    expr 0 != 2 # 1
    expr 0 == 2 # 0
    ```
  - **`$((expression))` / `$[expression]`** 双括号 / 方括号代替 expr 命令计算表达式的值
    - 使用 `$(( ))` / `$[ ]` 效果类似
    - `*` 乘法符号等不需要转义
    - `|` / `&` / `^` 表示二进制运算
    - `||` / `&&` 表示逻辑运算
    - `==` 用于判断相等
    ```py
    echo $(( 2 * 5 )) # 10

    echo $(( 2 | 5 )) # 7
    echo $(( 2 & 5 )) # 0
    echo $(( 2 ^ 5 )) # 7

    echo $(( 2 || 0 ))  # 1
    echo $(( 2 && 0 ))  # 0

    echo $(( 2 == 3 ))  # 0
    ```
  - **用于字符串**
    ```sh
    ''' length 计算字符串长度'''
    expr length 'foo'
    # 3

    ''' substr 抽取子字符串 '''
    expr substr 'foogoo' 4 6
    # goo

    ''' match 从第一个字符匹配正则表达式，返回匹配到的长度 '''
    expr match 'fooogoo' '.*g'
    # 5
    expr match 'fooogoo' '.*g*'
    # 7
    expr match 'fooogoo' 'goo'
    # 0

    ''' index 第一个匹配到任意一个字符的位置 '''
    expr index 'foogoo' 'o'
    # 2
    expr index 'foogoo' 'og'
    # 2
    ```
  - **计算时间间隔**
    ```shell
    DA=`date +%s`
    echo $DA  # 1510222879
    DB=`date +%s`
    echo $DB  # 1510224086

    TM=$(expr \( $DB - $DA \) / 60)
    echo $TM # 20
    TS=$(expr \( $DB - $DA \) % 60)
    echo $TS # 7

    echo $TM"m"$TS"s"  # 20m7s
    ```
***

# 字符串的长度 index 抽取 替换
  - $(#) 字符串长度
    ```shell
    STRING="this is a string"
    echo ${#STRING}
    # >output: 16
    ```
  - expr index 字符位置
    ```shell
    STRING="this is a string"
    SUBSTRING="hat"
    expr index "$STRING" "$SUBSTRING"   # 1 is the position of the first 't' in $STRING
    ```
  - :: 抽取字符串
    ```shell
    STRING="this is a string"
    POS=1
    LEN=3

    # 从 $STRING 中 $POS 处开始的 $LEN 个字符
    echo ${STRING:$POS:$LEN} # > his

    # 长度可以是负值，表示倒数第几个字符
    echo ${STRING:0:-1} # > this is a strin

    # 省略$LEN，抽取到结尾
    echo ${STRING:12} # > ring

    # 从结尾抽取，不能直接使用 ${STRING:-3}
    echo ${STRING:-3} # 输出的是整个字符串 this is a string
    echo ${STRING:0 - 3}  # > ing
    echo ${STRING:3 * -1} # > ing
    echo ${STRING:${#STRING} - 3} # > ing

    # Code to extract the First name from the data record
    DATARECORD="last=Clifford,first=Johnny Boy,state=CA"
    COMMA1=`expr index "$DATARECORD" ','` # 14 position of first comma
    CHOP1FIELD=${DATARECORD:0:$COMMA1}    # last=Clifford,
    CHOP2FIELD=${DATARECORD:$COMMA1}    # first=Johnny Boy,state=CA

    # index contains the searching char
    FOO='aa bb cc'
    echo ${FOO:0:$(expr index "$FOO" 'b')}  # aa b
    echo ${FOO:0:$(expr index "$FOO" ' ')}  # aa with a space
    echo ${FOO:0:$(expr $(expr index "$FOO" ' ') - 1)}  # aa
    ```
  - [@]// 字符串替换
    ```shell
    # 替换第一个字符串
    STRING="to be or not to be"
    echo ${STRING[@]/be/eat}    # to eat or not to be

    # 替换所有字符串
    STRING="to be or not to be"
    echo ${STRING[@]//be/eat}    # to eat or not to eat

    # 删除，全部替换为空
    STRING="to be or not to be"
    echo ${STRING[@]// not/}    # to be or to be

    # 替换开始位置的字符串
    STRING="to be or not to be"
    echo ${STRING[@]/#to be/eat now}  # eat now or not to be

    # 替换结尾位置的字符串
    STRING="to be or not to be"
    echo ${STRING[@]/%be/eat}    # to be or not to eat

    # 替换为shell命令输出
    STRING="to be or not to be"
    STRING2="${STRING[@]/%be/be on $(date +%Y-%m-%d)}"  # to be or not to be on 2017-01-05
    echo $STRING2
    ```
***

# if 判断条件语法
  - **格式**
    ```shell
    if [ expression ]; then
        # expression
    elif [ expression ]; then
        # expression
    else
        # expression
    fi
    ```
  - **expression 条件语法**
    - **[ ]** 相当于test命令别名，条件语句中不能有 || && 连接多个表达式
    - **[[ ]]** 是 [ ] 的扩展形式，支持更复杂的表达式
    - **test** 使用 -a 表示且， -o 表示或，! 表示非
    - **" "** 可以加也可以不加
    ```shell
    if [ "1" -ne "2" ]; then echo 6; fi
    if [ ! "1" -eq "2" ]; then echo 6; fi
    if [[ "1" -eq 1 ]]; then echo 6; fi
    if test "1" -ne "2"; then echo 6; fi
    if ! test "1" -eq "2"; then echo 6; fi

    if [ "1" -ne "2" ] && [ 1 -eq "1" ]; then echo 6; fi
    if [ "1" -ne "2" -a 1 -eq 1 ]; then echo 6; fi
    if [[ "1" -eq "2" || "1" -eq "1" ]]; then echo 6; fi
    if test "1" -eq "2" -o "1" -eq "1"; then echo 6; fi
    if test "1" -eq "2" || test "1" -eq "1"; then echo 6; fi

    if !grep "" ... ; then ... ; fi
    ```
  - **数字比较**
    - **-eq** 两个数值相等
    - **-ne** 两个数值不相等
    - **-gt** 第一个数大于第二个数
    - **-lt** 第一个数小于第二个数
    - **-le** 第一个数小于等于第二个数
    - **-ge** 第一个数大于等于第二个数
    ```shell
    test "1" -eq "2"  # 测试两个数据相等
    ```
  - **字符串比较** shell 脚本中的字符串存储的是字符串的内容，因此比较的是两个字符串的内容
    - 等号两侧需要有空格
    - 字符串两侧使用 "" 避免 shell 重新解释字符串内容
    ```shell
    test $str1 = $str2   # 测试字符串 str1 和 str2 相等，也可以使用==
    test -z $str3    # 测试字符串 str3 是空串
    test -n $str2    # 测试字符串 str2 是非空串
    test $str1 != $str2  # 测试字符串 str1 和 str2 不相等
    ```
  - **文件测试**
    - **-d** 文件是目录
    - **-e** 文件存在
    - **-f** 文件是普通文件
    - **-h / -L** 文件是符号链接
    - **-r** 文件可读
    - **-s** 文件长度大于 0
    - **-u** 文件设置了 suid 位
    - **-w** 文件可写
    - **-x** 文件是否可执行
    ```shell
    test -r test.txt  # 测试文件状态
    [ -w $1 -a $0 ]  #检查两个文件是否同时可写
    ```
  - **正则表达式比较 =~**
    ```shell
    # 判断字符串是数字
    FF=abc123
    if [[ ! $FF =~ ^[0-9] ]]; then echo "string"; else echo "number"; fi
    # string

    FF=123
    if [[ ! $FF =~ ^[0-9] ]]; then echo "string"; else echo "number"; fi
    # number

    FF=1ba23
    if [[ ! $FF =~ ^[0-9] ]]; then echo "string"; else echo "number"; fi
    # number
    ```
  - **`if [ -n $STR ]`** 在 `$STR` 为空时会返回 `True`
    ```sh
    aa=""
    if [ -n $aa ]; then echo 1; else echo 0; fi
    # 1
    test -n $aa; echo $?
    # 0
    if [ -z $aa ]; then echo 1; else echo 0; fi
    # 1
    ```
    原因在于当判断的字符串为空时，实际的表达式变为 `if [ -n ]`，此时 `-n` 将作为判断的字符串，判断结果为 `True`
    ```sh
    if [ -n ]; then echo 1; else echo 0; fi
    # 1

    if [ -asdfae ]; then echo 1; else echo 0; fi
    # 1
    ```
    解决方法可以在字符串两边加上引号，或使用 `[[  ]]`
    ```sh
    if [ -n "$aa" ]; then echo 1; else echo 0; fi
    # 0
    test -n "$aa"; echo $?
    # 1

    if [[ -n $aa ]]; then echo 1; else echo 0; fi
    # 0
    ```
***

# case 选择语句
  - 语法：
    ```shell
    case "$variable" in
      "$condition1" )
        command...
      ;;
      "$condition2" )
        command...
      ;;
    esac
    ```
  - 示例：
    ```shell
    mycase=1
    case $mycase in
      1) echo "You selected bash";;
      2) echo "You selected perl";;
      3) echo "You selected phyton";;
      4) echo "You selected c++";;
      5) exit;;
    esac
    ```
    ```shell
    # 判断字符串包含子串
    while read line; do
        case $line in
            *root*) echo root;;
            *leondgarse*) echo leondgarse;;
        esac;
    done < /etc/group;
    ```
***

# for while until 循环
  - for 语法：
    ```shell
    # basic construct
    for arg in [list]
    do
      command(s)...
    done
    ```
  - for 示例：
    ```shell
    for FILE in $(ls); do
      echo "File is : $FILE"
    done
    ```
    [list]可以是一个空格分割的字符串，不能是数组
    ```shell
    NUMBERS=(951 402 ...)
    # 此时$NUMBER代表的值是951，即只有第一个元素
    for num in $NUMBERS; do echo $num; done # --> 951

    # 遍历数组使用
    for num in ${NUMBERS[@]}; do echo $num; done
    # 或使用
    for ((i=0; $i<${#NUMBERS[*]}; i=$i+1)); do echo ${NUMBERS[i]}; done
    ```
  - 字符串数组与 for 循环
    ```shell
    FOO=(`ls`)  # 字符串数组
    echo $FOO   # 输出第一个元素
    echo ${FOO[@]}  # 输出全部元素，空格分隔
    for FILE in ${FOO[@]}; do echo $FILE; done  # 遍历输出全部元素
    for ((i=0; $i<${#FOO[@]}; i=$i+1)); do echo ${FOO[i]}; done # 遍历输出全部元素
    ```
    ```shell
    FOO=`ls`  # 空格分隔的字符串
    echo $FOO # 输出全部元素，空格分隔
    echo ${#FOO}  # --> 775，字符串中的字符数量 [ ? ]
    echo ${#FOO[@]} # --> 92，字符串的单词数量
    for FILE in $FOO; do echo $FILE; done  # 遍历输出全部元素，第一个元素是全部字符
    for ((i=0; $i<${#FOO[@]}; i=$i+1)); do echo ${FOO[i]}; done # [ ? ]
    ```
  - while 语法：
    ```shell
    # basic construct
    while [ condition ]
    do
      command(s)...
    done
    ```
  - while 示例：
    ```shell
    COUNT=4
    while [ $COUNT -gt 0 ]; do
      echo "Value of count is: $COUNT"
      COUNT=$(($COUNT - 1))
    done
    ```
  - until 语法：
    ```shell
    # basic construct
    until [ condition ]
    do
      command(s)...
    done
    ```
  - until 示例：
    ```shell
    COUNT=1
    until [ $COUNT -gt 5 ]; do
      echo "Value of count is: $COUNT"
      COUNT=$(($COUNT + 1))
    done
    ```
  - break / continue：
    ```shell
    # Prints out only odd numbers - 1,3,5,7,9
    COUNT=0
    while [ $COUNT -lt 10 ]; do
      COUNT=$((COUNT+1))
      # Check if COUNT is even
      if [ $(($COUNT % 2)) = 0 ] ; then
       continue
      fi
      echo $COUNT
    done
    ```
  - for / while 读取文件
    ```shell
    for ee in $(cat /etc/passwd); do echo $ee; done # 按照 换行符 / 空格划分 [ ??? ]
    while read line; do echo $line; done < /etc/passwd  # 按照换行符划分
    ```
***

# function 函数
  - 语法
    ```shell
    # basic construct
    [function] function_name [()] {
        action;
        [return int;]
    }
    ```
  - 调用直接使用函数名，可以传递参数
    ```bash
    function adder {
     echo "$(($1 + $2))"
    }

    # Pass two parameters to function adder
    adder 12 56         # 68
    ```
  - 返回值 return / echo
    - 函数中使用 **echo** 可以将值返回给变量 **r=$(func)**
    - 函数中使用 **return** 返回的值，只能通过 **echo $?** 得到
    ```shell
    function fAdd() {
        echo $1, $2
        return $(($1+$2))
    }

    fAdd 2 3  # 2, 3
    echo $?   # 5
    r=$(fAdd 2 3)
    echo $r   # 2, 3
    echo $?   # 5   
    ```    
  - 示例1，计算：
    ```shell
    function ENGLISH_CALC {
      case "$2" in
        "plus") echo "$1 + $3 = $(($1+$3))" ;;
        "minus") echo "$1 - $3 = $(($1-$3))" ;;
        "times") echo "$1 * $3 = $(($1*$3))" ;;
      esac
    }

    # testing code
    ENGLISH_CALC 3 plus 5
    ENGLISH_CALC 5 minus 1
    ENGLISH_CALC 4 times 6
    ```
  - 示例2，变量：
    ```shell
    #!/bin/bash
    echo "Script Name: $0"
    function func {
      for var in $*
      do
        let i=i+1
        echo "The \$${i} argument is: ${var}"
      done
      echo "Total count of arguments: $#"
    }
    ```
***

# getopts 命令行参数处理
  - **getopts** 和 **getopt** 功能相似但又不完全相同
    - **getopt** 是独立的可执行文件
    - **getopts** 是由 Bash 内置的，使用语法简单，不支持类似 `--help` 的长选项，不会重排所有参数的顺序
  - **getopts 格式**
    ```sh
    getopts option_string variable
    ```
    - **option_string 选项列表** 每个字母代表一个选项，`:` 表示该选项后面需要参数值
    - **variable 参数列表** 实际获取到的参数
    - 允许将不带参数值的选项堆叠到一起，如 `-ms`
  - **getopts 使用** 一般是在循环中使用
    - 循环开始时，getopts 检查一个命令行参数，判断是否以 `-` 开头的 **options** 中的字母
    - 配合使用 `case` 处理匹配到的选项，其中 **OPTARG** 保存当前选项的参数值，**OPTIND** 记录当前参数在参数列表中的位置，初始值是 1
    - 对于 **没有匹配到的选项**，会有警告信息，可以在 `option_string` 前面加上 `:` 不打印，`case` 中可以使用 `?` 进行相应处理
    - 当参数列表全部解析完，或者解析到不以 `-` 开头的参数，则退出循环
    ```sh
    #!/bin/bash
    while getopts h:ms option
    do
      case "$option" in
        h)
          echo "option:h, value $OPTARG"
          echo "next arg index:$OPTIND";;
        m)
          echo "option:m"
          echo "next arg index:$OPTIND";;
        s)
          echo "option:s"
          echo "next arg index:$OPTIND";;
        \?)
          echo "Usage: args [-h n] [-m] [-s]"
          echo "-h means hours"
          echo "-m means minutes"
          echo "-s means seconds"
          exit 1;;
      esac
    done

    echo "*** do something now ***"
    ```
    **运行结果**
    ```sh
    $ ./getopt.sh -h 100 -msd
    option:h, value 100
    next arg index:3
    option:m
    next arg index:3
    option:s
    next arg index:3
    ./getopt.sh: illegal option -- d
    Usage: args [-h n] [-m] [-s]
    -h means hours
    -m means minutes
    -s means seconds
    ```
  - **位置参数** getopts 解析完参数后，如果还有参数，可以使用 `shift` 跳过已经解析的参数，作为位置参数使用
    ```sh
    shift $((OPTIND - 1))
    echo $@
    ```
***

# read 用户输入
  - **将用户输入读入变量**
    ```shell
    # echo -n 指定不换行
    echo -n "Input a string: " && read USER_STRING
    echo $USER_STRING
    ```
  - **-p** 输入提示
    ```shell
    read -p "Input a string: " USER_STRING
    ```
    ```shell
    TARGET='./foo'
    read -p "Target file $TARGET exist, overrite it? (Y/n):" RESULT
    if [[ ! $RESULT =~ ^[yY] ]]; then
        echo 'Please give another name then'
    else
        rm -f $TARGET
    fi
    ```
  - **REPLY** 如果不指定变量名，read 的结果保存在 `$REPLY` 中，不是 `REPLAY`
    ```shell
    read
    echo $REPLY
    ```
  - **-t** 设置超时时间，超时返回非零
    ```shell
    read -t 5 -p "Input: " USER_STRING
    echo $? # 142
    ```
  - **-s** 隐藏用户输入
    ```shell
    read -s -p "Input: " USER_STRING
    ```
  - **time** 结合 `read` 计时
    ```shell
    time read # real	0m4.776s
    ```
  - **读取文件**
    ```shell
    cat /etc/passwd | while read line; do echo $line; done
    while read line; do echo $line; done < /etc/passwd
    ```
***

## 示例
## expr 计算浮点数除法
  - expr 只可以计算整数值运算，浮点数运算可以使用 `awk` / `bc` / `dc`
  ```sh
  function expr_dev_float() {
      aa=$1
      bb=$2
      if [ $# -ge 3 ]; then precision=$3; else precision=2; fi

      if [ $(expr index $aa '.') -eq 0 ]; then aa_float_len=0; else aa_float_len=$(expr $(expr length $aa) - $(expr index $aa '.')); fi
      if [ $(expr index $bb '.') -eq 0 ]; then bb_float_len=0; else bb_float_len=$(expr $(expr length $bb) - $(expr index $bb '.')); fi
      if [[ $aa_float_len -ne 0 || $bb_float_len -ne 0 ]]; then
          if [ $aa_float_len -gt $bb_float_len ]; then magnify=$aa_float_len; else magnify=$bb_float_len; fi
          aa=${aa[@]/'.'/}
          for ((i=$aa_float_len; i<$magnify; i=$i+1)); do aa=$aa'0'; done
          bb=${bb[@]/'.'/}
          for ((i=$bb_float_len; i<$magnify; i=$i+1)); do bb=$bb'0'; done
      fi

      pp=1
      fraction_zero=''
      for ((i=0; $i<$precision; i=$i+1)); do pp=$(($pp * 10)); fraction_zero=$fraction_zero'0'; done

      integer_p=$(expr $aa / $bb)
      if [ $integer_p -eq 0 ]; then integer_p_len=0; else integer_p_len=${#integer_p}; fi

      integer_fraction_p=$(expr $aa \* $pp / $bb)
      fraction_p=${integer_fraction_p:$integer_p_len}
      fraction_zero_len=$(($precision - ${#fraction_p}))

      echo $integer_p'.'${fraction_zero:0:$fraction_zero_len}$fraction_p
  }

  expr_dev_float 4 3 3 # 1.333
  expr_dev_float 1 30 3 # 0.033
  expr_dev_float 1.5 `expr_float 1 3 4` 2 # 4.50
  ```
  ```sh
  awk -v aa=1 -v bb=30 'BEGIN {printf "%.3f\n", aa / bb}' # 0.033
  echo 'scale=3; 1 / 30' | bc # .033
  python -c 'print("%.2f" % (1 / 3))' # 0.33

  expr_dev_float 1.5 `awk -v aa=1 -v bb=30 'BEGIN {printf "%.6f\n", aa / bb}'` 3  # 45.000
  expr_dev_float 1.5 `echo 'scale=6; 1 / 30' | bc` 3  # 45.000

  calc() { awk "BEGIN {print $*}"; }
  calc '1 + 2 * 3 / 4'  # 2.5
  ```
## shell 下载迅雷链接
  - [thunder_download.sh](./thunder_download.sh)
***
