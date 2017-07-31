# ___2017 - 01 - 04 Shell script___
- \#!/bin/sh
- \# shell 中的注释会注视调用“#”字符以后的所有内容,直到这一行结束
- 当前使用的bash：
  ```c
  $ ps | grep $$
  $ echo $SHELL
  $ which bash
  ```
- $? - The exit status of the last command executed.
- $$ - The process ID of the current shell. For shell scripts, this is the process ID under which they are executing.
- $! - The process number of the last background command.
- read 用于从用户输入
***

# <span style="color:#ff0000;">变量
  - 双引号，避免shell对字符串做拆分处理，可引用除了字符 $、反引号（\`）、\ 外的任意字符或字符串
    ```c
    echo *        # 输出当前路径下所有文件名
    echo "*"        # 输出*

    echo `ls -l`        # 空格/换行符等被解释为字符串拆分标志，输出时作为拆分后的字符串数组输出
    echo "`ls -l`"      # 保留输出时的所有格式信息
    ```
  - 单引号与双引号类似，不同的是shell会忽略任何引用值，即屏蔽单引号内的特殊字符的原本含义。
    ```c
    CAL="cal"
    echo $CAL        # 输出 cal
    echo '$CAL'        # 输出$CAL
    ```
  - 反引号（`）用于设置系统命令输出到变量，shell认为反引号中的内容是一个系统命令
    ```c
    echo '$CAL'        # 输出cal命令的输出，但会在同一行中输出
    echo "`$CAL`"        # 可以输出换行符
    ```
  - $ shell 脚本中的一个保留字，表示变量的开始，初始化变量时等号两边不能有空格
    ```c
    $string="hello world"
    ```
  - unset 命令清除一个变量的值
    ```c
    unset $string
    ```
  - {} 使用大括号引用变量，用于跟后面字符区分
    ```c
    echo "${var}foo"
    ```
  - "" 引号里面的变量输出时会保留所有空格
    ```c
    greeting='Hello    world!'
    echo $greeting" now with spaces: $greeting"
    # >output: Hello world! now with spaces: Hello    world!
    ```
  - \`\`或$() \`在~下面，将变量值替换为命令输出内容
    ```
    FILELIST=`ls`
    echo "$FILELIST"
    # >outout: prog.sh

    FileWithTimeStamp=/tmp/my-dir/file_$(/bin/date +%Y-%m-%d).txt
    echo "$FileWithTimeStamp"
    # >outout: /tmp/my-dir/file_2017-01-04.txt
    ```
  - () 创建字符数组
    ```
    my_array=(apple banana "Fruit Basket" orange)
    new_array[2]=apricot

    echo ${my_array[3]}        # 需要使用{} --> orange
    ```
  - ($(EXP)) 初始化为命令输出的数组
    ```
    FILELIST=($(ls))
    echo "${FILELIST[3]}"        # 输出第3个元素值
    echo "${FILELIST}"        # 输出第0个元素值
    ```
  - ${VAR[@]} ${VAR[\*]} 输出全部元素，将数组值转换为一个字符串
    ```
    echo "${FILELIST[@]}"
    ```
  - ${#VAR[@]} ${#VAR[\*]} 元素数目
    ```
    echo ${#my_array[@]}        # 输出元素数目 --> 4

    # adding another array element
    my_array[4]="carrot"        # value assignment without a $ and curly brackets
    echo ${#my_array[@]}        # --> 5
    echo ${my_array[${#my_array[@]}-1]}        # --> carrot
    ```
  - declare -i var  # 使用 declare 命令,明确指定变量 b 是一个整型变量
  - ${var:-defaultvalue}  # 引用该变量时，设置变量默认值
***

# <span style="color:#ff0000;">命令行参数
  - $0 脚本名称
  - $n 传递给脚本或函数的第 n 个参数
  - $# 传递给脚本 / 函数的变量数目
  - $@ / $* 传递给脚本 / 函数的所有参数
  - shift      # 所有的位置参数都向左移动一个位置，但$0不变
  - 在使用双引号""时，$@ / $* 行为不同：
    ```
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

# <span style="color:#ff0000;">计算
  - expr 3 \ * 2 expr 用于对一个表达式求值，乘法符号必须被转义
  - echo $? 显示结果
  - <a name="inner-text"></a>$((expression)) 双括号代替 expr 命令计算表达式的值
***

# <span style="color:#ff0000;">字符串 长度 / index / 抽取 / 替换
  - $(#) 字符串长度
    ```
    STRING="this is a string"
    echo ${#STRING}
    # >output: 16
    ```
  - expr index 字符位置
    ```
    STRING="this is a string"
    SUBSTRING="hat"
    expr index "$STRING" "$SUBSTRING"   # 1 is the position of the first 't' in $STRING
    ```
  - :: 抽取字符串
    ```
    STRING="this is a string"
    POS=1
    LEN=3
    echo ${STRING:$POS:$LEN}  #从$STRING中$POS处开始的$LEN个字符
    # >output: his

    echo ${STRING:12}     # 省略$LEN，抽取到结尾
    # >output: ring

    # Code to extract the First name from the data record
    DATARECORD="last=Clifford,first=Johnny Boy,state=CA"
    COMMA1=`expr index "$DATARECORD" ','` # 14 position of first comma
    CHOP1FIELD=${DATARECORD:0:$COMMA1}    # last=Clifford,
    CHOP2FIELD=${DATARECORD:$COMMA1}    # first=Johnny Boy,state=CA
    ```
  - [@]// 字符串替换
    ```
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

# <span style="color:#ff0000;">if 判断条件语法
  - 格式：
    ```
    if [ expression ]; then
            # expression
    else [ expression ]; then
            # expression
    elif
            # expression
    fi
    ```
  - expression 条件语法：
    ```
    [] 相当于test命令别名，条件语句中不能有 || &amp;&amp; 连接多个表达式
    [[]] 是[]的扩展形式，支持更复杂的表达式
    test 使用 -a 表示且， -o 表示或，! 表示非
    "" 可以加也可以不加

    if [ "1" -ne "2" ]; then echo 6; fi
    if [ ! "1" -eq "2" ]; then echo 6; fi
    if [[ "1" -eq 1 ]]; then echo 6; fi
    if test "1" -ne "2"; then echo 6; fi
    if ! test "1" -eq "2"; then echo 6; fi

    if [ "1" -ne "2" ] &amp;&amp; [ 1 -eq "1" ]; then echo 6; fi
    if [ "1" -ne "2" -a 1 -eq 1 ]; then echo 6; fi
    if [[ "1" -eq "2" || "1" -eq "1" ]]; then echo 6; fi
    if test "1" -eq "2" -o "1" -eq "1"; then echo 6; fi
    if test "1" -eq "2" || test "1" -eq "1"; then echo 6; fi

    if !grep "" ... ; then ... ; fi
    ```
  - 数字比较：
    ```
    test "1" -eq "2"  # 测试两个数据相等
    # -eq 两个数值相等
    # -ne 两个数值不相等
    # -gt 第一个数大于第二个数
    # -lt 第一个数小于第二个数
    # -le 第一个数小于等于第二个数
    # -ge 第一个数大于等于第二个数        
    ```
  - 字符串比较：
    ```
    # shell 脚本中的字符串存储的是字符串的内容,因此,比较的是两个字符串的内容
    test $str1 = $str2   # 测试字符串 str1 和 str2 相等，也可以使用==
    test -z $str3    # 测试字符串 str3 是空串
    test -n $str2    # 测试字符串 str2 是非空串
    test $str1 != $str2  # 测试字符串 str1 和 str2 不相等

    等号两侧需要有空格
    字符串两侧使用""避免shell是重新解释字符串内容
    ```
  - 文件测试：
    ```
    test -r test.txt  # 测试文件状态
    # -d 文件是目录
    # -e 文件存在
    # -f 文件是普通文件
    # -h / -L 文件是符号链接
    # -r 文件可读
    # -s 文件长度大于 0
    # -u 文件设置了 suid 位
    # -w 文件可写
    # -x 文件是否可执行

    [ -w $1 -a $0 ]  #检查两个文件是否同时可写
    ```
***

# <span style="color:#ff0000;">case 选择语句
  - 语法：
    ```
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
    ```
    mycase=1
    case $mycase in
      1) echo "You selected bash";;
      2) echo "You selected perl";;
      3) echo "You selected phyton";;
      4) echo "You selected c++";;
      5) exit
    esac
    ```
***

# <span style="color:#ff0000;">for / while / until 循环
  - for 语法：
    ```
    # basic construct
    for arg in [list]
    do
      command(s)...
    done
    ```
  - for 示例：
    ```
    for FILE in $(ls); do
      echo "File is : $FILE"
    done

    其中[list]可以是一个空格分割的字符串，不能是数组
            NUMBERS=(951 402 ...)
            for num in $NUMBERS; do ...         # 此时$NUMBER代表的值是951，即只有第一个元素

    遍历数组使用：
            for num in ${NUMBERS[@]}; do echo $num; done
            或：
            for ((i=0; $i<${#NUMBERS[*]}; i=$i+1)); do echo ${NUMBERS[i]}; done
    ```
  - while 语法：
    ```
    # basic construct
    while [ condition ]
    do
      command(s)...
    done
    ```
  - while 示例：
    ```
    COUNT=4
    while [ $COUNT -gt 0 ]; do
      echo "Value of count is: $COUNT"
      COUNT=$(($COUNT - 1))
    done
    ```
  - until 语法：
    ```
    # basic construct
    until [ condition ]
    do
      command(s)...
    done
    ```
  - until 示例：
    ```
    COUNT=1
    until [ $COUNT -gt 5 ]; do
      echo "Value of count is: $COUNT"
      COUNT=$(($COUNT + 1))
    done
    ```
  - break / continue：
    ```
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
***

# <span style="color:#ff0000;">function 函数
  - 语法：
    ```
    # basic construct
    function_name {
     command...
    }

    调用直接使用函数名，可以传递参数
    function adder {
     echo "$(($1 + $2))"
    }

    # Pass two parameters to function adder
    adder 12 56         # 68
    ```
  - 示例1，计算：
    ```
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
    ```
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

# <span style="color:#ff0000;">getopts 命令行参数处理
  - getopts,它不支持长选项 (--help)
  - getopts和getopt功能相似但又不完全相同，其中getopt是独立的可执行文件，而getopts是由Bash内置的,支持长选项以及可选参数
  - getopts options variable
    ```
    getopts的设计目标是在循环中运行，
    每次执行循环，getopts就检查下一个命令行参数，并判断它是否合法,即检查参数是否以-开头，后面跟一个包含在 options 中的字母
    如果是，就把匹配的选项字母存在指定的变量variable中，并返回退出状态0
    如果-后面的字母没有包含在options中，就在variable中存入一个 ？，并返回退出状态0
    如果命令行中已经没有参数，或者下一个参数不以-开头，就返回不为0的退出状态
    ```
  - getopts 允许把选项堆叠在一起（如 -ms）
  - 如要带参数，须在对应选项后加 :（如h后需加参数 h:ms）。此时选项和参数之间至少有一个空白字符分隔，这样的选项不能堆叠。
  - 如果在需要参数的选项之后没有找到参数，它就在给定的变量中存入 ? ，并向标准错误中写入错误消息。否则将实际参数写入特殊变量 ：OPTARG
  - 另外一个特殊变量：OPTIND，反映下一个要处理的参数索引，初值是 1，每次执行 getopts 时都会更新。
  - Example:
    ```
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

    Execut:
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
