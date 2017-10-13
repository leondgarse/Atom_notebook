# ___2016 - 10 - 28 Primier C++ plus Basic___
- C++头文件没有扩展名，对于C的头文件前面加c， math.h --> cmath
- 循环中在栈中创建的对象其地址相同，为避免这种情况可使用堆
- pthread_cond_wait
- nullptr        C++空指针
- static_assert 用于在编译阶段对断言进行测试
***

# <span style="color:#ff0000;">输入输出 cin / cout
  - 包含头文件iostream：
    ```c++
    #include <iostream>        //使用iostream而不是iostream.h，相对的，c标准库的头文件名加前缀c，如<cmath>
    ```
    并在使用前提供名称空间：
    ```c++
    using namespace std;        // 指定名称空间，此时可以使用cout表示std::cout
    或只指定cin / cout
    using std::cout;
    using std::cin;
    ```
    cin是istream类，cout是ostream类
  - 在程序中包含iostream将自动创建8个流对象(4个用于窄字符流，4个用于宽字符流)
    ```c++
    (w)cin / cout / cerr(无缓冲) / clog(缓冲)
    ```
## <span style="color:#ff8000;">输出 cout
  - std::cout << "Hi" << std::endl;        // 输出并换行
  - std::cout << "Hi" << std::flush;        // 刷新缓冲区，不换行，也可以使用flush(cout);
  - std::cout << value
    ```c++
    << " raised to the power of "
    << pow << ": \t"
    << result << std::endl;
    ```
    cout.put(); 输出单个字符
    ```c++
    cout.put('w').put(65);        // 输出65对应的ASCII码字符
    ```
    cout.write(); 显示整个字符串
    write()方法不会遇到\0停止，可以越界显示指定长度的字符串
    ```c++
    const char *strc1 = "greatwall";
    cout.write(strc1, 5);        // 显示strc1的前5个字符
    ```
## <span style="color:#ff8000;">输入 cin
  - char ch;
  - cin >> ch;        // 读取char值时，cin将忽略空格和换行符
    ```c++
    while(ch != '\n') { cin >> cin; }        // 循环将不会退出，因为跳过了换行符
    ```
    cin.get(ch);        // 接受空格和换行符

  - int score[10];
  - cin >> score;        // 读取从第一个非空字符开始，到与目标类型不匹配的第一个字符

  - cin >> hex;        // 将整数输入解释为16进制

  - fload fee;
  - cin > score >> fee;

  - cin.read(name, 10); 读取指定数目的字节并保存在目标位置
    ```
    不会在输入后加上'\0'
    最常与write()结合使用，用于文件的输入输出
    ```
## <span style="color:#ff8000;">cout 格式控制
  - iostream中的dec / hex / oct(控制符，endl也是一个控制符)来控制输出的进制：
    ```c++
    cout << hex;
    cout << "count = " << count << endl;
    ```
  - cout.width(3);        // 输出时显示的宽度，只在显示下一个数值时有效
    ```c++
    cout << '#';
    cout.width(12);
    cout << 12 << "#" << 24 << "#\n";

    输出：#(space * 12)12#24#        // width()只影响下一个输出，输出的12右对齐
    ```
  - cout.fill('*');        // 改变默认填充字符为*
  - cout.precision(3)        // 设置浮点数的显示精度，显示三位小数，默认为6，默认是指显示的总位数，默认不显示末尾的0
    ```c++
    float prc1 = 20.04;
    cout.precision(2);
    cout << "prc1 = " << prc1 << endl;

    输出：prc1 = 20
    ```
  - cout.setf(ios_base::showpoint);         // 显示小数点，使用默认的浮点格式时，显示末尾的0
    ```c++
    float prc1 = 20.04;
    cout.setf(ios_base::showpoint);
    cout << "prc1 = " << prc1 << endl;
    cout.precision(2);
    cout << "prc1 = " << prc1 << endl;

    输出：prc1 = 20.0400        // 默认6位精度
          prc1 = 20.
    ```
  - cout.setf(ios_base::fixed, ios_base::floatfield)；         // 限定输出使用定点表示
  - cout.setf(ios_base::boolalpha)        // 指定cout显示true / false 而不是 0 / 1：
    ```c++
    cout.setf(ios_base::boolalpha);
    cout << (2 > 3) << endl;
    ```
## <span style="color:#ff8000;">setf() / unset() 方法格式控制
  - unset()用于清除控制位
  - setf()函数第一个原型： fmtflags setf(fmtflags); 用来控制单个位控制的格式信息
    ```c++
    其中fmtflags是bitmask类型，用于存储格式标记
    参数用来指出用来设置哪一位
    返回值指出所有标记以前的设置

    ios_base::boolalpha        输入和输出bool值，可以为true / false
    ios_base::showbase        输出使用c++基数前缀 0, 0x
    ios_base::showpoint        显示末尾小数点
    ios_base::uppercase        16进制输出，使用大写字母
    ios_base::showpos        正数前面显示+
    ```
  - setf()函数第二个原型： fmtflags setf(fmtflags, fmtflags); 用来控制由多位控制的格式信息
    ```c++
    第一个参数指出需要设定的位
    第二个参数指出需要清除的位，通常为一组相关值
    如设置16进制输出时，需将16进制标志位置1，8进制 / 10进制标志位清0

    ios_base::basefield        dec / oct / hex        10 / 8 / 16 进制
    ios_base::floadfield        fixed / scientific        定点计数法 / 科学计数法
    ios_base::adjustfield        left / right / internal        左对齐 / 右对齐 / 符号左对齐，值右对齐

    设置左对齐：
            cout.setf(ios_base::left, ios_base::adjustfield);
    设置16进制：
            cout.setf(ios_base::hex, ios_base::basefield);
    设置浮点显示模式：
            cout.setf(0, ios_base::floatfield);

    也可以使用unset()：
            cout.unset(ios_base::floatfield);
    ```
## <span style="color:#ff8000;">标准控制符 （调用setf()）
  - 类似于 dec / hex / oct 可以直接用在cout中，控制输出格式
  - 打开左对齐和定点：
    ```c++
    cout << left << fixed;
    ```
  - 控制符：
    ```c++
    boolalpha                 setf(ios_base::boolalpha)         输入和输出bool值，可以为true / false
    noboolalpha         unsetf(ios_base::boolalpha)

    (no) showbase / showpoint / showpos / uppercase
    internal / left / right
    dec / hex / oct
    fixed / scientific
    ```
## <span style="color:#ff8000;">setfill / setw
  - 头文件iomapip
  - setprecision() 接受整型参数指定精度
  - setfill() 接受char参数指定填充符
  - setw() 接受整型参数指定字符宽度
  - 示例：
    ```c++
    for (int n = 10; n <= 100; n += 10) {
            cout << setw(6) << setfill('.') << n << setfill(' ')
                    << setw(12) << setprecision(3) << n*n
                    << setw(14) << setprecision(34) << sqrt(n)
                    << endl;
    }
    ```
## <span style="color:#ff8000;">保存 / 恢复格式状态
  - 在函数中修改格式信息后应在函数结束时恢复修改前的状态：
    ```c++
    streamsize prec = cout.precision(3);        //保存状态
    cout.precision(prec);                //恢复状态
    ios_base::fmtflags orig = cout.setf(std::ios_base::fixed);        //保存状态
    cout.setf(orig, std::ios_base::floatfield)；        //恢复状态
    ```
## <span style="color:#ff8000;">eof / fail / bad / good
  - cin / cout包含一个描述流状态的数据成员： eofbit / badbit / failbit
    ```c++
    cin操作达到文件末尾时，设置eofbit
    cin操作没有读取到指定字符时，设置failbit
    在一些无法诊断的失败破坏流时，设置badbit

    3个状态位全部为0时，说明状态正常
    ```
  - 位检测方法：
    ```
    good()         所有位都为0时，返回true
    eof()        eofbit被设置，返回true
    bad()        badbit被设置，返回true
    fail()        eofbit / failbit被设置，返回true [ ??? ]

    rdstate()                返回流状态
    exception()                返回位掩码，指出哪些标记会导致异常
    exception(iostate ex)        设置哪些状态将导致clear()引发异常
    clear(iostate s)        将流状态设置为s，s默认值为0，如果设置的位被设置为引发异常，将引发异常
    setstate(iostate s)        调用clear(rdstate() | s)，设置s的对应状态位，其他位不变
    ```
  - 检测到EOF后cin / 文件将eofbit 和 failbit 都设置为1，可通过成员方法eof()和fail()测试
  - 当cin出现在需要bool值的地方，将调用转换函数转换为bool值：
    ```
    while ( cin) { ... } <==> while (!cin.fail()) { ... } [ ??? ]
    while (cin >> input)

    而由于cin.get(char) 的返回值为cin，因此可使用：
            while (cin.get(ch)) { ... }
    判断输入是否成功

    使用 while(cin) 判断输入是否成功，这比 !cin.eof() / !cin.fail() 更通用，因为可以检测到如磁盘故障等失败原因
    ```
## <span style="color:#ff8000;">清除位状态
  - 如果流状态位被设置，将对后面的输入或输出关闭，直到位被清除：
    ```c++
    while (cin >> input) { sum += input; }
    cin >> input;        // won't work before clear fail bit

    使用clear()清除位:
            cin.clear();
    清空输入缓冲中剩余字符：
            while (!isspace(cin.get()))        // get rid of bad input until space
                    continue;
    或清空整行：
            while (cin.get != '\n')
                    continue;
    继续接受输入：
            cin >> input;
    ```

    ```c++
    while (cin >> input) { ... }
    if (cin.fail() && !cin.eof()) {        // failed because of mismatched input
            cin.clear();
            while (!isspace(cin.get())) continue;
    } else {        // other fault
            cout << "I/O error\n";
            exit(1);
    }
    cin >> input;        
    ```
## <span style="color:#ff8000;"> 输入 get() / getline()
  - get()
    ```c++
    cin.get(); 有多种重载方法
    cin.get(void); 返回类型是int，因此不能使用cin.get().get();，但可以用于：
            char ch = cin.get();
            cin.get(ch).get();
    cin.get(ch); 用于单个字符输入
    cin.get(name, 10); 字符串输入，在读取9个字符或遇到换行符时停止，不读取换行符，将其留在输入队列中，接下来的输入操作将首先看到换行符
            使用cin.get(name, 20).get()来跨过换行符，相较于getline()，使得检查错误更简单

    cin.get(name, 10, '#'); 指定接受字符的结束标志
    ```
  - getline()
    ```c++
    cin.getline(name, 10); 当遇到换行符或读取到指定数目(9个)的字符时停止接收，但不保存换行符，在存储时以空字符代替换行符

    istream& getline(char *buffer, streamsize num, char delim);
            指定接收字符串的结束符，但不接收该字符
    ```
  - 当get()读入空行后将设置失效位failbit，阻断接下来的输入，另一个问题是当输入字符串长度大于指定个数时，余下的字符将留在输入队列中，getline()还会设置失效位，使用命令恢复：
    ```c++
    if(!cin) {
            cin.clear();        //只进行复位并不清理缓冲区
            cin.ignore(200, '\n');        //清理最多200个字符的缓冲区，至'\n'为止
    }
    ```
## <span style="color:#ff8000;">peek / gcount / putback
  - peek()
    ```c++
    返回输入总的下一个字符，但不抽取字符，用于查看下一个字符
    cin.get(name, 25);
    if (cin.peek() != '\n') { ... } // 查看是否是读取了整行，还是达到字符数限制
    ```
  - gcount()
    ```c++
    返回上一次非格式化抽取方法读取的字符数，即get() / getline() / idmore() / read() 方法读取的字符数，而不是 >> 运算符抽取的字符数
    ```
  - putback() :=(
    ```c++
    将一个字符插入到输入流中，接受一个char参数，返回类型为 istream &

    ```
## <span style="color:#ff8000;">重定向
  - 输入重定向 < / 输出重定向 >
  - 对标准输出的重定向并不会影响 ceer / clog
    ```c++
    $ test < foo.in > foo.out
    $ 2>&1        // 标准错误重定向到标准输出
    ```
***

# <span style="color:#ff0000;">变量
  - short至少16位，int至少与short一样长
  - long至少32位，且至少与int一样长
  - long long至少64位，且至少与long一样
  - float至少32位，double至少48位，且不少于float；
  - wchar_t 宽字符类型，用于表示扩展字符集
  - 在头文件climits (/usr/include/c++/4.7/，c头文件位于/uer/include/) 中包含了关于整型限制的信息，在头文件float.h中可以找到浮点型的系统限制
  - alignof() 获得类型或对象的对齐要求，alignas控制对齐方式 :=(
## <span style="color:#ff8000;">初始化
  - 初始化不是赋值，初始化指创建变量并给它赋初始值，而赋值则是擦除对象的当前值并用新值代替。
    初始化：
    ```c++
    int ival(1024);        // 直接初始化，效率高
    int ival = 1024;        // 赋值初始化
    ```
  - 列表初始化：
    ```c++
    int x = {5};
    double y{2.75};
    short quar[5] {4, 5, 2, 3, 6};
    int *pa = new int[4] {2, 4, 6, 7};

    列表初始化可以防止将值赋给无法保存它的类型：
            char ch = {3.45};        // double-to-char, compile-time error
    ```
## <span style="color:#ff8000;">auto
  - 自动判断变量类型
    ```c++
    int a;
    auto b = a;

    ```
    可用于简化模板声明
## <span style="color:#ff8000;">decltype
  - 将变量的类型声明为表达式指定的类型
    ```c++
    decltype(x+y) xpy = x + y;

    ```
    可用于声明模板参数
## <span style="color:#ff8000;">volatile
  - 内存单元存储的值可能在程序代码以外发生改变
  - 如果编译器发现程序在几条语句中两次使用了某个变量值，可能不是让程序查找这个值两次，而是将其缓存到寄存器中，这种优化假设变量在两次取值期间不会改变
  - 声明为volatile则不会进行这种优化
## <span style="color:#ff8000;">mutable
  - 即使结构或类被声明为const，某个成员也可以被修改
  - struct data {
    ```c++
    char name[30];
    mutable int accesses;
    ```
    };

  - const data = { ... };
## <span style="color:#ff8000;">new
  - new关键字 typename * pn = new typename;        //在堆中分配空间，并使用delete释放该内存
  - 在编译时给数组分配内存被称为静态联编，但使用new时是在运行时创建，被称为动态联编
  - 内置标量类型
    ```c++
    int * p = new int(6);
    ```
  - 列表初始化常规结构 / 数组
    ```c++
    struct where {double x; double y; double z;};
    where * one = new where {2.1, 2.2, 2.3};
    int * arr = new int[4] {1, 2, 3, 4};
    也可用于单值：int * p = new int {6};
    ```
    ```c++
    eg: int * psome = new int[10];
            delete [] psome;        //如果new时使用了[]，则delete时也要有[]
    ```
    <br />
  - new失败时返回异常std::bad_alloc
## <span style="color:#ff8000;">定位new
  - 需要包含头文件 new #include <new>，在指定的内存地址分配空间：
    ```c++
    char buf [100];
    double *pd1 = new（buf） double[20];
    double *pd2 = new(buf + 20 * sizeof double) double[20];
    ```
  - 不跟踪哪些内存单元已被使用
  - 若是在栈中分配的空间，不能使用delete删除
  - 对于使用定位new分配的类，需要显示调用这些类的析构函数：
    ```c++
    JustTesting *pc1 = new(buf) JustTesting;
    JustTesting *pc2 = new(buf + sizeof(JustTesting)) JustTesting;
    ...
    pc3->~JustTesting();
    pc1->~JustTesting();         // 调用析构函数顺序与创建相反
    ```
## <span style="color:#ff8000;">using 变量别名
  - typedef const char * pc1;
  - using pc2 = const char * ;
  - typedef condt int *(* pa1)[10];
  - using pa2 = const int *(*)[10]; :=(
## <span style="color:#ff8000;">类型转换运算符
  - 更严格的控制类型转换，可以根据目的选择一个适合的运算符，而不是使用通用的类型转换
### <span style="color:#00ff00;">dynamic_cast
  - 在类层次中进行向上转换(派生类转为基类)，而不允许其他转换
    ```c++
    p1 = dynamic_cast <Type *> (p2);
    ```
    仅当p2是Type类或Type类的派生类时，返回一个 Type * 的指针，否则返回空指针
### <span style="color:#00ff00;">const_cast
  - 改变声明为const 或 volatile变量的值
  - 转换时类型必须是相同的，只能有const 或volatile特征的不同
    ```c++
    Type a;
    const Type b;

    const Type * pa = &a;
    const Type * pb = &b;

    Type *pc = const_cast <Type *> (pa);
    Type *pd = const_cast <Type *> (pb);

    ```
    此时pb可以用来修改a的值，但pd仍不能修改b的值
  - 转换时pa必须是Type类型的
### <span style="color:#00ff00;">static_cast
  - 用于所有可以隐式转换的类型，即可以指定派生类转换为基类，也可以基类转换为派生类
    ```c++
    static_cast<Type> (expression);
    ```
    也可以用于枚举与整型，double与int等类型转换
### <span style="color:#00ff00;">reinterpret_cast
  - 其他必须进行的类型转换
    ```c++
    struct data {short a; short b;};
    long value = 0xFFFF0008;
    data *pb = reinterpret_cast<data *> (value);
    cout << pb->a;
    ```
***

# <span style="color:#ff0000;">引用变量 (左值引用 / 右值引用)
  - 值传递：数据量小，且函数不改变参数值
  - 指针传递：数据对象是数组或结构
  - 引用传递：数据对象是类或结构
  - int a;
  - int & b = a;                // int & works as int * , indicates b is an alias of a,
  - int * const c = &a;        // This * c could be similar with b
  - const 引用与强制类型转换 【？】
  - 如果函数的返回值是引用类型，则不能返回生存期只在函数内部的变量，对已释放内存的引用将引起错误，因此只能是通过参数传递给它的对象
  - 函数使用引用参数：
    ```c++
    int & f_return_ref(&d) { ... return d; }
    ...
    {
            int a;
            int b;
            int c;

            b = f_return_ref(a);         // This indicating b = a;
            f_return_ref(a) = c;        // This indicating a = c;
    }
    ```
## <span style="color:#ff8000;">右值引用
  - 左值：一个变量名或解除引用的指针等，即一个可以出现在等号左边的表达式，程序可获取其地址
  - 传统的引用(左值引用)可以将引用关联到一个左值：
    ```c++
    int & b = a;
    ```
  - 右值：一个字面常量，计算表达式x+y，以及函数(不能是返回引用值的函数)等，即一个只能出现在等号右边的表达式，程序不能获取其地址
  - 右值引用可以将引用关联到一个右值，使用&&：
    ```c++
    int x = 10;
    int y = 20;

    int && r1 = 13;
    int && r2 = x + y;        // 关联到的是当前x+y的值，以后改变x / y不会影响r2
    double && r3 = std::sqrt(2.0);
    ```
    主要目的是实现移动语义
***

# <span style="color:#ff0000;">结构 / 共用体
  - C++允许在声明结构变量时省略关键字struct
  - C++的结构体成员可以有成员函数
  - 如果初始化时大括号内未包含任何东西，则各个成员都被设置为0
  - 结构体可以使用 = 来为同类型的结构体赋值，即使结构体中有数组也是可以的，这同样适用于C语言
  - C++结构与类具有相同的特性，区别在于结构的默认访问权限是public，类是private
  - 共用体允许有构造函数 / 析构函数
  - 匿名共用体：其成员将成为位于相同地址处的变量
    ```c++
    eg: struct widget {
            int type;
            union {
                    long id_num;
                    char id_char[20];
            };
    };
    ...
    widget prize;
    ...
    if (prize.type == 1)
            cin >> prize.id_num;
    else
            cin >> prize.char_num;
    ```
    <br />
***

# <span style="color:#ff0000;">数组的替代品 (模板类 vector / array / valarray)
  - 可将列表初始化用于vector / array
  - 不同于数组，相同大小的array类之间可以相互赋值
  - 都可以使用[]来访问元素
## <span style="color:#ff8000;">vector
  - STL类，类似于string，也是一种动态数组，在运行阶段设置长度，也可以附加新数据或插入新数据，存储于堆中需包含头文件vector
    ```c++
    vector<int> vi;
    vector<double> vd(n);        // 动态内存分配，n可以是变量
    vector<typeName> vt(n_elem);

    重载的[]运算符：vd[0] = 9;
    ```
## <span style="color:#ff8000;">array
  - 设计用于替代数组，与数组一样使用固定长度，效率高，方便安全，存储于栈中，需包含头文件array
    ```c++
    array<int, 5> ai = {1, 2, 3, 4, 5};        //长度不能是变量
    array<typeName, n_elem> arr;

    成员方法begin() / end()确定边界，at()方法可防止越界访问：
    ai.at(2) = 13;
    ```
  - 提供了多个STL方法，如begin() / end() / rbegin() / rend()，因此可以将STL方法用于array对象
## <span style="color:#ff8000;">valarray
  - 用于处理数值，提供更多的算数支持
  - 不能自动调整大小，但有一个resize()方法
  - 没有定义STL方法适用的begin() / end() 方法
    ```c++
    valarray<int> v1;                // An array of int, size == 0;
    valarray<double> v2(8);        // An array of double, size == 8;
    valarray<int> v3(10, 8);                 // An array of int, size == 8, init value == 10;
    double gpa[5] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1};
    valarray<double> v4(gpa, 4);        // An array of double, size == 4, init value using first 4 elements of gpa;

    operator[](); // 访问各个元素
    size();         // 返回元素数
    sum();         // 返回元素总和
    max();         // 返回最大元素
    min();         // 返回最小元素
    ```
  - valarray类重载了所有算术运算符：
    ```c++
    valarray<double> vad1(10), vad2(10), vad3(10);
    vad3 = vad1 + vad2;        // vad3中每个元素等于vad1与vad2中每个元素之和

    vad3 *= 2.5;        // vad3中每个元素扩大2.5倍
    vad3 = log(vad1);        // varray重载了log()运算符，可以直接用于对象
    vad3 = vad1.apply(log);        // apply()方法用于非重载方法，不能修改调用对象，返回一个包含结果的新对象

    ```
  - slice类
    ```c++
    slice类可以用作数组索引，这种下标指示功能可以用一位数组表示二维数组
    slice[起始索引, 索引数目, 跨距];
    如： slice[1, 4, 3]创建的对象表示选择1， 4， 7， 10四个元素

    vad2[slice(1, 4, 3)] = 10         // set selected elements to 10
    valarray<double> vad4(vad3[slice(3, 3, 1)]);        // initialize vad4 using vad3[3, 4, 5]
    ```
  - 其他特性：
    ```c++
    valarray<bool> vbool = vad3 > 9;        // vbool will be true / false result of if vad3 elements greater than 9
    ```
  - 用于STL方法：
    ```c++
    sort(vad1.begin(), vad1.end());        // NOT valid, no begin() / end() method
    sort(vad1, vad1+10);        // NOT valid, vad1 is not an address
    sort(&vad1[0], &vad1[10]);        // valid in most situation, not recommend

    c++提供了接受valarray对象作为参数的模板函数 begin() / end()：
            sort(begin(vad1), end(vad1));
    ```
***

# <span style="color:#ff0000;">循环 / 分支语句
  - 基于范围的for循环：
    ```c++
    为用于STL而设计，对 数组 / vector / array 的每个元素进行相同操作时：
    double prices[5] = { ... };
    for (double x : prices) { cout << x << endl; }
    for (double &x : prices) { ...} // 使用引用实现对元素的修改

    for (int x : {1, 2, 3, 4, 5}) { ... }

    vector<string> sv(5);
    for (auto &elem : sv) { show(elem); } // auto自动判断变量类型为string
    ```
***

# <span style="color:#ff0000;">string类
  - string类位于名称空间std中
  - 类设计让程序能够自动处理string的大小，即自动调整string对象的长度
  - string类的操作方法由头文件cstring提供，如 c语言中的strxxx() 方法、size()
  - string类的很多方法被重载，可以同时处理string类或c风格字符串
  - 通常c++将分配一个比实际字符串大的内存块，以便提供增大空间，如果字符串不断增大，程序将重新分配原来两倍大小的空间
  - string类虽然不是STL的组成部分，但设计时考虑到了STL，因此可以使用STL接口
## <span style="color:#ff8000;">构造函数
  - size_type是一个依赖于实现的整型，头文件string中定义
  - string::npos定义为字符串的最大长度，通常为unsigned int最大值
  - 构造函数类型
    ```c++
    string(const char * s)
    string(size_type n, char c)
    string(const string & str)
    string()
    // 初始化为s的前n个字符，可以超过s的长度，将复制s后面的内容到字符串结尾
    string(const char * s, size_type n)

    // 初始化为begin到end区间的字符，begin / end作用相当于指针，范围包括begin，不包括end
    template<class Iter> string(Iter begin, Iter end)

    // 初始化为str从pos到结尾，或从pos开始的n个字符    
    string(const string & str, size_type pos = 0, size_type n = npos)
    // 初始化为str，并可能修改str的值（移动构造函数）
    string(string && str) noexcept
    // 初始化为初始化列表il中的值
    string(initializer_list<cahr> il)
    ```
  - 用例：
    ```c++
    char ar[30] = 'For the love of a princess!'
    stringstr1(ar, 20);
    string str2(ar + 5, ar + 15);
    string str3(&str2[2], &str2[8]);
    string str4(str1, 3, 16);

    string str5 = {'p', 'i', 'a', 'n', 'o'};        // 初始化列表
    ```
## <span style="color:#ff8000;">cctype中的字符函数
  - isalnum(); isalpha(); iscntl(); isdigit(); isgraph(); islower(); isprint(); ispunct(); isspace(); isupper(); isxdigit(); tolower(); toupper();
## <span style="color:#ff8000;">string类运算符
  - string类重载了多个版本的 + / += / =，可用于string类 / c风格字符串 / char字符
    ```c++
    string one, two;
    one += 'a';
    two = '?';
    one += two;
    ```
  - 重载[]运算符，可以访问string中各个字符：
    ```c++
    string three = ":D";
    three[1] = 'P';
    ```
  - 重载 < / > / == / != 运算符，用于string类与sting / c风格字符串比较
## <span style="color:#ff8000;">string类输入
  - c风格字符串：
    ```c++
    char info[100];
    cin.getline(info, 100);
    cin.get(info, 100);
    ```
    string类：
    ```c++
    string str;
    cin >> str;        // string类重载的 >> 运算符
    getline(cin, str); // string类的getline方法

    两个版本的getline都有一个可选参数，可以指定字符界定输入边界（默认为\n），指定分界字符后，换行符将被视为普通字符：
    cin.getline(str, ':');
    getline(str, ':');

    string类的getline方法以及>>运算符将自动调整目标str的大小，使之刚好存储下输入的字符
    ```
  - 文件输入：
    ```c++
    ifstream fin;
    fin.open("foo");
    if (fin.is_open() == false)
            return;
    string item;
    getline(fin, item, ':');         // using ':' as terminating character.
    while(fin) {
            cout << item << endl;
            getline(fin, item, ':');
    }
    fin.close();
    ```
## <span style="color:#ff8000;">其他方法 size / length / find / capacity / reserve / c_str
  - size() / length()方法都可以返回字符串中的字符数
  - find()：
    ```c++
    size_type find(const string &str, size_type pos = 0) const        // 从pos处查找str，成功返回首次出现的位置索引，失败返回string::npos
    size_type find(const char * s, size_type pos = 0) const        // pos处查找s
    size_type find(const char * s, size_type pos = 0, size_type n) const // pos处查找s的前n个字符
    size_type find(char ch, size_type pos = 0) const                // 查找ch
    ```
    类似函数还有：
    ```c++
    rfind()         // 查找最后一次出现位置
    find_first_of()         // 查找参数中任何一个字符首次出现的位置
    find_last_of()         // 查找参数中任何一个字符最后出现的位置
    find_first_not_of()        // 查找参数中不包含字符首次出现的位置
    find_last_not_of()        // 查找参数中不包含字符最后出现的位置
    ```
  - capacity() 返回当前分配给字符串的实际内存大小，通常是比size()大，空字符串也有默认最小容量
  - reserve(new_size) 可以重新申请new_size长度的字符串，实际的内存大小也要大于new_size
  - c_str() 返回一个指向c风格字符串的指针，可以用于如open()等要求c风格字符串参数的方法中：
    ```c++
    string filename;
    ostream fout;
    fout.open(filename.c_str());
    ```
## <span style="color:#ff8000;">字符串的其他种类
  - string库实际是基于一个模板类的：
    ```c++
    template<class charT, class traits = char_traits<charT>, class Allocate = allocate<charT>>
    basic_string { ... };

    ```
    模板basic_string有四个具体化，并分别对应一个typedef名称：
    ```c++
    typedef basic_string<char> string;
    typedef basic_string<wchar_t> wstring;
    typedef basic_string<char16_t> u16string;
    typedef basic_string<char32_t> u32string;

    可以创建以上类型的字符串，或开发某种类似字符串的类
    traits类描述选定字符类型的特定情况，如比较算法等
    Allocate是一个管理内存分配的类，以上字符串类型的预定义模板具体化，都使用new / delete
    ```
## <span style="color:#ff8000;">内核格式化 (incore formating) sstream
  - ostringstream
    ```
    将信息写入对象，用于存储信息
    可以使用cout的方法

    ostringstream outstr;
    double price = 93.2;
    outstr.precision(2);
    outstr << "price is " << price << endl;

    str() 方法返回一个被初始化为缓冲区内容的字符串对象
    string mesg = outstr.str();
    使用str()方法将冻结该对象，不能再将信息写入该对象中
    ```
  - istringstream
    ```c++
    可以使用cin的方法读取istringstream对象中的数据
    可以使用string对象初始化

    string strw = "a serial of words";
    istringstream instr(strw);
    string word;
    while (instr >> word) { ...; cout << word << endl;}
    ```
***

# <span style="color:#ff0000;">函数 (多态 / 模板 / 后置返回类型)
## <span style="color:#ff8000;">默认参数
  - 在函数声明处使用 char * left(const char * str, int n = 1); 来告知程序可能的默认值
  - 必须从右向左添加默认值，要为某个参数添加默认值，则必须为它右边所有的参数提供默认值
  - 实参按从左到右赋给形参，而不能跳过 :=(
## <span style="color:#ff8000;">函数多态 / 函数重载
  - 同名，不同参数列表（函数特征标）
  - 如果参数数目 / 类型 / 顺序相同则特征标相同
  - 不区分类型引用和类型本身
  - 不区分返回值
  - const [?]
  - 名称修饰： 编译器根据参数列表的不同对函数名称作转换
## <span style="color:#ff8000;">函数模板
  - 将同一种算法用于不同类型的函数
  - template <typename T>        // 建立模板，可使用class代替typename，类型名随意
  - void Swap(T &a, T &b) { ... }         // 使用时与常规函数相同，编译器将自动生成相应的函数，最终代码不包含任何模板
  - void Swap(T &a, T &b, int a) { ... }        :=(// 重载的模板，并不需要所有形参类型都是泛型
### <span style="color:#00ff00;">具体化
  - 显式具体化： 指定使用特定的函数原型来处理代码，而不使用函数模板生成，代码中必须有该函数的定义，使其行为与模板代码不同
    ```c++
    具体化优先于常规模板，而非模板函数优先于具体化和常规模板
    template <> void Swap<job> (job &, job&);        //其中<job>可省
    ```
  - 显示实例化：指定使用模板生成特定参数类型的函数来处理代码，生成模板的一个指定类型的实例
    ```c++
    int a; double b;
    template void Add<double> (double, double);         // 指定使用double参数列表的实例来处理参数
    还可通过在程序中使用函数来创建：cout << Add<double> (x, m) << endl;
    ```
  - 在同一个文件中使用同种类型的显式实例化和显式具体化将出错
## <span style="color:#ff8000;">后置返回类型
  - 在函数名和参数列表后面指定返回类型，用于无法预先知道返回类型的情况
  - auto h(int x, float y) -> double        // auto是一个占位符，表示后置返回类型提供的类型
  - 结合decltype：
    ```c++
    template<typename T1, typename T2>
    auto gt(T1 x, T2 y) -> decltype(x + y)
    { ...; return x + y; }
    ```
## <span style="color:#ff8000;">可变参数模板
  - 创建接受可变数量参数的模板函数或模板类
  - 声明使用...
### <span style="color:#00ff00;">模板和函数参数包
  ```c++
  template <typename ... Args>        // Args is a template parameter pack
  void show_list(Args ... args) { ... }        // args is a function parameter pack

  如对于：
          show_list('s', 80, "sweet", 4.5);
  模板参数包Args为char, int, const char *, double
  函数参数包args类型为Args，且与模板参数包包含的类型列表匹配
  ```
### <span style="color:#00ff00;">展开参数包
  - 递归：
    ```c++
    为模板函数的第一个参数指定名称
    void show_list() {}        // definition for 0 parameter

    template <typename T, typename ... Args>
    void show_list(T value, Args ... args) {
            cout << value << ", ";
            show_list(args...);        // 每次递减一个参数，直到列表为空，结束递归
    }
    ```
  - 改进：
    ```c++
    使用引用，并定义显示单个元素的函数
    void show_list() {}        // definition for 0 parameter

    template <typename T>        // print '\n' for the last elament, instead of ", "
    void show_list(const T& value) {
            std::cout << value << '\n';
    }

    template <typename T, typename ... Args>
    void show_list(const T& value, const Args& ... args) {
            cout << value << ", ";
            show_list(args...);        // 每次递减一个参数，直到列表为空，结束递归
    }
    ```
***

# <span style="color:#ff0000;">名称空间
  - using编译指令：using namespace std; 表明可以使用std名称空间的名称，而不用使用std::前缀
  - using声明：使用特定的几个名称：
    ```c++
    using std:cout;
    using std:endl;
    using std:cin;
    ```
    一般来说using声明比using编译指令更安全，因为它只导入指定的名称，而using编译指令导入所有名称
  - 示例：
    ```c++
    namespace Jack {
        double pail;
        void fetch();
    }
    ```
    访问名称空间 Jack::pail = 12.34; (限定的名称)
  - 名称空间是开放的，即可以把名称或方法加入到已有的名称列表中
    ```c++
    namespace Jack {
            double score;
    }

    ```
    或提供名称列表中已有原型的函数代码
    ```c++
    namespace Jack {
            void fetch() { ...; }
    }
    ```
## <span style="color:#ff8000;">名称空间嵌套
  - namespace Jill {
    ```c++
    using namesapce std::cout;
    using namespace std::cin;
    namespace myth {
            void spire();
    }
    double water;
    ```
    }

  - 使用using namespace Jill::myth;使内部名称可用
  - 如果名称空间A包含名称空间B，导入A将同时导入B
## <span style="color:#ff8000;">别名
  - namespace Mymyth = Jill::myth;
## <span style="color:#ff8000;">未命名名称空间
  - namespace { ...; }
  - 无法在文件外使用该名称空间中的变量名称，相当于定义静态全局变量
***

# <span style="color:#ff0000;">对象和类
  - OOP最重要特性：抽象，封装和数据隐藏，多态，继承，代码的可重用性
  - 类设计尽量将公有接口和实现细节分开，公有接口表示设计的抽象组件，将实现细节放在一起并将它们与抽象分开被称为封装
  - 类声明中可以省略private关键字，这是类的默认访问控制
  - C++结构与类具有相同的特性，区别在于结构的默认访问权限是public，类是private
  - 类成员函数可以访问类的private组件，定义时使用作用域解析运算符 ( :: ) 来标识函数所属的类
  - Stock::update()称为方法的限定名，update()为全名的缩写，非限定名，只能在类作用域中使用
  - 其定义位于类声明中的函数将自动成为内联函数
  - 为避免参数与成员重名，通常可以在成员名前加m_，或加后缀_
## <span style="color:#ff8000;">构造函数
  - 用于在类创建时初始化成员变量，没有返回值(不是void)，不能通过对象来调用
    ```c++
    Stock::Stock(const string &co, long n = 0, double pr = 0.0) { ... }

    Stock food = Stock("Cabbage", 50, 1.25);
    Stock garment("Furry Mason", 50, 1.34);
    Stock *pstock = new Stock("Games", 18, 19.0); // 初始化一个匿名对象，将地址赋值给pstock
    Stock hot_tip = {"plus", 100, 45.0};         //C++11 列表初始化
    Stock jock {"Sport", 20, 3.0};

    Stock stocks[5] = {
            Stock("hello", 2, 3.3),
            Stock(),
            Stock("world", 3, 4.4)
    };
    ```
  - 为类定义了构造函数后，必须为它提供默认的构造函数，即给已有的构造函数的参数提供默认值，或定义一个不接收任何参数的构造函数
    ```c++
    Stock st; // Calling the default constructed function. Not Stock st(); this is declaring a function!
    ```
  - explicit：
    ```c++
    接受一个参数的构造函数允许将对象初始化为一个值：
            Stock stock1 = "hello";
    可以使用关键字explicit禁止单参数构造函数导致的自动转换：
            explicit Stock::Stock(const string &co, long n = 0, double pr = 0.0) { ... }
    但仍可以显示转换：
            stock2 = Stock("hello");
    ```
## <span style="color:#ff8000;">析构函数
  - 对象过期时完成清理工作，如delete释放new分配的内存
  - Stock::~Stock() { ... }        //使用 int main() { { ... ;} return 0; } 形式的程序结构可在main函数结束前执行到析构函数
## <span style="color:#ff8000;">const成员函数
  - 函数不修改类中的成员值
  - 声明 void show() const;
  - 定义 void stock::show() const { ... }
## <span style="color:#ff8000;">this指针
  - 示例：
    ```c++
    const Stock & Stock::topval(const Stock &s) const // 函数不修改参数值，不修改类中的成员值，且返回一个const引用
    {
        if (s.total_val > this->total_val)
            return s.total_val;
        else
            return * this;        // this是一个指针，使用*this返回对象本身
    }
    ```
  - 类中的成员函数至少有一个参数this指针，当作为线程实现函数时，可修饰为static以去掉this指针，但这会丢失函数的多态性，另一种方法如下：
    ```c++
    static void *recieveData(void *par) {
            //静态函数不能直接调用非静态成员函数，需this指针
            ((ProducterThread*)par)->run();
    }
    void start() { pthread_create(&tid, 0, receiveData, this); }
    virtual void run() { ... }        //具体的线程实现
    ```
## <span style="color:#ff8000;">类中的常量成员
  - 可以使用枚举：
    ```c++
    class bakery {
    private:
            enum {month = 12};
            double costs[month];
    ...
    };
    ```
  - 或使用静态成员变量：
    ```c++
    static const int month = 12;
    ```
## <span style="color:#ff8000;">静态成员变量
  - static int num_obj;
  - 类中的静态成员变量只创建一个变量副本，也就是类的所有对象共享一个静态成员
  - 静态成员变量可以在类声明之外，方法文件中使用使用单独的语句进行初始化：
    ```c++
    int MyString::num_obj = 0;
    ```
## <span style="color:#ff8000;">静态成员函数
  - 不能通过对象调用静态成员函数，通过类名和作用域解析运算符来调用
    ```c++
    static int HowMany() { ... }
    int count = MyString::HowMany();
    ```
    静态成员函数不能使用this指针，智能使用静态数据成员
## <span style="color:#ff8000;">作用域为类的枚举
  - 使用class / struct定义，使用时名称要求显式限定
    ```c++
    enum class egg {Small, Medium, Large, Jumbo};
    enum struct apple {Small, Medium, Big};

    egg choice = egg::Medium;
    apple pick = apple::Medium;
    ```
  - 不能进行隐式类型转化
    ```c++
    int choice2 = choice;         // Not allowed
    ```
    可使用：
    ```c++
    enum class:: short pizza {Small, Medium, Large, Jumbo};
    ```
## <span style="color:#ff8000;">运算符重载
  - 定义格式：Time operator +(const Time & t) const;
  - 使用：
    ```c++
    total = time1 + time2; 等效于 total = time1.operator+(time2);
    t4 = t1 + t2 + t3; 等效于 t4 = t1.operator+(t2.operator+(t3));
    ```
  - 重载后的运算符必须至少有一个操作数是用户定义的类型，使用运算符时不能违反运算符原来的句法规则
  - 不能创建新的运算符，某些运算符不能被重载
  - 大部分运算符可以通过成员函数或非成员函数重载，但 =, [], (), -> 只能通过成员函数重载
  - 如果使用A = 2 * B; 这不同于A = B * 2; 编译器不能使用成员函数调用来替换该表达式，可以使用友元函数或如下方式(非成员函数重载，但不访问私有数据)：
    ```c++
    Time operator*(int n, const Time & t) { return t * n; }
    ```
  - 为区分++运算符的前缀与后缀版本，c++将operator++()作为前缀版本，operator++(int)作为后缀版本，其中的参数不会用到
### <span style="color:#00ff00;">重载 [] 运算符
  - operator []( )将重载[]运算符
  - char & MyString::operator [] (int i) {return str[i]; }         // 返回char &类型，可以给特定元素赋值
  - 对于对象 const MyString answer("future"); 如果只有上述定义 cout << answer[1]; 将出错，因此需要定义一个const对象使用的operator[]定义：
  - const char MyString::operator [] (int i) const { return str[i]; }
## <span style="color:#ff8000;">友元函数
  - 允许非成员函数访问类的私有成员
  - 声明： friend Time operator*(int n, const Time & t ); // 类中声明，表明虽然有声明，但不是成员函数，不能使用成员运算符来调用，但具有成员函数访问权限
  - 定义： Time operator*(int n, const Time & t ) { ... }   // 不使用friend，不使用Time::限定符

  - A = 2 * B 将等效于 A = operator*(2, B);
  - 实现 cout << "time: " << time << endl; 调用，通过重载 << 运算符实现，若是通过成员函数重载 <<，则格式需要限定为：
    ```c++
    time << cout;
    ```
    可通过友元函数重载：
    ```c++
    ostream & operator<<(ostream & os, const Time & t) { os << ...; return os; }
    ```
  - 共同的友元：
    ```c++
    函数需要访问两个类的私有数据，可以将函数作为两个类的友元
    class Analyzer;         // forward declaration
    class Prob {
            friend void sync(Analyser & a, const Probe & p);        // sync a to p
            friend void sync(Probe & p, const Analyser & a);        // sync p to a
    };

    class Analyzer {
            friend void sync(Analyser & a, const Probe & p);        // sync a to p
            friend void sync(Probe & p, const Analyser & a);        // sync p to a
    };

    inline void sync(Analyser & a, const Probe & p) { ... }        // sync a to p
    inline void sync(Probe & p, const Analyser & a) { ... }        // sync p to a
    ```
## <span style="color:#ff8000;">类的强制类型转换函数
  - 将类对象转换为int / double等基本类型
    ```c++
    operator int();
    operator double();

    ```
    转换函数必须是类方法，不能有返回类型，不能有参数
    ```c++
    class Stonewt {
    private:
            double pounts;
    public:
            operator double () const;
    }
    ...
    Stonewt::operator double() const { return pounts; }
    ...

    Stonewt po{2.34};
    double height = (double)po;
    double weight = po;
    ```
    <br />
  - explicit关键字也可以用于转换函数，将禁止隐式转换：
    ```c++
    explicit operator double () const;

    double height = (double)po;         // allowed
    double weight = po;                 // not allowed
    ```
## <span style="color:#ff8000;">特殊成员函数
  - C++自动提供以下成员函数：
    ```c++
    默认构造函数
    默认析构函数
    复制构造函数
    赋值运算函数
    地址运算函数

    移动构造函数
    移动赋值运算符
    ```
## <span style="color:#ff8000;">复制构造函数
  - 用于将一个对象复制到新创建的对象中，只用于初始化
  - 当函数按值传递对象或返回对象时，都将调用复制构造函数
  - 通常的原型：Class_name(const Class_name &) { ... }
  - 功能：默认的复制构造函数逐个复制成员的值
    ```c++
    Stock food = Stock("Cabbage", 50, 1.25);

    Stock drink = food;
    Stock fruit(food);
    Stock milk = Stock(food);
    Stock *meat = new Stock(food);
    ```
### <span style="color:#00ff00;">在构造函数使用new分配内存的类中将存在问题：
  ```c++
  class MyString {
  private:
          char * str;
          ...
  public:
          MyString(const char * s);
          ~MyString();
          ...
  };

  MyString::MyString(const char * s) { str = new char[5]; ... } // new分配内存
  MyString::~MyString(delete[] str; ... )        // delete释放内存

  void func_val(MyString string) { ... } // 按值传递函数调用

  int main() { MyString st1("hello world"); func_val(st1); return 0; }
  ```
  - 按值传递的函数调用默认复制构造函数string.str = st1.str, 函数返回时调用string的析构函数会释放该内存，实际导致st1的内存被释放
  - 解决： 提供一个显示复制构造函数，尤其是类中有使用new分配内存的指针成员时，以复制指向的数据
  - 另外，可以声明一个private的复制构造函数，以便对于不希望有复制操作的类，便于追踪错误，这种情况将调用类的私有方法，产生错误
## <span style="color:#ff8000;">赋值运算函数
  - 将已有的对象赋值给另一个对象时将使用赋值运算符
  - 初始化对象时Stock drink = food; 将使用复制构造函数 （根据实现，也可能通过先创建临时变量，再调用赋值运算函数）

  - 通常的原型：Class_name & Class_name::operator = (const Class_name &)
  - 赋值运算符只能由类成员函数重载
  - 在调用 vegetable = food; 时，会有类似复制构造函数的问题，因此对于需要new分配空间的指针成员，显示定义赋值运算函数
  - 由于目标对象可能使用了以前分配的数据，所以应先使用delete[]来释放这些数据
  - 函数应避免将对象赋给自身
  - 函数返回一个调用对象的引用
    ```c++
    MyString & amp; MyString::operator = (const MyString & mstr) {
            if (this == & mstr)        // 根据地址判断是否相同
                    return * this;
            delete[] str;
            ...
            return * this;        
    }
    ```
## <span style="color:#ff8000;">移动语义 (移动构造函数 / 移动复制运算符)
  - 用于接管其他对象数据的所有权，如函数返回值创建的临时对象，避免重复分配内存
  - 如果定义了移动构造函数 / 移动赋值运算符，右值引用的参数将调用它们，否则调用复制构造函数 / 复制赋值运算符
  - 默认的移动构造函数与移动赋值运算符工作方式与复制版本类似
  - 在有移动语义之前对于：
    ```c++
    Object 1 = (Object2 + Object3);
    ```
    将首先创建返回值的临时对象，再调用复制构造函数，然后销毁临时对象
  - 在引入移动语义之后，一般编译器会进行优化，即使没有使用移动语义，也可以直接将所有权转移给Object1，消除额外的复制工作
### <span style="color:#00ff00;">移动构造函数
  - 要实现移动语义，可定义两个复制构造函数：
    ```c++
    一个是常规复制构造函数，使用const左值引用作为参数，关联到左值实参，执行深度复制
    另一个是移动构造函数，使用右值引用作为参数，关联到右值实参，让编译器知道不需要复制，只调整记录，参数不能为const
    ```
  - Foo.h
    ```c++
    class Foo {
    private:
            int n;
            char * pc;
    public:
            Foo();
            explicit Foo(int k);
            Foo(int k, char ch);
            Foo(const Foo & f);        // regular copy constructor
            Foo(const Foo && f);        // move constructor
            ~Foo();

            Foo operator +(conft Foo & f) const;
            ...
    };
    ```
  - Foo.c
    ```c++
    Foo::Foo() { n =0; pc = nullptr; }
    Foo::Foo(int k) : n(k) { pc = new char[n]; }
    Foo::Foo(int k, char ch) : n(k) {
            pc = new char[n];
            for (int i = 0; i < n; i++) { pc[i] = ch; }
    }
    Foo::Foo(const Foo & f) : n (f.n) {
            pc = new char[n];
            for (int i = 0; i < n; i++) { pc[i] = f.pc[i]; }
    }

    Foo::Foo(const Foo &&) : n (f.n) {
            pc = f.pc;
            f.pc = nullptr;        // give old object nullptr in return, delete[] cannot be used on same address twice.
            f.n = 0;
    }
    Foo::~Foo() { delete[] pc; }

    Foo Foo::operator +(conft Foo & f) const {
            Foo temp = Foo(n + f.n);
            for (int i = 0; i < n; i++) { temp[i] = pc[i]; }
            for (int i = f.n; i < temp.n; i++) { temp[i] = f.pc[i-n]; }
            return temp;
    }
    ```
  - test.c
    ```c++
    Foo one(10, 'x');
    Foo two(20, 'y');

    Foo three = one;        // calls copy constructor
    Foo four(one + two);        // calls operator+(), move constructor

    ```
### <span style="color:#00ff00;">移动赋值运算符
  - 同样使用右值引用，将源目标的所有权转让给目标，形参不能是const：
    ```c++
    Foo & Foo::operator =(const Foo & f) {        // copy assignment
            if (this == &f) { return * this; }
            delete[] pc;
            n = f.n;
            pc = new char[n];
            for (int i = 0; i < n; i++) { pc[i] = f.pc[i]; }
            return * this;
    }

    Foo & Foo::operator =(Foo && f) {                // move assignment
            if (this == &f) { return * this; }
            delete[] pc;
            n = f.n;
            pc = f.pc;
            f.pc = nullptr;
            f.n = 0;
            return * this;
    }
    ```
### <span style="color:#00ff00;">强制移动
  - 对于类似 Foo Object1 = Object2; 使用左值引用的场景，将调用复制构造函数，若要强制调用移动构造函数，可以使用：
    ```c++
    static_cast<> 运算符将对象强制转换成 Foo &&类型
    std::move() 函数，将左值引用转换为右值引用，在头文件utility中定义

    Foo Object1, Object2;
    Foo Object3 = Object1 + Object2;        // calls move construstor
    Object3 = Object1;                // copy assignment
    Object3 = Object1 + Object2;        // move assignment
    Object3 = std::move(Object1);        // forcced move assignment

    ```
    对于使用了move()的场景，如果没有定义右值引用对应的函数(移动赋值运算符)，将调用左值引用的函数(复制赋值运算符)
## <span style="color:#ff8000;">构造函数中使用new
  - 1.如果在构造函数中使用new来初始化指针成员，则应在析构函数中使用delete。
  - 2.new和delete必须相互兼容。new对应于delete，new[]对应于delete[]。
  - 3.如果有多个构造函数，则必须以相同的方式使用new，要么都带中括号，要么都不带。因为只有一个析构函数，因此所有的构造函数都必须与它兼容。可在一个构造函数中将指针初始化为空，这是因为delete可以用于空指针。
  - 4.应定义一个复制构造函数，通过深复制将一个对象初始化为另一个对象，具体地说，复制构造函数应分配足够的空间来存储复制的数据，并复制数据，而不仅仅是数据的地址。另外，还应该更新所有受影响的静态类成员
  - 5.应当定义一个赋值操作符，通过深复制将一个对象复制给另一个对象，具体地说，该方法应完成这些操作，检查自我赋值的情况，释放成员指针以前指向的内存，赋值数据而不仅仅是数据的地址，并返回一个指向调用对象的引用
## <span style="color:#ff8000;">成员初始化列表
  - 用于初始化函数中，对象创建时在函数体执行之前初始化参数
  - 对于const成员变量，只能使用这种方法来初始化，因为函数体执行时会创建const常量，此时无法再进行赋值操作
  - 对于被声明为引用的类成员也必须使用初始化列表语法
  - 初始化顺序按照成员声明顺序，而不是在初始化列表中的顺序
  - 对于类成员mem1初始化为val值, mem2为0，mem3为n*2：
    ```c++
    class_name::classname(type_name val) : mdata(val), mem2(0), mem3(n*2) { ... }
    ```
  - 示例：
    ```c++
    class Player {
    private:
            string firstName;
            string lastName;
            bool hasTable;
    public:
            Player(const string &fn = "none", const string &ln = "none", bool ht = false);
            ...;
    };
    ...

    Player::Player(const string &fn, const string &ln, bool ht) : firstName(fn), lastName(ln), hasTable(ht) { ... }
    ```
  - 类内初始化：
    ```c++
    class Session {
            int mem1 = 10;
            double mem2 {19.66};
            short mem3;
    public:
            Session() {}        // #1
            Session(short s) : mems(s) {}        // #2
            Session(int n, double d, short s) : mem1(n), mem2(d), mem3(s) {}        // #3
    ...
    };

    使用成员初始化列表的构造函数将使用列表值覆盖这些默认初始值，因此第三个构造函数覆盖了类内成员初始化值
    ```
## <span style="color:#ff8000;">委托构造函数 (一个构造函数的定义中使用另一个构造函数)
  - 如果类定义了多个构造函数，可以在一个构造函数的定义中使用另一个构造函数，可用于成员初始化列表：
    ```c++
    class Notes {
    private:
            int n;
            double x;
            string s;
    public:
            Notes();
            Notes(int k);
            Notes(int k, double y);
            Notes(int k, double y, string t);
    ...
    };

    Notes::Notes(int k, double y, string t) : n(k), x(y), s(t) { ... }
    Notes() : Notes(0, 0.01, "Ah") { ... }
    Notes(int k) : Notes(k, 0.01, "Ah") { ... }
    Notes(int k, double y) : Notes(k, y, "Ah") { ... }
    ```
## <span style="color:#ff8000;">默认的方法default / 禁用的方法 delete
  - default用于声明特殊成员函数的默认版本，如在提供了构造函数的情况下，自动生成默认的构造函数，只能用于6个特殊成员函数：
    ```c++
    Class Someclass {
    public:
            Someclass(Someclass &&);
            Someclass() = default;        // use default constructor
            Someclass(const Someclass &) = default;
            Someclass & operator =(const Someclass &) = default;
    ...
    };
    ```
  - delete用于禁止编译器使用特定方法，只用于查找匹配函数，使用它们将导致编译错误，如禁止复制对象，禁止特定的转换：
    ```c++
    Class Someclass {
    public:
            Someclass() = default;        // use default constructor
            Someclass(const Someclass &) = delete;        // disable copy constructor
            Someclass & operator =(const Someclass &) = delete;        // disable copy assignment operator
            Someclass(Someclass &&) = default;        // use default move constructor
            Someclass & operator =(Someclass &&) = default;        // use default move assignment operator
    ...
            void redo(double);        
            void redo(int) = delete;        // refusing int parameter, not allowed to convert to double value
    };

    如果在启用移动方法的同时禁用复制方法，因为移动引用的操作智能关联到右值，以下：
            Someclass one, two;
            Someclass three(one);        // not allowed, one is a lvalue
            Someclass four(one + two);        // allowed
    ```
***

# <span style="color:#ff0000;">类继承 （公有继承）
  - 派生类 ----> is-a ----> 基类
  - 派生类对象存储了基类的数据成员，可以使用基类的方法
  - 基类的私有成员也将成为派生类的一部分，但只能通过基类的公有和保护方法访问
  - 派生类中的成员名称优先于直接或间接继承来的同名名称
  - 如果派生类没有重新定义函数，将使用该函数的基类版本
## <span style="color:#ff8000;">声明
  ```c++
  class RatedPlayer : public Player { // 公有派生
  private:
      unsigned int rating;        // add a new data member
  public:
      RatedPlayer(unsigned int r = 0, const string &fn = "none", const string &ln = "none", bool ht = false);
      RatedPlayer(unsigned int r, const Player &pl);
      unsigned int Rating() const { return rating; }         // add a new method

  };
  ```
## <span style="color:#ff8000;">派生类的构造函数 / 析构函数
  - 派生类需要自己的构造函数，可以根据需要添加额外的数据成员和成员函数，必须给新成员和继承的成员提供数据
  - 创建派生类对象时首先创建基类，基类对象应当在进入派生类构造函数之前创建，C++使用成员初始化列表语法来完成：
    ```c++
    RatedPlayer::RatedPlayer(undigned int r, const string &fn, const string &ln, bool ht)
            : Player(fn, ln, ht) { rating = r; }        // 除非使用默认构造函数，否则应显示调用基类构造函数
    RatedPlayer::RatedPlayer(unsigned int r, const Player &pl)
            : Player(pl), rating(r) {}                // 调用基类的复制构造函数，若没有定义将使用隐式的成员复制构造函数
    ```
  - 派生类对象过期时，将首先调用派生类的析构函数，然后调用基类的析构函数
    ```c++
    Player::~Player() { delete[] name;}
    RatedPlayer::~RatedPlayer() {delete[] style;}
    ```
## <span style="color:#ff8000;">基类 / 派生类 指针 / 引用
  - 基类 指针 / 引用 可以在不进行显示类型转换的情况下 指向 / 引用 派生类对象，基类 指针 / 引用只能用于调用基类方法 （向上强制转换upcasting）
  - 基类引用定义的函数或指针参数可用于基类对象或派生类对象
  - 派生类对象可用于基类对象初始化，或将派生类对象赋给基类对象：
    ```c++
    RatedPlatyer olaf1(1840, "Olaf", "Loaf", true);
    Player olaf2(olaf1);        // 调用基类的复制构造函数
    Player winner; winner = olaf1;        // 调用基类的赋值运算符
    ```
## <span style="color:#ff8000;">派生类的复制构造函数 / 重载赋值运算符
  - 如果派生类中使用了动态内存分配，派生类的复制构造函数需要显示调用基类的复制构造函数：
    ```c++
    RatedPlayer::RatedPlayer(const RatedPlayer & rp)
            :Player(rp) { ... }
    ```
  - 对于重载赋值运算符有同样要求：
    ```c++
    RatedPlayer & RatedPlayer::operator = (const RatedPlaer & rp) {
            if (this == & rp)
                    return * this;
            Player::operator=(rp);         // or * this = rp; recursive invocation
            ...
            return * this;
    }
    ```
## <span style="color:#ff8000;">派生类访问基类的友元函数
  - 基类友元函数：
    ```c++
    class Player {
    public:
            friend std::ostream & operator << (std::ostream & os, const Player & pl);
    ...
    };

    std::ostream & operator << (std::ostream & os, const Player & pl) {
            ... ;
            return &os;
    }

    ```
    派生类友元函数：
    ```c++
    class RatedPlayer : public Player {
    public:
            friend std::ostream & operator << (std::ostream & os, const RatedPlayer & rp);
    ...
    };


    std::ostream & operator << (std::ostream & os, const RatedPlayer & rp) {
            os << (const Player &)rp;
            ... ;
            return &os;
    }
    ```
  - 友元不是成员函数，不能通过作用域解析运算符调用，解决方法是使用强制类型转换，以在匹配远行时选择正确的函数
  - 另一种更好的强制类型转换方法：os << dynamic_cast<const Player &> (rp);
## <span style="color:#ff8000;">多态 （多态公有继承）
  - 多态公有继承：同一个方法在基类与派生类中行为是不同的，取决于调用该方法的对象
  - 实现机制：
    ```c++
    在派生类中重新定义基类的方法
    使用虚方法
    ```
  - class Player { ..., void viewAcct() const; };
  - class RatedPlayer { ..., void viewAcct() const; };

  - void Retedplayer::viewAcct() const {Player::viewAcct(); ... }
  - 重新定义继承的方法不是重载，如果在派生类中重新定义函数，将隐藏同名的基类函数，不管参数特征标如何：
    ```c++
    如果重新定义继承方法，应确保与原来的原型完全相同，但如果返回类型是基类引用或指针，可以修改为指向派生类的 （返回类型协变）
    如果基类声明被重载了，应在派生类中重新定义所有的基类版本

    class Player {
    public:
            virtual void func() const;
            virtual void func(int n) const;
            virtual void func(double d) const;
    ...
    };

    class Player {
    public:
            virtual void func() const;
            virtual void func(int n) const;
            virtual void func(double d) const;
    ...
    };
    如果只定义了一个方法，其他两个将被隐藏，派生类对象无法使用它们

    ```
    可以在派生类中将基类的公有方法在private中声明，以隐藏该方法
## <span style="color:#ff8000;">虚方法
  - 关键字virtual，只用于类声明的方法原型中
  - 当方法是通过引用 / 指针而不是对象调用时：
    ```c++
    如果没有使用virtual，程序将根据引用/指针类型来选择方法；
    如果使用了virtual，将根据引用或指针指向的对象的类型来选择方法

    对于：
    Player pl1( ... );
    RatedPlayer pl2( ... );
    Player &pl_ref1 = pl1;
    Player &pl_ref2 = pl2;

    如果方法 viewAcct() 不是virtual:
    pl_ref1.viewAcct();         // using Player::viewAcct()
    pl_ref2.viewAcct();         // using Player::viewAcct()

    如果方法 viewAcct() 是virtual:
    pl_ref1.viewAcct();         // using Player::viewAcct()
    pl_ref2.viewAcct();         // using RatedPlayer::viewAcct()
    ```
  - 经常在基类中将派生类会重新定义的方法声明为虚方法，在派生类中将自动成为虚方法，在派生类声明中使用关键字virtual来指出哪些函数是虚函数
  - 为基类声明一个虚析构函数，确保释放派生对象时按正确的顺序调用析构函数
  - 构造函数与友元函数不能是需函数，析构函数应当是虚函数
  - 通常应为基类提供一个虚析构函数，即使并不需要
  - 如果需要，可以在友元函数中调用虚方法来解决
### <span style="color:#00ff00;">动态联编
  - 函数名联编：将源代码中的函数调用解释为执行特定的函数代码块
  - 静态联编：在编译过程中进行联编，效率更高，非虚方法使用
  - 动态联编：程序运行时选择代码块，因为对于虚方法，在编译时使用哪个函数是不确定的
### <span style="color:#00ff00;">虚函数工作原理
  - 对于每个有虚方法定义的基类，新增一个隐藏成员，该成员保存一个指向函数地址数组的指针，这种数组称为虚函数表，存储了类声明的虚函数地址，
  - 在派生类继承时，也将创建另一个隐藏成员：
    ```c++
    如果没有重新定义虚函数，数组中将保存基类的虚函数地址
    如果重新定义了虚函数，数组中将保存新定义虚函数的地址
    如果新增虚函数，数组中将添加新的虚函数地址
    ```
    因此使用虚函数将：
    ```c++
    为每个对象新增一个成员
    为每个类新建一个虚函数表
    为每一测虚函数调用新增一次查找地址操作
    ```
## <span style="color:#ff8000;">访问控制 protected
  - 派生类成员可以直接访问基类的保护成员，类外只能通过公有类成员来访问protected部分中的类成员
  - 对于数据成员最好还是使用private，不要使用protected
  - 对于成员函数，使派生类能访问外部不能使用的函数
## <span style="color:#ff8000;">抽象基类 abstract base class， ABC
  - 当类声明中包含纯虚函数时该类将称为抽象基类
  - 不能创建该类的对象，只能用作基类
  - 纯虚函数 pure virtual function： 声明结尾处为=0，提供未实现的函数
    ```c++
    virtual double Area() const = 0;
    在类中可以不定义该函数，也可以在实现文件中提供方法的定义
    ```
## <span style="color:#ff8000;">继承构造函数 (using)
  - using 可以用于让名称空间中的函数可用：
    ```c++
    namespace Box {
            int fn(int) { ... }
            double fn(double) { ... }
    ...
    };
    using Box::fn;
    使fn的所有重载版本都可用
    ```
  - 也可以用于派生类中使用基类方法：
    ```c++
    class C1 {
    public:
            int fn(int) { ... }
            double fn(double) { ... }
    ...
    };
    class C2 {
    public:
            using C1::fn;
            double fn(double) { ... }
    ...
    };

    C2 c2;
    int k = c2.fn(3);        // uses C1::fn(int)
    double d = c2.fn(2.4);        // uses C2::fn(double)
    ```
  - 这种方法也可以用于构造函数，让派生类继承基类的所有构造函数(默认构造函数 / 复制构造函数 / 移动构造函数除外)：
    ```c++
    calss BS {
            int q;
            double w;
    public:
            BS() : q(0), w(0) {}
            BS(int k) : q(k), w(0) {}
            BS(double x) : q(0), w(x) {}
            BS(int k, double x) : q(k), w(x) {}
    ...
    };

    class DR : public BS {
            short j;
    public:
            using BS::BS;
            DR() : j(-1) {}        // DR needs its own default constructor
            DR(double x) : BS(2*x), j(int(x)) {} // [ ??? ]
            DR(int i) : j(-1), BS(i, 0.5*i) {}
    ...
    };

    int main {
            DR o1;        // use DR()
            DR o2(18.81);        // use DR(double) instead of BS(double)
            DR o3(10, 1.8);        // use BS(int, double)
    ...
    }

    优先使用与派生类构造函数特征标匹配的构造函数
    ```
## <span style="color:#ff8000;">虚方法 override / final
  - 派生类中的同名方法，如果特征标不同，将隐藏基类中所有的同名方法，使其在派生类中不可用
  - override用于在派生类中指定覆盖基类中的虚方法，如果特征标不同，将引起编译错误：
    ```c++
    class Action {
    public:
            virtual void f(char) const { ... }
    ...
    };
    class Bingo : public Action {
    public:
            virtual void f(char) const override { ... }        // over ride base version
            virtual void f(char *) const override { ... }        // compile error
    ...
    };
    ```
  - final 用于禁止派生类覆盖特定的虚方法：
    ```c++
    virtual void f(char) const final { ... }        // forbid re-define of f()
    ```
***

# <span style="color:#ff0000;">保护继承
## <span style="color:#ff8000;">私有继承
  - 使用私有继承，基类的公有与保护 成员 / 方法 将成为派生类的私有成员 / 方法
  - 获得实现，但不获得接口， 实现 has-a 关系
  - private是默认值，因此省略访问限定符将导致私有继承
  - class Student : private std::string, private std::valarray<double> { ... };
  - 这里的私有继承为类提供了两个无名的子对象成员，因此使用私有继承时，将使用类名与作用域解析运算符来调用方法
  - 与在类声明中直接包含子类 class Student {private: std::string name; ... }; 对比：
    ```c++
    私有继承只能使用一个子类的对象
    私有继承可以访问原有类的保护成员
    私有继承可以重新定义基类的虚函数
    ```
  - 构造函数：
    ```c++
    class Student : private std::string, private std::valarray<double> {
    private:
            typedef std::valarray<double> ArrayDb;
    public:
            Student() : std::string("Null name"), ArrayDb() {}
            // explicit 避免接受一个参数的构造函数将对象初始化为一个值
            ecplicit Student(const std::string & str) : std::string(std), ArrayDb() {}
            ecplicit Student(int n) : std::string("Null name"), ArrayDb(n) {}
            Student(const char * str, const double *pd, int n)
                    : std::string(str), ArrayDb(pd, n) {}        // 使用类名而不是成员名
    ... };
    ```
  - 访问基类方法：
    ```c++
    double Student::Averrage() const {
            if (ArrayDb::size() > 0)
                    return ArrayDb::sum() / ArrayDb::size();
            else
                    return 0;
    }
    ```
  - 访问基类对象：
    ```c++
    const string & Student::Name() const {
            return (std::string &) * this;
    }
    通过强制类型转换，将this指针转换为基类对象
    ```
  - 访问基类友元函数：
    ```c++
    std::ostream & os operator << (std::ostream & os, const Student & st) {
            os << "Name is: " << (const std::string &) st << "\n";
            ...
            return os;
    }
    ```
## <span style="color:#ff8000;">保护继承
  - 基类的保护与公有成员 / 方法都将称为派生类的保护成员 / 方法
  - 当从派生类派生出另一个类时，使用私有继承时，将不能使用基类的方法，因为基类的方法成为了私有方法，使用保护继承时将继续可以使用
## <span style="color:#ff8000;">using 重新定义访问权限
  - 使用保护派生或私有派生时，要让基类的方法在派生类外面可用

  - 方法一，定义一个使用基类方法的派生类方法：
    ```c++
    double Student::sum() const { return std::valarray<double>::sum(); }

    ```
    方法二，使用using声明指出派生类可以使用基类的特定成员：
    ```c++
    class Student : private std::string : private std::valarray<double> {
    public:
            using std::valarray<double>::max;        // 没有括号 / 函数特征标 / 返回类型
            using std::valarray<double>::min;
    };
    ```
***

# <span style="color:#ff0000;">多重继承 MI
  - 主要问题：
    ```c++
    从两个不同的类继承同名方法
    从两个或更多相关基类继承同一个类的多个实例
    ```
## <span style="color:#ff8000;">一般继承关系
  ```c++
  class Worker {
  public:
      void show() const;
  ...};

  class Singer : public Worker {
  public:
      void show() const;
  ...};

  class Writer : public Worker {
  public:
      void show() const;
  ...};
  ```
  - 派生类 class SingerWriter : public Singer, public Writer { ... }; 将存在两个Worker对象，此时需要通过类型转换指定使用的Worker：
    ```c++
    SingerWriter sw;
    Worker * ps = ( Singer * ) &sw;
    Worker * pw = ( Writer * ) &sw;
    但这种方法将使对象包含两个Worker对象
    ```
## <span style="color:#ff8000;">虚基类
  - 关键字virtual （与虚函数无关，只为了不填加新的关键字）
  - 虚基类使从多个类派生出的对象只继承一个基类对象
    ```c++
    class Worker { ... };
    class Singer : virtual public Worker { ... };
    class Writer : public virtua Worker { ... };

    派生类 class SingerWriter : public Singer, public Writer { ... };
    继承的Singer与Writer对象共享一个Worker对象，因此可以使用多态
    ```
  - 使用虚基类的派生类需要显示调用基类的构造函数：
    ```c++
    SingerWriter(const Worker &wk, int p = 0, int v = Singer::other)
            : Worker(wk), Writer(wk, p), Singer(wk, v) {}
    若使用：
    SingerWriter(const Worker &wk, int p = 0, int v = Singer::other)
            : Writer(wk, p), Singer(wk, v) {}
    将不能使用wk初始化Worker，而是会调用Worker的默认构造函数
    ```
  - 混合使用virtual：
    ```c++
    class Singer : virtual public Worker { ... };
    class Dancer : virtual public Worker { ... };
    class Writer : public Worker { ... };
    class Speaker : public Worker { ... };

    派生类 class MutiWorker : public Singer, public Dancer, public Writer, public Speaker { ... }; 将包含三个Worker类子对象
    ```
## <span style="color:#ff8000;">多重继承中的同名方法
  - 可以使用作用域解析运算符指定使用哪个方法：
    ```c++
    SingerWriter newhire( ... );
    newhire.Singer::show();
    ```
    更好的方法是在SingerWriter中重新定义新的同名方法，并指明使用哪个版本的方法：
    ```c++
    void SingerWriter::show() const {
            Singer::show();
            Writer::show();
            ...
    }
    ```
***

# <span style="color:#ff0000;">友元类
  - 定义一种关系，既不是公有继承的is-a关系，也不是包含或私有/保护继承的has-a关系，比如电视机 + 遥控器的关系
  - 友元类的所有方法可以访问原始类的所有私有和保护成员
  - 友元声明可以位于公有 / 私有 / 保护部分
## <span style="color:#ff8000;">使用示例
  ```c++
  class Tv {
  public:
          friend class Remote;
          enum {TV, DVD};
          bool volup();
  ...
  };

  // friend class refering Tv class, should after the declaration of Tv.
  class Remote {
  private:
          int mode;
  public:
          Remote(int m = Tv::TV) : mode(m) {}
          bool volup(Tv & t) { return t.volup(); }         // using Tv function.
  ...
  };

  int main() {
          Tv s42;
          s42.volup();

          Remote white;
          white.volup(s42);

          Tv s58;
          white.volup(s58);
  ...
  }
  ```
## <span style="color:#ff8000;">友元成员函数
  - 将特定的类成员成为另一个类的友元，而不必让整个类成为友元
  - 使用其他类成员函数作为友元的类，需要在声明之前已经有改函数的声明，因此需要注意声明的顺序
    ```c++
    class Tv;         // forwad declaration

    class Remote {
    private:
            int mode;
    public:
            enum {TV, DVD};
            Remote(int m = TV) : mode(m) {}
            boll volup(Tv & t);        // can not call Tv public function here.
            void set_chan(Tv & t, int chan);        // declaration of the funtion will be friend.
    ...
    };

    class Tv {
    public:
            friend void Remote::set_chan(Tv & t, int chan);         // friend function
            enum {TV, DVD};
            bool volup();
    private:
            int channel;
    ...
    };

    // Remote methods as inline function
    inline bool Remote::volup(Tv & t) { return t.volup(); }         // Tv::volup() function has been declared, and calling only public method.
    inline void Remote::set_chan(Tv & t, int chan) { t.channel = chan; } // friend function using private member of Tv
    ...
    ```
***

# <span style="color:#ff0000;">异常
  - c++异常是对程序运行过程中发生的异常情况的一种响应，通过throw 值/对象 来表达异常
  - 当异常发生时程序的默认处理方式是程序终止
  - 异常发生时，如果当前函数不能捕获该异常，将继续向上层调用者传递
  - 引发异常时，编译器总是创建一个临时拷贝，即使catch块中指定的是引用，因为函数结束时将释放自动变量，导致上层调用者无法引用到异常发生时创建的变量
  - 在catch快中仍使用引用，使基类引用可以匹配派生类对象
## <span style="color:#ff8000;">try / catch / throw语法
  - 可以定义一个异常类，用来区分不同的异常：
    ```c++
    class bad_hmean {
    private:
            double a;
    public:
            bad_hmean(double v1 = 0) : a(v1) {}
            void msg();
    ...
    };
    ```
    <br />
  - 在可能出现异常的代码上使用try - catch关键字：
    ```c++
    int main() {
            try {
                    foo1();
                    foo2();
            }
            catch(bad_hmean & bh) {        // exception type, could be class type
                    bh.msg();                 // exception handle method.<br />                        ...
            }
            catch(const char * s) {
                    ...
            }
            catch (...) {                        // catch all
                    ...
            }
            ...
    }

    使用catch(...) 代表捕捉所有异常，该捕捉方式必须放在所有捕捉之后，越宽泛的捕捉方式后放
    ```
  - throw引发异常：
    ```c++
    foo1() {
            if (a == -b)
                    throw "bad arguments: a == -b.";        // throw exception, match catch(const char * s)
            ...
    }
    foo2() {
            if (a < 0)
                    throw bad_hmean(a);
            try {
                    foo1();
            } catch(const char * s) {
                    ...
                    throw;                // re-throws excption to up caller
            }
            ...
    }
    ```
## <span style="color:#ff8000;">异常规范（c++11建议不使用）
  - 告知该函数可能发生哪些异常：
    ```c++
    void foo();                 // may throw any type exception 该函数可能抛出任何异常*/
    void foo() throw (int, double, const char*);        // may throw int / double / const char* exception
    void foo() throw();         // doesn't throw an excption
    void foo() noexcept;         // doesn't throw an excption
    ```
  - 在带异常规范的函数中引发的异常需要跟规范中某一类型匹配，否则将引起程序异常终止
  - set_unexpected()函数用于修改默认行为
## <span style="color:#ff8000;">栈解退
  - 函数调用跳转时，程序将调用函数的指令地址放入栈中，被调用函数执行完毕时从该地址继续执行，并释放其自动变量，如果是类，则调用析构函数。
  - 函数出现异常终止时，程序也会释放栈中内存，但不会在释放栈的第一个返回地址后停止，而是继续释放栈，直到找到一个位于try块中的返回地址。
  - 之后函数将跳转到块尾的异常处理程序，这个过程称为栈解退
  - 以下程序中：
    ```c++
    void test1(int n) {
            double * ar = new double(n);
            ...
            if (oh_no)
                    throw exception();
            ...
            delete[] ar;
            return;
    }
    栈解退时将删除ar，且无法调用末尾的delete[]，导致内存泄漏
    ```
    未避免这种内存泄漏，可以捕获该异常，并在catch中包含一些清理代码：
    ```c++
    void test2(int n) {
            double * ar = new double(n);
            ...
            try {
                    if (oh_no)
                            throw exception();
            } catch (exception &ex) {
                    delete[] ar;
                    throw;
            }
            ...
            delete[] ar;
            return;
    }
    ```
    或使用智能指针模板
## <span style="color:#ff8000;">exception类
  - \#include <exception>
  - 其他异常可以使用exception类作为基类，虚方法what()返回一个字符串，派生类可以重新定义它
    ```c++
    class bad_hmean : public std::exception {
    public:
            const char * what() { return "bad argument for hmean."; }
    ...
    };
    ```
### <span style="color:#00ff00;">stdexcept异常类
  - \#include <stdexcept>
  - 定义了logic_error类和runtime_error类，exception类派生而来，用作两类异常的基类
  - 这两个类都定义了接受一个string对象的构造方法，用于what()方法返回的字符串
    ```c++
    class logic_error : public exception {
    public:
            explicit logic_error(const string * what_arg);
            ...
    };
    ```
  - logic_error：
    ```c++
    描述逻辑错误，通过合理的变成可以避免这种错误
    domain_error： 参数不在定义域范围内
    invalid_argument：参数值无效
    length_error：没有空间执行所需操作，如string.append()
    out_of_bounds：索引错误，如operator[]方法
    ```
  - runtime_error：
    ```c++
    运行期间难以预计的错误
    range_error
    overflow_error
    underflow_error
    ```
  - 用例：
    ```c++
    try { ... }
    catch(out_of_bounds & oe) { count << oe.what() << endl; }
    catch(logic_error & le) { count << le.what() << endl; }
    catch(exception & e) { count << e.what() << endl; }
    ```
### <span style="color:#00ff00;">bad_alloc异常和new
  - 使用new导致的内存分配失败问题，c++最新处理方式是引发bad_alloc异常
    ```c++
    try {
            Big * pb = new Big(1000);
            ...
    } catch(bad_alloc * ba) {
            cout << ba.what() << endl;
            ...
    }
    ```
  - 很多代码都是针对new失败返回空指针设计的，为此提供了一个失败时返回空指针的new：
    ```c++
    int * pa = new(std::nothrow)int[5000];
    ```
### <span style="color:#00ff00;">自定义异常类继承exception类
  ```c++
  class Sales {
  public:
          class bad_index : public logic_error {
          private:
                  int ix;
          public:
                  explicit bad_index(int n, const string & s = "Index error in Sales.\n");
                  virtual ~bad_index() {}
          };
          virtual double & operator[](int i);
          virtual double operator[](int i) const;
  ...
  };

  Sales::bad_index::bad_index(int n, const string & s) : logic_error(s), ix(n) {}

  double & Sales::operator[](int i) {
          if (i < 0 || i > MAX)
                  throw bad_index(i);
          ...
  }
  ```
## <span style="color:#ff8000;">未捕获异常
  - 未捕获异常不会导致程序立刻终止，将调用函数terminate()，默认情况下terminate()函数调用abort()
  - 可通过set_terminate()函数（头文件exception）修改terminate()默认行为：
    ```c++
    typedef void (*terminate_handler)();
    terminate_handler set_terminate(terminate_handler f) no_except;
    ```
  - 使用：
    ```c++
    void myQuit() {
            cout << "Terminate due to unexpected exception!\n"
            exit(5);
    }

    set_exception(myQuit);
    ```
***

# <span style="color:#ff0000;"> 文件输入 / 输出
## <span style="color:#ff8000;">类继承关系
  - streambuf类为缓冲区提供了内存，并提供了用于填充缓冲区 / 访问缓冲区内容/ 刷新缓冲区 / 管理缓冲区内存的类方法
  - ios_base类表示流的一般特征，如是否可读取，是二进制流还是文本流等
  - ios类基于ios_base类，并包括一个指向streambuf对象的指针成员
  - ostream类派生自ios类，提供输出方法
  - istream类也是派生自ios类，提供输入方法
  - iostream类基于ostream和istream类，因此继承了输入和输出方法
  - wistream / wostream类都是wchar_t的具体化，wcout对象用于输出宽字符流

  - 文件使用ifstream / ofstream类，在头文件fstream中定义，派生自iostream类
  - fstream类用于同步文件I/O

  - 字符串流ostringstream派生自ostream类，istringstream派生自istream类，在头文件sstream中定义
  - cin cout: #include <iostream>
    ```c++
    istream outstream
    ```
    字符串流：#include <sstream>
    ```c++
    istringstream ostringstream
    ```
    文件流：#include <fstream>
    ```c++
    ifstream ofstream
    ```
## <span style="color:#ff8000;">基本的文件 I/O 操作：
  - 创建ofstream对象来管理输出流：
    ```c++
    #include <fstream>
    ofstream outFile;        // create object for output

    ```
    将对象与特定文件关联：
    ```c++
    outFile.open("foo");        //associate with a file
    ofstream fout2("foo");        // create & associate

    ```
    使用iostream方法来使用该对象，ostream是ofstream的基类，因此可以使用所有ostream类方法
    ```c++
    fout << "bar";

    ```
    对象过期时，会自动关闭文件连接，也可以显示关闭
    ```c++
    outFile.close();
    ```
  - ifstream读文件：
    ```c++
    ifstream fin;
    fin.open("foo");

    char ch;
    fin >> ch;

    string str;
    fin.getline(str, 80);

    fin.close();
    ```
  - fstream用于读写：
    ```c++
    ftream finout;
    finout.open("foo", ios_base::in | ios_base::out);
    ```
## <span style="color:#ff8000;">流状态检查
  - 打开一个不存在的文件，将设置failbit位
  - 检查open是否成功：
    ```
    fin.open("foo");
    if (fin.fail()) { ... }        // open attempt failed
    if (!fin.good()) { ... }        // open attempt failed
    if (!fin) { ... }                // open attempt failed

    if (!fin.is_open()) { ... }         // open attempt failed
    is_open()方法能够检测以不合适的文件模式打开文件等错误
    ```
## <span style="color:#ff8000;">文件模式
  - openmode 描述读 / 写 / 追加等文件使用方式，bitmask类型
    ```c++
    ios_base::in         打开文件用于读取
    ios_base::out         打开文件用于写入
    ios_base::ate         打开文件，并将文件指针移到文件末尾
    ios_base::app         追加方式打开文件，值允许在文件末尾添加内容
    ios_base::trunc         如果文件存在，则截短文件，即删除文件原有内容
    ios_base::binary         二进制文件
    ```
  - ifstream.open() 默认使用ios_base::in
  - ofstream.open() 默认使用ios_base::out | ios_base::trunc  :=(
## <span style="color:#ff8000;">read / write
  ```c++
  struct planet {
          char name[20];
          double population;
          double g;
  };

  ...
  planet pl;
  ofstream fout;
  fout.open("foo", ios_base::out | ios_base::app | ios_base::binary);
  fout.write((char *)&pl, sizeof pl);
  ...
  ifstream fin;
  fin.open("foo", ios_base::in | ios_base::binary);
  fin.read((char *)&pl, sizeof pl);
  ```
## <span style="color:#ff8000;">随机存储 seekg / seekp / tellg / tellp
  - seekg() 用于输入指针
  - seekp() 用于输出指针
  - seekg() 原型：
    ```c++
    basic_stream<char T, traits> &seekg(off_type, ios_base::seekdir);
    basic_stream<char T, traits> &seekg(pos_type);

    对于char类型，等同于
    istream &seekg(streamoff, ios_base::seekdir);
    istream &seekg(streampos);
    ```
  - seekg()使用：
    ```c++
    fin.seekg(30, ios_base::beg);        // 30 bytes beyond the beginning
    fin.seekg(-1, ios_base::cur);        // backup one bytes
    fin.seekg(0, ios_base::end);        // go to the end of the file

    fin.seekg(112);        // 112 bytes beyond the beginning
    ```
  - tellg() / tellp()
    ```c++
    tellg() 返回输入流文件指针当前位置
    tellp() 返回输出流文件指针当前位置

    对于fstream对象，两个指针相同
    ```
***

# <span style="color:#ff0000;">类模板
## <span style="color:#ff8000;">声明
  ```c++
  template <typename Type> // or olde version: template <class Type>
  class Stack {
  private:
          enum{ MAX = 10};
          Type item[MAX];
          int top;
  public:
          Stack();
          bool isempty();
          bool push(const Type & item);
  ...
  };

  template <typename Type>
  Stack<Type>::Stack() { top = 0; }

  template <typename Type>
  bool Stack<Type>::isempty() { return top == 0; }

  template <typename Type>
  bool Stack<Type>::push(const Type & item) { ... }
  ```
  - 每个模板成员函数都使用相同的模板声明打头，并使用泛型名Type替代类中使用的类型名，类限定名改为Stack<Type>
  - 模板不是类的成员，只是C++的编译指令，说明如何生成类和函数定义，因此不能将模板函数实现放在独立的实现文件中
  - 模板函数不能单独编译，必须与特定的模板实例化请求一起使用
  - 模板常用作容器类
  - 必须显示地提供所需的类型，这与函数模板是不同的：
    ```c++
    Stack<std::string> st;
    Stack<const char *> stc;
    ```
## <span style="color:#ff8000;">非类型参数 （表达式参数）
  - template <class T, int n>         // int n称为表达式参数
  - class StackTP { ... };

  - StackTP<double, 12> stp;
  - 表达式参数可以是整型，枚举，指针或引用，因此不能使用 double m, 但可以使用 double * pm 或 double & rm
  - 实例化模板时，表达式参数必须是常量表达式
  - 每种表达式参数的不同值都将生成自己的模板：
    ```c++
    StackTP<double, 12> stp12;
    StackTP<double, 13> stp13;
    将生成两个独立的类声明
    ```
## <span style="color:#ff8000;">模板的多功能性
### <span style="color:#00ff00;">模板用作基类或其他模板的类型参数
  ```c++
  template <typename Type>
  class ArrayTP { ... };

  template <typename Type>
  class GrowArrayTP : ArrayTP<Type> { ... };

  template <typename Type>
  class StackTP {
  Private:
          ArrayTP<Type> ap;
  ...
  };

  ArrayTP< StackTP<int> > asi;
  ```
### <span style="color:#00ff00;">递归调用
  ```c++
  StackTP< StackTP<int, 5>, 10 > stp_squ;
  相当于二维数组： int stp_squ[10][5];
  ```
### <span style="color:#00ff00;">多个模板参数
  ```c++
  template <typename T1, typename T2>
  class PairTP {
  private:
          T1 a;
          T2 b;
  public:
          T1 & first();
          T2 & second();
          T1 first() const;
          T2 second() const;
          PairTP(const T1 & aval, const T2 &bval) : a(aval), b(bval) {}
          PairTP();
  };

  template <typename T1, typename T2>
  T1 & PairTP<T1, T2> :: first() { return a; }

  template <typename T1, typename T2>
  T2 & PairTP<T1, T2> :: second() { return b; }

  int main() {
          PairTP<string, int> rating[3] = {
                  PairTP<string, int> ("hello", 3),
                  PairTP<string, int> ("aloha", 4),
                  PairTP<string, int> ("hola", 5)
          };
          int joins = sizeof(rating) / sizeof(PairTP<string, int>);
          ...
          return 0;
  }
  ```
### <span style="color:#00ff00;">默认类型模板参数
  - 为类型参数提供默认值
    ```c++
    template<typename T1, typename T2 = int> class Topo { ... };
    ```
    类似的也可以为非类型参数提供默认值
## <span style="color:#ff8000;">模板的具体化
  - 如果有多个模板可以选择，编译器将选择具体化程度最高的版本
### <span style="color:#00ff00;">隐式实例化
  - 最常用 StackTP<double, 12> stp;
  - 编译器在需要对象之前不会生成类的隐式实例化声明
### <span style="color:#00ff00;">显示实例化
  - template class StackTP<string, 100>;
  - 在需要对象之前，编译器根据模板生成类的声明
### <span style="color:#00ff00;">显示具体化
  - template<> class StackTP<const char * , 100> { ... }
  - 对于特定类型的定义，不使用模板生成，如定义对于char * 类型的参数使用strcmp进行比较
### <span style="color:#00ff00;">部分具体化
    - 对于 template <typename T1, typename T2> class PairTP { ... };
    - 声明 template<T1> class PairTP<T1, int> { ... }；
    - 也可以提供指针特殊版本部分具体化现有模板：
      ```c++
      template <typename T> class Stack { ... };
      template <typename T*> class Stack { ... };        
      ```
    - 其他应用：
      ```c++
      template <class T1, class T2, class T3> class Trino { ... };
      template <class T1, class T2> class Trino<T1, T2, T2> { ... };
      template <class T1> class Trino<T1, T1*, T1*> { ... };
      ```
## <span style="color:#ff8000;">成员模板
  ```c++
  template <typename T>
  class beta {
  private:
          template <typename V>
          class hold {
          priavte:
                  V val;
          public:
                  hold(V v = 0) : val(V) {}
                  V value() const { return val; }
                  ...
          };
          hold<T> q;                // template object
          hold<int> n;        // template object
  public:
          beta(T t, int i) : q(t), n(i) {}
          template <typename U>
          U blab(U u, T t) { return (n.value() + q.value()) * u / t; }
          ...
  };

  int main() {
          beta<double> guys(3.5, 3);
          std::count << guys.blab(10, 2.3) << std::endl;        // return int vale
          std::count << guys.blab(10.0, 2.3) << std::endl;        // return double value
          ...
          return 0;
  }
  ```
  - 也可以在beta模板中声明 hold 类和 blab 方法，并在beta模板外面定义它们：
    ```c++
    template <typename T>
    class beta {
    private:
            template<typename V>
            class hold;
            hold<T> q;
            hold<int> n;
    public:
            beta(T t, int i) : q(t), n(i);
            template <typename U>
            U blab(U u, T t);
            ...
    };

    // member definition
    template <typename T>
       template <typename V>        // NOT equaling template <typename T, typename V>
          class beta<T> :: hold { ... };

    template <typename T>
       template <typename U>
          U beta<T> :: blab(U u, T t) { ... }
    ```
## <span style="color:#ff8000;">模板类型用作模板参数
  - 将模板类型用作模板参数：
    ```c++
    template <template <typename T> class Thing>
    class Crab { ... };

    假设如下声明 Crab<King> legs;
    为使该声明有效，模板参数King必须是一个模板类，其声明与模板参数Thing匹配：
            template <typename T>
            class King { ... };
    ```
  - 示例：
    ```c++
    template <template <typename T> class Thing>
    class Crab {
    private:
            Thing<int> s1;
            Thing<double> s2;
    ...
    };

    int main() {
            Crab<Stack> nebula;
            ...
    }
    ```
  - 混合使用：
    ```c++
    template <template <typename T> class Thing, typename U, Typename V>
    class Crab2 {
    private:
            Thing<U> s1;
            Thing<V> s2;
    ...
    };
    ```
## <span style="color:#ff8000;">模板类和友元
### <span style="color:#00ff00;">非模板友元
  - 在模板类中将一个常规函数声明为友元：
    ```c++
    template <typename T>
    class HasFriend {
    public:
            friend void counts();
            friend void report(HasFriend<T> &);
    ...
    };
    ```
  - report()本身并不是模板函数，只是使用一个模板作参数，因此必须为要使用的模板定义显示具体化：
    ```c++
    void report(HasFriend<int> & hs) { ... }
    void report(HasFriend<double> & hs) { ... }
    ```
### <span style="color:#00ff00;">约束模板友元
  - 友元的类型取决于类被实例化时的类型，使类的每一个具体化都获得与友元匹配的具体化
  - 类外声明模板的具体化
  - 首先在类的前面声明两个模板函数：
    ```c++
    template <typename T> void counts();
    template <typename T> void report(T &);
    ```
    在类声明中声明为友元：
    ```c++
    tempalte <typename TT>
    class HasFriend {
    public:
            friend void counts<TT>();
            friend void report<>(HasFriend<TT> &);
    ...
    };

    对于report()，也可以使用friend void report< HasFriend<TT> >(HasFriend<TT> &); 因为可以从函数参数推断出模板类型，因此<>中类型可省
    ```
    为友元提供模板定义：
    ```c++
    template <typename T>
    void counts() { ... }
    template <typename T>
    void report(T & hf) { cout << hf.item << endl; ... }
    ```
### <span style="color:#00ff00;">非约束模板友元
  - 类内部声明模板，创建非约束类友元函数，即友元模板类型参数与类模板类型参数是不同的：
    ```c++
    template <typename T>
    class ManyFriends {
    ...
            template <typename C, typename D> friend void show2(C &, D &);
    };
    ```
## <span style="color:#ff8000;">模板别名
  - typedef：
    ```c++
    typedef std::array<double, 12> arrayd;
    arrayd ad;
    ```
  - using：
    ```c++
    using arrayd = std::array<double, 12>;
    ```
  - 使用模板提供一系列别名(模板部分具体化)：
    ```c++
    template <typename T>
       using arrtype = std::array<T, 12>; // template to create multiple aliases.

    arrtype<int> hour;
    arrtype<std::string> months;

    ```
## <span style="color:#ff8000;">模板中的嵌套类
  ```c++
  template <typename Item>
  class QueueTP {
  private:
          class Node {
          public:
                  Item item;
                  Node * next;
                  Node(const Item & i) : item(i), next(nullptr) {}
          };
          Node * front;
          Node * rear;
  ...
  };

  类外使用Node: QUeueTP<double>::Node node;
  ```
***

# <span style="color:#ff0000;">运行阶段类识别RTTI (Runtime Type Identification)
  - 运行阶段判断对象类型，如调用某些派生类的特定方法
  - 只能将RTTI用于包含虚函数的类层次结构，只有对于这种类层次结构，才应该将派生对象的地址赋给基类指针
## <span style="color:#ff8000;">dynamic_cast运算符
  - 用于判断是否可以安全的将对象的地址赋给特定类型的指针（派生类赋给基类）
    ```c++
    Type *p = dynamic_cast<Type *> (pt);
    ```
    如果可以安全转换，返回对象地址，否则返回空指针
  - 通常，如果pt类型是Type或是Type直接或间接派生来的，则可以安全转换
    ```c++
    class Grand {
    public:
            virtual void speak() const;
    ... };
    class Super : public Grand {
    public:
            virtual void speak() const;
            virtual void say() const;
    ... };
    class Magnigicent : public Super {
    public:
            virtual void speak() const;
            virtual void say() const;
    ... };

    pg = get_one_random();         // return any type among Grand / Super / Magnificent
    pg.speak();                         // always works
    if (ps = dynamic_cast< Super * >(pg))        // judg if pg is Super / Magnificent
            ps.say();
    ```
    <br />
  - 用于引用时，由于没有与空指针对应的引用值，将抛出bad_cast异常（头文件typeinfo）：
    ```c++
    try {
            Type & pr = dynamic_cast<Type &> (pt);
            ...
    } catch( bad_cast &bc ){ ... }
    ```
## <span style="color:#ff8000;">typeid运算符和type_info类
  - typeid运算符用于判断两个对象是否为同种类型，返回一个type_info对象的引用
    ```c++
    typeid(Magnificent) == typeid( * pg )
    如果pg是空指针，将引起bad_typeid异常（头文件typeinfo）
    ```
  - type_info结构存储了有关特定类型的信息，重载了 == 和 != 运算符用于类型比较
  - type_info类包含一个name()成员，返回一个因实现而异的字符串，通常为类名称
    ```c++
    cout << "Now processing type: " << typeid(* pg).name() << endl;
    ```
  - 使用typeid()只能显示的判断某一种类型，对于派生类应考虑尽量使用dynamic_cast用于判断
***

# <span style="color:#ff0000;">智能指针模板类
  - 类似于指针的类对象，帮助管理动态内存分配（头文件memory）
  - 三个智能指针模板 auto_ptr / unique_ptr / shared_ptr 都定义了类似指针的对象，可以将new获得的地址赋给这种对象（auto_ptr在c++11已摒弃）
  - 当智能指针过期时，其析构函数将使用delete来释放内存，因此无须记住稍候手动来释放
  - 智能指针在很多方面类似常规指针：
    ```c++
    使用解除引用运算符*ps
    访问成员方法ps->show()
    赋给指向相同类型的常规指针
    赋给另一个同类型的智能指针对象
    ```
## <span style="color:#ff8000;">创建
  - 构造函数类似：
    ```c++
    template <class X> class auto_ptr {
    public:
            explicite auto_ptr(X*p = 0);
    ... };
    ```
    ```c++
    unique_ptr<double> pdu(new double);
    shared_ptr<string> pds(new string);

    shared_ptr<string> animal[5] = {shared_ptr<string> (new string"Fox"), shared_ptr<string> (new string"Wolf"), ... };
    unique_ptr<double []>pda(new double[5]);         // ONLY allowed for unique_ptr, which has a reloaded version using new[] / delete[]
    ```
    ```c++
    shared_ptr<double> pd;
    double *p = new double;

    pd = p;                 // NOT allowed (implicit conversion)
    pd = shared_ptr<double>(p);        // allowed
    shared_ptr<double> pshared(p);         // allowed
    ```
  - 应避免用于非堆内存：
    ```c++
    string vacation("I wandered lonely as a cloud.");
    shared_ptr<string>pvac(&vacation);                // NOT allowed [ ??? ]

    pvac 过期时将把delete运算符用于非堆内存

    应使用：shared_ptr<string> pvac2(new string("I wandered lonely as a cloud."));
    ```
## <span style="color:#ff8000;">智能指针类的赋值运算
  - 如果不做特殊处理，指向同一个对象的两个智能指针在过期时将删除同一个对象两次，这是不能接受的
  - 解决方法：
    ```c++
    建立所有权(ownership)概念，只有一个智能指针可以拥有它，拥有所有权才可以删除对象，赋值操作将转移所有权
            unique_ptr / auto_ptr采用该策略
    使用引用计数(reference counting)跟踪智能指针数，赋值时技术加1，指针过期时计数减1，计数为0时才调用delete
            shared_ptr采用该策略
    ```
  - auto_ptr:
    ```c++
    auto_ptr<string> pa1(new string("auto"));
    auto_ptr<string> pa2 = pa1;        // allowed
    cout << *pa1;                // will trigger core dump!

    调用转移所有权后的auto_ptr指针将引起运行时错误
    ```
  - unique_ptr:
    ```c++
    unique_ptr<string>pu1(new string("auto"));
    unique_ptr<string>pu2 = pu1;        // NOT allowed

    unique_ptr指针只允许存在一份，赋值操作将引起编译时错误，因此比auto_ptr更安全

    如果unique_ptr的赋值操作右值是一个临时变量，将允许这种操作：
            unique_ptr<string>pu3 = unique_ptr<string>(new string("unique"));

            unique_ptr<string> demo(cahr *s) { unique_ptr<string> temp(new string(s)); return temp;}
            unique_ptr<string> pu4 = demo("unique");

    还可以使用std::move()将一个unique_ptr指针赋给另一个：
            unique_ptr<string> ps1, ps2;
            ps1 = demo("unique1");
            ps2 = move(ps1);        // [ ??? How about ps1 now? ]
    ```
  - shared_ptr:
    ```c++
    shared_ptr用于需要多个指针指向同一个对象的情况，如指针数组中标示最大/最小元素
    shared_ptr包含一个构造函数，可以将右值unique_ptr转换为shared_ptr，但需要满足unique_ptr的赋值条件：
            unique_ptr<string> pu5(new string("unique"));
            shared_ptr<string> spp1(pu5);        // NOT allowed
            shared_ptr<string> spp2 = demo("shared"); // allowed
    ```
***

# <span style="color:#ff0000;">标准模板库STL (Standard Template Library)
  - 提供一组表示容器 / 迭代器 / 函数对象 / 算法的模板
  - 是一种泛型编程(generic programming)模式，而不是面向对象编程   :=(
## <span style="color:#ff8000;">模板类vector (用于示例容器类)
  ```c++
  class elem { string name; int rating; };
  vector<elem> books;
  vector<elem> books_backup(books); // copy a new vetor
  ```
## <span style="color:#ff8000;">内存管理 Allocator
  - 各种STL容器模板都可以接受一个可选参数，指定使用哪个分配器对象来管理内存：
    ```c++
    template< class T, Allocator = allocate<T> >
      class vector { ... };
    默认使用allocate<T>类，使用new / delete
    ```
## <span style="color:#ff8000;">STL算法
  - 对算法进行分类的方式之一是按结果放置的位置，有些就地完成，有些创建拷贝
  - 有些算法有两个版本，STL的约定是，复制版本的名称以 _copy 结尾  __
  - 有些算法根据将函数应用于容器元素得到的结果来执行操作，通常以 _if 结尾  __
## <span style="color:#ff8000;">算法组
  - 非修改式序列操作：头文件algorithm，操作不修改容器内容，如find() / for_each()
  - 修改式序列操作：头文件algorithm，transform() / random_shuffle() / copy()
  - 排序和相关操作：头文件algorithm，sort()等
  - 通用数字操作：头文件numric，通常都是数组的操作特性
## <span style="color:#ff8000;">STL容器基本方法
  - 除了分配空间外，所有的STL容器都提供了一些基本方法：
    ```c++
    size()        // 返回容器中元素数目
    swap()        // 交换两个容器内容
    begin()         // 返回指向容器中第一个元素的迭代器
    end()         // 返回一个表示超过容器尾的迭代器

    cbegin() / cend()         // begin() / end() 的const版本
    ```
## <span style="color:#ff8000;">迭代器
  - 一个广义上的指针，可以进行解引用 * / 递增(++)等操作
  - 最好避免直接使用迭代器，而应尽可能使用STL函数，如for_each()或基于范围的for循环
  - 为不同的容器提供统一的接口，每个容器都定义了一个合适的迭代器，是一个名为iterator的typedef
    ```c++
    vector<double> :: iterator pd;
    vector<double> scores;
    pd = scores.begin();
    *pd = 1.23;
    pd++;
    ```
  - 也可以使用自动类型判断：
    ```c++
    auto pda = scores.begin();
    ```
  - 遍历容器：
    ```c++
    for (pd = scores.begin(); pd != scores.end(); pd++)
            cout << *pd << endl;<br />
    []方法：
    for (int i = 0; i < scores.size(); i++)
            cout << scores[i] << endl;
    ```
## <span style="color:#ff8000;">初始化列表 initializer_list模板类
  - 成员函数begin() / end() 用于获得列表范围
  - 容器类包含将initializer_list<T>作为参数的构造函数：
    ```c++
    std::vector<int> vi {1, 3, 2, 4, 5, ... };
    等价于显示的将列表指定为构造函数参数
    std::vector<int> vi ({1, 3, 2, 4, 5, ... });
    ```
  - 使用示例：
    ```c++
    double average(const std::initializer_list<double> & ril) {
            double sum = 0;
            for (auto p = ril.begin(); p != ril.end(); p++) {
                    sum += * p;
            }
            double ave = 0.0;
            if (ril.size() > 0)
                    ave = sum / ril.size();
            return ave;
    }

    cout << average({2, 3, 4}) << endl;
    cout << average(vi) << endl;
    ```
## <span style="color:#ff8000;">STL特殊方法 push_back / erase / insert
  - 只有某些STL容器类才有的方法
  - push_back() 将元素添加到矢量末尾
    ```c++
    vector<double> scores;
    double temp;
    while (cin >> temp && temp >= 0)
            scores.push_back();
    cout << "size: " << scores.size() << endl;
    ```
  - erase() 删除指定区间的元素，接受两个迭代器参数
    ```c++
    scores.erase(scores.begin(), scores.begin() + 2);         // 删除弟0，1个元素，删除区间表示[p1,p2)
    ```
  - insert() 接受三个迭代器参数，第一个指定新元素插入位置，第二个和第三个参数指定区间，通常是另一个容器对象的一部分
    ```c++
    old_v.insert(old_v.begin(), new_v.begin() + 1, new_v.end());         // 将new_v中除第一个元素插入到old_v第一个元素前面
    ```
## <span style="color:#ff8000;">非成员函数 for_each / random_shuffle / sort / copy / transform
  - 对于搜索 / 排序等算法定义的通用算法，使之可以适用于不同的容器类
  - 即使有非成员函数，STL有时也会定义一个相同功能的成员函数，因为特定算法比通用算法效率高，如vector类的swap()效率比非成员函数swap()高
  - 非成员函数允许交换两个不同类型容器的内容
### <span style="color:#00ff00;">for_each()
  - 接受3个参数，前两个指定容器中区间的迭代器，最后一个是指向函数的指针或函数对象
  - for_each()函数将被指向的函数用于容器区间中的各个元素
  - 被指向的函数不能修改容器元素的值
    ```c++
    template <class ImputIterator, class Function>
    Function for_each(InputIterator first, InputIterator second, Function f);
    ```
  - 可用于替代for循环：
    ```c++
    void show(elem) { ... }

    vector<elem> newbooks;
    vector <elem> :: iterator pr;
    for (pr = newbooks.begin(); pr != newbooks.end(); pr ++) { show(*pr); }

    替换为：
    for_each(newbooks.begin(), newbooks.end(), show);
    ```
    还可以使用基于范围的for循环：
    ```c++
    for (auto &x : newbooks) { show(x); }
    ```
    使用for循环时可以修改元素的值
### <span style="color:#00ff00;">random_shuffle()
  - 接受两个指定区间的迭代器参数，并随机排列其中的元素
  - 要求容器类允许随机访问，如vector类
### <span style="color:#00ff00;">sort()
  - 可以接受两个指定区间的迭代器参数，并使用为存储在容器中的类型元素定义的 < 运算符，对区间中的元素进行升序排列
    ```c++
    vector<int> rating;
    sort(rating.begin(), rating.end());
    ```
  - 如果容器元素是自定义的类型，则需要定义能够处理该类型的operator<()运算符
    ```c++
    bool elem::oeprator<(const elem & e1, const elem & e2) { ... }
    sort(newbooks.begin(), newbooks.end());
    ```
  - 如果要按降序或其他排列顺序进行排序，可以传递第三个参数，一个指向用于排序函数的指针，该函数返回值可以转换为bool，false表示顺序不正确
    ```c++
    bool worsethan(const elem & e1, const elem & e2) { ... }
    sort(newbooks.begin(), newbooks.end(), worsethan);
    ```
### <span style="color:#00ff00;">copy()
  - 三个参数，前两个迭代器参数表示要复制的范围，第三个迭代器参数表示复制到目标容器的位置
  - copy()函数将覆盖目标容器中已有的内容
  - 目标容器可写，并要求有足够大的空间，copy()不能自动根据发送值调整目标容器的长度
### <span style="color:#00ff00;">transform()
  - 可以接受四个参数，前两个指定容器的迭代器参数，第三个指定结果复制到哪里的迭代器，最后一个函数符，用于区间中每个元素：
    ```c++
    const int MAX = 8;
    double arrd[MAX] = { ... };
    vector<double> gr8(srrd, srrd + MAX);
    ostream_iterator<double, char> out(cout, " ");         // 一个输出迭代器，指定使用char输出double值到屏幕，每个值用" "分隔
    transform(gr8.begin(), gr8.end(), out, sqrt);        // 计算每个元素平方根，并输出到屏幕
    ```
    也可以接受五个参数，第三个迭代器参数指定另一个容器的起始位置，第五个参数接受两个参数的函数符：
    ```c++
    double add(double x, double y) { return x + y; }
    vector<double> m8(MAX);
    transform(gr8.begin(), gr8.end(), m8.begin(), out, add);        // 计算gr8与m8两个容器中元素的和，输出到cout
    ```
## <span style="color:#ff8000;">STL算法用于常规数组
  - 迭代器是广义的指针，而指针是迭代器，因此STL算法可以使用指针来对基于指针的非STL容器进行操作：
    ```c++
    const int SIZE = 10;
    double da[SIZE];
    sort(da, da + SIZE);        // 第二个参数指向最后一个元素后面的地址

    int cast[10] = {5, 6, 2, 7, 3, ... };
    vector<int>dice(10);
    copy(cast, cast + 10, dice.begin());
    ```
  - 只要定义适当的迭代器，超尾元素指示等，STL算法也可以用于自定义的数据形式
***

# <span style="color:#ff0000;">泛型编程 (迭代器 / 容器)
  - 面向对象关注的是编程的数据方面，而泛型编程关注的是算法方面，旨在编写独立于数据类型的代码
  - 模板使算法独立于存储的数据类型，迭代器使算法独立于使用的容器类型
  - 应尽可能用通用术语表达算法，并基于算法要求，设计基本迭代器特征与容器特征
## <span style="color:#ff8000;">迭代器实现
  - 泛型编程旨在实现使用相同的接口处理不同的容器类型，因此需要迭代器这样的通用表示，来遍历容器中的值
  - 对于查找find()功能迭代器基本要求：
    ```c++
    应能够对迭代器进行解引用操作*p
    应能够将迭代器赋给另一个 p = q
    应能够将两个迭代器进行比较，检查是否相等 p == q / p!= q
    应能够遍历容器中所有元素++p / p++
    ```
    ```c++
    struct Node { double item; Node * p_next; };
    class Iterator {
            Node * pt;
    public:
            Iterator() : pt(0) {}
            Iterator(Node * p) : pt(p) {}

            double operator *() { return pt->item; }
            Iterator & operator ++() { pt = pt->p_next; return * this; }        // for ++it
            Iterator operator ++(int) {        // for it++, parameter int is for no use. return a temp var, it cannot be reference
                    Iterator temp = * this;
                    pt = pt->p_next;
                    return temp;
            }
            // ... operator =(); operator ==(); operator !=(); etc
    };

    Iterator find(Iterator begin， Iterator end, const double &val) {
            for (auto ar = begin; ar != end; ar++)
                    if (*ar == val) return ar;
            return end;
    }
    ```
  - 对于随机访问功能，需要定义 + 操作，实现 P+10这样的表达
## <span style="color:#ff8000;">迭代器类型
  - STL定义了5种迭代器概念：输入迭代器 / 输出迭代器 / 正向迭代器 / 双向迭代器 / 随机访问迭代器
  - 尽可能使用要求最低的迭代器
  - 迭代器类型只是概念上的定义，有时使用[改进]来表示概念上的继承关系，使用[模型]来表示概念的具体实现
  - 对于find()只需要读取数据，可使用输入迭代器：
    ```c++
    template<class InputIterator, class T>
    ImputIterator find(InputIterator first, InputIterator last, const T & val);
    ```
    对于sort()需要读写数据，并能随机访问，使用随机访问迭代器：
    ```c++
    template<class RandomIterator>
    void sort(RandomIteratorfirst, RandomIteratorlast);
    ```
### <span style="color:#00ff00;">输入迭代器
  - 输入是相对于程序说的，即来自容器的信息被视为输入
  - 输入迭代器可以访问容器中所有值(++运算符)，但不一定能让程序修改值
  - 是单向迭代器，可以递增，但不能递减
### <span style="color:#00ff00;">输出迭代器
  - 输出指将信息从程序传输给容器
  - 只写，不能保证可读
  - 单向迭代器
### <span style="color:#00ff00;">正向迭代器
  - 只使用++算法
  - 与输入 / 输出迭代器不同的是，正向迭代器总是按相同的顺序遍历，将正向迭代器递增后，仍可对前面的迭代器解引用操作，并得到相同的值
  - 可读写，也可以使用const限定为只读
### <span style="color:#00ff00;">双向迭代器
  - 支持前缀和后缀递减运算符(--)
### <span style="color:#00ff00;">随机访问迭代器
  - 实现 + / - / += / -= / [] / < / > / <= / >=等操作
  - a, b为随机访问迭代器变量，n为整数，则可以：
    ```c++
    a + n / n + a / a - n / a[n] / b - a / a > b
    ```
## <span style="color:#ff8000;">STL提供的迭代器模型
  - STL预定义迭代器提高了算法的通用性，如对于copy()算法，使用这些迭代器将不仅可以将信息从一个容器赋值到另一个容器，还可以：
    ```c++
    将信息从容器赋值到输出流 (ostream_iterator)
    将信息从输入流复制到容器 (istream_iterator)
    将信息插入到另一个容器 (insert_iterator)
    ```
### <span style="color:#00ff00;">ostream_iterator
  - 一个表示输出流的迭代器，它是一个适配器(adapter)，可以将其他接口转化为STL使用的接口
  - 通过以下声明来创建这种迭代器：
    ```c++
    #include <iterator>
    ostream_itrerator<int, char> out_iter(cout, " ");

    out_iter现在是一个接口，可以使用cout来显示信息
    第一个模板参数(int)指出别发送给输出流的数据类型
    第二个模板参数(char)指出输出流使用的数据类型 char / wchar_t
    构造函数第一个参数(cout)指出要使用的输出流，可以使用文件
    构造函数第二个字符串参数(" ")指出发送给输出流的每个数据项后的分隔符
    ```
  - 使用：
    ```c++
    out_iter++ = 15;        // woks like cout << 15 << " ";
    将15和" "组成的字符串发送给cout，并为下一次输出做好准备
    ```
  - 将copy()用于迭代器：
    ```c++
    vector<int> dice(10);
    copy(dice.begin(), dice.end(), out_iter);        // copy vector to output stream
    这将显示dice整个容器的内容

    也可以创建匿名的迭代器：
    copy(dice.begin(), dice.end(), ostream_iterator<int, char> (cout, " "));
    ```
### <span style="color:#00ff00;">istram_iterator
  - 一个输入迭代器概念的模型，使istream输入可以作为迭代器接口
  - 定义copy()的输入范围：
    ```
    copy(istream_iterator<int, char>(cin), istream<int, char>(), dice.begin());

    istream_iterator的两个模板参数指出要读取的数据类型，与输入流使用的字符类型
    构造函数参数cin表示使用cin管理的输入流
    省略构造函数参数表示输入失败

    该copy()操作将从输入流中读取，直到文件结尾 \ 类型不匹配 \ 出现其他输入故障为止
    ```
### <span style="color:#00ff00;">反向迭代器 reverse_iterator
  - 执行递增操作将导致递减，为了简化对已有函数的使用，实现反向操作
  - 对于vector类
    ```
    rbegin() 成员方法返回指向超尾元素的反向迭代器，与end()方法返回值相同，但类型不同
    rend() 成员方法返回指向第一个元素的反向迭代器，与begin()方法返回值相同，但类型不同
    ```
  - 反向显示内容：
    ```c++
    copy(dice.rbegin(), dece.rend(), out_iter);        // display in reverse order
    ```
  - 解引用( * )操作的特殊补偿：
    ```c++
    由于dice.rbegin()返回超尾元素，因此不能执行解引用操作，同时dice.rend()返回第一个元素位置，不能作为超出范围指示器
    反向迭代器通过先递减再解除引用解决这个问题，即如果反向迭代器rp指向位置6，则*rp将是位置5的值

    vector<int>::reverse_iterator ri;
    for (ri = dice.rbegin(), ri != dice.rend(), ri++) cout << *ri << endl;
    ```
### <span style="color:#00ff00;">插入迭代器 back_insert_iterator / front_insert_iterator / insert_iterator
  - back_insert_iterator 将元素插入到容器尾部，只能用于允许在尾部快速插入的容器 (快速插入指的是一个时间固定的算法)，vector类符合这种要求
  - front_insert_iterator 将元素插入到容器前端，只能用于允许在起始位置做时间固定插入的容器类型，vector类不符合这种要求，queue类符合
  - insert_iterator 将元素插入到insert_iterator构造参数指定的位置前面，没有算法的要求，但前两个完成任务更快
  - 声明：
    ```c++
    将容器类型作为迭代器模板参数，实际容器标示符作为构造函数参数：
            vector<int> dice;
            back_insert_iterator< vector<int> > back_iter(dice);
    back_insert_iterator的构造函数将假设传递给它的类型有一个push_back()方法，用于重新调整容器大小
            vector<int> :: puch_back();        // vector puch_back() method

    对于insert_iterator还需要一个指示插入位置的构造函数参数：
            insert_iterator< vector<int> > insert_iter(dice, dice.begin());
    ```
  - copy()不能自动根据发送值调整目标容器的长度，以下调用假设dice有足够大的空间：
    ```c++
    int cast[10] = {5, 6, 2, 7, 3, ... };
    vector<int>dice(10);
    copy(cast, cast + 10, dice.begin());

    三种插入迭代器可以将复制转换为插入，插入将添加新的元素而不会覆盖原有的内容，并使用自动内存分配来确保有足够的空间
    string s1[4] = { "copy", ... };
    string s2[2] = { "back_insert", ... };
    string s3[2] = { 'insert', ... };
    vector<string> words(4);

    copy(s1, s1+4, words.begin());        // s1
    copy(s2, s2+2, back_insert_iterator< vector<string> >(words));        // s1 s2
    copy(s3, s3+2, insert_iterator< vector<string> >(words, words.begin()));        // s3 s1 s2
    ```
## <span style="color:#ff8000;">容器种类
  - 容器类型是用于创建具体容器对象的模板
  - 存储在容器中的类需要是可复制构造的和可赋值的，容器过期时，存储在容器中的数据也将过期
  - 可以用 == 来比较不同类型的容器，因为容器重载的 == 使用迭代器来比较内容，因此如果dqueue与vector对象内容相同，则它们是相等的
  - X::value_type 通常指出了存储在容器中的值类型
  - X::key_type 通常指出了键类型
  - 11个容器类型：
    ```c++
    序列（数组/链表）：deque / list / queue / priority_queue / stack / vector
    关联容器（树数据结构）：map / multimap / set / multiset
    bitset
    ```
    c++11新增：
    ```c++
    单向链表（序列）： forward_list
    无序关联容器（哈希表数据结构）： unordered_map / unordered_multimap / unordered_set / unordered_multiset
    且不再将bitset视为容器
    ```
### <span style="color:#00ff00;">时间复杂度
  - 编译时间：操作将在编译时完成，执行时间为0
  - 固定时间：操作发生在运行阶段，但独立于对象中的元素数目
  - 线性时间：执行时间与元素数目成正比
## <span style="color:#ff8000;">容器类型 （序列）
  - 序列是基本容器概念的一种改进(deque / forward_list / list / queue / priority_queue / stack / vector)
  - 序列概念要求迭代器至少是正向迭代器，其元素按严格的线性顺序排列
### <span style="color:#00ff00;">vector
  - 最简单的一种序列模型，是数组的一种表示
  - 提供自动内存管理，提供随机访问( [] / at() )，在尾部添加和删除元素的时间是固定的( push_back() / pop_back() )
  - 在头部或中间插入元素的复杂度为线性时间，没有定义 push_front() / pop_front() 方法
  - 除序列外，vector还是可反转容器概念模型，新增两个类方法 rbegin() / rend()，返回反向迭代器
### <span style="color:#00ff00;">deque
  - 表示双端队列(double-ended queue)，头文件deque
  - 实现类似vector，提供了随机访问
  - 从deque对象的头部插入/删除元素复杂度是固定时间，定义了 push_front() / pop_front() 方法
### <span style="color:#00ff00;">list
  - 表示双向链表，可以双向遍历
  - list对象在任何位置进行插入 / 删除的时间都是固定的
  - 也是可反转容器，但不支持数组表示法和随机访问，因此不能使用STL非成员方法sort()
  - 链表专用成员函数：
    ```c++
    void merge(list<T, Alloc> & x) 将链表x与调用链表合并，两个链表必须已经排序，合并后经过排序的链表保存在调用链表中，x将为空
    void remove(const T &val) 从链表中删除val的所有实例
    void remove_if(Function) 接受一个返回bool值的函数对象，应用于区间中每个元素，返回true则删除
    void sort() 使用 < 运算符对链表进行排序
    void splice(iterator pos, list<T, Alloc> & x) 将链表x的内容插入到pos前面，x将为空
    void unique() 将连续的相同元素压缩为一个元素

    sort() / unique() / merge()方法还拥有接受另一个参数的版本，用来指定用于比肩元素的函数
    remove() 方法也可以接受另一个参数，用来指定确定是否删除元素的函数

    remove_if():
            bool tooBig(int n) { return n > 100; }
            list<int> scores;
            ...
            scores.remove_if(tooBig);
    ```
### <span style="color:#00ff00;">forward_list
  - 单向链表，只需要正向迭代器，不可反转
  - 相比于list，简单紧凑，但功能更少
### <span style="color:#00ff00;">queue
  - queue模板类是一个适配器类，让底层类(默认为deque类)展示典型的队列接口
  - 不允许随机访问，不允许遍历队列
  - 队列方法：
    ```c++
    bool empty() const 测试是否为空
    size_type size() const 检查元素数目
    T & front() 返回队首元素的引用
    T & back() 返回队尾元素的引用
    void push(const T & x) 队尾添加元素
    void pop() 队首删除元素，不检索数据
    ```
### <span style="color:#00ff00;">priority_queue
  - 是另一个适配器类，与queue区别在于，最大的元素被移到队首，默认底层使用vector类
  - 可以提供一个可选的构造函数参数，用于确定哪个元素放到队首的比较方法
    ```c++
    priority_queue<int> pq1;        // default version
    priority_queue<int> pq2(greater<int>); // use greater<int> to order

    greater<int>是一个预定义的函数对象
    ```
### <span style="color:#00ff00;">stack
  - 与queue类似，也是一个适配器类，提供典型的栈接口，默认使用vector类
  - 压入元素到栈顶，从栈顶弹出元素，查看栈顶值
    ```c++
    T & top()
    void push(const T & x)
    void pop()
    ```
## <span style="color:#ff8000;">容器类型 （关联容器）
  - 对容器概念的另一个改进，将值与键关联到一起，使用键来查找值
  - 通常是使用某种树实现的，查找速度更快，但不能指定插入位置，因此插入方法insert()只指定要插入的信息，不指定位置
### <span style="color:#00ff00;">集合 set （头文件set）
  - set 值与键类型相同，键是唯一的，对于set来说，值就是键
  - multi_set 类似于set，只是可能有多个值的键相同
  - 声明：
    ```c++
    set<string> A;
    set<string, less<string>> B;
    第二个模板参数是可选的，指示用来进行排序的函数，默认情况下使用模板less<>
    ```
  - 使用：
    ```c++
    const int N = 6;
    string s1[N] = {"for", "the", "love", "for", ...};
    set<string> A(s1, s1+N);
    ostream_iterator<string, char> out(cout, " ");
    copy(set.begin(), set.end(), out);
    键是唯一的，且初始化后集合会被排序
    ```
  - STL通用函数
    ```c++
    集合可以有并集 / 交集操作，STL提供了一些通用函数，set对象自动满足使用这些算法的条件，即容器是经过排序的

    set_union()函数获得两个集合的并集，接受五个迭代器参数，前两个定义第一个集合区间，后两个定义第二个集合区间，最后一个是输出迭代器
            set_union(A.begin(), A.end(), B.begin(), B.end, ostream_iterator<string, char>(cout, " "));
    如果将输出指定为其他集合C，则不能使用C.begin(),因为：
            集合将键视为常量，因此C.begin()不能用作输出迭代器
            set_union将覆盖容器中原有数据，要求目标有足够空间
    因此应使用insert_iterator:
            set_union(A.begin(), A.end(), B.begin(), B.end, insert_iterator< set<string> >(C, C.begin()));

    set_intersection() 获得两个集合的交集
    set_difference() 获得两个集合的差

    lower_bound()接受一个迭代器参数，返回一个指向集合中第一个不小于键参数成员的迭代器
    upper_bound()接受一个迭代器参数，返回一个指向集合中第一个大于键参数成员的迭代器
    可以使用这两个方法返回一个区间：
            copy(C.lower_bound("gost"), C.upper_bound("for"), ostream_iterator<string, char>(cout, " "));
    ```
### <span style="color:#00ff00;">图 map （头文件map）
  - map 值与键类型不同，键是唯一的
  - multi_map 类似于map，只是一个键可以与多个值相关联
  - 声明：
    ```c++
    multimap<int, string> codes;        // 键类型为int，存储的值类型为string
    可以接受第三个可选模板参数，用于对键进行排序的函数
    ```
  - 使用：
    ```c++
    实际的值类型将键类型与值类型结合到一起，STL使用模板类pair<class T, class U>将其存储到一个对象中pair<const keytype, datetype>
    对于codes，对象，值类型为pair<const int, string>

    pair<const int, string> item(213, "Los Angeles");
    codes.insert(item);
    codes.insert(pair<const int, string>(214, "New York"));
    codes.insert(multimap<int, string>::value_type(215, "Washington"));

    对于pair对象，使用first / second两个成员来访问其两个部分：
            cout << item.first << " " << item.second << endl;
    ```
  - 成员函数：
    ```c++
    count() 接受键作为参数，返回具有该键的元素数目
    lower_bound() / upper_bound() 接受键作为参数，原理与set类的类似

    equal_range() 用键作为参数，返回两个迭代器，指示该键的区间，返回值封装在一个pair对象中
            pair<multimap<int, string>::iterator, multimap<int, string>::iterator> range = equal_range(718);
            mulimap<int, string>::iterator it;
            for (it = range.first, it != range.second, it++) cout << (* it).second << endl;
    或使用auto自动类型推断：
            auto range = equal_range(718);
            for (auto it = range.first, it != range.second, it++) cout << (* it).second << endl;
    ```
  - map类数组表示法：
    ```c++
    map类可以用数组表示法，将键用作索引来访问存储的值
    codes[216] = "Hawaii";

    插入216时，先在codes中查找主键为216的项，如果没发现：
            先将一个新的对象插入codes，键是216，值是一个空字符串
            插入完成后，将字符串赋为"Hawaii"
    使用这种赋值方法比较直观，但如果元素是类对象，则开销比较大

    用于获取值：string city = codes[215];
    只有该键存在时才返回正确的值，否则会自动插入一个实例，值为初始化值
    可以通过find() / count()方法来发现一个键是否存在：
            if (codes.find(215) != codes.end()) { cout << "215 exist!" << endl; }
    ```
***

# <span style="color:#ff0000;">函数对象
  - 可以以函数方式与()结合使用的任意对象，包括函数名 / 函数指针 / 重载了()运算符的类
  - 如对于STL方法for_each()原型:
    ```c++
    template <class ImputIterator, class Function>
    Function for_each(InputIterator first, InputIterator second, Function f);

    第三个参数接受函数参数，用于处理容器数据，但无法提前指导数据类型，因此不能使用函数指针
    此处定义的Function可以是一个一元函数对象
    对于
            vector<elem> newbooks;
            for_each(newbooks.begin(), newbooks.end(), show);
    Function表示的类型是 void ( * )(const elem &)

    Function也可以表示重载了()运算符的类
    ```
## <span style="color:#ff8000;">函数符概念
  - 生成器 generator：不用参数可调用的函数符
  - 一元函数 unary function：一个参数可调用的函数符
  - 二元函数 binary function：两个个参数可调用的函数符

  - 谓词 predicate：返回bool值的一元函数
  - 二元谓词 binary predicate：返回bool值的二元函数
  - 如STL方法sort()接受二元谓词参数
  - list成员方法remove_if()接受谓词参数
## <span style="color:#ff8000;">二元函数转换为一元函数对象
  - 假设有一个接受两个参数的模板函数：
    ```c++
    template<typename T>
    bool tooBig(const T & val, const T & lim) { return val > lim; }
    ```
    则无法将其用于接受谓词参数的函数，如list成员方法remove_if()
  - 可以使用类将其转换为单个参数的函数对象(函数适配器)：
    ```c++
    template<typename T>
    class TooBig {
    private:
            T limit;
    public:
            TooBig(const T & t) : limit(t) {}
            bool operator ()(const T & val) { return val > limit; }
    };

    TooBig<int> tb100(100);        // 原先函数的第二个参数用于构建函数对象
    if (tb100(x)) { ... }        // 等同于 tooBig<int>(x, 100)

    list<int> scores;
    ...
    scores.remove_if(tb100);
    scores.remove_if(TooBig<int>(200));
    ```
## <span style="color:#ff8000;">STL预定义的函数符
  - 头文件function定义了多个模板类函数对象
  - 对于所有内置的算数运算符 / 关系运算符 / 逻辑运算符，STL都提供了等价的函数符
    ```c++
    + plus / - minus / * multiplies / / divides / % modulus / - negate
    == equal_to / != not_equal_to / > greater / < less / >= greater_equal / <= less_equal
    && local_and / || local_or / ! local_not
    ```
  - plus<>使用：
    ```c++
    #include <function>
    plus<double> addd;
    double y = addd(2.2, 3.3);

    transform(gr8.begin(), gr8.end(), m8.begin(), out, plus<double>());        // STL非成员函数
    ```
  - 自适应函数符：
    ```c++
    自适应生成器 / 自适应一元函数 / 自适应二元函数 / 自适应谓词 / 自适应二元谓词
    ```
## <span style="color:#ff8000;">函数适配器 binder1st / binder2nd类
  - binder1st类：
    ```c++
    将二元函数的第二个参数与特定值相关联，转换为一元函数
    对于二元函数f2(x, y)：
            binder1st(f2, val) f1;
    使用f1(x)时，等价于f2(x, val)

    STL提供了bind1st()函数简化类binder1st类的使用，对于二元函数multiplies：
            bind1st(multiplies<double>(), 2.5)        // 将是一个与2.5作乘法运算的函数
            transform(gr8.begin(), gr8.end(), out, bind1st(multiplies<double>(), 2.5));
    ```
  - binder2nd类：
    ```
    与binder1st类似，只是将参数赋给第二个参数
    bind2nd()函数用于简化binder2nd类使用
    ```
***

# <span style="color:#ff0000;">lambda (匿名函数)
  - 使函数定义位于使用的地方附近
  - lambda可以访问作用域内的任何动态变量
## <span style="color:#ff8000;">lambda作为函数参数
  - 对于接受函数指针或函数符的函数，可以使用匿名函数定义lambda作为其参数
    ```c++
    [] (int x) { return x % 3 == 0; }        // --> bool f3(int x) { return x % 3 == 0; }

    使用 [] 代替函数名
    返回类型相当于使用decltype根据返回值自动推断得到，这里为 bool，如果没有返回语句，则为void
    仅当lambda语句仅由一条返回语句组成时,自动类型推断才管用，否则应使用返回类型后置语法：
            [] (double x) -> double { double y = 3.14; return x * y; }
    ```
    ```c++
    std::vector<int> number(10);
    std::srand(std::time(0));
    std::generate(number.begin(), number.end(), std::rand);        // STL function, generate an array using the third function parametre

    int count = count_if(number.begin(), number.end(),
            [] (int x) { return x % 3 == 0; }); // STL function, return the elements number which the third function parameter returns true
    ```
## <span style="color:#ff8000;">指定名称的lambda函数
  - lambda函数也可以指定名称：
    ```c++
    auto mod3 = [] (int x) { return x % 3 == 0; }        // mod3 is a name of a lambda function

    可以像常规函数一样使用指定名称的lambda函数：
            bool result = mod3(x);
    ```
## <span style="color:#ff8000;">lambda捕获作用域内的变量
  - 将变量名放在[]内以捕获要使用的变量：
    ```c++
    [z] 按值访问变量z
    [&z] 按引用访问变量z

    [=] 按值访问所有动态变量
    [&] 按引用访问所有动态变量

    [x, &z] 按值访问x，按引用访问z
    [=, &z] 按引用访问z，按值访问所有其他变量
    [&, x] 按值访问x，按引用访问所有其他变量
    ```
    ```c++
    std::vector<int> vi(10);
    int count13 = 0;
    std::for_each(vi.begin(), vi.end(),
            [&count13] (int x) { count13 += (x % 13 == 0); });        // count elements which could be divided excatly by 13

    int count 3 = 0;
    std::for_each(vi.begin(), vi.end(),
            [&] (int x) { count3 += (x % 3 == 0); count13 += (x % 13 == 0); }); // count elements which could be divided excatly by 3 & 13
    ```
    <br />
***

# <span style="color:#ff0000;">包装器 wrapper
  - 也叫适配器 adapter，用于给给其他编程接口提供更一致或更合适的接口，如 bind1st / bind2nd
    ```c++
    模板bind，可替代bind1st / bind2nd
    模板mem_fn，可以将成员函数作为常规函数进行传递
    模板reference_wrapper，可以创建行为像引用，但可以被复制的对象
    模板function，可以以统一的方式处理多种类似于函数的形式
    ```
## <span style="color:#ff8000;">包装器function
  - 将常规函数 / 重载()运算符的对象 / lambda函数统一成同样的接口
    ```c++
    std::function <double(int, char)> fd;        // 创建一个名为fd的function对象，接受两个参数，返都double类型
    ```
  - 如对于函数模板：
    ```c++
    template <typename T, typename F>
    T use_f(T v, F f) {
    ...
            return f(v);
    }
    调用常规函数 / 重载()运算符的对象 / lambda函数类型的F，将创建不同的模板实例
    ```
  - 可以创建不同的function对象：
    ```c++
    function <double (double)> ef1 = dub;        // 常规函数指针
    function <double (double)> ef2 = Fq(10.0);        // 函数对象
    function <double (double)> ef3 = [] (double u) { return u * u; };        // lambda函数

    cout << use_f(12.1, ef1); << endl;
    cout << use_f(12.1, ef2); << endl;
    cout << use_f(12.1, ef3); << endl;
    将只创建一个use_f实例
    ```
  - 也可以使用typedef + 临时对象：
    ```c++
    typedef function <double (double)> fdd;
    cout << use_f(12.1, fdd(dub)); << endl;
    cout << use_f(12.1, fdd(Fq(10.0))); << endl;
    ```
  - 也可以将use_if()第二个参数声明为function包装器对象类型：
    ```c++
    #include <functional>
    template <typename T>
    T use_if(T v, std::function<T (T)> f) {
    ...
            return f(v);
    }

    cout << use_f<double>(12.1, dub); << endl;
    cout << use_f<double>(12.1, Fq(10.0)); << endl;
    cout << use_if<double>(12.1, [] (double u) { return u * u; }) << endl;
    ```
***

# <span style="color:#ff0000;">c++11新增的其他内容 (并行编程 / 新增的库)
## <span style="color:#ff8000;">并行编程
  - 关键字thread_local
    ```c++
    支持线程化的内存模型，将变量声明为静态存储，其持续性与特定线程相关，线程过期时变量也将过期
    ```
    原子操作(atomic operation)库头文件atomic
  - 线程支持库头文件thread / mutex / condition_variable / future
## <span style="color:#ff8000;">新增的库
  - random 随机数工具
  - chrono 提供了处理时间间隔的途径
  - tuple 广义的pair对象，可存储任意多个类型不同的值
  - ratio 有理数算数库
  - regex 正则表达式支持库
