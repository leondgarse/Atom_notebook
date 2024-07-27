- 2013 - 06 - 11（<curses.h>）
- $ sudo apt-get install libncurses5-dev
- 1).        initscr();        //一般curses程序必须先呼叫的函数，启动curses模式，以呼叫endwin()来暂停
  ```python
  cbreak();        //开启后除ctrl与delete仍被视为特殊字符外，其他字符被存储在buffer，直到输
                          入RETURN或NEWLINE
  nonl();        //输入资料是RETURN是否被对应为NEWLINE，输出资料时NEWLINE是否对应为
                          RETURN，默认开启
  noecho();        //从键盘输入时是否显示在屏幕上，默认开启
  intrflush(stdscr, FALSE);        //TRUE时，当输入中断字元时反应较快速，但可能造成屏幕错乱
  keypad(stdscr, TRUE);        //开启后可以使用键盘上一些特殊字元，如方向键等
  refresh();        //刷新屏幕内容，否则更改内容不会在实际屏幕上显示
 
  LINES/COLS：curses初始化时读入，分别为当前屏幕的最大行号与最大列号，使用时注意： 
          当移动到LINE行输出时，会直接在当前位置后面输出，不会发生光标移动； 
          当移动到COLS列输出时，会跳转到下一行行首输出； 
          所以可以使用的最后一行与最后一列分别是LINES-1/COLS-1 
- 2).        #define KEY_DOWN        0402                /* down-arrow key */ 
  ```python
  #define KEY_UP                0403                /* up-arrow key */ 
  #define KEY_LEFT                0404                /* left-arrow key */ 
  #define KEY_RIGHT                0405                /* right-arrow key */ 
  #define KEY_HOME                0406                /* home key */ 
  #define KEY_BACKSPACE        0407                /* backspace key */ 
  #define KEY_F0                        0410                /* Function keys. Space for 64 */ 
  #define KEY_F(n)                (KEY_F0+(n))        /* Value of function key n */ 
  #define KEY_DL                        0510                /* delete-line key */ 
  #define KEY_DC                0512                /* delete-character key */ 
  #define KEY_CLEAR                0515                /* clear-screen or erase key */ 
  #define KEY_EOS                0516                /* clear-to-end-of-screen key */ 
  #define KEY_EOL                0517                /* clear-to-end-of-line key */ 
  #define KEY_NPAGE        0522                /* next-page key */ 
  #define KEY_PPAGE        0523                /* previous-page key */ 
 
  [TAB] \t; [ENTER] \r; [ESC] 27; [BACKSPACE] 127 
- 3).        字符以及字符串操作： 
  ```python
  特殊图形字符，通常是绘图或者制表，这些常量都是以ACS_开头 
  chtype ch;                //无符号长整型，高位部分储存字符的额外属性信息，如色彩等 
 
  int addch(ch);        //在光标位置输出字符ch，同时光标右移一个位置 
  int echochar(ch);        //我那成输出后不用调用refresh即可完成字符输出 
  int addstr();        //光标位置输出字符串str，同时后移光标位置，如果字符串的长度超出了屏幕的大 
                                          小，字符串会被截断 
  int printw(fmt, [, arg...]);        //格式化输出 
  int getch();        //从键盘读入一个字符 
  int ungetch(ch);        //返回字符到输入队列
  int getstr(str);        //从键盘终端接收字符串，必须以'\n'结尾，存储时'\n'被空字符串代替 
  int scanw(fmt, [, argptr...]);        //格式化输入 
  int insch(ch);        //在光标位置插入字符，之后光标不会移动(此处不同于教程) 
  int delch();        //删除光标当前位置字符，同时其后面字符顺序左移 
  int insertln();        //整行插入 
  int deleteln();        //正行删除 
  box(win, ch1, ch2);        //自动画方框，ch1垂直方向所用字元，ch2水平方向所用字元
- 4).        字符属性： 
  ```python
  一般作为attrset的参数，或者当字符传递给函数时可以直接将这些属性与字符取“|” 
          A_NORMAL: 标准的显示模式. 
          A_BLINK: 闪烁属性. 
          A_BOLD: 加粗属性. 
          A_DIM: 半透明属性. 
          A_REVERSES: 反显属性。 
          A_STANDDUT: 高亮度显示属性. 
          A_UNDERLINE: 加下划线. 
          A_ALTCHARSET: 可代替字符集. 
          COLOR_PAIR(n): 字符的背景和前景属性. 
  int attron(attrs);        //设置指定的属性，会与当前的属性设置叠加，且会影响之后的所有文本字符 
  int attrset();        //设置当前的属性，用新属性替代原属性，attrset(0)将关闭所有属性 
  int attroff();        //关闭某个属性 
 
  int standout();        //在当前屏幕上打开高亮度显示属性 
  int standent();        //关闭所有属性 
- 5).        光标操作： 
  ```python
  int move(y, x);        //移动逻辑光标的绝对位置，屏幕的行宽和列宽定义为(LINES - 1, COLS - 1) 
          移动光标与显示字符结合：mvfunc(y, x, [arg,...]); 返回值与没有mv时一样 
  getyx(win, y, x);        //得到光标当前位置 
  void mvcur(last_y, last_x, new_y, new_x);        //移动物理光标 
  curs_set(int);        //参数为0/1/2，分别表示光标的状态为隐藏、正常、高亮 
- 6).        清除屏幕： 
  ```python
  int clear();        //清除整个屏幕，移动光标到(0, 0)，并调用clearok()函数，调用refresh完成清除 
  int erase();        //不会自动调用clearok()函数，清除不彻底 
  int clrtoeol();        //清除光标位置到该行末尾的所有内容 
  int clrtobot();        //清除光标当前位置到屏幕底端的所有内容 
- 7).        颜色属性： 
  ```python
  has_solors();        //如果终端支持彩色显示，返回TRUE，否则返回FALSE 
  can_change_clors();        //如果终端支持改变默认的颜色表，返回TRUE，否则返回FALSE 
 
  int start_color();        //如果终端不支持彩色，返回ERR，使用颜色时，首先必须调用这个函数 
  int init_pair(pair_num, front_color, back_color);        //改变颜色配对表条目中的颜色定义 
  int init_color(color_num, r, g, b);         //改变默认颜色的RGB分量 
 
- 8).        窗口： 
  ```python
  自定义窗口函数与标准屏幕交互的函数基本是相同的，比如窗口刷新用的通常是wrefresh()， 
          爽口清除函数werae()和wclear()，局部清除函数wclrt()和wclrtoeol()。 
  WINDOW结构体常用成员： 
          _cury, _curx: 当前窗口中光标的位置。 
          _begy, _begxt: 当前窗口的左上角相对于标准屏幕的位置。 
          _palent: 当前窗口的父窗口。 
          _parx, _pary子窗口的左上角相对于父窗口的位置。 
          _attrs: 当前窗口的属性。 
          _bkgd: 当前窗口的背景字符。 
 
  创建和删除： 
  WINDOW *newwin(int lines, cols, begin_y, begin_x); 
          //若lines或cols为0，函数将自动创建一个范围包括整个终端屏幕的窗口 
  int delwin(WINDOW *win); 
          //清除窗口内存，在主窗口删除之前必须先删除与它关联的所有子窗口 
  WINDOW *subwin(WINDOW *win, int lines, cols, begin_y, begin_x); 
          //创建子窗口，通过WINDOW结构体中的_parent指针，可以得到子窗口的父窗口 
                  win父窗口的指针，begin_y/begin_x子窗口左上角相对于标准屏幕的位置 
          由于子窗口在父窗口之中，所以对其中任一个的改变都会同时影响到它们 
  WINDOW *derwin(WINDOW *win, int lines, cols, begin_y, begin_x); 
          //不同于subwin()，derwin()的begin_y/begin_x是子窗口左上角相对于父窗口的位置 
 
  输入和输出： 
          窗口中输入和输出函数名称通常是由标准屏幕的操作函数在头部加上“w”组合而成，同时将 
                  窗口的指针作为第一个参数传递。 
  窗口坐标： 
          getbegyx(win, y, x)函敷用来获取指定窗口的起始坐标，坐标值保存在变量 y, x中。 
          gemaxyx(win, y, x)函数用来获取窗口的最大坐标。 
          gelparyx(win, y, x)函数用来获取子窗口相对于父窗口的起始坐标。 
          getyx(win, y, x)函数获取当前窗口中的光标位置。 
                  //需要注意的是坐标(y, x)的前面并没有'&amp;'符号 
 
  窗口复制： 
          int overlay(WINDOW *srcwin, WINDOW *dstwin); 
                  //非破坏性复制，即不会复制原窗口中的空字符，通常用来从重叠窗口中建立组合屏幕 
                          注意这个函数不能适用于非关联窗口 
          int overwrite(WINDOW *srcwin, WINDOW *dstwin); 
                  //破坏性复制，完全清空目标窗口内容 
          int copywin(srcwin, dstwin, int sminrow, smincol, 
                          dminrow, dmincol, dmaxrow, dmaxcol, overlay); 
                  //可以将原窗口中的热和部分复制到目标窗口的任何部分 
  移动窗口： 
          int mvwin(win, y, x);        //编程示例 
  激活窗口： 
          void touchwin(win);        //编程示例 
  窗口装饰： 
          int box(win, vert, hort); 
                  //vert是垂直方向的字符，通常为ACS_VLINE，为双字节字符； 
                          hort是水平方向的字符，通常为ACS_HLINE，为双字节字符。 
                          使用box(win, 0, 0)即按通常效果绘制边框 
          int border(win, ls, rs, ts, bs, tl, tr, bl, br); 
                  //后面参数分别对应：窗口左边字符/窗口右边字符/窗口上边字符/窗口下边字符 
                          窗口左上角字符/窗口右上角字符/窗口左下角字符/窗口右下角字符 
          int hline(chtype ch, int n); 
          int vline(chtype ch, int n); 
                  //分别为绘制水平线/绘制垂直线，对应有whline/wvline 
 
  设置窗口标志： 
          void leaveok(WINDOW *win, bool state);        //refresh后光标的位置 
          void scrollok(WINDOW *win, bool state);        //是否允许滚动，编程示例 
          void clearok(WINDOW *win, bool state);        //如果设置了_clear，那么对于refresh()的
                                          每次调用都会自动清除屏幕，而不管是哪个窗口在调用 
  窗口刷新： 
          int wrefresh(WINDOW *win); 
                  //对wnoutrefresh与doupdate的轮流调用 
          int wnoutrefresh(WINDOW *win); 
                  //将指定窗口的内容复制到虚拟屏幕的数据结构中 
          int doupdate(WINDOW *win); 
                  //刷新虚拟屏幕与实际屏幕的不同之处 
           
          直接调用wrefresh()是一种较低效率的刷新。如果直接分开调用wnoutrefresh()和 
                  doupdate()，刷新效率比使用wrefresh()本身直接刷新更有效。通过对每一个窗口调 
                  用 wnoutrefresh()，然后只集中一次doupdate()，这样需要输出的字符的总数量和总 
                  的处理时间得到最小化。 
 
  窗口重画： 
          redrawwin()/wredrawwin()/wredrawln() 
  屏幕转储 
          int putwin(WINDOW *win, FILE *filep);        //编程示例 
                  //将窗口中的所有数据复制到filep指定的已经打开的文件中 
          WINDOW *getwin(FILE *filep); 
                  //返回由filep中的内容创建的窗口指针 
          int scr_dump(const char *filename); 
          int scr_restore(const char *filename); 
                  //针对标准屏幕的窗口转储操作，与上面两个函数不能混合使用 
  窗口使用示例 
- 9).        基垫 
  ```python
  基垫是与标准屏幕没有直接关联的，它可以非常大，也可以完全不可见，大部分的窗口操作 
          函数也都可以用在基垫上，但创建/刷新函数会有所不同。 
  创建与销毁： 
          WINDOW *newpad(int lines, cols); 
          WINDOW *subpad(WINDOW *pad, int lines, cols, begin_y, begin_x); 
                  //创建子基垫 
  刷新基垫： 
          基垫传间后与标准屏幕的相对位置并不固定，因此不能使用wrefresh()来刷新 
          int prefresh(WINDOW *pad, int prow, pcol, sminrow, smincol,
                                  smaxrow, smaxcol); 
                  //prow/pcol分别是基垫中需要刷新部分的左上角位置， 
                          sminrow/smincol/smaxrow/smaxcol指定标准屏幕上显示基垫的矩形区域 
                  prefresh()也是pnoutrefresh()与doupdate的联合调用 
                  编程示例 
 
  ```
  <br />
- 10).        鼠标： 
  ```python
  考虑到可移植性问题，对于那些使用鼠标的操作最好同时能够提供键盘支持 
  事件： 
          BUTTON1_PRESSED                按下鼠标键1 (~4) 
          BUTTON1_RELEASED                松开鼠标键1 (~4) 
          BUTTON1_CLICKED                单击鼠标键1 (~4) 
          BUTTON1_DOUBLE_CLICKED双击鼠标键1 (~4) 
          BUTTON1_TRIPLE_CLICKED        三击鼠标键1 (~4) 
          BUTTON_SHIFT/CTRL/ALT        在鼠标状态改变期间按下SHIFT/CTRL/ALT键 
          ALL_MOUSE_EVENT                报告所有鼠标状态 
          REPORT_MOUSE_POSITION        鼠标移动 
 
          MENVENT结构体： 
                  typedef struct { 
                          short id;                //消息来源的不同设备 
                          int x, y, z;                //事件发生时鼠标位置，该位置是相对于屏幕左上角的 
                          mmask_t bstate;        //事件发生时鼠标按键的状态 
                  }MEVENT; 
 
  鼠标操作函数： 
          mmask_t mousemask(mmask_t newmask, mmask_t *oldmask); 
                  //初始化鼠标系统，所有的鼠标操作中第一个调用的函数，通知系统截获并处理参数 
                          newmask指定的鼠标事件，原有的时间保存到oldmask中(可为NULL)；如果对所 
                          有鼠标事件都关心可使用ALL_MOUSE_EVENT事件；如果需要关闭所有事件，可 
                          使用0 
                  eg：处理所有鼠标双击事件 
                          mousemask(BUTTON1_DOUBLE_CLICKED|BUTTON2_DOUBLE_CLICKED| 
                                          BUTTON3_DOUBLE_CLICKED|BUTTON4_DOUBLE_CLICKED|, 
                                                  old_mask); 
          int mouseinterval(int erval); 
                  //设置鼠标一次点击的时间间隔，单位是毫秒，默认为1/5秒 
          int getmouse(MEVENT *envnt); 
                  //使用getch()和wgetch()从鼠标接收输入，如果是鼠标输入，则其返回KEY_MOUSE 
                          常量，随后可以使用getmouse()函数从鼠标事件队列中获取下一个鼠标事件 
                  eg： 
                          MEVENT event; 
                          ch = getch(); 
                          if (ch = KEY_MOUSE) { 
                                  if (getmouse(&amp;event) == OK) 
                                          /*event handler*/ 
                          } 
          int ungetmouse(MEVENT *envnt); 
                  //将KEY_MOUSE时间返回给getch()和wgetch()函数的输入队列，同时将鼠标事件返回 
                          给鼠标事件队列 
          bool wenclose(WINDOW *win, int y, int x); 
                  //判断参数中的(x, y)坐标是否在指定的窗口中，其中的(x, y)参数通常是getmouse获取 
                          的MEVENT结构体中的鼠标位置 
          bool wmouse_trafo(const WINDOW *win, int *pY, *pX, bool to_screen); 
                  //如果to_screen为TRUE则将窗口坐标转化为屏幕相对坐标，为FALSE则相反 
 
          鼠标程序开发步骤： 
                  1).        使用mousemask()函数初始化需要获取的鼠标事件 
                  2).        循环使用getch()/wgetch()函数获取键盘或者鼠标输入，当返回KEY_MOUSE时， 
                          则表明是鼠标输入 
                  3).        通过getmouse()获取触发的鼠标事件，根据具体的时间进行处理 
          编程示例 
