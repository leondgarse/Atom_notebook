# ___2013 - 06 - 06 Linux进程___
***

# 基于Linux的开发
  - 操作系统特点：并行，多任务
  - cpu 调度系统：FIFO，短进程优先，时间片轮换
  - 一个进程可有一个或多个线程
    - 进程是系统分配资源的最小单位
    - 线程是cpu调度的最小单位
  - 程序与进程：一对多，一个程序可以有多个进程并发执行
    - 进程：程序一次执行的过程，是一个动态的程序执行过程，包括创建，调度，消亡
    - 进程除包括程序的正文段与用户数居段（指令和数据）
    - 还包括运行时的系统数据段（程序计数器值，CPU的所有寄存器值以及存储临时数据的进程堆栈）
  - 进程类型：交互进程，批处理进程，守护进程
  - 进程运行状态：就绪态，运行态，阻塞态（可中断，不可中断），暂停态，僵尸态 --> （wait/waitpid） --> 死亡态
  - 进程的执行模式：用户模式 --> （中断或系统调用） --> 内核模式
***

# 调用 pid_t fork(void)
  - 创建4GB的虚拟空间
  - 将原进程的堆栈、数据段复制到新的4GB空间，代码段共享
  - 原进程与复制出的进程都从fork()调用的下一条语句执行
  - 函数返回值：父进程中返回子进程PID，子进程中返回0，出错返回-1
  - 父进程结束之后，子进程的父进程变为init进程
***

# int exec() 函数族
  - fork()函数创建子进程，在子进程中调用exec()函数族以加载执行另一个函数，其参数要以NULL结尾。
  - execlp/execvp当指定filename作为参数时，如果包含/则视为路径名， 否则就按PATH环境变量，在其指定的各目录中搜索可执行文件
***

# wait 函数
  - 阻塞进程，直到任一个子进程结束或接收到一个信号为止
    ```c
    pid_t wait(int *status)
    ```
  - 等待特定某个子函数结束，其参数pid可为 > 0，-1，0，< -1
    ```c
    pid_t waitpid(pid_t pid, int *status, int options)
    ```
  - 轮询方式等待父进程结束
    ```c
    while (getppid() != 1)
            sleep();
    ```
***

# void exit(int status) 函数
  - 相对于_exit()，exit()函数会调用退出处理程序，清理I/O缓冲
  - 可以在每个子进程块的执行结束后添加exit(0)
***

# alarm 函数
  ```c
  #include <unistd.h>
  unsigned int alarm(unsigned int seconds);
  ```
  - 返回0或以前设置的闹钟时间的剩余秒数
  - 每个进程只能有一个闹钟时间，如果在调用alarm时，以前已为该进程设置过闹钟时钟，而且它还没超时，则将该闹钟的余留值作为本次alarm函数调用的返回值
  - 如使用 return alarm(0); 将取消之前的闹钟时钟，返回其余留值
***

# sleep 函数
  ```c
  #include <unistd.h>
  unsigned int sleep(unsigned int seconds);
  ```
  - 当调用进程捕捉到一个信号，并从信号处理函数返回时，sleep函数返回未睡够的秒数
***

# system 函数
  ```c
  #include <stdlib.h>
  int system(const char *command);
  ```
  - 在函数中执行一个shell命令，如果command是一个空指针，则仅当命令处理命令可用时，system返回非0值，这一特征可用来确定系统上是否支持system函数
  - 在其实现中调用了fork、exec、和waitpid，因此有这三者的返回值
***

# 进程组 setpgid / getpgrp
    ```c
    #include <unistd.h>
    int setpgid(pid_t pid, pid_t pgid);         
            //设置进程组ID，可加入一个现有的组或者创建一个新组，
                    如果其两个参数相等，则该进程成为进程组组长；
                    如果pid是0，则使用调用者的pid；如果pgid是0，则由pid指定的进程ID将用作进程组ID
            一个进程只能为它自己或它的子进程设置进程组ID，一般会在fork之后调用此函数，
                    使父进程设置其子进程的进程组ID，并且使子进程设置其自己的进程组ID，以保证其生效
    pid_t getpgrp(void);                //得到进程组ID
            //每个进程组都有一个组长进程，组长进程的标识是，其进程组ID等于其进程ID
    ```
***

# 10). 会话
    ```python
    会话(session)是一个或多个进程组的集合。
    #include <unistd.h>
    pid_t setsid(void);
            //创建一个新会话，并设置新的进程组ID
              如果调用函数不是一个进程组的组长，则此函数创建一个新会话：
                    该进程变成新会话首进程、该进程成为新进程组的组长进程、该进程没有控制终端
              如果该进程是一个进程组的组长，则此函数出错返回，
                    所以通常先调用fork，然后使其父进程终止，而子进程继续。因为子进程继承了
                    父进程的进程组ID，而其进程ID是新分配的，保证子进程不会是一个进程组的组长
    pid_t getsid(pid_t pid);
            返回调用进程的会话首进程的进程组ID
    ```
***

# 11). Linux创建守护进程示例
    ```python
    #include <stdio.h>

    void init_daemon(void)
    {
    /****创建守护进程****/
            pid_t pid;
            //第一步：fork创建子进程
            if((pid = fork()) < 0)
                    err_quit("fork error %s time", "first");
            else if(pid > 0)
                    exit(0);
            //第二步：创建新会话
            if(setsid() < 0)
                    err_quit("setsid error %s time", "first");
            //第三步：子进程继续运行，父进程结束时将会产生SIGHUP信号
            //                        忽略此SIGHUP信号，并用fork创建子进程
            signal(SIGHUP, SIG_IGN);

            if(pid = fork < 0)
                    err_quit("fork error %s time", "second");
            else if(pid > 0)
                    exit(0);
            //第四步：创建新的进程组
            if(setpgrp() < 0)
                    err_quit("setpgrp error %s time", "first");
            //第五步：关闭所有文件描述符
            int i, max_fd = sysconf(_SC_OPEN_MAX);
            for(i = 0; i < max_fd; i++)
                    close(i);
            //第六步：消除umask影响
            umask(0);
            //第七步：改变当前目录为根目录
            chdir("/");
            //第八步：重新定向标准IO描述符
            open("dev/null", O_RDWR);
            dup(0, 1);
            dup(0, 2);
            /****创建守护进程完成****/
    }
    ```
***

# 12). 使用syslog来记录守护进程的LOG
    ```python
    用户空间的守护进程klogd，运行后消息将追加到/var/log/<messages>
    大多数用户进程（守护进程）调用syslog(3)函数以产生日志消息，并是消息发送至UNIX域 数据包套接字/dev/log
    #include <syslog.h>
    void openlog(const char *ident, int option, int facility);
            //可选，如果不调用openlog，则在第一次调用syslog时，自动调用openlog
                    ident一般是程序的名称，可将它加入到每则日志文件中
                    option指定许多选项的位屏蔽LOG_CONS/LOG_NDELAY/
                            LOG_NOWAIT/LOG_ODELAY/LOG_PERROR/LOG_PID（见man手册）
                    facility(设施)参数的目的是可以让配置文件说明，来自不同设施的消息将以不同的
                            方式进行处理，若为0，则可将设施作为priority参数的一个部分进行说明
    void syslog(int priority, const char *format, ...);
            //产生一个日志消息，其priority参数是facility与level的组合
                    format参数以及其他参数传至vsprintf以便进行格式化，
                    在format中，每个%m都先被代换成对应与errno值的出错消息字符串
    void closelog(void);
            //可选，只是关闭层被用于与syslog守护进程通信的描述符
    ```
***

# 13). 文件锁
    ```python
    #include <sys/file.h>
     int flock(int fd, int operation);
            //operation选项：LOCK_SH共享锁/读锁，LOCK_EX互斥锁/写锁，
            //        LOCK_UN解锁，LOCK_NB不能获得指定锁时不阻塞。
    ```

    ```python
    #include <unistd.h>
    #include <fcntl.h>
    int fcntl(int fd, int cmd, ... /* arg */ );
    int fcntl(int fd, int cmd, struct flock *lock);
            //对文件的部分上锁。
    ```
