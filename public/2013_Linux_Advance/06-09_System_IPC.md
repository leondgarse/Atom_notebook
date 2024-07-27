- 2013 - 06 - 09（System IPC）
- System IPC 对象：（基于内核来实现，一旦创建将永久保存）
  ```python
  共享内存：数据交换
  消息队列：数据交换
  信号量/信号灯：同步与互斥问题

  IPC对象cmd共通参数：IPC_STAT读取消息队列的属性；
                  IPC_SET设置消息队列的属性；
                  IPC_RMID删除消息队列。

  获取键值：
          #include <sys/types.h>
          #include <sys/ipc.h>
          key_t ftok(const char *pathname, int proj_id);
                  //将一个路径名与一个pid转化为一个System V IPC key
  ```
***

# <span style="color:#ff0000;">1). 共享内存
    ```python
    最为高效的进程间通信方式，进程可以直接读写这块内存，而不需要进行数据拷贝；
            但需要依靠某种同步机制，如互斥锁和信号量。

    #include <sys/ipc.h>
    #include <sys/shm.h>

    int shmget(key_t key, size_t size, int shmflg);
            // 分配size大小的共享内存空间；
              若为创建新段（服务器进程），则必须指定size，通常为系统页长的整数倍，
              创建新段时，内容初始化为0；
              如果是引用一个现存的段（客户进程），则将size指定为0；
              key键值为IPC_PRIVATE用于创建新IPC对象；
              shmflg值为IPC_CREAT/IPC_EXCL | (mode_flags)。

    void *shmat(int shmid, const void *shmaddr, int shmflg);
            // 将shmid所标识的共享内存空间映射到调用者的地址空间；
              shmaddr将共享内存映射到指定地址，若为NULL，则表示自动分配；
              shmflg为SHM_RDONLY表示只读，0默认为读写，没有只写标志；
              失败返回（void *）-1

    int shmdt(const void *shmaddr);
            // 解除映射

    int shmctl(int shmid, int cmd, struct shmid_ds *buf);
            // 共享内存操作，cmd: IPC_RMID(删除对象)
    ```
    <br />
***

# <span style="color:#ff0000;">2). 消息队列
    ```python
    #include <sys/types.h>
    #include <sys/ipc.h>
    #include <sys/msg.h>

    int msgget(key_t key, int msgflg);
            // 创建或打开一个消息队列，获得消息队列id号，flag值为IPC_CREAT/IPC_EXCL |
                     (mode_flags)

    int msgsnd(int msqid, const void *msgp, size_t msgsz, int msgflg);
            // msgp是一个调用者定义的结构体指针：
                    struct msgbuf {
                            long mtype;    /* message type, must be > 0 */
                            char mtext[1];  /* message data */
                    };
              msgsz为正文部分即mtext大小
              msgflg：IPC_NOWAIT消息没有发送完函数也会立即返回，0发送完返回

    ssize_t msgrcv(int msqid, void *msgp, size_t msgsz, long msgtyp, int msgflg);
            // 读取消息后删除，并将正文部分存储在msgp中；
              msgtyp参数 == 0，则接收队列中的第一条消息；
                    > 0 则接收队列中第一条mtype == msgtyp 的消息；
                    < 0 则接收队列中第一条mtype小于等于msgtyp绝对值，且mtype类型值小的消息。

    int msgctl(int msqid, int cmd, struct msqid_ds *buf);
            // cmd：IPC_STAT读取消息队列的属性，保存与buf中；
                    IPC_SET设置消息队列的属性，取自buf中；
                    IPC_RMID删除消息队列
    ```
***

# <span style="color:#ff0000;">3). 信号量/信号灯
    ```python
    进程或线程间同步的机制
    二值信号灯：资源可用时值为1，不可用时为0；
    计数信号灯：统计资源个数，其值代表可用资源数

    等待操作是等待信号灯的值变为大于0，然后将其值减1；
    释放操作则相反，用来唤醒等待资源的进程或线程。

    System V 的信号灯由内核维护，是一个或多个信号灯的集合，其中每一个都是单独的计数信号灯。

    #include <sys/types.h>
    #include <sys/ipc.h>
    #include <sys/sem.h>

    int semget(key_t key, int nsems, int semflg);
            //获得一个信号灯集的id；
              nsems为集合中信号灯的数量，若为创建新集合（服务器进程），则必须指定nsems，
                    如果是引用一个现存的集合（客户进程），则将nsems指定为0；
              semflg值为IPC_CREAT/IPC_EXCL | (mode_flags)。

    int semop(int semid, struct sembuf *sops, unsigned nsops);
            // semid要操作的信号灯集id号；
              sops调用程序需定义结构体如下形式（<sys/sem.h>中已定义）：
                    struct sembuf{
                            unsigned short sem_num; /* 要操作的信号灯的编号*/
                            short     sem_op;  /* 0等待；1资源释放，V操作；
                                            -1资源分配，P操作*/
                            short     sem_flg; /* operation flags 0，IPC_NOWAIT,SEM_UNDO*/
                    };
              nsops要操作的信号灯个数，对应于结构体数组sops的数量。

    int semctl(int semid, int semnum, int cmd, .../*union semun arg*/);
            // semid要操作的信号灯集id号；
              semnum要修改的信号灯编号，其值在0 ~ nsems-1；
              cmd：GETVAL获取信号灯的值（semval），SETVAL设置信号灯的值，IPC_RMID删除信号灯集
                    依cmd命令，调用程序需定义共用体：
                    union semun {
                            int val;                /* Value for SETVAL */
                            struct semid_ds *buf;        /* Buffer for IPC_STAT, IPC_SET */
                            unsigned short *array;        /* Array for GETALL, SETALL */
                            struct seminfo *__buf;        /* Buffer for IPC_INFO
                                            (Linux-specific) */
                    };
    ```
