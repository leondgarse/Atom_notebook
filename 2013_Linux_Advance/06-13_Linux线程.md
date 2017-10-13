- ___2013 - 06 - 13（Linux线程）___
***

# <span style="color:#ff0000;">基本概念
  - 为了减少进程间上下文切换时的系统开销，引入了轻量级进程的概念，也就是线程
  - 实现将原本单进程的串行化任务处理，变为相互独立的任务处理之间的交叉进行Linux里同样用task_struct来描述线程，线程和进程都参与统一的调度 
 
  - 多线程通过第三方的线程库来实现：New Posix Thread Library(NPTL)
  - 可以将POSIX特征测试宏_POSIX_THREADS用于#ifdef测试，以确定是否支持线程 
***

# <span style="color:#ff0000;">线程资源
  - 在同一进程中创建的线程共享该进程的地址空间 
  - 多个线程间共享以下资源：可执行指令、静态数据、进程中打开的文件描述符、信号处理函数 当前工作目录 
  - 每个线程私有的资源：线程ID(TID)、PC(程序计数器)和相关寄存器、错误号(errno) 、堆栈(局部变量、返回地址)、信号掩码和优先级、执行状态和属性 
***

# <span style="color:#ff0000;">线程相关函数
  - \#include <pthread.h> 
  - int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine) (void *), void *arg); 
    ```python
    创建新线程，成功返回0， 出错返回错误编号，但不会设置errno 
    thread储存返回的线程ID，attr指示创建线程的属性，为NULL则默认属性 
    void *(*start_routine) (void *)指示线程中执行的函数返回值为void*，参数也为void* 
    arg参数表示传给线程执行函数的参数地址 ，如果需要传递的参数不止一个， 那么需要把这些参数放到一个结构中 
 
    eg: int err; 
        if ((err = pthread_create(&amp;ntid, NULL, thr_func, NULL)) != 0) 
            err_exit("cannt create thread: %s\n", strerror(err));                         
    ```
  - void pthread_exit(void *retval); 
    ```python
    结束线程，若在线程中调用exit()则会结束整个进程，可返回void*型参数 
    如果线程被取消，由retval指定的内存单元就置为PTHREAD_CANCELED 
    ```
  - int pthread_join(pthread_t thread, void **retval); 
    ```python
    阻塞调用线程，等待ID号为thread的线程结束，并将其返回值放到retval里， 
    成功返回0， 出错返回错误编号，但不会设置errno 
    ```
  - int pthread_cancel(pthread_t thread); 
    ```python
    取消一个线程的执行，成功返回0， 出错返回错误编号，但不会设置errno 
    ```
  - pthread_t pthread_self(void); 
    ```python
    获取自身线程ID ，无符号长整型 
    ```
  - int pthread_equal(pthread_t t1, pthread_t t2); 
    ```python
    比较两个线程ID，相等返回0，否则返回非0 
    实现的时候可能用一个结构体来代表pthread_t数据类型， 因此必须使用函数来进行两个线程ID的比较 
    ```
***

# <span style="color:#ff0000;">线程间同步（信号量）
  - \#include <semaphore.h> 
  - int sem_init(sem_t *sem, int pshared, unsigned int value); 
    ```python
    sem初始化的信号量 
    pshared信号量共享的范围（0：线程间使用，非0：进程间使用） 
    val信号量初值 
    ```
  - int sem_wait(sem_t *sem); 
    ```python
    P操作，获得资源 
    ```
  - int sem_post(sem_t *sem); 
    ```python
    V操作，释放资源 
    ```
***

# <span style="color:#ff0000;">线程间互斥（互斥锁）
  - \#include <pthread.h> 
  - int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutex‐attr_t *mutexattr); 
    ```python
    对锁进行初始化，要用默认的属性初始化互斥量，只需把mutexattr设置为NULL 
    还有一种初始化方法，即把互斥量置为常量PTHREAD_MUTEX_INITIALIZER（只对静态分配的互斥量）。 
    ```
  - int pthread_mutex_lock(pthread_mutex_t *mutex); 
  - int pthread_mutex_unlock(pthread_mutex_t *mutex); 
  - int pthread_mutex_trylock(pthread_mutex_t *mutex); 
    ```python
    尝试对互斥锁进行加锁，若互斥量处于未锁住状态，则将锁住互斥量，返回0； 
    若不能锁住互斥量，则返回EBUSY 
    ```
  - int pthread_mutex_destroy(pthread_mutex_t *mutex); 
    ```python
    如果动态分配互斥量（如通过malloc函数），那么在释放内存前需要调用该函数 
    ```
  - 死锁： 
    ```python
    如果一个线程试图对同一个互斥量加锁两次，就会陷入死锁状态；
    或者两个线程都在相互请求另一个像乘占有的资源，也会产生死锁。 
 
    可以通过小心的控制互斥量加锁的顺序来避免死锁的产生，例如，总是在对互斥量B加锁之前锁住互斥量A；
    或者使用pthread_mutex_trylock接口避免死锁，即如果返回成功则继续进行，
            否则先释放已经占有的资源，做好清理工作，然后过一段时间重新尝试。 
    ```
***

# <span style="color:#ff0000;">线程属性
  - \#include <pthread.h> 
  - int pthread_attr_init(pthread_attr_t *attr); 
    ```python
    以默认值初始化线程属性配置，如果要修改其中个别属性的值，需要调用其他函数 
    ```
  - int pthread_attr_destroy(pthread_attr_t *attr); 
    ```python
    去除对pthread_attr_t结构的初始化，并用无效值初始化属性对象 
    ```
***

# <span style="color:#ff0000;">pread / pwrite
  - 线程中对所有线程共享的文件描述符进行读写操作应使用，以保证操作的原子性，并且不对文件指针进行更新 
