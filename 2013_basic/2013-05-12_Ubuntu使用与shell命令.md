# ___2013 - 05 - 12（Ubuntu使用与shell命令）___
***

# <span style="color:#ff0000;">参数
  - 查看gcc版本号：gcc --version
  - 查看linux内核版本号：uname -a
  - exit 35     # 添加一个 exit 退出命令
  - 比较两个排序后的文件内容 comm
  - mp3info查找音频文件，并删除比特率大于320的
    ```c
    mp3info -x -p "%r#%f\n" *.mp3 | grep 320 | cut -d '#' -f 2- | sed 's/ /\\ /g' | xargs rm {} \;
    ```
  - 挂载ISO文件
    ```c
    sudo mount -o loop /media/leondgarse/GrandFiles_Seag/Operating_Systems/cn_windows_7_ultimate_with_sp1.iso /media/cdrom0/
    ```
  - 挂载squashfs
    ```c
    sudo mount -o loop /media/leondgarse/GrandFiles_Seag/Operating_Systems/squashfs_backup/2017-01-19_201732.squashfs /media/cdrom0/
    ```
  - 格式化为FAT32，-I选项指定整个盘，NTFS格式使用mkfs.ntfs
    ```c
    sudo mkfs.vfat -F 32 -I /dev/sdc
    ```
## <span style="color:#ff8000;">apt-get
  - apt-get --purge remove ...... （完全删除）
  - apt-get -f install        （修复依赖关系）
  - apt-get install -d foobar （只下载不安装）
    ```c
    Q: The following packages have been kept back
    A: sudo apt-get -u dist-upgrade
    ```
    ```c
    Q: Unable to lock directory /var/lib/apt/lists/
    A: sudo rm /var/lib/apt/lists/lock
    ```
  - 404 error while apt-get install libssl-dev
    ```c
    check if
            $ sudo apt-get update
    meet any error like: Failed to fetch http://ppa.launchpad.net/ 404 Not Found

    remove that ppa from update-manager:
            $ /usr/bin/python3 /usr/bin/software-properties-gtk

    or remove it from /etc/apt/sources.list or /etc/apt/sources.list.d
    ```
  - Error sudo: add-apt-repository: command not found
    ```c
    To fix this error, you have to install the software-properties-common:
    $ sudo apt-get install software-properties-common
    This is all. Now your command for adding PPAs works like a charm.
    If you want to find out how I have fixed this error by myself, without external / Google help, read further.
    I have search with apt-file for the add-apt-repository and found out in which package is the command located.
    Apt file searches for files, inside packages and tells you in what package the file you had searched is located.
    It is not installed by default, so you need to do this:
    $ sudo apt-get install apt-file &amp;&amp; apt-file update
    This is how you use apt-file for fishing files inside packages:
    $ apt-file search add-apt-repository<br />        python-software-properties: /usr/bin/add-apt-repository<br />        python-software-properties: /usr/share/man/man1/add-apt-repository.1.gz
    So, indeed, it is in the python-software-properties package.
    ```
  - If you prefer to use the command line or if there is no graphical installer available you can use this command as an administrator:
    ```c
    apt install teamviewer_11.0.xxxxx_i386.deb
    Older systems (Ubuntu 14.04, Debian 7 and below)
    Run this command:
    dpkg -i teamviewer_11.0.xxxxx_i386.deb
    In case dpkg indicates missing dependencies, complete the installation by executing the following command:
    apt-get install -f
    ```
## <span style="color:#ff8000;">echo
  - echo $? 打印终止状态
    ```
    exit(1)表示发生错误后退出程序， exit(0)表示正常退出。
    ```
## <span style="color:#ff8000;">ps
  - -a 显示有其他用户所拥有的进程的状态，
  - -x 显示没有控制终端的进程状态，
  - -j 显示与作业有关的信息：会话ID、进程组ID、控制终端以及终端进程组ID
  - ps aux 与 ps -aux 是不同的命令， ps -aux 将试图打印用户名为“x”的进程，如果该用户不存在则执行ps aux命令，并输出一条警告信息，ps -axj等是同样的结果
## <span style="color:#ff8000;">df / du / dd
  - df命令用于查看一级文件夹大小、使用比例、档案系统及其挂入点：
    ```c
    $ df -h        // -h表示Human-readable输出
    ```
    du命令用于查询文件或文件夹的磁盘使用空间
    ```c
    $ du -hd1        // -d1表示深度为1，若直接使用不带参数的du命令，将会循环列出所有文件和文件夹所使用的空间
    $ du -h --max-depth=1
    ```
  - 硬盘互刻
    ```c
    df                        //查看当前系统躲在硬盘设备节点
    ls /dev/sd*                        //列出当前主机内所有硬盘
    dd if=/dev/sdb of=/dev/sda        //将sdb硬盘内容复制到sda
    ```
## <span style="color:#ff8000;">date
  - 将日期转换为星期：
    ```c
    date -d "Jan 1 2000" +%A
    ```
## <span style="color:#ff8000;">head / tail
  - 显示文件中间几行内容:
    ```c
    cat -n hug-tool.txt | head -n 10 | tail -n +5
    cat -n hug-tool.txt | sed -n '5,10p'
    ```
***

# <span style="color:#ff0000;">配置
  - 使用 PS1=user$: 命令临时更改显示的命令提示符
    ```c
    PS1='[\u@\h: $PWD]# '

    ```
    ubuntu不能man pthread库函数：
    ```c
    sudo apt-get install manpages-posix-dev
    ```
## <span style="color:#ff8000;">grub配置文件
  - grub配置文件/etc/default/grub与/etc/grub.d目录下的对应文件，如修改分辨率、等待时间等可通过修改/etc/default/grub实现
  - 修改grub背景图片：
    ```c
    sudo cp xxx.jpg /boot/grub/back.jpg
    sudo update-grub重启即可
    ```
  - 更改grub背景主题：
    ```c
    将下载的主题文件解压到/boot/grub/themes文件夹中（没有的自己创建）
    然后修改/etc/default/grub
    加入：GRUB_THEME="/boot/grub/themes/******/theme.txt"（主题名自己设置）
    然后sudo grub-update
    ```
## <span style="color:#ff8000;">环境变量
  - 修改：sudo vi /etc/environment添加，或者vi ~/.bashrc添加
    ```c
    source /etc/environment 是配置生效
    ```
  - 误操作环境变量文件/etc/environment，会造成无法登录的状况，ubuntu13.04下字符界面使用
    ```c
    命令：/usr/bin/sudo /usr/bin/vi /etc/environment
    ```
## <span style="color:#ff8000;">SSH
  - Ubuntu使用SSH访问远程Linux服务器： $ ssh leondgarse@192.168.7.11
  - ssh配置文件：man ssh_config
### <span style="color:#00ff00;">ssh key fingerprint
  - $ ssh-keygen -lf ~/.ssh/id_rsa.pub
    ```c
    -l means "list" instead of create a new key
    -f means "filename"

    With newer versions of ssh-keygen, run ssh-keygen -E md5 -lf <fileName> if you want the same format as old
    ssh-keygen -lf also works on known_hosts and authorized_keys files.
    ```
  - ssh-add -l is very similar but lists the fingerprints of keys added to your agent.
### <span style="color:#00ff00;">Escape character:
  - ~?        显示所有
  - ~.        退出SSH连接
  - ~~        输入~
### <span style="color:#00ff00;">解决ssh的" Write failed: Broken pipe"问题
  - < Q > 用 ssh 命令连接服务器之后，如果一段时间不操作，再次进入 Terminal 时会有一段时间没有响应，然后就出现错误提示：
    ```c
    Write failed: Broken pipe
    只能重新用 ssh 命令进行连接。
    ```
    < A >
    ```c
    方法一：如果您有多台服务器，不想在每台服务器上设置，只需在客户端的 ~/.ssh/ 文件夹中添加 config 文件，并添加下面的配置：
    ServerAliveInterval 60

    方法二：如果您有多个人管理服务器，不想在每个客户端进行设置，只需在服务器的 /etc/ssh/sshd_config 中添加如下的配置：
    ClientAliveInterval 60

    方法三：如果您只想让当前的 ssh 保持连接，可以使用以下的命令：
    $ ssh -o ServerAliveInterval=60 user@sshserver

    If you use tmux + ssh, you can use the following configuration file to make all the ssh session keep alive:
    [tonyaw@qdbuild3 ~]$ cat ~/.ssh/config
    Host *
    ServerAliveInterval 60
    ```
### <span style="color:#00ff00;">ssh-add :Could not open a connection to your authentication agent
  - 若执行ssh-add /path/to/xxx.pem是出现这个错误:Could not open a connection to your authentication agent，则先执行如下命令即可：
    ```c
    ssh-agent bash
    ```
### <span style="color:#00ff00;">Save ssh output to a local file
  - ssh user@host | tee -a logfile
### <span style="color:#00ff00;">no matching key exchange method found. Their offer: diffie-hellman-group1-sha1
  - possible solution
    ```c
    The problem isn't the cipher as much as the key exchange.
    Newer open ssh dropped support (by default) for "insecure" key exchanges (SHA1) which are all that are supported by older ios/etc. gear. <br />        I've been updating code on boxes where possible to eliminate this issue but it's really an easy fix. <br /><br />        In /etc/ssh/ssh_config: <br /><br />        Host * <br />        GSSAPIAuthentication yes <br />        KexAlgorithms +diffie-hellman-group1-sha1 <br /><br />        That will add the old kex to your ssh (outbound) and should work ok.
    ```
### <span style="color:#00ff00;">no matching host key type found. Their offer: ssh-dss
  - possible solution
    ```c
    The recent openssh version deprecated DSA keys by default.
    You should pursuit your GIT provider to add some reasonable host key. Relying only on DSA is not a good idea.
    As a workaround, you need to tell your ssh client that you want to accept DSA host keys, as described in the official documentation for legacy usage.
    You have few possibilities, but I recommend to add these lines into your ~/.ssh/config file:
    Host your-host
      HostkeyAlgorithms +ssh-dss
    ```
### <span style="color:#00ff00;">ssh: connect to host 135.251.168.141 port 22: Connection refused
  - apt-get install openssh-server
## <span style="color:#ff8000;">samba
### <span style="color:#00ff00;">samba 的安装
  - $ sudo apt-get install samba smbfs samba-common smbclient
### <span style="color:#00ff00;">创建 Samba 配置文件
  - 保存现有的配置文件: $ sudo cp /etc/samba/smb.conf /etc/samba/smb.conf.bak
  - 打开现有的文件: $ sudo vim /etc/samba/smb.conf
  - 在 smb.conf 最后添加
    ```c
    [username]
    path = /home/username
    available = yes
    browseable = yes
    public = yes
    writable = yes
    ```
### <span style="color:#00ff00;">重启 samba 服务器
  - $ sudo /etc/init.d/smbd reload (修改过 smb.conf 的话要执行一次)
  - $ sudo /etc/init.d/smbd restart
### <span style="color:#00ff00;">查看目标服务器所有的共享目录
  - $ smbclient -L 192.168.7.11 -U leondgarse%123456
### <span style="color:#00ff00;">将目标服务器的共享目录挂载到/media/samba目录下
  - $ sudo mount -t cifs -o username=leondgarse,password=123456 //192.168.7.11/leondgarse /media/samba/
### <span style="color:#00ff00;">开机自动启动samba服务
  - sudo vi /etc/init/samba.conf
  - 加入 start on (local-filesystems and net-device-up)
  - 关闭： sudo sed -i 's/start on/# &/' /etc/init/smbd.conf
## <span style="color:#ff8000;">TFTP
  - tftp / tftpd 设置TFTP 服务
    ```c
    sudo apt-get install tftp tftpd
    sudo apt-get install openbsd-inetd

    sudo mkdir /tftpboot
    sudo chmod 777 /tftpboot -R

    sudo vi /etc/inetd.conf
    在里面填入如下一行:
    tftp dgram udp wait nobody /usr/sbin/tcpd /usr/sbin/in.tftpd /tftpboot

    新建 /etc/default/tftpd-hpa
    #Defaults for tftpd-hpa
    RUN_DAEMON="yes"
    OPTIONS="-l -s /tftpboot"

    $ sudo /etc/init.d/openbsd-inetd reload
    $ sudo /etc/init.d/openbsd-inetd restart
    ```
  - tftp-hpa / tftpd-hpa 设置TFTP 服务
    ```c
    查看源中tftp相关的应用：
    apt-cache search tftpd

    安装tftp-hpa tftpd-hpa:
    sudo apt-get install tftpd-hpa tftp-hpa

    查看tftp服务状态
    sudo service tftpd-hpa status
    或
    netstat -a | grep tftp	# 没有输出

    默认的配置文件：
    /etc/default/tftpd-hpa

    默认tftp根路径：
    /srv/tftp

    配置：
    sudo cp /etc/default/tftpd-hpa /etc/default/tftpd-hpa.ORIGINAL	# 备份
    sudo vi /etc/default/tftpd-hpa
    配置项
        TFTP_OPTIONS="--secure --create"	# 支持创建新文件
        TFTP_DIRECTORY="/tftpboot"	# 修改根路径

    修改根目录权限
    sudo chown -R tftp /tftpboot

    重启服务
    sudo service tftpd-hpa restart

    上传 / 下载
    tftp 127.0.0.1 -c put foo
    tftp 127.0.0.1 -c get foo
    ```
  - tftp中put 命令Access Violation错误：Error code 2: Access violation
    ```c
    tftp服务器缺少必要的身份验证，要上传文件，必须是服务器中已存在同名的文件，且该文件权限允许被覆盖
    首先在服务中创建一个与要上传的文件同名的文件，并更改权限为777
    $ touch a
    $ chmod 777 a
    ```
  - 上传二进制文件时错误 Check data fail, upload failed
    ```c
    linux下tftp默认格式是ascii，尝试指定mode 为 binary
        tftp -m binary 127.0.0.1 -c put foo
    ```
## <span style="color:#ff8000;">Checking conflict IP
  - $ sudo apt-get install arp-scan
  - $ arp-scan -I eth0 -l | grep 192.168.1.42
    ```c
    192.168.1.42 d4:eb:9a:f2:11:a1 (Unknown)
    192.168.1.42 f4:23:a4:38:b5:76 (Unknown) (DUP: 2)
    ```
## <span style="color:#ff8000;">Service running on server
  - Use nmap tool to know which ports are open in that server. nmap is a port scanner. Since it may be possible that ssh server is running on a different port. nmap will give you a list of ports which are open.
    ```c
    $ nmap myserver
    ```
    Now you can check which server is running on a given port. Suppose in the output of nmap, port 2424 is open. Now you can which server is running on 2424 by using nc(netcat) tool.
    ```c
    $ nc -v -nn myserver portno
    ```
    Suppose the output of 2424 port is:
    ```c
    myserver 2424 open
    SSH-2.0-OpenSSH_5.5p1 Debian-4ubuntu5
    ```
    This means ssh is running on 2424.
## <span style="color:#ff8000;">通过 DNS 来读取 Wikipedia 的词条
  - dig +short txt <keyword>.wp.dg.cx
## <span style="color:#ff8000;">Ubuntu11.04+ 中开机打开小键盘
  - 解决方法
    ```c
    $ sudo apt-get install numlockx
    $ sudo vi /etc/lightdm/lightdm.conf
    末尾添加 greeter-setup-script=/usr/bin/numlockx on

    For Ubuntu Gnome and Xubuntu XFCE (GDM)：
    $ sudo apt-get install numlockx
    $ sudo gedit /etc/gdm/Init/Default

    末尾添加：
    if [ -x /usr/bin/numlockx ]; then
            /usr/bin/numlockx on
    fi
    ```
## <span style="color:#ff8000;">Ubuntu下汇编方法
  - as / objdump
    ```c
    $ vi hello.s
    $ as -o hello.o hello.s
    $ ld -s -o hello hello.o
    $ ./hello
    反汇编：$ objdump -D hello
    ```
## <span style="color:#ff8000;">Windows下拷贝ubuntu镜像到u盘，会造成文件名被截短，在安装过程中提示md5验证失败
  - 解决： 将镜像文件在ubuntu下挂载后复制到u盘
## <span style="color:#ff8000;">注销用户
  - kill / pkill / pgrep
    ```c
    $ killall gnome-session                // 结束gnome-session进程
    $ pkill -KILL -u {username}        // 给用户名为{username}的进程发送-KILL信号
    $ pgrep -u {username} -l        // 查找当前进程中用户名为{username}的进程，并列出进程pid与名称
    $ pkill -kill -t pts/1                // 注销指定的远程终端
    ```
## <span style="color:#ff8000;">禁用PrintScreen截屏
  - 系统设置 ---> 键盘 ----> 快捷键 ----> 截图
## <span style="color:#ff8000;">恢复/克隆的系统中用户文件(图片/文档等)未出现在【位置】列表中，且图标是默认文件夹图标
  - 创建软连接
    ```shell
    ln -fs /media/D/Users/edgerw/* ~/

    ln -s /media/leondgarse/GrandFiles_Seag/Downloads/ ~/
    ln -s /media/leondgarse/GrandFiles_Seag/Documents/ ~/
    ln -s /media/leondgarse/GrandFiles_Seag/Music/ ~/
    ln -s /media/leondgarse/GrandFiles_Seag/Pictures/ ~/
    ln -s /media/leondgarse/Videos_Seag/ ~/Videos
    ```
  - xdg-user-dirs-gtk-update
    ```c
    $ xdg-user-dirs-gtk-update         //xdg-user-dirs用于在不同的语言下自动创建一些经常用到的目录

    若不成功，则可尝试修改语言为英文，再改回中文：
    export LANG=en_US
    xdg-user-dirs-gtk-update
    export LANG=zh_CN.UTF-8
    xdg-user-dirs-gtk-update

    如果在执行xdg-user-dirs-gtk-update命令时选择了不再提示，可执行一下命令恢复：
    echo zh_CN > ~/.config/user-dirs.locale
    ```
## <span style="color:#ff8000;">迁移用户文件夹
  - 方法
    ```c
    vi ~/.config/user-dirs.dirs 填入相应路径
    创建目标路径软连接到用户目录
    ```
## <span style="color:#ff8000;">Ubuntu系统的一种备份还原方法
  - 备份：
    ```c
    备份已安装软件包列表
        sudo dpkg --get-selections > package.selections
    备份Home下的用户文件夹，如果Home放在额外的分区則不需要
    备份软件源列表，将/etc/apt/文件夹下的sources.list拷贝出来保存即可
    ```
  - 还原：
    ```c
    复制备份的Sources.list文件到新系统的/etc/apt/目录，覆盖原文件，并替换（Ctrl+H）文档中的intrepid为jaunty，
    然后更新软件源sudo apt-get update。
    重新下载安装之前系统中的软件（如果你安装的软件数量比较多，可能会花费较长时间）
      sudo dpkg --set-selections < /home/user/package.selections && apt-get dselect-upgrade
    最后将备份的主文件夹（/home/用户名）粘贴并覆盖现有主文件夹
    ```
  - rsync
    ```shell
    # ucloner_cmd.py, functions.py
    sudo rsync -av / --exclude-from=/home/sevendays19/local_bin/rsync_execlude_file /media/sevendays19/75fc86d3-cca4-40bf-86bd-3acebba610c2/

    # make_system_dirs
    cd /media/sevendays19/75fc86d3-cca4-40bf-86bd-3acebba610c2/
    sudo mkdir /proc /sys /tmp /mnt /media /media/cdrom0

    # generate_fstab
    sudo cp /etc/fstab etc/
    vi etc/fstab
    sudo touch etc/mtab

    # change_host_name
    sudo vi etc/hostname
    sudo vi etc/hosts

    # fix_resume
    cd ~/UCloner-10.10.2-beta1/program/sh/
    sudo ./fix_resume.sh /media/sevendays19/75fc86d3-cca4-40bf-86bd-3acebba610c2/ /dev/sdb3

    # install_grub2
    sudo ./install_grub.sh /media/sevendays19/75fc86d3-cca4-40bf-86bd-3acebba610c2/ /dev/sdb
    ```
## <span style="color:#ff8000;">ubuntu 12.04 开机自动挂载windows分区
  - 查看UUID # blkid
    ```c
    /dev/sda1: UUID="064CE4C44CE4AF9B" TYPE="ntfs"
    /dev/sda2: UUID="46D07D1ED07D1601" TYPE="ntfs"
    /dev/sda5: UUID="0bc1ef30-260c-4746-88c4-fd6c245882ea" TYPE="swap"
    /dev/sda6: UUID="53bc0a32-b32e-4f85-ad58-e3dbd9a3df41" TYPE="ext4"
    /dev/sda7: UUID="2fc00def-79d7-421e-92f1-e33e46e74c66" TYPE="ext4"
    ```
  - 查看分区 # fdisk -l
    ```c
    Disk /dev/sda: 112.8 GB, 112774965760 bytes
    255 heads, 63 sectors/track, 13710 cylinders, total 220263605 sectors
    Units = 扇区 of 1 * 512 = 512 bytes
    Sector size (logical/physical): 512 bytes / 512 bytes
    I/O size (minimum/optimal): 512 bytes / 512 bytes
    Disk identifier: 0xd10cd10c

      设备 启动   起点     终点   块数  Id 系统
    /dev/sda1  *    2048  46139391  23068672  7 HPFS/NTFS/exFAT
    /dev/sda2    46139392  119539711  36700160  7 HPFS/NTFS/exFAT
    /dev/sda3    119541758  220262399  50360321  5 扩展
    /dev/sda5    119541760  123539805   1999023  82 Linux 交换 / Solaris
    /dev/sda6    123541504  181200895  28829696  83 Linux
    /dev/sda7    181202944  220262399  19529728  83 Linux

    Partition table entries are not in disk order
    ```
  - 创建挂载点
    ```c
    在/media下创建C,D,E三个目录，命令如下：
    $ sudo mkdir /media/C
    $ sudo mkdir /media/D
    $ sudo mkdir /media/E
    ```
  - 编辑/etc/fstab文件
    ```c
    # vi /etc/fstab

    在这个文件中加入如下信息
    # These is used for auto mount Windows disks on boot up.
    UUID=064CE4C44CE4AF9B /media/C ntfs defaults,codepage=936,iocharset=gb2312 0 0
    UUID=46D07D1ED07D1601 /media/D ntfs defaults,codepage=936,iocharset=gb2312 0 0
    UUID=629AFA8D9AFA5D4B /media/E ntfs defaults,codepage=936,iocharset=gb2312 0 0
    ```
## <span style="color:#ff8000;">mtd 设备
  - cd /run/user/1000/gvfs/mtp:host=%5Busb%3A003%2C003%5D/
## <span style="color:#ff8000;">JPEG error
  - Not a JPEG file: starts with 0x89 0x50
  - The file is actually a PNG with the wrong file extension. "0x89 0x50" is how a PNG file starts. Rename it to png
## <span style="color:#ff8000;">swap
  - How do I add a swap file?
    ```
    Note: btrfs does not support swap files at the moment. See man swapon. and btrfs Faq
    Create the Swap File:
    We will create a 1 GiB file (/mnt/1GiB.swap) to use as swap:
    sudo fallocate -l 1g /mnt/1GiB.swap
    fallocate size suffixes: g = Giga, m = Mega, etc. (See man fallocate).
    If fallocate fails or it not available, you can use dd:
    sudo dd if=/dev/zero of=/mnt/1GiB.swap bs=1024 count=1048576
    We need to set the swap file permissions to 600 to prevent other users from being able to read potentially sensitive information from the swap file.
    sudo chmod 600 /mnt/1GiB.swap
    Format the file as swap:
    sudo mkswap /mnt/1GiB.swap
    Enable use of Swap File
    sudo swapon /mnt/1GiB.swap
    The additional swap is now available and verified with: cat /proc/swaps
    Enable Swap File at Bootup
    Add the swap file details to /etc/fstab so it will be available at bootup:
    echo '/mnt/1GiB.swap swap swap defaults 0 0' | sudo tee -a /etc/fstab
    Example of making a swap file
    This is an example of making and using a swap file on a computer with no swap partition.
    $ sudo fallocate -l 1g /mnt/1GiB.swap
    $ sudo chmod 600 /mnt/1GiB.swap
    $ sudo mkswap /mnt/1GiB.swap
    Setting up swapspace version 1, size = 1048576 kB
    $ sudo swapon /mnt/1GiB.swap
    $ cat /proc/swaps
    Filename                                Type            Size    Used    Priority
    /home/swapfile                          file            1048576 1048576 -1
    $ echo '/mnt/4GiB.swap swap swap defaults 0 0' | sudo tee -a /etc/fstab
    $ reboot
    $ free -h
                  total        used        free      shared  buff/cache   available
    Mem:            15G        9.3G        454M        4.0G        5.8G        1.9G
    Swap:          1.0G        1.0G          0B
    Disable and Remove a Swap File
    Disable the swap file from the running system and the delete it:
    sudo swapoff /mnt/1Gib.swap
    sudo rm /mnt/1Gib.swap
    Remove the swap file details from fstab:
    gksudo gedit /etc/fstab
    Removing the swap file line
    /mnt/1GiB.swap swap swap defaults 0 0
    ```
  - What is swappiness and how do I change it?
    ```
    The swappiness parameter controls the tendency of the kernel to move processes out of physical memory and onto the swap disk. Because disks are much slower than RAM, this can lead to slower response times for system and applications if processes are too aggressively moved out of memory.
    swappiness can have a value of between 0 and 100
    swappiness=0 tells the kernel to avoid swapping processes out of physical memory for as long as possible
    swappiness=100 tells the kernel to aggressively swap processes out of physical memory and move them to swap cache
    The default setting in Ubuntu is swappiness=60. Reducing the default value of swappiness will probably improve overall performance for a typical Ubuntu desktop installation. A value of swappiness=10 is recommended, but feel free to experiment. Note: Ubuntu server installations have different performance requirements to desktop systems, and the default value of 60 is likely more suitable.
    To check the swappiness value
    cat /proc/sys/vm/swappiness
    To change the swappiness value A temporary change (lost on reboot) with a swappiness value of 10 can be made with
    sudo sysctl vm.swappiness=10
    To make a change permanent, edit the configuration file with your favorite editor:
    gksudo gedit /etc/sysctl.conf
    Search for vm.swappiness and change its value as desired. If vm.swappiness does not exist, add it to the end of the file like so:
    vm.swappiness=10
    Save the file and reboot.
    What is the priority of swap containers?
    The Linux kernel assigns priorities to all swap containers. To see the priorities that the Linux Kernel assigns to all the swap containers use this command.
    cat /proc/swaps
    Priorities can be changed by using the swapon command or defined in /etc/fstab. Consult the manual page of swapon for more info
    man swapon
    Should I reinstall with more swap?
    Definitely not. With the 2.6 kernel, "a swap file is just as fast as a swap partition."(Wikipedia:Paging, LKML).
    Why is my swap not being used?
    My swap is not being used! When I issue the free command, it shows something like this:
    tom@tom:~$ free
                 total       used       free     shared    buffers     cached
    Mem:        515980     448664      67316          0      17872     246348
    -/+ buffers/cache:     184444     331536
    Swap:       674688          0     674688
    Note: This regards mainly swap on hard disk partitions, but it could help anyway. In these examples /dev/hda8 is considered as swap.
    Swap may not be needed
    Start many memory consuming applications (e.g. Gimp, web browsers, LibreOffice etc) and then issue the free command again. Is swap being used now?
    Ubuntu Desktop uses Swap to Hibernate (PC off, no power needed, program states saved). If Hibernation is important to you, have more swap space than ram + swap overflow.
    Is there a swap partition at all?
    Use this command to see all partitions
    sudo fdisk -l
    You should be able to see something like this in the output
    /dev/hda8            4787        4870      674698+  82  Linux swap / Solaris
    If not, you either need to create a swapfile or create a swap partition. To create a swap partition you can
    boot from your Ubuntu install CD, create a swap partition out of the free space on your hard disk and then interrupt your installation.
    use Cfdisk.
    Enabling a swap partition
    In case you do have a swap partition, there are several ways of enabling it.
    Use the following command
    cat /etc/fstab
    Ensure that there is a line link below. This enables swap on boot.
    /dev/hda8       none            swap    sw              0       0
    Then disable all swap, recreate it, then re-enable it with the following commands.
    sudo swapoff -a
    sudo /sbin/mkswap /dev/hda8
    sudo swapon -a
    Empty Swap
    Even if you have lots of RAM and even if you have a low swappiness value, it is possible that your computer swaps. This can hurt the multitasking performance of your desktop system.
    You can use the following script to get the swap manually back into RAM:
    Place the script e.g. /usr/local/sbin:
    gksudo gedit /usr/local/sbin/swap2ram.sh
    Copy-paste the script into the file:
    #!/bin/sh

    mem=$(LC_ALL=C free  | awk '/Mem:/ {print $4}')
    swap=$(LC_ALL=C free | awk '/Swap:/ {print $3}')

    if [ $mem -lt $swap ]; then
        echo "ERROR: not enough RAM to write swap back, nothing done" >&2
        exit 1
    fi

    swapoff -a &&
    swapon -a
    Save and close gedit
    Make the script executable:
    sudo chmod +x /usr/local/sbin/swap2ram.sh
    Execute:
    sudo /usr/local/sbin/swap2ram.sh
    ```
***

# <span style="color:#ff0000;">软件
  - google-chrome --enable-webgl --ignore-gpu-blacklist
## <span style="color:#ff8000;">自动更新无法下载adobe flashplayer
  - sudo dpkg --configure -a
  - mv /路径/install_flash_player_11_linux.i386/libflashplayer.so ~/.mozilla/plugins/
## <span style="color:#ff8000;">wireshark配置
  - 初始安装完没有权限抓包，对此，运行命令：
    ```c
    sudo dpkg-reconfigure wireshark-common
    选yes创建wireshark用户组
    把需要运行wireshark的用户加入到wireshark用户组：
    sudo usermod -a -G wireshark $USER
    之后重新以该用户身份登录即可
    ```
  - Couldn’t run /usr/sbin/dumpcap in child process: Permission denied
    ```c
    $ sudo vi /etc/group
    将用户加入wireshark组
    ```
## <span style="color:#ff8000;">png图片形式文档转文字
  - 安装软件：
    ```c
    gocr、tesseract-ocr、libtiff-tools
    安装tesseract中文语言包tesseract-ocr-chi-sim
    ```
  - tif文件转文字tif-->text，直接使用tesseract命令即可，如：
    ```c
    tesseract a.tif a.txt -l chi_sim
    其中tif图片文件可由shutter截图得到
    ```
## <span style="color:#ff8000;">compiz
  - 如果因为compiz配置出现问题导致桌面不能显示，可进入/usr/bin目录下执行./ccsm
  - 或尝试在用户目录下分别进入.cache、.config、.gconf/apps下删除对应compiz项
    ```c
    rm .cache/compizconfig-1/ .config/compiz-1/ .compiz/ -rf
    ```
## <span style="color:#ff8000;">VLC显示中文字幕
  - 首先启动VLC，按Ctrl+P,
  - 左下角的显示设置 选 全部，依次点开 ：
    ```c
    视频－字幕／OSD－文本渲染器 右侧的字体栏中，选择一个中文字体
    /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
    ```
    接着点开：
    ```c
    输入／编码－其它编码器－字幕 右侧的 字幕文本编码 选 GB18030
    ```
    然后 把 自动检测 UTF－8 字幕 格式化字幕 前面的勾去掉
  - 保存
## <span style="color:#ff8000;">minicom无法保存配置
  - $ cd
  - $ ls -a                // 查看是否有.minirc.dfl文件
  - $ rm .minirc.dfl
## <span style="color:#ff8000;">gedit中文乱码
  - 打开终端输入：
    ```c
    gsettings set org.gnome.gedit.preferences.encodings auto-detected "['GB18030', 'GB2312', 'GBK', 'UTF-8', 'BIG5', 'CURRENT', 'UTF-16']"
    gsettings set org.gnome.gedit.preferences.encodings shown-in-menu "['GB18030', 'GB2312', 'GBK', 'UTF-8', 'BIG5', 'CURRENT', 'UTF-16']"
    ```
## <span style="color:#ff8000;">安装emerald
  - sudo add-apt-repository ppa:noobslab/themes
  - sudo apt-get update
  - sudo apt-get install emerald
## <span style="color:#ff8000;">Install new cursor theme
  - 0- Download the cursor theme.
  - 1- Locate the file ThemeName.tar.gz downloaded. It's probably in your Downloads folder.
  - 2- Right click on it and left click on "extract here". You will see the folder ThemeName.
  - 3- Move the theme folder to ~/.icons/. Open a terminal and you can use the following command line.
    ```c
    Quote: mv Downloads/ThemeName ~/.icons/
    < ON GNOME 3.x and UNITY >
    ```
  - 4- To update the cursor theme and cursor size: Search for and install Dconf Editor in Software Center.
  - 5- Use next Command, changing theme's name and the current cursor size to your custom cursor size xx.
    ```c
    Could also use 24, 32, 40, 48, 56 or 64 pixels.

    Quote: gsettings set org.gnome.desktop.interface cursor-theme ThemeName &amp;&amp; gsettings set org.gnome.desktop.interface cursor-size xx

    If you wish choose the cursor theme and cursor size, using the Dconf editor graphic interface (GUI),
    then launch it, an go org-->gnome-->desktop-->interface.
    ```
  - 6- Finally, create or edit the ~/.Xresources file using next command:
    ```c
    Quote: gedit ~/.Xresources
    Add the following two lines. Change ThemeName and xx to match size defined in previous step:
    Quote:
    Xcursor.theme: ThemeName
    Xcursor.size: xx
    ```
  - 7- Save, close and reboot.
## <span style="color:#ff8000;">Conky
  - Ubuntu 14.04安装Conky Manager
    ```c
    sudo add-apt-repository ppa:teejee2008/ppa
    sudo apt-get update
    sudo apt-get install conky-manager
    ```
  - 卸载该软件：
    ```c
    sudo apt-get remove conky-manager
    去除PPA：
    sudo apt-get install ppa-purge
    sudo ppa-purge ppa:teejee2008/ppa
    ```
## <span style="color:#ff8000;">Install SKY
  - Setup TEL.RED repository
    ```c
    $ sudo add-apt-repository 'deb http://repos.tel.red/ubuntu stable non-free'
    ```
    Update repositories metadata
    ```c
    $ sudo apt-get update
    ```
    Install 'sky' application package
    ```c
    $ sudo apt-get install -y sky
    ```
  - If 'Sky 2.0' was installed earlier via tel.red repo, to keep your system clean, issue following command to erase deprecated tel.red repository entry:
    ```c
    $ sudo sed -i '/deb https\?:\/\/.*\btel.red\b/d' /etc/apt/sources.list

    1. Ensure APT works with HTTPS and CA certificates are installed
    $ sudo apt-get install apt-transport-https ca-certificates

    2. Add TEL.RED Debian repository:
    * Ubuntu 16.04 xenial
    sudo sh -c 'echo deb https://tel.red/repos/ubuntu xenial non-free > /etc/apt/sources.list.d/telred.list'

    3. Download and register TEL.RED software signing public key
    $ sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 \
      --recv-keys 9454C19A66B920C83DDF696E07C8CCAFCE49F8C5

    4. Refresh apt repository metadata
    $ sudo apt-get update

    5. Install sky
    $ sudo apt-get install -y sky
    ```
## <span style="color:#ff8000;">ibus自动清除选中的文本
  - 系统输入法选择为ibus时会自动清除选中的文本，如果是英文输入法就没有这个问题
  - 解决方法：
    ```
    终端中 ibus-setup
    勾掉 在应用窗口中启用内嵌编辑模式(Embed preedit text in application window)
    ```
## <span style="color:#ff8000;">7z compress & extract
  - 解压缩7z文件
    ```c
    7za x phpMyAdmin-3.3.8.1-all-languages.7z -r -o./
    参数含义：
    x  代表解压缩文件，并且是按原始目录树解压（还有个参数 e 也是解压缩文件，但其会将所有文件都解压到根下）
    -r 表示递归解压缩所有的子文件夹
    -o 是指定解压到的目录，-o后是没有空格的
    ```
  - 压缩文件／文件夹
    ```
    7za a -t7z -r Mytest.7z /opt/phpMyAdmin-3.3.8.1-all-languages/*
    参数含义：
        a  代表添加文件／文件夹到压缩包
        -t 是指定压缩类型，这里定为7z，可不指定，因为7za默认压缩类型就是7z。
        -r 表示递归所有的子文件夹
        注意：7za不仅仅支持.7z压缩格式，还支持.tar.bz2等压缩类型的，用-t指定即可
    ```
## <span style="color:#ff8000;">evolution
  - 相关文件夹
  ```shell
  du -hd1 .local/share/evolution/
  du -hd1 .config/evolution/
  du -hd1 .cache/evolution/
  ```
