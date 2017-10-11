# ___2017-10-09_Netbackup___
***

# 目录
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [___2017-10-09_Netbackup___](#2017-10-09netbackup)
- [目录](#目录)
- [Jabber](#jabber)
- [Server](#server)
- [Other Links](#other-links)
- [Trouble shooting](#trouble-shooting)

<!-- /TOC -->
***

# Jabber
  ```python
  北京的项目 netbackup
  安装 server
  ```
***

# Server
  - Server: csfbld01.ih.lucent.com User / pass: judywa / CV0012213
    ```bash
    ssh judywa@csfbld01.ih.lucent.com / CV0012213
    ```
  - Location: /u/judywa/CSF-BACKUP/
    - LCM part code: /u/judywa/CSF-BACKUP/LCM/src/src/files  (This part transition has been done)
    - CBUR part code: /u/judywa/CSF-BACKUP/backup/src/src (This part will be transitioned this Friday)
    ```bash
    export __AONTUSER=( $(whoami) )
    alias PRINTF_CYAN='printf "\033[36m%s\n\033[m" $1'

    export __LCMPATH="/u/$__AONTUSER/CSF-BACKUP/LCM/src/src/files"
    export __CBURPATH="/u/judywa/CSF-BACKUP/backup/src/src"

    alias LCM='cd $__LCMPATH && PRINTF_CYAN `pwd -P` && ls'
    alias CBUR='cd $__CBURPATH && PRINTF_CYAN `pwd -P` && ls'
    ```
  - Lab IP: 135.248.249.138    root/newsys
  - Log to OAM node via command:  ssh -i /root/key.pem  10.195.104.31  or  ssh -i /root/key.pem  10.195.104.32
    ```bash
    ssh root@135.248.249.138 / newsys
    ssh -i /root/key.pem  10.195.104.31
    ssh -i /root/key.pem  10.195.104.32
    ```
  - license key
    ```python
    DEXZ-KPPO-8TCN-434O-4O4O-4M77-7777-763R-PO8
    ```
***

# Other Links
  - [nanaal](http://135.3.42.110/~nanaal/)
  - [demo video](http://135.3.42.110/~nanaal/BR-deploy-and-usage-demo.avi)
***

# Trouble shooting
- **Q: configurePorts:  WmcPortsUpdater failed with exit status 254**
  - A: NetBackup Web Services on the master server require port 1024 or higher.
    ```python
    $ ./configurePorts -status                                                                                                                
    Current Http Port: 80
    Current Https Port: 443
    Current Shutdown Port: 8105
    ```
  - Use the configurePorts command in the following format to re-configure a port:
    ```python
    configurePorts -httpPort http_port | -httpsPort https_port | -shutdownPort shutdown_port
    ```
  - Port sets for NetBackup Web Services
    | Port set   | HTTP port | HTTPS port | shutdown port |
    | ---------- | --------- | ---------- | ------------- |
    | First set  | 8080      | 8443       | 8205          |
    | Second set | 8181      | 8553       | 8305          |
    | Third set  | 8282      | 8663       | 8405          |
  - Command
    ```python
    $ ./configurePorts -httpPort 8080 -httpsPort 8443 -shutdownPort 8105                                                                      
    Old Shutdown Port: 8105
    New Shutdown Port: 8105
    Old Http Port: 80
    New Http Port: 8080
    Old Https Port: 443
    New Https Port: 8443

    $ ./configurePorts -status
    Current Http Port: 8080
    Current Https Port: 8443
    Current Shutdown Port: 8105
    ```
