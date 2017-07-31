

- # Documents
  Digitmap syntax

  https://ct.web.alcatel-lucent.com/scm-lib4/show-entry.cgi?number=3HH-08371-0002-DFZZA

	Which customer use which ONT type and Configuration mode

	https://ct.web.alcatel-lucent.com/scm-lib4/show-entry.cgi?number=3FE-27984-ABAA-DFZZA
	

- # List all Useful Doc numbers for Team Knowledge Record


Hi All,



As we discussed on group meeting, I created one binder 3HH-14000 for our team.

And the related useful record are upload on weblib. You can search 3HH-14000 to get all of them.



1.       Doc 3HH-14000-6001-DFZZA QD ONT Voice Name List   

²  Record all ONT team members information



2.       Doc 3HH-14000-6002-DFZZA ONT onboard AllInOne handbook    

²  Send to newhire when he or she is on board.



3.       Doc 3HH-14000-6003-DFZZA  Qingdao ONT Team Competence Data  

²  Please collect information and update this document for your team members



4.       Doc 3HH-14000-6005-DFZZA ONT Knowledge Pool

²  Please let your team members update this document when there are some experience.



5.  Please use doc number for QD RCA activity from 3HH-14000-61xx-DFZZA to 3HH-14000-64xx-DFZZA.

²  For FCU RCA, please upload 3HH-12869-1…-DFZZA



６. Please use doc number for internal PreLA from 3HH-14000-65xx-DFZZA to 3HH-14000-68xx-DFZZA.


Branch & AONT


(1)     MS branch

²  BCM GPON: G-240W-X/G-240G-C/G-010G-A/vCPE

²  BCM g.fast: F-240W-A/F-010G-A/F-010G-B/vCPE

²  BCM XGPON: XG-240W-A/XS-240W-A/…

²  BCM EPON: E-240G-A/E-240G-B/E-240W-A/…

²  IOT

²  LANTIQ GPON: G-010S-A

²  MTK GPON: G-240W-F/G-240G-E/…



(2)     BRL branch

²  BRL GPON  (BRL): I-240G-A/I-240G-D/B-0404G



(3)     TWG1 branch

²  BCM NGPON2 FPGA (TWG1): TW-080GX-A/TW-240GX-A



(4)     BCM GPON SFU NAR branch

²  G-440G-A/G-240G-A/G-821M-A/G-881G-A



(5)     BCM GPON China

²  I-240E/I-240W-Q/…



(6)     MTK GPON China

²  G-120W/G-140W/…


Voice Tool


Hi all,



大家原来用的都是R400电脑，现在换了新电脑需要重装voicetool，

voicetool的licence manager已经换人了，不是tom philips，换成了NIES Chris Chris.Nies@alcatel-lucent.com。

另外老外给的licence改成了XXXXXXXXXXXXXXXXXXXXX.lic，需要rename成’lservrc’才能使用。



如果安装的是VERSION: 3.6.24，在win7下运行应该会出现MFC crash，voicetool无法运行，需要将附件中的voice.exe替换掉安装目录中的voice.exe。



Thanks and Best Regards,

Jia Mason


SSID  -->  password


ALHN-64ea  -->  2801626214

ALHN-641d  -->  8195217876

ALHN-636e  -->  0328221137

ALHN-641a  -->  1890152011

ALHN-650a  -->  6466087472

ALHN-660b  -->  8306560640

ALHN-E0E4  -->  7686289736

ALHN-6097  -->  2594317273

ALHN-E0E4  -->  4278602388

ALHN-E0E4  -->  8639873105

VIET:

<UserName ml="64" rw="RW" t="string" v="vtadmin"></UserName>

<Password ml="64" rw="RW" t="string" v="bHjJfYjUoXGGOMvIaanu8Q==" ealgo="ab"></Password>

vtadmin / 123456


AONT knowledge


===================================================================================

>>>> tftp

上传文件到tftp server：tftp -p -l ftpprofile.xml 192.168.1.64

                     tftp -p -l cvp.log.bak 192.168.1.64

从tftp server下载文件：tftp -g -r a 192.168.1.64


>>>> telnet

telnet 192.168.1.254 \ ONTUSER:SUGAR2A041, root:root


>>>> crontab

AONT3@:/tmp # crontab -c /tmp /tmp/crontab.file

AONT3@:/tmp # crontab -c /tmp -l


>>>> share folder:

QD_ISAMV_Share_Forum/Feature_DEV_Practice/


>>>> trace debug

/configs/omcidebug --> 设定trace debug level

/tmp/omci.log --> trace debug 输出的文件


>>>> user login

user# enable

user# shell SHayuBCont88


>>>> ONT user / pwd

ONTUSER : SUGAR2A041

root : root


>>>> Execute upgrade command on the console

CFE> bn 192.168.1.66:kernel_boot.img.w j

CFE> bn 192.168.1.66:rootfs_sqsh.img.w r                


>>>>

CFE> getp

CFE> setp 3FE56756BAAA

CFE > ritool dump

CFE> b  -->  GPON Serial Number  -->  .


>>>>

PS1='[\u@\h: $PWD]# '


>>>>


touch /configs/cfgmgr_bootdebug


cfgcli -rall && sync && sync && reboot


>>>> cfgcli reloading xml from /usr/etc/alcatel/config/precfg/

cfgcli -s InternetGatewayDevice.Services.VoiceService.1.Capabilities.SIP.X_ALU-COM_XML_File_Name_Path ftpprofile_bcm240.xml


>>>>

cfgcli -s InternetGatewayDevice.X_Authentication.WebAccount.Password 12345

cfgcli -s InternetGatewayDevice.X_Authentication.Account.Password 12345



>>>>

route del -net 21.1.6.0 netmask 255.255.255.0

route add -net 21.1.6.0 netmask 255.255.255.0 dev pon_v18_0_1


brdl serial


BRLT 串口命令行烧写image的方法：



打开TFTP server 配置server 地址192.168.1.54 和 文件指定的up/down load 目录



启动ont到uboot 命令行模式执行如下命令：



tftp 192.168.1.54:uImage

blnprog linux



tftp 192.168.1.54:ubifs.img   //老版本是这个文件

or

tftp 192.168.1.54:rootfs_sqsh.img   //新版本是这个文件

blnprog rootfs

===================================================================================

Operator-ID     VOIP configure mode

ALCL            TR104

ALCO            OMCI

XXXX            OMCI

0000            TR104

===================================================================================

AONT30:/ # ritool --help

AONT30:/ # ritool dump  /* Mnemonic / PartNumber / MACAddress */

AONT30:/ # ritool set

AONT30:/ # ritool set OperatorID ... /* 配置 VOIP config 方式 OMCIV1 / TR069 */

AONT30:/ # cat /usr/etc/buildinfo  /* 查看系统信息 */

AONT30:/ # swug --info /* Get the upgrade build info */

AONT30:/ # oflt phy mode set 1 auto auto /* 打开LAN口 */

AONT30:/ # cfgcli -e WebAccount. /*  查看web页面的用户名密码，结尾有个“.”*/

                                               AdminGPON / ALC#FGU
                                               superadmin / 12345

                                               ENTB: Administrador / AdminGPON2013etb



登录网页界面 -> Maintain -> Firmware Upgrade -> Select File -> [名称对应cat /usr/etc/buildinfo -> IMAGEVERSION==3FE56557 前面字段]


AONT3@:/ # cd /configs/alcatel/config /* xml文件位置 */

AONT3@:/ # ls -l /usr/etc/alcatel/config/precfg


AONT3@:/ # cvpcli dbg spt ? /* log info */


AONT3@:/ # omcli omciMgr profile /configs/alcatel/config/   /* ont update xml */

AONT3@:/ # cvpcli spfl PrintPflData
AONT3@:/ # cvpcli pal showVSP


AONT3@:/ # cfgcli -g InternetGatewayDevice.X_ASB_COM_ONUType.type


AONT3@:/ # ftpget -u 'ont' -p 'ont' 21.1.4.126 /configs/alcatel/config/sky_all_2.xml sky_all_2.xml

cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.DigitMap "02XXXXXXX|9R(0[2-7]XXXXXXXX|00X.T"


//打开lan口

oflt phy mode set  1 auto auto



//mirror

halmgr lp set_lp_mirror 1 5 1


ritool set OnuMode 0002

cfgcli -g InternetGatewayDevice.X_ASB_COM_ONUType.type

cfgcli -g InternetGatewayDevice.X_ASB_COM_VOIPCfgType.type
