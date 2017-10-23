
- Feature Information
  - RCR: ALU02371974 VOICE Requirements for phase 4b
  - IR: ALU02403835
  - [Binder: 3HH-14840-AAAA](https://ct.web.alcatel-lucent.com/scm-lib4/view.cgi?number=3HH-14840-AAAA-ADZZA-01P01&no_index_sheet)
  - FDT: hg clone ssh://hg@135.251.206.233/HD_R58_FDT1480
# RTP PCM
  - [codec decode](https://github.com/twstx1/codec-for-audio-in-G72X-G711-G723-G726-G729/tree/master/G711_G721_G723)
  - BCM 的endpoint的说明文档
    - 在usersapce 可以调用vrgEndptPacket（）函数传递rtp包到DSP去处理
    - 反向，BCM提供钩子函数，在userspace 可以通过注册endptpacketcallback 函数得到SLIC送到DSP的转化为rtp的包
    - BCM 提供了一种codec叫linear pcm的模式，和G711以及G729是类似的，好像是PCM直接加上RTP的包头，这种方式不需要转化。直接填充包头就可以转为RTP
    - 这就需要我们在摘机后，在3G doogle模式下对底层的配置，不要配置以前的codec，配置为linear pcm模式
    - 不知道华为的设备，pcm的速率是多少，BCM的底层提供了两种，一种128的窄带，一种256的宽带
    ```c
    ErrCodeC gvhw_broadcom_driver::initiateCall(unsigned int chanID, Gen_phoneCodec coder_i, unsigned int pTime_i)；
    ErrCodeC  gvhw_broadcom_driver::configureDataChannel(unsigned int chanID,
                    Gen_pktMode pkt_mode_i,
                   Gen_phoneCodec coder_i,
                    int pTime_i,
                    Gen_echo lec_i,
                    Gen_VADCNG vadCng_i,
                    Gen_jbSize *jbSize_pi ,
                    Gen_payloadType *payloadType_pi)

    /* Voice codec types */
    typedef enum
    {
        ...
        CODEC_LINPCM128,  /* Narrowband linear PCM @ 128 Kbps * /
        CODEC_LINPCM256,  /* Wideband linear PCM @ 256 Kbps * /
        ...
    }
    ```
  - Linetest 命令code 接口
    ```c
    Generic_Voice_Hardware_wrapper/src/ONU/src/isamv_itf.cpp
    ```
  - sdk 对RTP包头的定义
    ```c
    typedef struct {

    #if BOS_CFG_BIG_ENDIAN
        VRG_UINT8  version:2;   /* protocol version * /
        VRG_UINT8  p:1;         /* padding flag * /
        VRG_UINT8  x:1;         /* header extension flag * /
        VRG_UINT8  cc:4;        /* CSRC count * /
        VRG_UINT8  m:1;         /* marker bit * /
        VRG_UINT8  pt:7;        /* payload type * /
    #elif BOS_CFG_LITTLE_ENDIAN
        VRG_UINT8  cc:4;        /* CSRC count * /
        VRG_UINT8  x:1;         /* header extension flag * /
        VRG_UINT8  p:1;         /* padding flag * /
        VRG_UINT8  version:2;   /* protocol version * /
        VRG_UINT8  pt:7;        /* payload type * /
        VRG_UINT8  m:1;         /* marker bit * /
    #else
       #error "BOS_CFG_xxx_ENDIAN not defined!"
    #endif
        VRG_UINT8  seq[2];      /* 16-bit sequence number * /
        VRG_UINT8  ts[4];       /* 32-bit timestamp * /
        VRG_UINT8  ssrc[4];     /* synchronization source * /
    } RTPPACKET;
    ```
  - Command used to change codec and payload type
    ```shell
    testConfDataChannel - <connId> <ptime> <codec> <pktmode> <lec> <vadCng> <payltype> configure data of channel
    ```
  - BCM 发出的RTP payload type 120对应的是
    ```c
    CODEC_LINPCM128,  /* Narrowband linear PCM @ 128 Kbps */
    ```
# PreLA
## The traceable available requirements as from the RCR
  - Background : 
    - Customer : Vodafone Spain
    - VOICE Requirements for phase 4b.
    - Requirement description
      ```markdown
      The ONT shall support the mandatory Voice requirements for initial deployment
      as described in below document : 3HH-08371-0602-DFZZA(VF Spain Voice Features Analysis) (doc 1, 1p03)
      1p03 may need to be updated according to 3FC-40133-B294-TQZZA (doc 3, 1p06)
      There is still another doc 4 (ux-playground_OtherSettings_analysis_v4 ) to provide supplementary specification for some tr104 nodes in doc1 and doc3.
      Discussed with Johan, the final scope should be judged by doc3.
      https://ct.web.alcatel-lucent.com/scm-lib4/show-entry.cgi?number=3HH-08371-0602-DFZZA&actions=yes
      ```
    - ONT type : G-240W-C
  - 1/  3G Dongle
    ```markdown
    See above doc 1,
    -    Tab/sheet : VoiceAnalysisTCD
            See Column I (Internal comment MDA) , select "USB Dongle, Voice related"  
    Additional, see below document : Vodafone Spain Requirements for HGUs v094 (doc 2, 1p03)
        https://ct.web.alcatel-lucent.com/scm-lib4/show-entry.cgi?number=3FC-88066-AAAA-DSZZA
        See section 10. HSPA Solution
    ```
  - Additional Huawei 3G Dongle technique document:
    - Integration Userguide for Huawei Dongles V1.1.pdf
    - Guide to Kernel Driver Integration in Embedded Linux & Android for Huawei Dongles.pdf
    - HUAWEI UMTS Datacard Modem AT Command Interface Specification_V2.17.pdf
  - 2/ TR-104/TR-098 datamodel , support of additional TR-104 objects required
    ```markdown
    See above doc 1,
    See Tab/sheet : VoiceAnalysisTR98std
    See tab/sheet : Voice AnalysisTR98prop
    ```
  - 3/ CLI commands
    ```markdown
    See above doc 1
    See Tab/sheet : VoiceAnalysisCLI
    CLI commands Show Voice status and Show voice port
    ```
  - 4/ SIP Server Redundancy
    ```markdown
    The ONT shall support the SIP Server Redundancy procedure as described in below
    document : Vodafone Spain Requirements for HGUs v094 (doc 2, 1p03)
        https://ct.web.alcatel-lucent.com/scm-lib4/show-entry.cgi?number=3FC-88066-AAAA-DSZZA
       
        See section 4.1.1 : Proxy Server Configuration
    Additional, see above doc 1,
    - See Tab/sheet : All TCDs applicable to VFES
            Reqt Description : Voice Timer T1 (to switch to secondary SBC) : 32 seconds
            Reqt Description : Voice Timer T2 (to switch to primary SBC) : 60 seconds
    ```
  - 5/  Ziba Playground
    ```markdown
    See above doc 1,
    - See Tab/sheet : Voice AnalysisTCD
    - Column B/subsection : General Conditions
      Reqt description :
      Ziba Playground
        (https://vf2.ux-playground.com/gateways/view/37/23/56/overview)
        MUST be used as guideline for implementation of WUI.
    ```
## foo
  - WEBGUI requirement
    - Registration Retry Interval(T1) && Primary Proxy Retry Interval(T2)
    - Requires: Need to configure T1&&T2 via WEBGUI
  - CLI Command requirement
    - show vocie status:
    - Requires: Need to support below CLI command to show registration status and the network used to provide the service
  - show voice port:
    - Requires: Need to support below CLI command to show port status
## Customer requirement
  - This feature is to meet up with the Vodafone Spain’s requirement. The following is the improvement on AONT voice:
    - The Voice related WEBGUI page should be modified according to the requiement of VFS.
    - The customer could set some new supported tr104 parameters via TR069.(will list detailed in later chapter)
    - Voice should support backup between USB Dongle and Wan, and also some additional requirement such as need to play “FAST BUSY TONE” after POTSa has answered one incoming when two POTS(POTSa and POTSb) connected to both FXS. (will list detailed in later chapter)
    - The ONT could support two new CLI command to get more voice information.(Show  voice status | Show voice port)
    - The ONT could use SIP Server Redundancy. Two timer T1 and T2 required.
## Scope of the RCR
  - AONT types: G-240W-C
  - Customer: Vodafone Spain
  - VOIP mode: TR069
  - Mgnt interfaces : WEBGUI/TR069
  - SW branch: MS branch
## HSPA requirement
  - Mobile Backup(switch mechanism between HSPA and WAN)
    - HG Factory default configuration: Voice traffic towards the HSPA module should use the 3G/2G available mobile network.
    - HG REBOOT scenario: After each reboot, HG shall be able to recognize where to forward voice traffic(to HSPA or WAN). If WAN is active, data goes through WAN interface as well as voice traffic. If HG is rebooted but WAN is inactive, HSPA module        shall be used for both data (through WUI) and voice (automatically) traffic.
    - Switch from HSPA to WAN: HG shall be able to automatically switch from HSPA to WAN(ATA VoIP)line as soon as ATA module registers to SBC(IMS network). In case there is a GSM active call and Voice registers to SBC, HG shall wait the end of the call before switching (automatically) to WAN line.
    - Switch from WAN to HSPA1: In case ATA module lost its connection/registration to IMS network, HG shall be able to automatically switch voice traffic to HSPA module that use 3G/2G mobile network depending on the better radio coverage.
    - Switch from WAN to HSPA2:  If HG makes a call and receives no response it must try to register. If it looses the registration it must switch automatically to VoCS.
  - Mobile Backup(STS management)
    - In case of HSPA line enabled, the STS foreseen by VONV are the STS managed by VONV Mobile Network.
  - Mobile Backup(POTS management)
    - POTS management for incoming call: If there are two POTS(POTSa and POTSb) connected to both FXS, the user can answer      to incoming call using independently POTSa or POTSb. If the user answers with POTSa, “FAST BUSY TONE” SHALL be heard on using independently POTSa or POTSb. If the user answers with POTSa, “FAST BUSY TONE” SHALL be heard on call using independently POTSa or POTSb. If the user answers with POTSa, “FAST BUSY TONE” SHALL be heard on using independently POTSa or POTSb. If the user answers with POTSa, “FAST BUSY TONE” SHALL be heard on POTSb(and viceversa).
    - POTS management for outgoing call: if there two POTS(POTSa and POTSb) connected to both FXS, the user can make a call using independently POTSa or POTSb. If the user makes a call with POTSa, “FAST BUSY TONE” SHALL be heard on POTSb(and viceversa) .
  - General(emergency numbers)
    - It is possible to performs calls to emergency numbers when there is no SIM in the dongle.
    - It is possible to performs calls to emergency numbers when the SIM is locked.
## TR98 Standard Parameters
  - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.CallingFeatures.CallerIDName
    - Requires: Need to support this new tr104 node
    - Progress: not supported yet, but in ALU02329107([TELMEX] additional TR-104 objects), it will be implemented as read only, and the real Display name used for the From-header should be used for this TR104 object. Display name used for the From-header should be used for this TR104 object.
  - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.ReceivePacketLossRate
  - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.FarEndPacketLossRate
  - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.RoundTripDelay
  - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.AverageReceiveInterarrivalJitter
  - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.AverageFarEndInterarrivalJitter
  - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.AverageRoundTripDelay
  - Requires: Need to support tr104 node below about Stats.
  - Progress: Now we have supported the following per session tr104 node
    - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_ReceivePacketLossRate
    - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_ReceiveInterarrivalJitter
    - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_FarEndPacketLossRate
    - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_AverageReceiveInterarrivalJitter
    - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_AverageFarEndInterarrivalJitter
    - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_AverageRoundTripDelay
    - But as VFS required and also discussed with PA, we need to implement the per line parameters, now we only implement per session private parameters, before in Frank’s opinion, stats per session will be more reasonable. If implement it per line, we may need to calculate the average for all sessions per line? Need to discuss and confirm this with PA further. parameters, before in Frank’s opinion, stats per session will be more reasonable. If implement it per line, we may need to calculate the average for all sessions per line? Need to discuss and confirm this with PA further.
## TR98 X Parameters
  - T1 timer(Registration Retry Interval) and T2 timer(Primary Proxy Retry Interval)
    - Requires: Need to implement new T1&&T2 timer to configure new sip redundancy mechanism
    - Progress:
      - Need to define new tr104 parameter.
      - Need to discuss the detailed behavior for T1&&T2(refer Open Points part later).
  - InternetGatewayDevice.Services.VoiceService.{i}.PhyInterface.{i}.X_PortStatus
    - Requires: Need to support new tr104 node to get port status
    - Progress: Need to support the following port status:
      - Diabled | enabled
      - On Hook
      - Off Hook
      - WaitonHook = Parking, lets consider this status”Waitonhook” is parking, and lets use the rules of our CVP to move a subscriber to  - parking.
      - Calling – Outgoing Call
      - Ringing – Incoming Call
      - Talking – Call established(incoming or outgoing)
    - NOTE: need to discuss the mapping , will track this in open points.
  - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.CallingFeatures.X_CallTransferProv
    - Requires: Need to implement tr104 node below
    - Progress: The following tr104 node with the same function has been supported. Suppose we don’t need extra effort.
      - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.CallingFeatures.X_ALU-COM_CallTransferProvision
  - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.CallingFeatures.X_CallTransferActivate
    - Requires: Need to implement tr104 node below
    - Progress: The following tr104 node with the same function has been supported. Suppose we don’t need extra effort.
      - InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.CallingFeatures.CallTransferEnable
  - Indicate current voice network which is VoIP or VoCS.
    - Requires: Need to confirm the name of tr104 node
    - Progress: I propose define the tr104 node as follows and need PA’s confirmation:
      - InternetGatewayDevice.X_ASB_COM_VoiceNetwork.type
    - NOTE:
      - X_SessionExpires shifted to HD_R5801 (cfr. Doc4)
      - X_interfaceName not required (cfr. Doc4)
## WEBGUI requirement
  - Registration Retry Interval(T1) && Primary Proxy Retry Interval(T2)
    - Requires: Need to configure T1&&T2 via WEBGUI
    - Progress: Clear, wait for corresponding tr104 implementation. Checked(https://vf2.ux-playground.com/gateways/view/37/23/58/phone-settings), Display format is as follows:
  - Modify logic of Line in Phone Number Status
    - Requires: Need to modify logic map of Line from “.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Status” to .VoiceService.{i}.PhyInterface.{i}.X_PortStatus
    - Progress: Clear. Need to modify .cgi script.
## CLI Command requirement
  - show vocie status:
    - Requires: Need to support below CLI command to show registration status and the network used to provide the service.
    - Progress: Need to confirm the format of this command, as we don’t have the interface in ONT now.
      ```python
      Actually now we have the following tr104 cli command to get the status:
      cfgcli –g InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Status
      Currently, we have supported these values: “Up”,”Registering”,”Error”,”Testing”,”Disabled”.
      In TR98 Standard Parameters part, we still need to support new range quiescent.
      Need to confirm if we can use this tr104 command as alternative command.
      ```
  - show voice port:
    - Requires: Need to support below CLI command to show port status
    - Progress:
      ```python
      Need to confirm the format of this command, as we don’t have the interface in ONT now.
      Actually now we have the following tr104 cli command to get the status:
      cfgcli –g InternetGatewayDevice.Services.VoiceService.{i}.PhyInterface.{i}.X_PortStatus
      In TR98 X Parameters part, we will implement this.
      Need to confirm if we can use this tr104 command as alternative command.
      ```
## Detailed SIP server redundancy requirement(T1 && T2)
  - New timer objects(T1,T2) are to be introduced.
  - T1 and T2 should be configurable via HTTP, CLI and TR-069.
  - Please note that there are already timer T1 and T2(same name but different behaviors) according to RFC3261 defined in the system.
  - T1: Range [?], default value 32s.
  - T2: Range [?], default value 60s.
  - T1 reflects the maximum time a SIP entity will wait to receive a final response to a sip request.  The timer is started upon transmission of the sip request. Upon the expiry of T1, the sip server is considered as OOS. Upon the receipt of any response(timer expiry is also considered as a final response), T1 is to be stopped.
   - T2 indicates the time period by which the system sends a REGISTER to the primary server to verify its availability . This period is measured from the moment the primary sip server has failed(T1 expiry or error code). It will be stopped by the successful response to the REGISTER message from the primary server.
  - It’s supposed that for this RCR both BGHM and FGHM MUST be disabled. Only timer expiry will trigger a failover action.
  - Both two timers should be passed to SIPM module.
  - We need to define a T2MonitoringRegister type for the server’s availability verification.
  - T1 is started upon transmission of the sip request. Upon the expiry of T1, the sip server is considered as OOS.
  - In-dialog message?
  - If it is detected that the primary server has failed and no line is working at HSPA mode, the system should try to switch all idle lines to the secondary server.
  - If it is detected that the secondary server has failed and no line is working at HSPA mode, the system should try to switch all idle lines to HSPA mode.
  - A SIP POTS termination is switched over to the secondary server or HSPA mode from the moment it is in “idle” state. Ongoing dialogs and transactions are not transferred to the usual SIP core or 3G dongle, and this neither at the signaling plane (SIP) nor at the media plane (RTP).
  - Fail-over to the HSPA mode shall ONLY be triggered from the moment none of the “usual” SIP servers are still reachable (neither primary nor secondary SIP server(s) are still reachable i.e. the SIP UA becomes isolated from its ‘usual’ SIP core).
  - For calls ongoing at the moment the SIP UA becomes isolated  from its “usual” SIP core  the following strategy applies : ongoing dialogs and transactions shall not be transferred to the HSPA mode, and this neither at the signaling plane (SIP) nor at the media plane (RTP). The ongoing dialogs and transactions will be maintained along the current signaling and media path until any termination interaction happens.
  - If a failover is done and the secondary server is in-service and replies 200OK to the REGISTER, T2 timer is to be started.
  - If a switching to HSPA action is done and the T2 is still running, restart it.
  - The system tries to send a REGISTER for a line to the usual SIP core from the moment this line is in “idle” state.
  - Upon the expiry of T2, a REGISTER(type= T2MonitoringRegister) is to be sent to the primary server. Should the T2 polling be restarted immediately. If the primary server replies a 200OK, the T2 is to be stopped. If the primary server is still OOS, keep T2 polling.
  - Upon the expiry of T2, if it’s detected that the primary server is OOS by T2MonitoringRegister and lines are working via secondary server, just ignore this OOS event.
  - Upon the expiry of T2, if it’s detected that the primary server is OOS by T2MonitoringRegister of ONE line and the other line is still in a call via the secondary server, no need to send a T2MonitoringRegister for this line this time.
  - If the system is working at HSPA mode, T2 is to be started. Upon the expiry of T2, the system will send REGISTER messages to the primary server for those lines which are not involved in connected call. If the primary server doesn’t reply the REGISTER, the system will try to send REGISTER to the secondary server. Should the sip server replies 200OK then all lines are switched back to this sip server, and then for those lines with calls the system will switch them back after the call is released. If both sip servers are still OOS, T2 timer keeps running and there is no need to send REGISTER for those lines with calls anymore.
  - After a switching action is done, the system MUST not send de-register to the previous sip server.
