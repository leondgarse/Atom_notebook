
# Feature Information
  - RCR: ALU02371974 [Vodafone Spain] [ONT] : VOICE -  Requirements  (phase 4b, incl. 3G Dongle/Redundancy) - part 2
  - IR: ALU02403835
  - [Binder: 3HH-14840-AAAA](https://ct.web.alcatel-lucent.com/scm-lib4/view.cgi?number=3HH-14840-AAAA-ADZZA-01P01&no_index_sheet)
  - FDT: hg clone ssh://hg@135.251.206.233/HD_R58_FDT1480

  - RCR: ALU02409655 [Vodafone Spain] [ONT] : VOICE -  Requirements phase 4b-part1
  - IR: ALU02414621
  - CINS: 3HH-14894-AAAA
  - [Binder: 3HH-14894-AAAA](https://ct.web.alcatel-lucent.com/scm-lib4/view.cgi?number=3HH-14894-AAAA-ADZZA-01P01&no_index_sheet)
  - [Grromer](https://ct.web.alcatel-lucent.com/scm-lib4/show-entry.cgi?number=3HH-09255-0014-ASZZA)
  # - [Grromer](http://aww.sh.bel.alcatel.be/metrics/datawarehouse/query/Groomer_PB_309.cgi?PBA=2&cont=1&PREFIX=FAGR&RCR=ALU02409655)
***

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
        VRG_UINT8  version:2;   // protocol version
        VRG_UINT8  p:1;         // padding flag
        VRG_UINT8  x:1;         // header extension flag
        VRG_UINT8  cc:4;        // CSRC count
        VRG_UINT8  m:1;         // marker bit
        VRG_UINT8  pt:7;        // payload type
    #elif BOS_CFG_LITTLE_ENDIAN
        VRG_UINT8  cc:4;        // CSRC count
        VRG_UINT8  x:1;         // header extension flag
        VRG_UINT8  p:1;         // padding flag
        VRG_UINT8  version:2;   // protocol version
        VRG_UINT8  pt:7;        // payload type
        VRG_UINT8  m:1;         // marker bit
    #else
       #error "BOS_CFG_xxx_ENDIAN not defined!"
    #endif
        VRG_UINT8  seq[2];      // 16-bit sequence number
        VRG_UINT8  ts[4];       // 32-bit timestamp
        VRG_UINT8  ssrc[4];     // synchronization source
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
***

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
    - Progress: not supported yet, but in ALU02329107([TELMEX] additional TR-104 objects), it will be implemented as read only, and the real Display name used for the From-header should be used for this TR104 object.
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
      Currently, we have supported these val.ues: “Up”,”Registering”,”Error”,”Testing”,”Disabled”.
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
***

# FS doc
## Per line call stat parameters
  - Vodafone Spain is requesting the support of following TR-104 standard parameters that we do not support ‘as is’ now
    ```python
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.ReceivePacketLossRate
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.FarEndPacketLossRate
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.RoundTripDelay
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.AverageReceiveInterarrivalJitter
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.AverageFarEndInterarrivalJitter
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Stats.AverageRoundTripDelay
    ```
  - Nokia does support similar parameters defined per session instead of per line as requested by VFS (inline with TR-104 spec), and this is confirmed by the PICS
    ```python
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_ReceivePacketLossRate
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_FarEndPacketLossRate
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_RoundTripDelay
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_AverageReceiveInterarrivalJitter
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_AverageFarEndInterarrivalJitter
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Session.{i}.X_ALU-COM_AverageRoundTripDelay
    ```
  - For basic call
    - There is no difference in case of a basic call, therefore my recommendation towards R&D will be that we link the values of the proprietary parameters to the standard parameter and then we will support VFS requirement.
    - Furthermore RBC/CU confirmed that basic call is sufficient.
  - For multiparty calls (CH / CW)
    - The data is collected for each of the sessions and not that there is single data per line.
    - In the TR-104 spec there is no explicit reference to how multiparty calls should be handled. The definition only talks about ‘current call’.
    - We only need to specify ourselves what to represent in case of multiparty calls.
    - What is required from VFSP for basic call (reuse of what we have, only mapping to the standard parameters)
    - For multiparty call, what I would propose is to report the stats for the active session, but also that I check with R&D if this is indeed the easiest implementation. If not, we (BU) can make another proposal.
  - For multiparty calls (CC)
    - For conference scenario, that both session are active
    - Choose session 1 as line stat.
    - If session 1 is empty, report session 2 as line stat.
## PortStatus
  - The requirement to support InternetGatewayDevice.Services.VoiceService.{i}.PhyInterface.{i}.X_PortStatus
  - The requested port status values are listed below as per VFS specification.
    - Diabled | enabled
    - On Hook
    - Off Hook
    - WaitonHook = Parking, lets consider this status “Waitonhook” is parking, and lets use the rules of our CVP to move a subscriber to parking.
    - Calling - Outgoing Call
    - Ringing - Incoming Call
    - Talking – Call established (incoming or outgoing)
  - It seems to me that we should be able to do a mapping based on call state in CVP.
  - It seems we may need another value for ‘releasing’, and for some I have no good mapping (howlertone, idle, unknown).
  - I am also wondering when we would have the value of ‘enabled’ as when ‘enabled’ I expect one of the other values will be reported.
  - For CallState:
    ```md
    | Call State in CVP | Call State in TR-104 | X_PortStatus |
    | ----------------- | -------------------- | ------------ |
    | Onhook            | Idle                 | OnHook       |
    | Offhook           | Idle                 | OffHook      |
    | idleOffhook       | Idle                 | WaitOnHook   |
    | Dialing           | Calling              | Calling      |
    | Calling           | Connecting           | Calling      |
    | Connected         | In Call              | Talking      |
    | ringring          | Ringing              | Ringing      |
    | ringback          | Ringing              | Ringing      |
    | releasing         | Disconnecting        | ???          |
    | Busytone          | Connecting           | Calling      |
    | Howlertone        | Idle                 | ???          |
    | Reorder           | In Call              | Talking      |
    | Dialtone_soc      | In Call              | Talking      |
    | Dialtone_third    | In Call              | Talking      |
    | Unknow            | Idle                 | ???          |
    | Idle              | Idle                 | ???          |
    ```
  - As there is no value for X_PortStatus that indicates a state related to releasing, I think we can map ‘releasing’ and ‘howlertone’ to Talking
  - When is CVP in state ‘unknown’ (is this an error state?), or in state ‘idle’? If possible, I would report onhook, offhook or waitonhook if we can derive this from other information.
  - 3HH-14012-AGAA-DFZZA-04P01-FS_ALU02162847_AONT Voice_Call History Statistics via HTTP or WEBGUI.ppt
    ```c
    Idle : This state is returned if one of the following conditions are fulfilled :
    Subscriber is on-hook
    Subscriber is off-hook and gets Howler tone
    Subscriber is off-hook and in parking state
    Subscriber is off-hook and gets dial tone
    Subscriber is off-hook and no match found in the digit map for the dialed number (Final state preceded by “Disconnecting” state)
    Subscriber is in a state that is not one of the other states defined in this overall summary
    From the sending of a CANCEL request till the release of all internal resources allocated for the call attempt. (Final state preceded by “Ringing”)
    From the sending of a BYE request till the release of all internal resources allocated for the call attempt. (Final state precede by “In Call”)
    Calling : This state is returned if the following condition is fulfilled :
    Subscriber is off-hook, has started dialing a DN and the dialing is not yet completed.
    Connecting : This state is returned if the following condition is fulfilled :
    The dialing has completed, A match was found in the digit map and an initial invite has been sent. No final response received yet from the SIP core.
    Special dial tome received as a result of having put a call on hold
    dialing a DN to set-up another party
    ```
  - Code
    ```c
    pal_call_status_breaking = 104    // ALU01962350

    pal_fn_getcallstatus(lineId, &callStatus);
    else if (callStatus == VM_CALL_STATUS_UNKNOWN)
      pConvert->setCallStatus("Unknown");
    else
      pConvert->setCallStatus("Idle");  
    ```
## TR104 Definition of voice network type(VoIP / VoCS):
  - For network, there will be a need for a new parameter that is linked with the use of GPON mode (VoIP) or HSPA mode (VoCS).
  - The parameter name we suggest using "InternetGatewayDevice.X_ASB_COM_VoiceNetwork.type", refer to “InternetGatewayDevice.X_ASB_COM_VOIPCfgType.type”, please confirm whether it is ok.
  - [JDV] Note that the mode (GPON, HSPA) can be different for voice and data services (criteria to switch between the 2 modes are different for voice and data). Therefore, I think the parameter needs to be at a lower level in the datamodel. It could be e.g.
  - InternetGatewayDevice.Services.VoiceService.{i}.X_ASB_COM_VoiceNetwork.type
## TR104 Definition for T1 timer(Registration Retry Interval) & T2 timer(Primary Proxy Retry Interval):
  - Need to implement new tr104 parameter T1&&T2 timer to configure new sip redundancy mechanism, please give the definition.
  - [JDV] T1 and T2 are from earlier SIP server redundancy requirement (v094) and they do not re-appear in the new one (v095) in which other timers are introduced for managing the SIP server redundancy.
  - The support of these parameters in TR104 of course needs to be aligned with the level of implementation of the SIP server redundancy mechanism in HDR58 which is currently under discussion.
## Implementation for CallerIDName:
  - Need to support this new tr104 standard node: InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.CallingFeatures.CallerIDName
  - It is not supported yet, but in ALU02329107([TELMEX] additional TR-104 objects), it will be implemented as read only. The real Display name used for the From-header should be used for this TR104 object. But as ALU02329107 hasn’t been planned, we need to implement it in Vodafone phase 4b firstly, so need to confirm whether the solution is ok or not.
  - [JDV] The easy part is to support this parameter in the TR104 model. The real question is what the ONT needs to do with this info. I have asked clarification from RBC/CU to define/specify use of this.
  - The feedback I got was that this parameter should be used when sending outgoing INVITE message, similar to the use of SIP display name in case of OMCIv2.
  - SIP display name: This ASCII string attribute defines the customer id used for the display attribute in outgoing SIP messages. The default value is null (all zero bytes). (R, W) (mandatory) (25 bytes)
  - According to the requirement doc v095, p65:
    ```md
    The HG SHALL remove any configuration related to Caller ID Name (Display Name), in order to avoid that final user can change the default Caller Id (Telephone Number) with any other string not  verified.
    ```
    We think should better define CallerIDName as readonly.
  - what is the reason to have this readonly? Why would an operator via ACS not be allowed to change the CallerIDName?
  - I think this was the proposal in Telmex context but I don’t see why, and I also don’t see why this would impact the implementation a lot.
  T1 timer(Registration Retry Interval) and T2 timer(Primary Proxy Retry Interval):
  InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.CallingFeatures.CallerIDName

  show voice status:
  Requires: Need to support CLI command to show registration status and the network used to provide the service.
  Solution for show voice status:
  The function will be same as the following command: cfgcli –g InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.Status
  Add corresponding command in cli command framework.

  show voice port status:
  Requires: Need to support CLI command to show port status.
  Solution for show voice port status:
  The function will be same as the following command: cfgcli –g InternetGatewayDevice.Services.VoiceService.{i}.PhyInterface.{i}.X_PortStatus
  Add corresponding command in cli command framework.
## Comments
  ```md
  slide 9: As indicated in RCR, the webgui needs to follow the ux-playground
  RCR states: Ziba Playground (https://vf2.ux-playground.com/gateways/view/37/23/56/overview) MUST be used as guideline for implementation of WUI.
  This was already a requirement in phase4 (HDR5701), but not everything was covered there.
  To be checked that now there is compliance, e.g. not sure if e.g. HDR5701 already has 3 user views (basic, expert, admin).
  Other requirement, voice status covered, 'Other settings' not explicitly mentioned or refering to 3HH-08371-0602-DFZZA (required parameters already part of generic webgui (except VAD support), but now list (for the delta versus HDR5701) is according to ux-playground and 3HH-08371-0602-DFZZA.

  slide 39-40: see also comment for slide 9. compliance with ux-playground (https://vf2.ux-playground.com/gateways/view/37/23/56/overview), Other settings according to 3HH-08371-0602-DFZZA for the delta versus HDR5701.
  ```

# X_ALU-COM_SessionExpires
## Requirement
  ```python
  Slide 7, X_Session_Expires is no longer an open point, confirmed phase 4b requirement.

  slide 7,please clarify the scope for this RCR
  VoiceProfile.{i}.SIP.X_SessionExpires, which is in TR98 but not in TR181 list.
  In tr098 standard not in 181 standard or in Tr98 VDS requirement not in 181 VDS requirement?
  what will be in 4b phase 1, 4b phase 2, phase 5?

  Session Timer Expires (s)
  InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.SIP.X_ALU-COM_SessionExpires

  Need to add new TR098 node .VoiceService.{i}.VoiceProfile.{i}.SIP. X_ALU-COM_SessionExpires, which will be defined in the ASB self define sheet.
  Default value for DB is the same as GSIP default value.
  Solution:
  There are two GSIP parameters - session_timer_max and session_timer_min, which have mapping relationship with them.
  Reuse the GMI message MSG_CVP_SET_SYSTEM_ATTRIBUTE_REQ (defaultSessionExpires) to configure these parameters to CVP.
  When system bootup, aligned with current implementation for preconfig, GSIP parameter value will override DB.


  .VoiceService.{i}.VoiceProfile.{i}.SIP.X_ALU-COM_SessionMinSE
  Above two parameters are new added, which will be defined in the ASB self define sheet.
  There are two GSIP parameters - session_timer_max and session_timer_min, which have mapping relationship with them. Reuse the GMI message MSG_CVP_SET_SYSTEM_ATTRIBUTE_REQ (defaultMinSE and DefaultSessionExpires) to configure these parameters to CVP.
  When system bootup, aligned with current implementation for preconfig, GSIP parameter value will override DB(X_SessionExpires and X_SessionMinSE).
  Default value for DB is the same as GSIP default value.
  These two parameters are also as part of “Other Settings” in WebGUI.

  ```
  ```python
  Session Timer Expires (s) and Session Timer Min-SE (s)
  .VoiceService.{i}.VoiceProfile.{i}.SIP. X_ALU-COM_SessionExpires(will added in VFS phase 4b)
  .VoiceService.{i}.VoiceProfile.{i}.SIP.X_ALU-COM_SessionMinSE
  Above two parameters are new added, which will be defined in the ASB self define sheet.
  There are two GSIP parameters - session_timer_max and session_timer_min, which have mapping relationship with them. Reuse the GMI message MSG_CVP_SET_SYSTEM_ATTRIBUTE_REQ (defaultMinSE and DefaultSessionExpires) to configure these parameters to CVP.
  When system bootup, aligned with current implementation for preconfig, GSIP parameter value will override DB(X_SessionExpires and X_SessionMinSE).
  Default value for DB is the same as GSIP default value.
  These two parameters are also as part of “Other Settings” in WebGUI.

  Links for RCR: ALU02409655:

  Need to add new TR098 node .VoiceService.{i}.VoiceProfile.{i}.SIP.X_ALU-COM_SessionExpires, which will be defined in the ASB self define sheet.
  Default value for DB is the same as GSIP default value.
  Solution:
  There are two GSIP parameters - session_timer_max and session_timer_min, which have mapping relationship with them.
  Reuse the GMI message MSG_CVP_SET_SYSTEM_ATTRIBUTE_REQ (defaultSessionExpires) to configure these parameters to CVP.
  When system bootup, aligned with current implementation for preconfig, GSIP parameter value will override DB.
  ```
## X_CT_COM_SessionUpdateTimer FR
  ```python
  ******** SETUP DESCRIPTION ********
  7360 FX
  one FANT-F board
  one NGLT-C board
  NT Redundancy: No
  LT Redndancy: No
  EMS: CLI
  LT Slot: 1
  NT slot:  a
  OLT SWver: L6GPAA55.078
  ONT Type: G-240W-B
  Test Version: 3FE56773AFFA26 (HDR55)
  Test Equip: ATE SIP server
  ******** Summary of problem:********
  With the help of Iris and sunny , Session-Expires header and Min-SE in The UPDATE used to refresh session are hardcode.
  1. Min-SE
  the configuration of MinSE via TR069 is not forwarded to CVP. A specific value 600 is forwarded to CVP.  we previously support this parameter, but to solve a bulk call issue. The implementation of this parameter is deleted by a colleague, and he set a specific value 600.
  2. Session-Expire
  Cfgmgr intends to use the value of X_CT-COM_SessionUpdateTimer which is not node for VFS. But when it send values to session-expire, it hardcode as 1800.
  ******** DETAILED DESCRIPTION ********
  STEPS:
  1. Configure ONT with VFSP
  2. Make a long duration call to check how long sends update message.
  3. Modify GSIP session_time_max, the update still sends 900s later after call is established.

  EXPECTATION:
  Min-SE uses value of node X_ASB_COM_VOICECONFIG.MinSE
  Session-expire uses value of GSIP parameter.

  RESULT:
  GSIP parameter session_time_max is not work.
  The Min-SE is not different from value got from serial.

  ******** REPRODUCTION NOTES ********
  Can be reproduced every time.

  ******** Trace & Debug ********
  Please check the attachment [xml_issue.txt]  

  ******** END *********
  ```
  ```python
  [Problem]
  Session-Expires header and Min-SE in the UPDATE message to refresh session are not consistent with configured values via XML.

  [Root Cause]
  Besides session_timer_max and session_timer_min in XML file, TR-104 parameters X_CT_COM_SessionUpdateTimer and X_ASB_COM_VOICECONFIG.MinSE can also configure the values of Session-Expires header and Min-SE header in SIP message. In our code logic, these two TR-104 parameters have been fixed to some specific values (1800 and 600) when configuring them from CFGMGR to CVP. Therefore, XML configuration of these two GSIP parameters will always be override by these two fixed value in CVP module.

  [Solution]
  For X_CT_COM_SessionUpdateTimer, it is not supported by our system anymore. We unset the bitmask of this parameter when configuring it from CFGMGR to CVP to make sure that in CVP the final used value is always from XML configuration.
  For X_ASB_COM_VOICECONFIG.MinSE, we will deprecate this parameter from our system. The deprecation will be divided into two steps. In this FR, we will only handle the first step. The second step will be handled in later release. In this FR we will solve this issue as follows.
  (1) We still keep it in TR-104 data model in code logic;
  (2) Customer can configure/retrieve it via TR069. No error will be returned to customer.
  (3) We ignore value of this parameter by not configuring it from CFGMGR to CVP in code logic.We will unset the bitmask of this parameter when configuring it from CFGMGR to CVP to realize this item.

  In second step, we will remove this parameter from supported TR104 data model completely. At that time, customer will not be able to configure/retrieve this parameter anymore.

  Divide this issue into two steps is because we must gave the opportunity and time to the customer to adapt its legacy procedures.For the legacy customer still to run its current procedures (which contain the configuration of other parameters too) without getting an error for that one parameter, causing the entire procedure to fail or to be aborted due to such error case of that one parameter. Step1 is a very temporary solution/way of working as to give the operator the time to adapt everything as necessary.
  And of course the legacy customer must immediately use the correct GSIP XML file settings.

  [IMPACT]
  For both broadcom and broadlight AONTs, only GSIP parameters session_timer_min and session_timer_min will be workable after this FR. Corresponding TR-104 parameters will not be workable anymore.

  [Changeset]
  ccc9a2d5860b

  [Review Board Link]
  http://135.251.206.105/r/20291/
  ```
## GMI
  - cfgmgr rts中会发送GMI消息到VM，以以下消息为例: MSG_CVP_SET_SIPSERVER_REQ,相应cfgmgr接口config_sipServerList，相应节点
    ```c
    InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.SIP.OutboundProxy
    ```
  - VM部分
    - voip_mgnt_ipcmsghandler() -> default分支 -> voip_msg_handler() -> 走MSG_VOIP_CVP_START<x<MSG_VOIP_CVP_END分支 -> cvp_msg_convert() ->  check MSG_CVP_SET_SIPSERVER_REQ 分支 -> handleMsg() 根据消息不同创建的类进行函数搜索，比如VMCVPSipServer::handleMsg
## cfgcli get
  ```c++
  CfgRet rts_VoiceProfSipObject(VoiceProfSipObject *newObj, const VoiceProfSipObject *currObj, const InstanceId *iid, const int flag)
      return config_voiceProfSipObj(newObj);
      --> CfgRet config_voiceProfSipObj(VoiceProfSipObject *voiceProfSipObj)
          if(voiceProfileSip.X_CT_COM_SessionUpdateTimer > 0)
          pop_voipMsgHeader((voipExtMsgHeader_t*)SystemAttributeReqMsg,
              MSG_CVP_SET_SYSTEM_ATTRIBUTE_REQ,
              sizeof(vm_system_attribute_set_req_t));
          pop_configVpsSystemAttributeReq(&voiceProfileSip,(vm_system_attribute_set_req_t *)RTS_VOIPMSG_BODY(SystemAttributeReqMsg));
          --> void pop_configVpsSystemAttributeReq(VoiceProfSipObject* voiceProfSipObj, vm_system_attribute_set_req_t* pMsg)
              pMsg->DefaultSessionExpires = voiceProfSipObj->X_CT_COM_SessionUpdateTimer;
              pMsg->bitmask |= VOIPCONFIGSYSTEMATTRIBUTE_DEFAULT_SESSION_EXPIRES;
          pop_configVoiceProfSipReq(obj,pReqMsg);

  void VMCVPSystemAttribute::handleMsg(mg_api_id_et msgId, vm_u8* pPayload)
      case MSG_CVP_SET_SYSTEM_ATTRIBUTE_REQ:
      handleCfg((vm_system_attribute_set_req_t*)pPayload);
      --> void VMCVPSystemAttribute::handleCfg(vm_system_attribute_set_req_t* req)
          fillSystemAttributeObj(req);
          --> void VMCVPSystemAttribute::fillSystemAttributeObj(...)
              DefaultSessionExpires = req->DefaultSessionExpires;
          result = (cfgSystemAttribute() == VM_TRUE);
          --> bool VMCVPSystemAttribute::cfgSystemAttribute()
              convertVMVsp(vsp);
              --> void VMCVPSystemAttribute::convertVMVsp(VMVsp& configData)
                  if (vmIsBitSet(bitmask, VOIPCONFIGSYSTEMATTRIBUTE_DEFAULT_SESSION_EXPIRES))
                  configData.addMember(VMVsp::E_VMVSP_DEFAULT_SESSTION_EXPIRES, ... )

  void VMLinePflConvert::handle(vm_u32 lineId, ProvisioningDataId_et objType)
      if (!configVsp(vsp))
      --> bool VMLinePflConvert::configVsp(VMVsp& vsp)
          if ((ftpPfl.lineData.session_timer_max >= 90) && (ftpPfl.lineData.session_timer_max <= 65535))
          vsp.addMember(VMVsp::E_VMVSP_DEFAULT_SESSTION_EXPIRES, (unsigned char*) & (ftpPfl.lineData.session_timer_max), sizeof(vm_u32), PROV_SRC_FTPPFL);

  void SipUserAgent::OnAdminStatusUp(SipMgntElement* pSuper)
      sendConfigDataToPAL(vspName);
      --> void SipUserAgent::sendConfigDataToPAL(const std::string& vspName) const
          voipData.dSessionExpire = pVspData->DefaultSessionExpires;

  bool OBJVoiceProfileSip::configVMVsp()
      bool OBJVoiceProfileSip::configVMVsp()
      vsp.addMember(VMVsp::E_VMVSP_DEFAULT_SESSTION_EXPIRES, (unsigned char*)&tmpSessionUpdateTimer, sizeof(vm_u32), PROV_SRC_TR069);
  ```
## cfgcli set
  ```c++
  CfgRet stl_VoiceProfSipObject(VoiceProfSipObject *cfgObj, const InstanceId *iid)

  vm_bool voip_getLineData(voip_profile_t* data, FtpProfileStruct* lineProfile)
      voip_setSessionTimer(lineProfile->lineData.session_timer_max, &(data->sessionUpdateTimer));
      --> void voip_setSessionTimer(vm_u32 sessionTimer, vm_u32_t* data)
          data->value = sessionTimer / 60;

  void VoIPPreCfgApply(const char* xmlPath, char flagGetZeromgrXml)
      if(VoIPPreCfg.sessionUpdateTimer.isValid)
          strcpy(precfg_attr_tbl[PRECFG_SessionUpdateTimer].datamodel_path, VoIP_PRECFG_PATH_VOICEPROFILE"SIP.X_CT-COM_SessionUpdateTimer");
  ```
***

# Refresh call history
## Requirement
  ```python
  For the mentioned call history objects, the situation is a little complex.
  ²  Currently, we only support to obtain these objects via WEBGUI. We don’t support to obtain neither the TR-104 objects nor the TR181 objects via ACS server for them.

  ²  To implement the function of obtaining them via ACS server, there are two alternative solutions.
  Ø  Solution1:
  On each of the GPV request from ACS server, we retrieve the latest call history records from CVP via GMI and return the required value to ACS.
  This solution will get one problem. When getting “InternetGatewayDevice.Services.VoiceService” via ACS, the operation will time out. The root cause is that getting value of each call history parameter of each call record will always cause inviting CVP via GMI. It is quiet time costly. (Related FR: ALU02366861)

  Ø  Solution2:
      On each of the GPV request from ACS server, we retrieve the call history data from cfgmgr DB. To support this solution, we need add an additional private TR-104 parameter to synchronize the call history records in cfgmgr DB with PMC buffer.
      If adopting this solution, we need restrict in customer spec document that: If customer wants to obtain the newest call history data, he need first update the cfgmgr DB via the new added TR104 parameter before he GPV for call history parameters. Otherwise, the GPV of call history data will only relate to the calls made before the last update operation via the new TR-104 parameter.
  ```
## FOO
  ```c++
  "InternetGatewayDevice.Services.VoiceService.%d.VoiceProfile.%d.Line.%d.",
  rts_VoiceLineObject,
  --> CfgRet rts_VoiceLineObject(VoiceLineObject *newObj, const VoiceLineObject *currObj, const InstanceId *iid, const int flag)
      return config_voiceLineObj(newObj,currObj);
      --> CfgRet config_voiceLineObj(VoiceLineObject *voiceLineObj, const VoiceLineObject *curvoiceLineObj)
          return voiceLineObjCfg(voiceLineObj, curvoiceLineObj, acAttrValue);


  CfgRet get_voiceHistSessionObj(VoiceHistSessionObject* voiceHistObj, int flag)
      if (iid.instance[iid.currentDepth - 1] != 1)
      return CFG_RET_SUCCESS;

  ```
***

# Test added node
  ```c
  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Enable
  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Enable
  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.2.Enable
  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.SIP.X_ALU-COM_SessionExpires

  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Enable Enabled
  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Enable Enabled
  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.2.Enable Enabled

  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_Hist_Session.1.SessionStartTime
  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_Hist_Session.1.CallingNumber
  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_Hist_Session.2.CallingNumber
  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_Hist_Session.5.CallingNumber

  cfgcli dump InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.
  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_CallHistory
  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_CallHistory 5
  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.2.X_ALU-COM_CallHistory 5
  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_CallHistory 10
  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.2.X_ALU-COM_CallHistory 10

  Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.X_ALU-COM_CallHistoryRefresh
  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_CallHistoryRefresh
  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_CallHistoryRefresh True
  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_CallHistoryRefresh False

  cvpcli pmc getAllCallAnyData 0

  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Stats.ResetStatistics
  cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Stats.ResetStatistics True

  cfgcli -g InternetGatewayDevice.Services.VoiceService.1.PhyInterface.1.Tests.X_ALU-COM_MDT_TestMode

  tr181 -g Device.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_CallHistoryRefresh
  tr181 -g Device.Services.VoiceService.1.VoiceProfile.1.SIP.X_ALU-COM_SessionExpires

  tr181 -s Device.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_CallHistoryRefresh True
  tr181 -s Device.Services.VoiceService.1.VoiceProfile.1.SIP.X_ALU-COM_SessionExpires 1800
  tr181 -s Device.Services.VoiceService.1.VoiceProfile.1.Line.1.X_ALU-COM_CallHistory
  ```
  ```c
  comcli -m cfgmgr -d 2 -o 1
  cvpcli dbg spt -m VOPM -l 0
  cvpcli dbg spo module on
  cvpcli dbg log stdout
  comcli -s
  cvpcli dbg spt -m VOPM -l 255
  ```

SE: delete if
RC: keep reset?
RC: TR181


http_ms/src/cgi/call_history_statistics.c


8DM-02690-000B-PBZZA tr181
8DM-02690-0005-PBZZA tr104
3HH-08371-0201-DFZZA
14:14

104的先别更新
14:14

你们先更新181
14:14

104的需要先把对应release的column创建出来
14:14

之前的都乱了

IR ALU02414621:[AONT] RCR ALU_02409655 Tr181 mapping for X_ALU-COM_SessionExpires and X_ALU-COM_CallHistoryRefresh
[Requirements]
  RCR ALU_02409655 Tr181 mapping for X_ALU-COM_SessionExpires and X_ALU-COM_CallHistoryRefresh.
  Replace macro STATS_PER_CALL_HIST with STATS_PER_CALL_HIST.

[Root Cause]
  Customer Requirements.

[Solution]
  Tr181 mapping logic for X_ALU-COM_SessionExpires and X_ALU-COM_CallHistoryRefresh.
  STATS_PER_CALL_HIST is available for  Makefile, and in code it's mapped to STATS_PER_CALL_HIST.

[IMPACT]
  No impact.

[Files]
apps/private/libs/mapper/Makefile
apps/private/libs/mapper/tr181.xml
apps/private/libs/mapper/voice.c

InternetGatewayDevice.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.SIP.URI
ALU02409655 – VOICE Requirements for phase 4b part1

cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Stats.ReceivePacketLossRate
cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Stats.FarEndPacketLossRate
cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Stats.RoundTripDelay
cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Stats.AverageReceiveInterarrivalJitter
cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Stats.AverageFarEndInterarrivalJitter
cfgcli -g InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.Stats.AverageRoundTripDelay
cfgcli -g InternetGatewayDevice.Services.VoiceService.1.PhyInterface.1.X_ALU-COM_PortStatus
cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Line.1.CallingFeatures.CallerIDName
cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.Enable Enabled
cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.SIP.X_ALU-COM_SessionExpires 1800
cfgcli -s InternetGatewayDevice.Services.VoiceService.1.VoiceProfile.1.X_ALU-COM_CallHistoryRefresh True


>>>> refresh_voiceHistSessionStat:507 Here we are, pRespMsg->returned_statistics_bitmask = 3, pRespMsg->countersValueSize = 0

X_ALU_COM_CallHistory


cfgcli -g InternetGatewayDevice.Services.VoiceService.1.PhyInterface.1.X_ALU-COM_PortStatus
Device.Services.VoiceService.{i}.VoiceProfile.{i}.Line.{i}.X_ALU-COM_CallHistoryRefresh
