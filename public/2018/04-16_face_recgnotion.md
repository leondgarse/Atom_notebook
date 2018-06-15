# ___2018 - 04 - 16 Face Recognition___
***

# Opencv 视频处理
## 环境
  ```shell
  conda create -n opencv
  conda info --envs
  source activate opencv

  conda install -yc conda-forge opencv
  pip install --upgrade pip

  pip install face_recognition
  pip install baidu-aip
  ```
## 使用
  ```python
  client.identifyUser(groupId, cv2.imencode('.jpg', frame[:, :, ::-1])[1].tobytes(), options)
  known_face_encodings
  unknown_encording = fr.face_encodings(fr.load_image_file("chai_jinlong.png"))[0]
  unknown_encording = fr.face_encodings(fr.load_image_file("/home/leondgarse/Pictures/Selection_002.png"))[0]
  fr.face_distance(known_face_encodings, unknown_encording)
  image = get_file_content('/home/leondgarse/Pictures/Selection_002.png')
  client.identifyUser(groupId, image, options)
  ```
***

# 百度人脸识别 Aip
## 简介
  - [Server SDK 资源](https://ai.baidu.com/sdk#bfr)
  - [Python SDK 文档](http://ai.baidu.com/docs#/Face-Python-SDK/top)
  - **人脸检测** 检测人脸并定位，返回五官关键点，及人脸各属性值
  - **人脸比对** 返回两两比对的人脸相似值
  - **人脸查找** 在一个人脸集合中找到找到相似的人脸，由一系列接口组成，包括人脸识别、人脸认证、人脸库管理相关接口（人脸注册、人脸更新、人脸删除、用户信息查询、组列表查询、组内用户列表查询、组间复制用户、组内删除用户）
  - 安装人脸识别 Python SDK
    ```python
    # pip
    pip install baidu-aip
    # setuptools
    python setup.py install
    ```
  - 人脸识别 Python SDK目录结构
    ```python
    ├── README.md
    ├── aip                   # SDK目录
    │   ├── __init__.py       # 导出类
    │   ├── base.py           # aip基类
    │   ├── http.py           # http请求
    │   └── face.py           # 人脸识别
    └── setup.py              # setuptools安装
    ```
## 新建AipFace
  - AipFace 是人脸识别的 Python SDK 客户端，为使用人脸识别的开发人员提供了一系列的交互方法
  - 常量 `APP_ID` 在百度云控制台中创建，常量 `API_KEY` 与 `SECRET_KEY` 是在创建完毕应用后，系统分配给用户的，均为字符串，用于标识用户，为访问做签名验证，可在AI服务控制台中的应用列表中查看
  ```python
  from aip import AipFace
  APP_ID = "11108357"
  API_KEY = 'PrurvVg3MUluX61sviVlSGha'
  SECRET_KEY = 'VLKac9zzkcdkUcjSVZhwmPwEWLDLtQ92'
  client = AipFace(APP_ID, API_KEY, SECRET_KEY)

  """ 读取图片 """
  def get_file_content(filePath):
      with open(filePath, 'rb') as fp:
          return fp.read()
  ```
## 人脸检测
  - 检测请求图片中的人脸，返回人脸位置、72个关键点坐标、及人脸相关属性信息
  - 典型应用场景：如 **人脸属性分析**，**基于人脸关键点的加工分析**，**人脸营销活动** 等
  - 五官位置会标记具体坐标；72个关键点坐标也包含具体坐标，但不包含对应位置的详细位置描述
  - **参数**
    - **image**	图像数据，base64编码，要求base64编码后大小不超过4M，最短边至少15px，最长边最大4096px,支持jpg/png/bmp格式
    - **max_face_num**	最多处理人脸数目，默认值1
    - **face_fields**	包括age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities信息，逗号分隔，默认只返回人脸框、概率和旋转角度
  - **返回参数**
    - **result_num** 人脸数目
    - **face_probability**	人脸置信度，范围0-1
    - **occlusion**	人脸各部分遮挡的概率,[0, 1],0表示完整，1表示不完整
  - **示例**
    ```python
    image = get_file_content('/home/leondgarse/workspace/face_recognition/examples/two_people.jpg')

    """ 调用人脸检测 """
    client.detect(image);

    """ 如果有可选参数 """
    options = {}
    options["max_face_num"] = 2
    options["face_fields"] = "age"

    """ 带参数调用人脸检测 """
    client.detect(image, options)
    ```
## 人脸比对
  - 该请求用于比对多张图片中的人脸相似度并返回两两比对的得分，可用于判断两张脸是否是同一人的可能性大小
  - 典型应用场景：如人证合一验证，用户认证等，可与您现有的人脸库进行比对验证
  - **参数**
    - **images** base64编码后的多张图片数据，半角逗号分隔，单次请求总共最大20M
    - **ext_fields** 返回质量信息，取值固定:目前支持qualities(质量检测)。(对所有图片都会做改处理)
    - **image_liveness**
      - faceliveness,faceliveness - 对比对的两张图片都做活体检测
      - faceliveness - 对第一张图片不做活体检测、第二张图做活体检测
      - faceliveness, - 对第一张图片做活体检测、第二张图不做活体检测
    	- 需要用于判断活体的图片，图片中的人脸像素面积需要不小于100px*100px，人脸长宽与图片长宽比例，不小于1/3
    - **types**	请求对比的两张图片的类型，示例：“7,13”
      - 12表示带水印证件照：一般为带水印的小图，如公安网小图
      - 7表示生活照：通常为手机、相机拍摄的人像图片、或从网络获取的人像图片等
      - 13表示证件照片：如拍摄的身份证、工卡、护照、学生证等证件图片，需要确保人脸部分不可太小，通常为100px*100px
  - **返回参数**
    - **score** 比对得分
    - **faceliveness** 活体分数，如0.49999。单帧活体检测参考阈值0.393241，超过此分值以上则可认为是活体，活体检测接口主要用于判断是否为二次翻拍，需要限制用户为当场拍照获取图片；推荐配合客户端SDK有动作校验活体使用
  - **示例**
    ```python
    images = [
        get_file_content('/home/leondgarse/workspace/face_recognition/examples/obama.jpg'),
        get_file_content('/home/leondgarse/workspace/face_recognition/examples/obama2.jpg'),
    ]

    """ 调用人脸比对 """
    client.match(images);

    """ 如果有可选参数 """
    options = {}
    options["ext_fields"] = "qualities"
    options["image_liveness"] = ",faceliveness"
    options["types"] = "7,13"

    """ 带参数调用人脸比对 """
    client.match(images, options)
    ```
## 人脸识别
  - 用于计算指定组内用户，与上传图像中人脸的相似度。识别前提为您已经创建了一个人脸库
  - 典型应用场景：如人脸闸机，考勤签到，安防监控等
  - 人脸识别返回值不直接判断是否是同一人，只返回用户信息及相似度分值
  - 推荐可判断为同一人的相似度分值为 **80**，可以根据业务需求选择更合适的阈值
  - 人脸库、用户组、用户、用户下的人脸层级关系如下所示
    ```python
    |- 人脸库
       |- 用户组一
          |- 用户01
             |- 人脸
          |- 用户02
             |- 人脸
             |- 人脸
             ....
       |- 用户组二
    ```
  - **参数**
    - **group_id** 用户组id，标识一组用户（由数字、字母、下划线组成），长度限制128B，如果需要将一个uid注册到多个group下，group_id需要用多个逗号分隔，每个group_id长度限制为48个英文字符
    - 产品建议：根据业务需求，可以将需要注册的用户，按照业务划分，分配到不同的group下，例如按照会员手机尾号作为groupid，用于刷脸支付、会员计费消费等，这样可以尽可能控制每个group下的用户数与人脸数，提升检索的准确率
    - **image** 图像数据，base64编码，要求base64编码后大小不超过4M，最短边至少15px，最长边最大4096px,支持jpg/png/bmp格式
    - **ext_fields** 特殊返回信息，多个用逗号分隔，取值固定: 目前支持faceliveness(活体检测)，需要用于判断活体的图片，图片中的人脸像素面积需要不小于100px*100px，人脸长宽与图片长宽比例，不小于1/3
    - **user_top_num** 返回用户top数，默认为1，最多返回5个
  - **返回参数**
    - **uid**	匹配到的用户id
    - **scores**	结果数组，数组元素为匹配得分，top n，得分[0,100.0]
  - **示例**
    ```python
    groupId = "examples,hpe_faces"
    image = get_file_content('/home/leondgarse/workspace/face_recognition/examples/two_people.jpg')
    image = get_file_content('/home/leondgarse/workspace/face_recognition/examples/knn_examples/train/rose_leslie/img1.jpg')

    """ 调用人脸识别 """
    client.identifyUser(groupId, image)

    """ 如果有可选参数 """
    options = {}
    options["ext_fields"] = "faceliveness"
    options["user_top_num"] = 3

    """ 带参数调用人脸识别 """
    client.identifyUser(groupId, image, options)
    ```
## 人脸认证
  - 用于识别上传的图片是否为指定用户，即查找前需要先确定要查找的用户在人脸库中的id
  - 典型应用场景：如人脸登录，人脸签到等
  - 人脸认证与人脸识别的差别
    - 人脸识别需要指定一个待查找的人脸库中的组
    - 人脸认证需要指定具体的用户id即可，不需要指定具体的人脸库中的组
    - 实际应用中，人脸认证需要用户或系统先输入id，这增加了验证安全度，但也增加了复杂度，具体使用哪个接口需要视业务场景判断
  - **参数**
    - **uid** 用户id（由数字、字母、下划线组成），长度限制128B
    - **group_id** 用户组id，标识一组用户（由数字、字母、下划线组成），长度限制128B，如果需要将一个uid注册到多个group下，group_id需要用多个逗号分隔，每个group_id长度限制为48个英文字符
    - **image**	图像数据，base64编码，要求base64编码后大小不超过4M，最短边至少15px，最长边最大4096px,支持jpg/png/bmp格式
    - **top_num**	返回用户top数，默认为1
    - **ext_fields**	特殊返回信息，多个用逗号分隔，取值固定: 目前支持faceliveness(活体检测)，需要用于判断活体的图片，图片中的人脸像素面积需要不小于100px*100px，人脸长宽与图片长宽比例，不小于1/3
  - **返回参数**
    - **result** 结果数组，数组元素为匹配得分，top n。 得分范围[0,100.0]，推荐得分超过 **80** 可认为认证成功
    - **faceliveness** 活体分数，如0.49999，单帧活体检测参考阈值 **0.393241**，超过此分值以上则可认为是活体
  - **示例**
    ```python
    uid = "obama"
    groupId = "examples,hpe_faces"
    image = get_file_content('/home/leondgarse/workspace/face_recognition/examples/examples/obama.jpg')

    """ 调用人脸认证 """
    client.verifyUser(uid, groupId, image);

    """ 如果有可选参数 """
    options = {}
    options["top_num"] = 3
    options["ext_fields"] = "faceliveness"

    """ 带参数调用人脸认证 """
    client.verifyUser(uid, groupId, image, options)
    ```
## M:N 识别
  - 待识别的图片中，存在多张人脸的情况下，支持在一个人脸库中，一次请求，同时返回图片中所有人脸的识别结果
  - **参数**
    - **group_id** 用户组id，标识一组用户（由数字、字母、下划线组成），长度限制128B。如果需要将一个uid注册到多个group下，group_id需要用多个逗号分隔，每个group_id长度限制为48个英文字符
    - **image** 图像数据，base64编码，要求base64编码后大小不超过4M，最短边至少15px，最长边最大4096px,支持jpg/png/bmp格式
    - **ext_fields** 特殊返回信息，多个用逗号分隔，取值固定: 目前支持faceliveness(活体检测)。注：需要用于判断活体的图片，图片中的人脸像素面积需要不小于100px*100px，人脸长宽与图片长宽比例，不小于1/3
    - **detect_top_num** 检测多少个人脸进行比对，默认值1（最对返回10个）
    - **user_top_num** 返回识别结果top人数”，当同一个人有多张图片时，只返回比对最高的1个分数（即，scores参数只有一个值），默认为1（最多返回20个）
  - **返回参数**
    - **uid**	匹配到的用户id
    - **scores** 结果数组，数组元素为匹配得分，得分[0,100.0]；个数取决于user_top_num的设定，推荐80分以上即可判断为同一人
    - **position** 人脸位置，如{top:111,left:222,width:333,height:444,degree:20}
  - **示例**
    ```python
    groupId = "examples,hpe_faces"
    image = get_file_content('/home/leondgarse/workspace/face_recognition/examples/two_people.jpg')

    """ 调用M:N 识别 """
    client.multiIdentify(groupId, image);

    """ 如果有可选参数 """
    options = {}
    options["ext_fields"] = "faceliveness"
    options["detect_top_num"] = 3
    options["user_top_num"] = 1

    """ 带参数调用M:N 识别 """
    client.multiIdentify(groupId, image, options)
    ```
## 人脸注册
  - 调用在线API接口添加或删除用户后，会延迟十分钟再展示到网页页面
  - 用于从人脸库中新增用户，可以设定多个用户所在组，及组内用户的人脸图片
  - 典型应用场景：构建您的人脸库，如会员人脸注册，已有用户补全人脸信息等
  - **参数**
    - **uid** 用户id（由数字、字母、下划线组成），长度限制128B
    - **user_info** 用户资料，长度限制256B
    - **group_id** 用户组id，标识一组用户（由数字、字母、下划线组成），长度限制128B。如果需要将一个uid注册到多个group下，group_id需要用多个逗号分隔，每个group_id长度限制为48个英文字符
    - **image** 图像base64编码，每次仅支持单张图片，图片编码后大小不超过10M，为保证后续识别的效果较佳，建议注册的人脸，为用户正面人脸
    - **action_type** 参数包含append、replace
      - 如果为“replace”，则每次注册时进行替换replace（新增或更新）操作，用新图替换库中该uid下所有图片
      - 默认为append操作，uid在库中已经存在时，对此uid重复注册时，新注册的图片默认会追加到该uid下
  - **返回参数**
    - 成功返回 **log_id**	请求唯一标识码，uint64 随机数
  - **示例**
    ```python
    uid = "rose_leslie"
    userInfo = "rose_leslie's info"
    groupId = "examples,hpe_faces"
    image = get_file_content('/home/leondgarse/workspace/face_recognition/examples/knn_examples/train/rose_leslie/img1.jpg')

    """ 调用人脸注册 """
    client.addUser(uid, userInfo, groupId, image);

    """ 如果有可选参数 """
    options = {}
    options["action_type"] = "replace"

    """ 带参数调用人脸注册 """
    client.addUser(uid, userInfo, groupId, image, options)
    ```
## 人脸更新
  - 用于对人脸库中指定用户，更新其下的人脸图像
  - 针对一个uid执行更新操作，新上传的人脸图像将覆盖此uid原有所有图像
  - 执行更新操作，如果该uid不存在时，会返回错误，如果添加了action_type:replace,则不会报错，并自动注册该uid，操作结果等同注册新用
  - **参数**
    - **uid** 用户id（由数字、字母、下划线组成），长度限制128B
    - **user_info** 用户资料，长度限制256B
    - **group_id** 更新指定groupid下uid对应的信息
    - **image** 图像数据，base64编码，要求base64编码后大小不超过4M，最短边至少15px，最长边最大4096px,支持jpg/png/bmp格式
    - **action_type** append	目前仅支持replace，uid不存在时，不报错，会自动变为注册操作；未选择该参数时，如果uid不存在会提示错误
  - **返回参数**
    - 成功返回 **log_id** 请求唯一标识码，uint64 随机数
  - **示例**
    ```python
    uid = "rose_leslie"
    userInfo = "rose_leslie's info"
    groupId = "examples,hpe_faces"
    image = get_file_content('/home/leondgarse/workspace/face_recognition/examples/knn_examples/train/rose_leslie/img2.jpg')

    """ 调用人脸更新 """
    client.updateUser(uid, userInfo, groupId, image);

    """ 如果有可选参数 """
    options = {}
    options["action_type"] = "replace"

    """ 带参数调用人脸更新 """
    client.updateUser(uid, userInfo, groupId, image, options)
    ```
## 人脸删除
  - 用于从人脸库中删除一个用户
  - 删除的内容，包括用户所有图像和身份信息
  - 如果一个uid存在于多个用户组内，将会同时将从各个组中把用户删除
  - 如果指定了group_id，则只删除此group下的uid相关信息
  - **参数**
    - **uid** 用户id（由数字、字母、下划线组成），长度限制128B
    - **group_id** 删除指定groupid下uid对应的信息
  - **返回参数**
    - 成功返回 **log_id** 请求唯一标识码，uint64 随机数
  - **示例**
    ```python
    uid = "rose_leslie"

    """ 调用人脸删除 """
    client.deleteUser(uid);

    """ 如果有可选参数 """
    options = {}
    options["group_id"] = "hpe_faces"

    """ 带参数调用人脸删除 """
    client.deleteUser(uid, options)
    ```
## 用户信息查询
  - 用于查询人脸库中某用户的详细信息
  - **参数**
    - **uid** 用户id（由数字、字母、下划线组成），长度限制128B
    - **group_id** 选择指定group_id则只查找group列表下的uid内容，如果不指定则查找所有group下对应uid的信息
  - **返回参数**
    - **uid**	匹配到的用户id
    - **user_info**	注册时的用户信息
    - **groups** 用户所属组列表
  - **示例**
    ```python
    uid = "rose_leslie"

    """ 调用用户信息查询 """
    client.getUser(uid);

    """ 如果有可选参数 """
    options = {}
    options["group_id"] = "examples"
    # options["group_id"] = "hpe_faces"

    """ 带参数调用用户信息查询 """
    client.getUser(uid, options)
    ```
## 组内用户列表查询
  - 用于查询用户组的列表
  - **参数**
    - **start**	默认值0，起始序号
    - **num**	返回数量，默认值100，最大值1000
  - **返回参数**
    - **result_num** 返回个数
    - **result** group_id列表
  - **示例**
    ```python
    """ 调用组列表查询 """
    client.getGroupList();

    """ 如果有可选参数 """
    options = {}
    options["start"] = 0
    options["num"] = 50

    """ 带参数调用组列表查询 """
    client.getGroupList(options)
    ```
## 组内用户列表查询
  - 用于查询指定用户组中的用户列表
  - **参数**
    - **group_id** 用户组id（由数字、字母、下划线组成），长度限制128B
    - **start** 默认值0，起始序号
    - **num** 返回数量，默认值100，最大值1000
  - **返回参数**
    - **result_num** 返回个数
    - **result** user列表
    - **uid**	用户id
    - **user_info**	用户信息
  - **示例**
    ```python
    groupId = "hpe_faces"

    """ 调用组内用户列表查询 """
    client.getGroupUsers(groupId);

    """ 如果有可选参数 """
    options = {}
    options["start"] = 0
    options["num"] = 50

    """ 带参数调用组内用户列表查询 """
    client.getGroupUsers(groupId, options)
    ```
## 组间复制用户
  - 用于将已经存在于人脸库中的用户复制到一个新的组
  - **参数**
    - **src_group_id** 从指定group里复制信息
    - **group_id** 用户组id，标识一组用户（由数字、字母、下划线组成），长度限制128B，如果需要将一个uid注册到多个group下，group_id需要用多个逗号分隔，每个group_id长度限制为48个英文字符
    - **uid** 用户id（由数字、字母、下划线组成），长度限制128B
  - **返回参数**
    - 成功返回 **log_id** 请求唯一标识码，uint64 随机数
  - **示例**
    ```python
    srcGroupId = "examples"
    groupId = "hpe_faces"
    uid = "obama"

    """ 调用组间复制用户 """
    client.addGroupUser(srcGroupId, groupId, uid);
    ```
## 组内删除用户
  - 用于将用户从某个组中删除，但不会删除用户在其它组的信息
  - 当用户仅属于单个分组时，本接口将返回错误，请使用人脸删除接口
  - **参数**
    - **group_id** 用户组id，标识一组用户（由数字、字母、下划线组成），长度限制128B。如果需要将一个uid注册到多个group下，group_id需要用多个逗号分隔，每个group_id长度限制为48个英文字符
    - **uid** 用户id（由数字、字母、下划线组成），长度限制128B
  - **返回参数**
    - 成功返回 **log_id** 请求唯一标识码，uint64 随机数
  - **示例**
    ```python
    groupId = "hpe_faces"
    uid = "obama2"
    """ 调用组内删除用户 """
    client.deleteGroupUser(groupId, uid)
    ```
## 身份验证
  - 质量检测（可选）活体检测（可选）公安验证（必选）
  - **参数**
    - **image** 图像数据，base64编码，要求base64编码后大小不超过4M，最短边至少15px，最长边最大4096px,支持jpg/png/bmp格式
    - **id_card_number** 身份证号（真实身份证号号码）。我们的服务端会做格式校验，并通过错误码返回，但是为了您的产品反馈体验更及时，建议在产品前端做一下号码格式校验与反馈
    - **name** utf8，姓名（真实姓名，和身份证号匹配）
    - **quality** 判断图片中的人脸质量是否符合条件。use表示需要做质量控制，质量不符合条件的照片会被直接拒绝
    - **quality_conf** 人脸质量检测中每一项指标的具体阈值设定，json串形式，当指定quality:use时生效
    - **faceliveness** 判断活体值是否达标。use表示需要做活体检测，低于活体阈值的照片会直接拒绝
    - **faceliveness_conf** 人脸活体检测的阈值设定，json串形式，当指定faceliveness:use时生效。默认使用的阈值如下：{faceliveness：0.834963}
    - **ext_fields** 可选项为faceliveness，qualities。选择具体的项，则返回参数中将会显示相应的扩展字段。如faceliveness表示返回结果中包含活体相关内容，qualities表示返回结果中包含质量检测相关内容
  - **返回参数**
    - result	是	float	与公安小图相似度可能性，用于验证生活照与公安小图是否为同一人，有正常分数时为[0~1]，推荐阈值0.8，超过即判断为同一人
    - No permission to access data
      ```python
      {'error_msg': 'No permission to access data', 'error_code': 6}
      ```
  - **示例**
    ```python
    image = get_file_content('/home/leondgarse/Downloads/webwxgetmsgimg.jpg')
    idCardNumber = "0101010101001"

    name = "whoami"

    """ 调用身份验证 """
    client.personVerify(image, idCardNumber, name);

    """ 如果有可选参数 """
    options = {}
    options["quality"] = "use"
    options["quality_conf"] = "{\"left_eye\": 0.6, \"right_eye\": 0.6}"
    options["faceliveness"] = "use"
    options["faceliveness_conf"] = "{\"faceliveness\": 0.834963}"
    options["ext_fields"] = "qualities"

    """ 带参数调用身份验证 """
    client.personVerify(image, idCardNumber, name, options)
    ```
## 在线活体检测
  - 人脸基础信息，人脸质量检测，基于图片的活体检测
  - **参数**
    - **image** 图像数据，base64编码，要求base64编码后大小不超过4M，最短边至少15px，最长边最大4096px,支持jpg/png/bmp格式
    - **max_face_num** 最多处理人脸数目，默认值1
    - **face_fields** 如不选择此项，返回结果默认只有人脸框、概率和旋转角度，可选参数为qualities、faceliveness
      - qualities：图片质量相关判断
      - faceliveness：活体判断
      - 如果两个参数都需要选择，请使用半角逗号分隔
  - **返回参数**
    - **faceliveness** 活体分数，face_fields 包括 faceliveness 时返回
    - **type** 真实人脸/卡通人脸置信度
  - **示例**
    ```python
    image = get_file_content('/home/leondgarse/workspace/face_recognition/examples/knn_examples/train/rose_leslie/img2.jpg')

    """ 调用在线活体检测 """
    client.faceverify(image)

    """ 如果有可选参数 """
    options = {}
    options["max_face_num"] = 2
    options["face_fields"] = "qualities,faceliveness"

    """ 带参数调用在线活体检测 """
    client.faceverify(image, options)
    ```
***

# Face Recognition
## Installation
  - [github 地址 face_recognition](https://github.com/ageitgey/face_recognition.git)
  ```shell
  pip install face_recognition
  ```
  https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
## Command-Line Interface
  - `face_recognition` - Recognize faces in a photograph or folder full for photographs.
    ```shell
    # A folder with person you already know. There should be one image file for each person with the files named according to who is in the picture
    # A second folder with the files you want to identify
    face_recognition train_set/ test/
    ```
    You can do that with the --tolerance parameter. The default tolerance value is 0.6 and lower numbers make face comparisons more strict
    ```shell
    face_recognition --tolerance 0.54 train_set/ test/
    ```
    If you want to see the face distance calculated for each match in order to adjust the tolerance setting, you can use --show-distance true
    ```shell
    face_recognition --show-distance true train_set/ test/
    ```
    If you are using Python 3.4 or newer, pass in a --cpus <number_of_cpu_cores_to_use> parameter
    ```shell
    # --cpus -1 to use all CPU cores in your system
    face_recognition --cpus 4 train_set/ test/
    ```
  - `face_detection` - Find faces in a photograph or folder full for photographs.
    ```shell
    face_detection  ./folder_with_pictures/
    ```
## get_frontal_face_detector in dlib
  ```python
  from dlib import get_frontal_face_detector
  import face_recognition as fr
  detector = get_frontal_face_detector()

  file_name = "two_people.jpg"
  image = fr.load_image_file(file_name)
  detector(image, 1)
  face_location = detector(image, 1)

  for ind, fl in enumerate(face_location):
      print("ind = %d, fl = %s" % (ind, fl))
      draw.rectangle(((fl.left(), fl.top()), (fl.right(), fl.bottom())), outline=(0, 0, 255))

  del draw
  pil_image.show()
  ```
## Find faces in pictures
  - Find all the faces that appear in a picture
  ```python
  import face_recognition as fr
  from PIL import Image, ImageDraw

  # file_name = "obama.jpg"
  file_name = "two_people.jpg"
  image = fr.load_image_file(file_name)
  # top, right, bottom, left
  face_location = fr.face_locations(image)

  pil_image = Image.open(file_name)
  draw = ImageDraw.Draw(pil_image)

  # left, top, right, bottom
  # draw.rectangle(face_location[0], outline=(0, 0, 255))
  for (top, right, bottom, left) in face_location:
      draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

  del draw
  pil_image.show()
  ```
  **只显示人脸部分**
  ```python
  file_name = "obama1.jpg"
  image = fr.load_image_file(file_name)
  face_location = fr.face_locations(image, model='hog')
  (top, right, bottom, left) = face_location[0]

  pil_image = Image.open(file_name)
  tt = pil_image.crop((left, top, right, bottom))
  tt.show()

  pil_image = Image.fromarray(image[top:bottom, left:right])
  pil_image.show()

  import cv2
  image = cv2.imread(file_name)
  pil_image = Image.fromarray(image[top:bottom, left:right, ::-1])
  pil_image.show()
  ```
## Find and manipulate facial features in pictures
  - Get the locations and outlines of each person’s eyes, nose, mouth and chin.
  ```python
  # file_name = "obama.jpg"
  file_name = "two_people.jpg"
  image = fr.load_image_file(file_name)
  face_landmarks_list = fr.face_landmarks(image)

  # Print the location of each facial feature in this image
  facial_features = [
      'chin',
      'left_eyebrow',
      'right_eyebrow',
      'nose_bridge',
      'nose_tip',
      'left_eye',
      'right_eye',
      'top_lip',
      'bottom_lip'
  ]

  pil_image = Image.fromarray(image)
  draw = ImageDraw.Draw(pil_image)

  for face_landmarks in face_landmarks_list:
      for facial_feature in facial_features:
          draw.line(face_landmarks[facial_feature], width=5)

  pil_image.show()
  ```
## Identify faces in pictures
  - Recognize who appears in each photo.
  ```python
  obama_image = fr.load_image_file("obama.jpg")
  biden_image = fr.load_image_file("biden.jpg")
  unknown_image = fr.load_image_file("obama1.jpg")

  biden_encording = fr.face_encodings(biden_image)[0]
  obama_encording = fr.face_encodings(obama_image)[0]
  unknown_encording = fr.face_encodings(unknown_image)[0]
  fr.compare_faces([obama_encording, biden_encording], unknown_encording, tolerance=0.4)
  # Out[137]: [True, False]

  fr.face_distance([obama_encording, biden_encording], unknown_encording)
  # Out[41]: array([0.34765424, 0.82590487])
  ```
  ```python
  chai_jinlong = fr.face_encodings(fr.load_image_file("chai_jinlong.png"))[0]
  dong_hangrui = fr.face_encodings(fr.load_image_file("dong_hangrui.png"))[0]
  du_huandeng = fr.face_encodings(fr.load_image_file("du_huandeng.png"))[0]
  nie_shuo = fr.face_encodings(fr.load_image_file("nie_shuo.png"))[0]
  sui_xiangrong = fr.face_encodings(fr.load_image_file("sui_xiangrong.png"))[0]
  zhang_ping = fr.face_encodings(fr.load_image_file("zhang_ping.png"))[0]
  zhao_yuliang = fr.face_encodings(fr.load_image_file("zhao_yuliang.png"))[0]
  zhou_yunfeng = fr.face_encodings(fr.load_image_file("zhou_yunfeng.png"))[0]

  unknown_encording = fr.face_encodings(fr.load_image_file("chai_jinlong.png"))[0]
  fr.compare_faces([chai_jinlong, dong_hangrui, du_huandeng, nie_shuo, sui_xiangrong, zhang_ping, zhao_yuliang, zhou_yunfeng], unknown_encording, tolerance=0.4)
  # Out[137]: [True, False]

  fr.face_distance([chai_jinlong, dong_hangrui, du_huandeng, nie_shuo, sui_xiangrong, zhang_ping, zhao_yuliang, zhou_yunfeng], unknown_encording)
  # Out[47]:
  # array([0.        , 0.57158912, 0.57613123, 0.55736657, 0.51305602, 0.51012086, 0.50903686, 0.51348426])
  ```
***

# 百度人脸识别 Aip 检测摄像头图像
  ```python
  import cv2
  import os
  from aip import AipFace
  from PIL import Image, ImageDraw
  import face_recognition as fr

  FRAME_PER_DETECT = 3
  FRAME_COMPRESS_RATE = 0.25
  FRAME_RESTORE_RATE = 1 / FRAME_COMPRESS_RATE
  APP_ID = "11108357"
  API_KEY = 'PrurvVg3MUluX61sviVlSGha'
  SECRET_KEY = 'VLKac9zzkcdkUcjSVZhwmPwEWLDLtQ92'
  client = AipFace(APP_ID, API_KEY, SECRET_KEY)

  groupId = "examples,hpe_faces"
  options = {}
  options["ext_fields"] = "faceliveness"
  options["detect_top_num"] = 3
  options["user_top_num"] = 2

  def get_file_content(filePath):
      with open(filePath, 'rb') as fp:
          return fp.read()

  def image_files_in_folder(folder):
      return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

  def display_face_locations(frame, face_locations, face_names):
      # Display face locations
      for (top, right, bottom, left), name in zip(face_locations, face_names):
          # Scale back up face locations since the frame we detected in was scaled to 1/4 size
          top *= FRAME_RESTORE_RATE
          right *= FRAME_RESTORE_RATE
          bottom *= FRAME_RESTORE_RATE
          left *= FRAME_RESTORE_RATE

          # Draw a box around the face
          cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

          # Draw a label with a name below the face
          cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX
          cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

  def parse_multiIdentify_result(result):
      # Found faces
      face_locations = []
      face_names = []
      group_names = []

      if result.get('result') != None:
          for face_result in result['result']:
              face_position = face_result['position']
              face_left = face_position['left']
              face_top = face_position['top']
              face_right = face_position['left'] + face_position['width']
              face_bottom = face_position['top'] + face_position['height']
              face_coord = [face_left, face_top, face_right, face_bottom]

              if face_coord in face_locations:
                  # Face already exist
                  print("face_coord = %s" % face_coord)
                  if face_result['scores'][0] < result['result'][face_locations.index(face_coord)]['scores'][0]:
                      # And this has a worse score, skip this
                      continue
              else:
                  # New face
                  face_locations.append(face_coord)

              face_name = "Unknown"
              group_name = "Unknown"
              if face_result['scores'][0] >= FACE_SCORE_TOLERANCE:
                  face_name = face_result['uid']
                  group_name = face_result['group_id']
              print("face_name = %s, score = %d" % (face_name, face_result['scores'][0]))
              face_names.append(face_name)
              group_names.append(group_name)

      return face_locations, face_names, group_names

  video_capture = cv2.VideoCapture(0)
  video_capture = cv2.VideoCapture("rtsp://admin:hpe.1234@192.168.136.205:554/MPEG-4/ch1/main/av_stream")
  ret, frame = video_capture.read()

  # Resize frame of video to 1/4 size for faster face recognition processing
  small_frame = cv2.resize(frame, (0, 0), fx=FRAME_COMPRESS_RATE, fy=FRAME_COMPRESS_RATE)
  # Convert the image from BGR color (which OpenCV uses) to RGB color (which aip uses)
  rgb_small_frame = small_frame[:, :, ::-1]
  Image.fromarray(rgb_small_frame).show()

  # model could be hog (faster) or cnn (more accurate)
  face_location = fr.face_locations(rgb_small_frame, model='cnn')

  if len(face_location) != 0:
      (top, right, bottom, left) = face_location[0]
      rgb_face_frame = rgb_small_frame[top:bottom, left:right]
  Image.fromarray(rgb_face_frame).show()

  result = client.multiIdentify(groupId, cv2.imencode('.jpg', rgb_face_frame)[1].tobytes(), options)
  print(result)
  face_locations, face_names, group_names = parse_multiIdentify_result(result)

  # Display the resulting image
  display_face_locations(frame, face_locations, face_names)

  pil_image = Image.fromarray(frame[:, :, ::-1])
  pil_image.show()

  cv2.imshow('Video', frame)

  video_capture.release()
  cv2.destroyAllWindows()
  ```
  **百度 Aip 输出结果**
  ```python
  In [29]: result
  Out[29]:
  {'result': [{'uid': 'two_people',
     'scores': [100],
     'group_id': 'examples',
     'user_info': '',video_capture = cv2.VideoCapture("rtsp://admin:hpe.1234@192.168.136.205:554/MPEG-4/ch1/main/av_stream")
     'position': {'left': 796.55651855469,
      'top': 91.270530700684,
      'width': 153,
      'height': 145,
      'degree': 12,
      'prob': 1}},
    {'uid': 'biden',
     'scores': [99.32218170166],
     'group_id': 'examples',
     'user_info': '',
     'position': {'left': 796.55651855469,
      'top': 91.270530700684,
      'width': 153,
      'height': 145,
      'degree': 12,
      'prob': 1}},
    {'uid': 'obama',
     'scores': [94.886238098145],
     'group_id': 'examples',
     'user_info': '',
     'position': {'left': 240.31817626953,
      'top': 66.275604248047,
      'width': 152,
      'height': 150,
      'degree': 6,
      'prob': 1}},
    {'uid': 'wang_guowei',
     'scores': [28.18116569519],
     'group_id': 'hpe_faces',
     'user_info': '',
     'position': {'left': 240.31817626953,
      'top': 66.275604248047,
      'width': 152,
      'height': 150,
      'degree': 6,
      'prob': 1}}],
   'result_num': 4,
   'ext_info': {'faceliveness': '0.024044899269938,4.0677969082026E-5'},
   'log_id': 4030489918041911}

  In [32]: result
  Out[32]:
  {'error_code': 216402,
   'error_msg': 'face not found',
   'log_id': 4041334861041911}

  {'error_msg': 'Open api qps request limit reached', 'error_code': 18}
  ```
***

# OpenCV 之 网络摄像头
## RTSP
  - RTSP (Real Time Streaming Protocol)，是一种语法和操作类似 HTTP 协议，专门用于音频和视频的应用层协议，和 HTTP 类似，RTSP 也使用 URL 地址
    ```python
    rtsp_addr = "rtsp://admin:a1234567@192.168.5.186:554/MPEG-4/ch1/main/av_stream"
    ```
  - 海康网络摄像头的 RTSP URL 格式如下
    ```python
    rtsp://[username]:[password]@[ip]:[port]/[codec]/[channel]/[subtype]/av_stream
    ```
    - **username** 用户名，常用 admin
    - **password** 密码，常用 12345
    - **ip** 摄像头IP，如 192.0.0.64
    - **port** 端口号，默认为 554
    - **codec** 视频编码模式，有 h264、MPEG-4、mpeg4 等
    - **channel** 通道号，起始为1，例如通道1，则为 ch1
    - **subtype** 码流类型，主码流为 main，辅码流为 sub
  - 大华网络摄像头的 RTSP URL 格式如下
    ```python
    rtsp://[username]:[password]@[ip]:[port]/cam/realmonitor?[channel=1]&[subtype=1]
    ```
    - **username、password、ip、port** 同上
    - **channel** 通道号，起始为1，例如通道2，则为 channel=2
    - **subtype** 码流类型，主码流为0（即 subtype=0），辅码流为1（即 subtype=1）
## VideoCapture 类
  - VideoCapture 类是 OpenCV 中用来操作视频流的类，可以在构造函数中打开视频，其参数支持以下三种类型
    - name of video file (eg. `video.avi`)
    - image sequence (eg. `img_%02d.jpg`, which will read samples like `img_00.jpg, img_01.jpg, img_02.jpg, ...`)
    - URL of video stream (eg. `protocol://host:port/script_name?script_params|auth`).
## VideoCapture 显示实时视频
  ```python
  import numpy as np
  import cv2

  cap = cv2.VideoCapture(0)

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('output.avi',fourcc, 20.0, (720,404))

  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret==True:
          frame = cv2.flip(frame,0)

          # write the flipped frame
          out.write(frame)

          cv2.imshow('frame',frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      else:
          break

  # Release everything if job is finished
  cap.release()
  out.release()
  cv2.destroyAllWindows()
  ```
***

# 活体检测 Facial Presentation Attack Detection
  - Facial Presentation Attack Detection
  - Anti-spoofing
  - Face Liveness
  - [NUAA Photograph Imposter Database](http://parnec.nuaa.edu.cn/xtan/data/nuaaimposterdb.html)
  - [The Replay-Attack Database](https://www.idiap.ch/dataset/replayattack)
  - [The Replay-Mobile Database](https://www.idiap.ch/dataset/replay-mobile)
  - [The MSU Mobile Face Spoofing Database (MFSD)](http://biometrics.cse.msu.edu/Publications/Databases/MSUMobileFaceSpoofing/index.htm#Download_instructions)
## Load NUAA data
  ```python
  import os
  import tensorflow.contrib.slim as slim
  from skimage.transform import resize
  import numpy as np
  import tensorflow as tf
  from skimage.io import imread
  from glob2 import glob

  def load_NUAA_data(data_path, data_scope='train', image_resize=0, limited_data_len=0):
    client_data_path = glob(os.path.join(data_path, "Client*[!.txt]"))[0]
    imposter_data_path = glob(os.path.join(data_path, "Imposter*[!.txt]"))[0]
    train_x_client_file = glob(os.path.join(data_path, "client_train*.txt"))[0]
    train_x_imposter_file = glob(os.path.join(data_path, "imposter_train*.txt"))[0]
    test_x_client_file = glob(os.path.join(data_path, "client_test*.txt"))[0]
    test_x_imposter_file = glob(os.path.join(data_path, "imposter_test*.txt"))[0]

    print('''
        client_data_path = %s,
        imposter_data_path = %s,
        train_x_client_file = %s,
        train_x_imposter_file = %s,
        test_x_client_file = %s,
        test_x_imposter_file = %s''' % (
        client_data_path, imposter_data_path,
        train_x_client_file, train_x_imposter_file,
        test_x_client_file, test_x_imposter_file))


    if data_scope == 'test':
        x_client_file = test_x_client_file
        x_imposter_file = test_x_imposter_file
    else:
        # Not specified as test, use train as default
        x_client_file = train_x_client_file
        x_imposter_file = train_x_imposter_file

    # Resize images if needed
    imread_from_line = lambda data_path, image_path: imread(os.path.join(data_path, image_path.strip().split(' ')[0].replace('\\', '/')))
    if image_resize != 0:
        imread_function = lambda data_path, image_path: resize(imread_from_line(data_path, image_path), (image_resize, image_resize), mode='reflect')
    else:
        imread_function = imread_from_line

    # Pick limits length of data if limited_data_len != 0, else output all
    truncated_line = lambda lines, out_len: np.array(lines)[np.random.permutation(len(lines))[:out_len]] if out_len != 0 else np.array(lines)

    x_client_lines = open(x_client_file).readlines()
    x_client_lines = truncated_line(x_client_lines, limited_data_len)
    x_client = [ imread_function(client_data_path, line) for line in x_client_lines ]

    x_imposter_lines = open(x_imposter_file).readlines()
    x_imposter_lines = truncated_line(x_imposter_lines, limited_data_len)
    x_imposter = [ imread_function(imposter_data_path, line) for line in x_imposter_lines ]

    x = np.stack(x_client + x_imposter)
    x = x.astype(np.float32)
    print('len(x_client) = %s, len(x_imposter) = %s, x.shape = %s' % (len(x_client), len(x_imposter), x.shape))
    # Out[50]: (3491, 64, 64)

    y = np.stack([[1, 0]] * len(x_client) + [[0, 1]] * len(x_imposter))
    y = y.astype(np.float32)
    print('y.shape = %s' % (y.shape, ))
    # Out[57]: (3491,)
    return x, y

  def accuracy_calc(prediction, y):
      tt = np.array(prediction)
      seperator_index = int(y[:, 0].sum())
      # Out[7]: 1743
      sum_on_client = np.sum(tt[:seperator_index] == 0)
      # Out[8]: 1266
      sum_on_imposter = np.sum(tt[seperator_index:] == 1)
      # Out[9]: 1557

      # accuracy
      accuracy = (sum_on_client + sum_on_imposter) / tt.shape[0]
      # Out[25]: 0.8060727585219135
      print('seperator_index = %d, sum_on_client = %f, sum_on_imposter = %f, accuracy = %f' % (seperator_index, sum_on_client, sum_on_imposter, accuracy))
  ```
  **data test**
  ```python  
  ''' Detectedface dataset '''
  data_path = '/home/leondgarse/workspace/facial_presentation_attack_detection/Detectedface/'
  client_data_path = os.path.join(data_path, 'ClientFace')
  train_x_client_file = os.path.join(data_path, 'client_train_face.txt')
  imread(os.path.join(client_data_path, '0001/0001_00_00_01_0.jpg')).shape
  # Out[13]: (203, 203, 3)
  imread(os.path.join(client_data_path, '0003/0003_01_00_01_168.jpg')).shape
  # Out[33]: (317, 317, 3)
  xx = [imread(os.path.join(client_data_path, line.strip().split(' ')[0].replace('\\', '/'))) for line in open(train_x_client_file).readlines()]
  tt = np.array([ii.shape for ii in xx])
  tt.shape
  # Out[83]: (1743, 3)

  np.unique(tt[:, 0])
  # Out[85]: array([131, 162, 203, 252, 317])

  {nn: np.sum(tt[:, 0] == nn) for nn in np.unique(tt[:, 0])}
  # Out[87]: {131: 11, 162: 330, 203: 598, 252: 601, 317: 203}

  train_x, train_y = load_NUAA_data(data_path, data_scope='train', image_resize=128)
  test_x, test_y = load_NUAA_data(data_path, data_scope='test', image_resize=128, limited_data_len=1500)

  ''' NormalizedFace dataset '''
  data_path = '/home/leondgarse/workspace/facial_presentation_attack_detection/NormalizedFace/'
  imread(os.path.join(data_path, 'ClientNormalized', '0001/0001_00_00_01_0.bmp')).shape
  # Out[3]: (64, 64)
  imread(os.path.join(data_path, 'ImposterNormalized', '0001/0001_00_00_01_0.bmp')).shape
  # Out[38]: (64, 64)

  train_x, train_y = load_NUAA_data(data_path, data_scope='train')
  test_x, test_y = load_NUAA_data(data_path, data_scope='test')
  ```
## Alexnet v1
  ```python
  ''' alexnet v1, slim and fully_connected layers for output '''
  def alexnet_inference_v1(input, num_classes, dropout_keep_prob=0.5):
      # Layer 1: Traditional: 64, 11, 4, another treatise one: 96, 11, 4
      # Change padding='SAME', if input.shape == [?, 64, 64, 1]
      conv1 = slim.conv2d(input, num_outputs=96, kernel_size=11, stride=4, padding='VALID')
      lrn1 = slim.nn.lrn(conv1, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
      pool1 = slim.max_pool2d(lrn1, kernel_size=3, stride=2)
      # Layer 2: Traditional: 192, 5, 1, another treatise one: 256, 5, 1
      conv2 = slim.conv2d(pool1, num_outputs=256, kernel_size=5, stride=1)
      lrn2 = slim.nn.lrn(conv2, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
      pool2 = slim.max_pool2d(lrn2, kernel_size=3, stride=2)
      # Layer 3: Traditional: 384, 3, 1, another treatise one: 384, 3, 1
      conv3 = slim.conv2d(pool2, num_outputs=384, kernel_size=3, stride=1)
      # Layer 4: Traditional: 256, 3, 1, another treatise one: 384, 3, 1
      conv4 = slim.conv2d(conv3, num_outputs=384, kernel_size=3, stride=1)
      # Layer 5: Traditional: 256, 3, 1, another treatise one: 256, 3, 1
      conv5 = slim.conv2d(conv4, num_outputs=256, kernel_size=3, stride=1)
      pool5 = slim.max_pool2d(conv5, kernel_size=3, stride=2)

      # Use fully_connected layers for output
      with slim.arg_scope([slim.fully_connected],
                      biases_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005)):
          flatten = slim.flatten(pool5)
          fc1 = slim.fully_connected(flatten, num_outputs=4096, scope='fc1')
          dropout1 = slim.dropout(fc1, dropout_keep_prob, scope='dropout1')
          fc2 = slim.fully_connected(dropout1, num_outputs=4096, scope='fc2')
          dropout2 = slim.dropout(fc2, dropout_keep_prob, scope='dropout2')
          fc3 = slim.fully_connected(dropout2, num_outputs=num_classes, activation_fn=None, scope='fc3')
          # softmax = slim.softmax(fc3)

      return fc3, flatten
  ```
## Alexnet v2
  ```python
  ''' alexnet v2, use conv2d instead of fully_connected layers'''
  def alexnet_inference_v2(input, num_classes, dropout_keep_prob=0.5):
      # Layer 1: Traditional: 64, 11, 4, another treatise one: 96, 11, 4
      # Change padding='SAME', if input.shape == [?, 64, 64, 1]
      conv1 = slim.conv2d(input, num_outputs=96, kernel_size=11, stride=4, padding='VALID')
      lrn1 = slim.nn.lrn(conv1, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
      pool1 = slim.max_pool2d(lrn1, kernel_size=3, stride=2)
      # Layer 2: Traditional: 192, 5, 1, another treatise one: 256, 5, 1
      conv2 = slim.conv2d(pool1, num_outputs=256, kernel_size=5, stride=1)
      lrn2 = slim.nn.lrn(conv2, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
      pool2 = slim.max_pool2d(lrn2, kernel_size=3, stride=2)
      # Layer 3: Traditional: 384, 3, 1, another treatise one: 384, 3, 1
      conv3 = slim.conv2d(pool2, num_outputs=384, kernel_size=3, stride=1)
      # Layer 4: Traditional: 256, 3, 1, another treatise one: 384, 3, 1
      conv4 = slim.conv2d(conv3, num_outputs=384, kernel_size=3, stride=1)
      # Layer 5: Traditional: 256, 3, 1, another treatise one: 256, 3, 1
      conv5 = slim.conv2d(conv4, num_outputs=256, kernel_size=3, stride=1)
      pool5 = slim.max_pool2d(conv5, kernel_size=3, stride=2)

      # Use conv2d instead of fully_connected layers.
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
          net_kernel_size = pool5.shape[1]
          net = slim.conv2d(pool5, 4096, [net_kernel_size, net_kernel_size], padding='VALID',
                            scope='fc6')
          net = slim.dropout(net, dropout_keep_prob, scope='dropout6')
          net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
          net = slim.dropout(net, dropout_keep_prob, scope='dropout7')
          net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            biases_initializer=tf.zeros_initializer(),
                            scope='fc8')

          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          return net, slim.flatten(pool5)
  ```
## alexnet nodel
  ```python
  def alexnet_model(features, labels, mode):
      xs = features['x']
      # xs.shape.ndims is same as len(xs.shape)
      if xs.shape.ndims == 3: xs = tf.expand_dims(xs, axis=-1)

      outputs = alexnet_inference(xs, num_classes=NUM_CLASSES)
      prediction = tf.argmax(outputs, axis=-1)
      if mode == tf.estimator.ModeKeys.PREDICT:
          predictions = {
              'alexnet_outputs': outputs,
              'predictions': prediction,
          }
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      # if labels.shape.ndims == 1: labels = tf.expand_dims(labels, axis=-1)
      # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels, axis=-1), logits=outputs))
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=outputs))
      # loss = tf.losses.mean_squared_error(tf.argmax(labels, axis=-1), prediction)
      global_step = tf.train.get_global_step()
      learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE)
      # train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
      train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
      # learning_rate_decay_fn = lambda learning_rate, global_step: tf.train.exponential_decay(learning_rate, global_step, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE)
      # train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(),
      #               optimizer='SGD',
      #               learning_rate=LEARNING_RATE_BASE)
      #               # learning_rate_decay_fn=None)

      return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction, loss=loss, train_op=train_op)
  ```
## Estimator Train test
  ```python
  ''' Detectedface dataset '''
  data_path = '/home/leondgarse/workspace/facial_presentation_attack_detection/Detectedface/'
  train_x, train_y = load_NUAA_data(data_path, data_scope='train', image_resize=128)
  test_x, test_y = load_NUAA_data(data_path, data_scope='test', image_resize=128, limited_data_len=1500)

  ''' NormalizedFace dataset '''
  # data_path = '/home/leondgarse/workspace/facial_presentation_attack_detection/NormalizedFace/'
  # train_x, train_y = load_NUAA_data(data_path, data_scope='train')
  # test_x, test_y = load_NUAA_data(data_path, data_scope='test')

  NUM_CLASSES = 2
  BATCH_SIZE = 256
  DECAY_STEPS = train_x.shape[0] / BATCH_SIZE
  DECAY_RATE = 0.99
  LEARNING_RATE_BASE = 0.001

  alexnet_inference = alexnet_inference_v1

  train_input_fn = tf.estimator.inputs.numpy_input_fn({'x': train_x}, train_y, batch_size=BATCH_SIZE, num_epochs=100, shuffle=True)

  estimator_alexnet = tf.estimator.Estimator(model_fn=alexnet_model)
  estimator_alexnet.train(input_fn=train_input_fn)

  eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x': train_x}, train_y, batch_size=BATCH_SIZE, num_epochs=None, shuffle=True)
  estimator_alexnet.evaluate(eval_input_fn)

  predict_input_fn = tf.estimator.inputs.numpy_input_fn({"x": train_x}, num_epochs=1, shuffle=False)
  pp = estimator_alexnet.predict(input_fn=predict_input_fn)
  tt = [ii['predictions'] for ii in pp]
  accuracy_calc(tt, train_y)

  predict_input_fn = tf.estimator.inputs.numpy_input_fn({"x": test_x}, num_epochs=1, shuffle=False)
  pp = estimator_alexnet.predict(input_fn=predict_input_fn)
  tt = [ii['predictions'] for ii in pp]
  accuracy_calc(tt, test_y)
  ```
## Sample test
  ```python
  ''' Sample test '''
  sample_data = lambda x, y, start, end: (np.vstack([x[start:end], x[-end-1:-start-1]]), np.vstack([y[start:end], y[-end-1:-start-1]]))
  xs, ys = sample_data(train_x, train_y, 0, 100)
  BATCH_SIZE = 20
  DECAY_STEPS = xs.shape[0] / BATCH_SIZE
  ts_input_fn = tf.estimator.inputs.numpy_input_fn({'x': xs}, ys, batch_size=BATCH_SIZE, num_epochs=None, shuffle=True)
  estimator_alexnet = tf.estimator.Estimator(model_fn=alexnet_model)
  estimator_alexnet.train(input_fn=ts_input_fn, steps=300)

  xst, yst = sample_data(train_x, train_y, 500, 600)
  ps_input_fn = tf.estimator.inputs.numpy_input_fn({"x": xs}, num_epochs=1, shuffle=False)
  ps_input_fn = tf.estimator.inputs.numpy_input_fn({"x": xst}, num_epochs=1, shuffle=False)
  pp = estimator_alexnet.predict(input_fn=ps_input_fn)
  tt = [ii['predictions'] for ii in pp]
  accuracy_calc(tt, yst)
  accuracy_calc(tt, ys)
  ```
## Additional SVM classifier
  ```python
  ''' Additional SVM classifier '''
  from sklearn.svm import SVC
  svm_c = [2 ** i for i in range(-8, 8, 1)]
  svm_gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
  svm_kernel = ['rbf', 'linear', 'poly', 'sigmoid']
  param_grid = {'C': svm_c, 'gamma': svm_gamma, 'kernel': svm_kernel}
  from sklearn.model_selection import GridSearchCV
  clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
  clf.fit(emb, np.argmax(train_y, axis=-1))

  test_x, test_y = load_NUAA_data(data_path, data_scope='test', image_resize=128, limited_data_len=1500)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn({"x": test_x}, num_epochs=1, shuffle=False)
  tt = list(estimator_alexnet.predict(input_fn=predict_input_fn))
  emb_unknown = np.array([np.reshape(ii['emb_features'], (-1)) for ii in tt])
  emb_unknown.shape
  pp = clf.predict(emb_unknown)
  accuracy_calc(pp, test_y)
  ```
## Result
  ```python
  ''' Train '''
  INFO:tensorflow:Using default config.
  WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpox1lsmtz
  INFO:tensorflow:Using config: {'_model_dir': '/tmp/tmpox1lsmtz', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7ff94bdc4208>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
  INFO:tensorflow:Calling model_fn.
  >>>> outputs.shape = (?, 2)
  INFO:tensorflow:Done calling model_fn.
  INFO:tensorflow:Create CheckpointSaverHook.
  INFO:tensorflow:Graph was finalized.
  INFO:tensorflow:Running local_init_op.
  INFO:tensorflow:Done running local_init_op.
  INFO:tensorflow:Saving checkpoints for 1 into /tmp/tmpox1lsmtz/model.ckpt.
  INFO:tensorflow:loss = 0.26773486, step = 1
  INFO:tensorflow:global_step/sec: 0.175117
  INFO:tensorflow:loss = 0.288006, step = 101 (571.049 sec)
  INFO:tensorflow:Saving checkpoints for 107 into /tmp/tmpox1lsmtz/model.ckpt.
  INFO:tensorflow:global_step/sec: 0.176473
  INFO:tensorflow:loss = 0.042064987, step = 201 (566.659 sec)
  INFO:tensorflow:Saving checkpoints for 213 into /tmp/tmpox1lsmtz/model.ckpt.
  INFO:tensorflow:global_step/sec: 0.17806
  INFO:tensorflow:loss = 0.14192937, step = 301 (561.607 sec)
  INFO:tensorflow:Saving checkpoints for 320 into /tmp/tmpox1lsmtz/model.ckpt.
  ...
  INFO:tensorflow:global_step/sec: 0.176783
  INFO:tensorflow:loss = 0.15262769, step = 4801 (565.666 sec)
  INFO:tensorflow:Saving checkpoints for 4815 into /tmp/tmpox1lsmtz/model.ckpt.
  INFO:tensorflow:global_step/sec: 0.177146
  INFO:tensorflow:loss = 0.26890928, step = 4901 (564.507 sec)
  INFO:tensorflow:Saving checkpoints for 4922 into /tmp/tmpox1lsmtz/model.ckpt.
  INFO:tensorflow:Saving checkpoints for 5000 into /tmp/tmpox1lsmtz/model.ckpt.
  INFO:tensorflow:Loss for final step: 0.05957412.
  ''' Evaluate '''

  ```

# Foo
  ```python
  >>> from skimage import data
  >>> from skimage.transform import rotate
  >>> image = data.camera()
  >>> rotate(image, 2).shape
  (512, 512)
  >>> rotate(image, 2, resize=True).shape
  (530, 530)
  >>> rotate(image, 90, resize=True).shape
  (512, 512)
  ```  
