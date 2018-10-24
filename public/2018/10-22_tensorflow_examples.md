# ___2018 - 10 - 22 Tensorflow Examples___
***

## How to Retrain an Image Classifier for New Categories
  - [How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)
  - https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/
  ```py
  import tensorflow_hub as hub

  hub_module = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
  module = hub.Module(hub_module)
  ```
  **测试**
  ```py
  height, width = hub.get_expected_image_size(module)

  image_file = './datasets/flower_photos/daisy/100080576_f52e8ee070_n.jpg'
  images = tf.gfile.FastGFile(image_file, 'rb').read()
  images = tf.image.decode_jpeg(images)

  sess = tf.InteractiveSession()
  images.eval().shape

  imm = tf.image.resize_images(images, (height, width))
  imm = tf.expand_dims(imm, 0)  # A batch of images with shape [batch_size, height, width, 3].
  plt.imshow(imm[0].eval().astype('int'))

  tf.global_variables_initializer().run()
  features = module(imm).eval()  # Features with shape [batch_size, num_features].
  print(features.shape)
  # (1, 2048)
  ```
  ```py
  def jpeg_decoder_layer(module_spec):
      height, width = hub.get_expected_image_size(module_spec)
      input_depth = hub.get_num_image_channels(module_spec)
      jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
      imm = tf.image.decode_jpeg(jpeg_data, channels=input_depth)

      imm = tf.image.convert_image_dtype(imm, dtype=tf.float32)
      imm = tf.expand_dims(imm, 0)
      imm = tf.image.resize_images(images, (height, width))

      return jpeg_data, imm
  ```
  **测试**
  ```py
  jj, ii = jpeg_decoder_layer(module)
  tt = sess.run(ii, {jj: tf.gfile.FastGFile(image_file, 'rb').read()})
  print(tt.shape)
  # (299, 299, 3)
  ```
  ```py
  CLASS_COUNT = 5
  def add_classifier_op(class_count, bottleneck_module, is_training, learning_rate=0.01):
      height, width = hub.get_expected_image_size(bottleneck_module)
      resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
      bottleneck_tensor = bottleneck_module(resized_input_tensor)
      batch_size, bottleneck_out = bottleneck_tensor.get_shape().as_list()  # None, 2048

      # Add a fully connected layer and a softmax layer
      with tf.name_scope('input'):
          bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[batch_size, bottleneck_out], name='BottleneckInputPlaceholder')
          target_label = tf.placeholder(tf.int64, [batch_size], name='GroundTruthInput')

      with tf.name_scope('final_retrain_ops'):
          with tf.name_scope('weights'):
              init_value = tf.truncated_normal([bottleneck_out, class_count], stddev=0.001)
              weights = tf.Variable(init_value, name='final_weights')
          with tf.name_scope('biases'):
              biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
          with tf.name_scope('dense'):
              logits = tf.matmul(bottleneck_input, weights) + biases

      final_tensor = tf.nn.softmax(logits, name='final_result')

      # The tf.contrib.quantize functions rewrite the graph in place for
      # quantization. The imported model graph has already been rewritten, so upon
      # calling these rewrites, only the newly added final layer will be
      # transformed.
      if is_training:
          tf.contrib.quantize.create_training_graph()
      else:
          tf.contrib.quantize.create_eval_graph()

      # If this is an eval graph, we don't need to add loss ops or an optimizer.
      if not is_training:
          return None, None, bottleneck_input, target_label, final_tensor

      with tf.name_scope('cross_entropy'):
          cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=target_label, logits=logits)

      with tf.name_scope('train'):
          optimizer = tf.train.GradientDescentOptimizer(learning_rate)
          train_step = optimizer.minimize(cross_entropy_mean)

      return (train_step, cross_entropy_mean, bottleneck_input, target_label, final_tensor)
  ```
  ```py
  flower_url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
  train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(flower_url), origin=flower_url)

  def load_image_train_test(data_path, test_rate=10):
      rr = {}
      for sub_dir_name in tf.gfile.ListDirectory(data_path):
          sub_dir = os.path.join(data_path, sub_dir_name)
          print(sub_dir)
          if not tf.gfile.IsDirectory(sub_dir):
              continue

          item_num = len(tf.gfile.ListDirectory(sub_dir))

          train_dd = []
          test_dd = []
          for item_name in tf.gfile.ListDirectory(sub_dir):
              hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(item_name)).hexdigest()
              percentage_hash = int(hash_name_hashed, 16) % (item_num + 1) * (100 / item_num)
              if percentage_hash < 10:
                  test_dd.append(os.path.join(sub_dir, item_name))
              else:
                  train_dd.append(os.path.join(sub_dir, item_name))
          rr[sub_dir_name] = {'train': train_dd, 'test': test_dd}

      return rr
  ```
## Advanced Convolutional Neural Networks
  - [Advanced Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/deep_cnn)
  - [tensorflow/models/tutorials/image/cifar10/](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/)
  - [tensorflow/models/tutorials/image/cifar10_estimator/](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)
***

# Sequences
## Recurrent Neural Networks
  - [Recurrent Neural Networks](https://www.tensorflow.org/tutorials/sequences/recurrent)
  - [tensorflow/models/tutorials/rnn/ptb/](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)
## Recurrent Neural Networks for Drawing Classification
  - [Recurrent Neural Networks for Drawing Classification](https://www.tensorflow.org/tutorials/sequences/recurrent_quickdraw)
  - [tensorflow/models/tutorials/rnn/quickdraw/](https://github.com/tensorflow/models/tree/master/tutorials/rnn/quickdraw)
## Simple Audio Recognition
  - [Simple Audio Recognition](https://www.tensorflow.org/tutorials/sequences/audio_recognition)
  - [tensorflow/tensorflow/examples/speech_commands/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands)
## Neural Machine Translation seq2seq Tutorial
  - [tensorflow/nmt](https://github.com/tensorflow/nmt)
***

# 数据表示 data representation
## Vector Representations of Words
## Improving Linear Models Using Explicit Kernel Methods
## Large-scale Linear Models with TensorFlow
***

# Non ML
## Mandelbrot set
## Partial differential equations
***

# GOO
  - [TensorFlow Hub](https://www.tensorflow.org/hub/)
  - [基于字符的LSTM+CRF中文实体抽取](https://github.com/jakeywu/chinese_ner)
  - [Matplotlib tutorial](http://www.labri.fr/perso/nrougier/teaching/matplotlib/)
  - [TensorFlow 实战电影个性化推荐](https://blog.csdn.net/chengcheng1394/article/details/78820529)
  - [TensorRec](https://github.com/jfkirk/tensorrec)
  - [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)

  ![](images/opt1.gif)
***

# translate
  ```py
  ! pip install translate

  from translate import Translator
  translator = Translator(to_lang='en', from_lang='zh')
  print(translator.translate('你好')) # Hello
  print(translator.translate('今天天气很不错啊')) # It's a nice day today.
  ```
***

# Chatbot
  - [百度中文词法分析 LAC](https://github.com/baidu/lac)
  - [百度问答系统框架 AnyQ](https://github.com/baidu/AnyQ)
  - [结巴中文分词](https://github.com/fxsjy/jieba)
```py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import jieba

def classification(predict_data="PPMessage"):
    target = []
    data1 = ["您好", "你多大了", "你是什么呀", "你怎么了", "你男的女的呀"]
    target.append("我是PPMessage智能客服,我的名字叫PP,我很面,就是一个小PP,所以叫PP,您有什么的问题需要问我?")
    data2 = ["我想买", "怎么部署PPMESSAGE", "怎么下载PPMessage", "ppmessage是什么意思"]
    target.append("这个问题涉及到我们的核心利益 ^_^,转人工客服吧?")

    X = []
    Y = []

    for data in data1:
        X.append(" ".join(jieba.lcut(data)))
        Y.append(0)

    for data in data2:
        X.append(" ".join(jieba.lcut(data)))
        Y.append(1)

    v = TfidfVectorizer()
    X = v.fit_transform(X)

    clf = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    clf.fit(X, Y)

    x = " ".join(jieba.lcut(predict_data))
    x = v.transform([x])
    y = clf.predict(x)

    print(target[y[0]])
    return

if __name__ == "__main__":
    classification()
```
## AI Chat Bot in Python with AIML
  - [AI Chat Bot in Python with AIML](https://www.devdungeon.com/content/ai-chat-bot-python-aiml)
  - [Github python-aiml](https://github.com/paulovn/python-aiml)
  - [Chatty Cathy](https://www.devdungeon.com/content/chatty-cathy)
  - **AIML** 是 Richard Wallace 开发的，最初是由 A.L.I.C.E. (Artificial Linguistics Internet Computer Entity) 发展而来，AIML 使用 XML 定义匹配规则与回复内容
  - **创建标准启动文件 std-startup.xml** 加载 AIML 的主入口文件
    ```xml
    <!-- 匹配一个模式并定义一个动作的基本文件，匹配模式 load aim b，然后启动 AIML，并加载对话文件 basic_chat.aiml -->

    <aiml version="1.0.1" encoding="UTF-8">
        <!-- std-startup.xml -->

        <!-- Category is an atomic AIML unit -->
        <category>

            <!-- Pattern to match in user input -->
            <!-- If user enters "LOAD AIML B" -->
            <pattern>LOAD AIML B</pattern>

            <!-- Template is the response to the pattern -->
            <!-- This learn an aiml file -->
            <template>
                <learn>basic_chat.aiml</learn>
                <!-- You can add more aiml files here -->
                <!--<learn>more_aiml.aiml</learn>-->
            </template>

        </category>

    </aiml>
    ```
  - **创建 AIML 对话文件 basic_chat.aiml**
    ```xml
    <!-- 匹配模式与回复内容 -->
    <aiml version="1.0.1" encoding="UTF-8">
    <!-- basic_chat.aiml -->

        <category>
            <pattern>HELLO</pattern>
            <template>
                Well, hello!
            </template>
        </category>

        <category>
            <pattern>WHAT ARE YOU</pattern>
            <template>
                I'm a bot, silly!
            </template>
        </category>

    </aiml>
    ```
  - **随机回复** 匹配规则可以使用 `*` 等占位符，随机回复使用 `<random>` 标签
    ```xml
    <!-- 匹配 One time I 开头的一句话 -->
    <category>
        <pattern>ONE TIME I *</pattern>
        <template>
            <random>
                <li>Go on.</li>
                <li>How old are you?</li>
                <li>Be more specific.</li>
                <li>I did not know that.</li>
                <li>Are you telling the truth?</li>
                <li>I don't know what that means.</li>
                <li>Try to tell me that another way.</li>
                <li>Are you talking about an animal, vegetable or mineral?</li>
                <li>What is it?</li>
            </random>
        </template>
    </category>
    ```
  - **使用已有的 AIML 文件** 使用自定义的 AIML 对话文件大约需要 10,000 条规则才能使对话比较真实，[Alice Bot website](www.alicebot.org/aiml/aaa/) 提供了一些定义好的对话文件，如 `std-65-percent.xml` 包含了 65% 的常用场景
  - **Python AIML** 通过创建 AIML 对象，并加载定义的 `启动文件 std-startup.xml` 与 `对话文件 basic_chat.aiml`，训练后可以用于对话
    ```sh
    # python2
    pip install aiml
    # python3
    pip install python-aiml
    ```
    ```py
    We create the startup file as a separate entity so that we can add more aiml files to the bot later without having to modify any of the programs source code. We can just add more files to learn in the startup xml file.

    import aiml

    # Create the kernel and learn AIML files
    kernel = aiml.Kernel()
    kernel.learn("std-startup.xml")
    kernel.respond("load aiml b")

    # Press CTRL-C to break this loop
    while True:
        print(kernel.respond(input("Enter your message >> ")))
    ```
  - **通过缓存加速加载过程** 当有很多个 AIML 文件时，训练过程会需要大量时间，可以通过将训练后的结果缓存到文件，加速启动过程
    ```py
    import aiml
    import os

    kernel = aiml.Kernel()

    if os.path.isfile("bot_brain.brn"):
        kernel.bootstrap(brainFile = "bot_brain.brn")
    else:
        kernel.bootstrap(learnFiles = "std-startup.xml", commands = "load aiml b")
        kernel.saveBrain("bot_brain.brn")

    # kernel now ready for use
    while True:
        print(kernel.respond(input("Enter your message >> ")))
    ```
  - **运行中重载 AIML** 通过 `load` 命令重新加载，如果需要保存中间数据，可以自定义命令处理
    ```py
    load aiml b
    ```
    ```py
    while True:
        message = input("Enter your message to the bot: ")
        print('message = %s' % message)
        if message == "quit":
            break
        elif message == "save":
            kernel.saveBrain("bot_brain.brn")
        else:
            bot_response = kernel.respond(message)
            # Do something with bot_response
            print(bot_response)
    ```
  - **会话与判定 Sessions and Predicates** AIML 可以通过会话 Session ID 管理不同用户的对话，通过 `respond` 的参数指定 sessionId
    ```py
    sessionId = 12345
    kernel.respond(input(">>>"), sessionId)
    ```
    Session ID 需要自己创建与管理，并保证每个用户的唯一性
    ```py
    sessionId = 12345

    # Get session info as dictionary. Contains the input
    # and output history as well as any predicates known
    sessionData = kernel.getSessionData(sessionId)

    # Each session ID needs to be a unique value
    # The predicate name is the name of something/someone
    # that the bot knows about in your session with the bot
    # The bot might know you as "Billy" and that your "dog" is named "Brandy"
    kernel.setPredicate("dog", "Brandy", sessionId)
    clients_dogs_name = kernel.getPredicate("dog", sessionId)

    kernel.setBotPredicate("hometown", "127.0.0.1")
    bot_hometown = kernel.getBotPredicate("hometown")
    ```
    在 AIML 会话文件中可以添加保存 **判定词 predicates** 的模板
    ```xml
    <aiml version="1.0.1" encoding="UTF-8">
        <category>
            <pattern>MY DOGS NAME IS *</pattern>
            <template>
                That is interesting that you have a dog named <set name="dog"><star/></set>
            </template>  
        </category>  
        <category>
            <pattern>WHAT IS MY DOGS NAME</pattern>
            <template>
                Your dog's name is <get name="dog"/>.
            </template>  
        </category>  
    </aiml>
    ```
    模拟对话
    ```py
    A: My dogs name is Max
    Bot: That is interesting that you have a dog named Max

    A: What is my dogs name?
    Bot: Your dog's name is Max.
    ```
  - [Additional References](www.alicebot.org/documentation/aiml-reference.html)
## AIML 标签元素
  - **Categories** 一条匹配规则的基本基本单元，至少包含一条 `pattern` 与 `template`
    ```xml
    <category>
        <pattern>WHAT IS YOUR NAME</pattern>
        <template>My name is Michael N.S Evanious.</template>
    </category>
    ```
  - **Patterns** 字符串表示的匹配规则，忽略大小写，可以使用 `*` 匹配多个字符
    ```py
    WHAT IS YOUR NAME
    WHAT IS YOUR *
    ```
  - **Templates** 对于匹配规则的回复内容，可以是字符串 / 变量 / 条件表达式 if-then-else / 随机回复
    ```py
    My name is John.
    My name is <bot name="name"/>.
    You told me you are <get name="user-age"/> years old.
    ```
    **srai** Symbolic Reduction in Artificial Intelligence 可以重定向到其他 pattern
    ```xml
    <category>
        <pattern>WHAT IS YOUR NAME</pattern>
        <template><![CDATA[My name is <bot name="name"/>.]]></template>
    </category>
    <category>
        <pattern>WHAT ARE YOU CALLED</pattern>
        <template>
            <srai>what is your name</srai>
        </template>
    </category>
    ```
## AIML 处理中文
  - 所有原生的 AIML 解释器都是面向英文设计的。它需要有空格来断字，并且在运行中会自动大写所有字母，并在回复中大写首字母
  - 如果要让机器人理解对话输入，需要在编写 AIML 时在汉字之间加上空格，并在运行时同样先给汉字插上空格再交付 AIML 解释器解析
  - PyAIML 的源码组织
    - AIML（XML）读取部分：AimlParser.py。使用 xml.sax 接口解析 XML文件读入对话逻辑和机器人回复信息
    - 人机对话输入输出接口：Kernel.py。提供运行期的 IO 接口，PyAIML 中最主要的 IO 接口是 respond 函数
    - 语言模式匹配部分：PatternMgr.py。利用 AIML 模式来匹配符合的输入字串。主要实现在 match 函数中
    - 其他支持模块
  - 中文空格断字 中文化 PyAIML 我们主要需要实现的是自动的中文空格断字 接下来就是把语言支持加入到 PyAIML 中。这并不难。这里主要需要添加中文切字的是 <pattern> 和 <sari> 标签的处理。同时也要为 <topic> 和 <that> 标签提供中文支持
    ```py
    """ LangSupport.py
        提供对中文空格断字的支持。
        支持 GB 及 Unicode 。
        LangSupport 对象的 input 函数能在中文字之间添加空格。
        LangSupport 对象的 output 函数则是去除中文字之间的空格。
    """
    from re import compile as re_compile
    from string import join as str_join

    findall_gb   = re_compile('[\x81-\xff][\x00-\xff]|[^\x81-\xff]+').findall
    findall_utf8 = re_compile('[\u2e80-\uffff]|[^\u2e80-\uffff]+').findall
    sub_gb       = re_compile('([\x81-\xff][\x00-\xff]) +(?=[\x81-\xff][\x00-\xff])').sub
    sub_utf8     = re_compile('([\u2e80-\uffff]) +(?=[\u2e80-\uffff])').sub
    sub_space    = re_compile(' +').sub

    LangSupport = type('LangSupport', (object, ),
            {'__init__': lambda self, encoding = 'ISO8859-1': self.__setattr__('_encoding', encoding),
             '__call__': lambda self, s: self.input(s),
             'input'   : lambda self, s: s,
             'output'  : lambda self, s: s } )

    GBSupport = type('GBSupport', (LangSupport, ),
            {'input' : lambda self, s:
                    str_join( findall_gb( type(s) == str and unicode(s, self._encoding) or s ) ),
             'output': lambda self, s:
                    sub_space(' ', sub_gb(r'\1', ( type(s) == str and unicode(s, 'UTF-8') or s ).encode(self._encoding) ) ) } )

    UnicodeSupport = type('UnicodeSupport', (LangSupport, ),
            {'input' : lambda self, s:
                    str_join( findall_utf8( type(s) == str and unicode(s, self._encoding) or s ) ),
             'output': lambda self, s:
                    sub_space(' ', sub_utf8(r'\1', ( type(s) == str and unicode(s, 'UTF-8') or s ).encode(self._encoding) ) ) } )
    ```
  - **判断中文字符**
    ```py
    def is_ascii_string(ss):
        for aa in ss:
            if aa >= 128:
                return False
        return True
    ```
```

AIML介绍（中文版）  

2010-01-12 23:47:45|  分类： 互联网伙伴机器人|字号 订阅

这个是google翻译的，虽然翻译的太直译了，大家就凑或看吧，不过英语好的还是建议看英文版的哦o(∩_∩)o...

美国博士理查德华莱士

AIML，或人工智能标记语言使人们能够进入聊天就雅丽自由软件技术为基础的机器人投入的知识。

AIML是由自由软件的Alicebot的社会，我1995-2000年期间。它最初是改编自非XML语法也称为AIML，形成了第一Alicebot，爱丽丝，人工语言因特网计算机实体的基础。

AIML，描述了数据对象的一类名为AIML对象和部分介绍了计算机程序的行为过程。 AIML对象，而在单位，叫作 主题 和 类别，其中包含任何分析或未解析数据。

分析数据是由字符，其中一些形态特征数据，其中一些形式AIML元素。 AIML元素封装刺激反应知识的文件中。在这些因素有时字符数据分析由AIML翻译，有时离开后经处理未解析的回应。

职类

在AIML知识的基本单位称为一个类别。每一类由一个输入的问题，一个输出答案，和一个可选的范围内。的问题，或刺激，被称为模式。 Theanswer，或反应，称为模板。可选的两种情况下被称为“说”和“主题。”在AIML模式语言很简单，包括文字，只有空间和通配符_和*.一语可以由字母和数字，但没有其他字符。该模式语言是区分不变。词是由一个分开的空间，而这样的话通配符功能。

在AIML第一个版本只允许每模式外卡的性质。该AIML 1.01标准允许在每个模式的多个通配符，但语言的目的是尽可能为手头的任务很简单，简单，甚至比一般表达式。该模板是AIML respons或答复。最简单的形式，模板只有平原，没有标记的文本组成。

更一般地说，AIML标记转变为一个小型计算机程序，可以保存数据的答复，启动其他程序，使有条件的反应，并递归调用模式匹配插入来自其他类别的答复。事实上，大多数AIML标签属于这个模板一边子语言。

AIML目前支持两种方式和其他语言的界面系统。在<system>的标签执行任何程序作为操作系统的shell命令访问，并插入在答复结果。同样，<javascript>标签允许模板内的任意脚本。该类别的可选范围内的部分由两个变种，称为<that>和<topic>。标签的<that>内出现的类别，其方式必须符合机器人的最后话语。记住最后一个话语是重要的，如果机器人问一个问题。标记显示的<topic>类别外，并收集一组类别在一起。该专题里面可以设置任何模板。

AIML是不完全一样的问题和答案简单的数据库。模式匹配“查询”语言是比简单的像SQL的东西。但是，一类模板可能包含递归<srai>标签，使输出不仅取决于匹配的类别之一，而且任何其他递归达成通过<srai>。

递推

AIML实现经营者与<srai>递归。对存在的任何协议的缩写，意。在“临时代办”人工智能的立场，但“S.R.”可能意味着“刺激反应”，“语法改写”，“象征性的减少”，“简单的递归”或“的同义词的决议。”在缩写的分歧反映了在多种应用中AIML <srai>。其中每个详细描述为低于款：

（1）。 象征性减少：减少复杂，简单的grammatic形式。

（2）。 分而治之：拆分为两个或多个分题的投入，并结合每个反应。

（3）。 别名：地图不同的方式说，同样的事情了同样的答复。

（4）。拼写或语法更正。

（5）。检测中的任意位置输入关键字。

（6）。 条件句： 某些形式的分支，可实施<srai>。

（7）。任何（1） - （6）组合。

<srai>的危险是，它允许botmaster创造无限循环。虽然构成一定的风险，以新手程序员，我们推测，包括<srai>远远比简单的迭代阻止任何可能的结构控制已经取代了它的标记。

（1）。象征性减少

象征性的减少是指简化成更简单的复杂的语法形式的进程。通常，在存储机器人知识类别的原子模式是最简单的术语说，例如，我们倾向于选择像“世卫组织”苏格拉底喜欢的“你知道谁苏格拉底是”当存储关于苏格拉底传记资料模式。

更复杂的形式很多，以简单的形式，减少使用象征性的减少设计AIML类别：

<category>

<pattern>你知道谁*是“/模式”

<template> <srai>是谁<star/>“/ srai”“/模板”>

“/”类别下方>

无论输入匹配这个模式中，部分绑定到*可分为与标记插入通配符<star/>答复。本分类减少任何形式的输入“你知道X是谁？”以“谁是X？”

（2）。分而治之

许多个别句子可减少到两个或两个以上subsentences，通过整合的答复分别组成的答复。句子与单词“是例如”开始，如果有多个单词，可视为subsentence“是的。”加上任何后续行动。

<category>

<pattern>是*“/模式”

<template> <srai>是“/ srai”<sr/>“/模板”>

“/”类别下方>

标记<sr/>只是一个<srai>缩写<star/>“/ srai”。

（3）。别名

该AIML 1.01标准不允许超过一类的模式。同义词是可能是最常见的应用<srai>。很多方法可以说同样的话减少为一个类别，其中包含的答复：

<category>

<pattern>你好“/模式”

<template>吃了吗？“/模板”>

“/”类别下方>

<category>

<pattern>您好“/模式”

<template> <srai>你好“/ srai”“/模板”>

“/”类别下方>

<category>

<pattern>您好有“/模式”

<template> <srai>你好“/ srai”“/模板”>

“/”类别下方>

<category>

<pattern>您好“/模式”

<template> <srai>你好“/ srai”“/模板”>

“/”类别下方>

<category>

<pattern>日HOLA“/模式”

<template> <srai>你好“/ srai”“/模板”>

“/”类别下方>

（4）。更正拼写和语法

一个最常见的客户端拼写的错误是“你的”当“你是使用”或“您”的目的。并不是每一个出现的“你”，但是应把“你”了。语法方面的小金额通常需要捕捞此错误：

<category>

<pattern>您的A *“/模式”

<template>我想你意思是“你”或“你”而不是“你的。”

<srai>你是一个<star/>“/ srai”

“/模板”>

“/”类别下方>

在这里，既纠正机器人客户端输入和语言教师的行为。

（5）。关键词

经常我们想编写一个AIML模板是由关键字的任何地方输入的句子出现激活。四个AIML类的一般格式是由酶联免疫吸附说明借用这个例子：

<category>

<pattern>母亲“/模式”

<template>告诉我更多关于你的家庭。 “/模板”>

“/”类别下方>

<category>

<pattern> _母亲“/模式”

<template> <srai>母亲“/ srai”“/模板”>

“/”类别下方>

<category>

<pattern>母亲_“/模式”

<template> <srai>母亲“/ srai”“/模板”>

“/”类别下方>

<category>

<pattern> _母亲*“/模式”

<template> <srai>母亲“/ srai”“/模板”>

“/”类别下方>

第一类的关键字都检测时，它本身出现，并提供通用的反应。第二类检测作为一个句子后缀关键字。第三检测到它作为一个输入句子前缀，终于在最后一类检测为缀关键字。最后三个类别使用每个<srai>链接到第一个，这样所有4起案件产生同样的答复，但它需要书面和只存储一次。

（6）。条件句

很可能写在AIML条件分支，只用<srai>标记。考虑三类：

<category>

<pattern>他是谁“/模式”

<template> <srai> WHOISHE <get name="he"/>“/ srai”“/模板”>

“/”类别下方>

<category>

<pattern> WHOISHE *“/模式”

<template>他<get name="he"/>。“/模板”>

“/”类别下方>

<category>

<pattern> WHOISHE未知“/模式”

<template>我不知道他是谁。“/模板”>

“/”类别下方>

只要上游“他”是初始化为“未知”的类别执行条件分支是否“他”已设置而定。作为对botmaster方便，AIML还通过<条件>标记同等的职责。

打靶

博特人士创建，并通过了监督学习型的循环过程称为定位。定位是一个循环纳入客户，漫游，和botmaster，其中客户的投入，没有发现完整的类别中记录匹配的机器人和目标发表了botmaster，谁然后创建适当的回应，其中最常见的问题开始。周期为目标，逐步产生一个更精确的机器人的个性。

在AIML写作艺术创造最缺省类别，提供不置可否的答复，支持多种输入明显。这里是一个很好的例子提供上发现alicebot，通用的邮件列表：

用户：在哪里可以找到与南方古猿网页？

爱丽丝：您想知道吗？

用户：错误的答案爱丽丝时间学习一种新线

爱丽丝：你会如何回答“在哪里可以找到与南方古猿网页？”

用户：我会说：“http://www.ancientskulls.net/australopithecus.html“

爱丽丝：谢谢，人类。试着问我了。

用户：在哪里可以找到与南方古猿网页？

爱丽丝： http://www.ancientskulls.net/australopithecus.html

这是一个非常好的例子。这是完全一样的目标时，botmaster的是老师。或者我应该说，这是一个AIML接口配置。当然，这种情况下提出，例如如何进行黑客和其他垃圾度外，杂草很多问题，但这可能只是对学习功能，而把在禁区沙箱，虽然它是许多聊天者自由劳动力的网页，可帮助聪明的机器人快速增长。

这是对学习问题简而言之症结所在。这一切都归结为编辑。无论是botmaster编辑优质答复首先，或者bot是“公开”，那么，教师的所有编辑所提供的所有的“自由”劳动蹩脚的答案。我个人的看法是，随着一个假设的学习机学习得像个孩子的语言能力相比，AIML定位的方法更有效。

背景

关键字“，”在AIML指机器人以前的话语。特别是，如果机器人的多段第一句回应，该值设置为序列中的最后一句。该“是在普通语言的使用激发了关键字”选择：

?：今天是昨天。

ç：这是没有意义的。

?：答案是3.1412926左右。

荤：这是很酷。

在AIML语法<that> ...</的“包围的模式相匹配的机器人以前的话语。一个常见的应用<that>是发现是，没有任何问题：

<category>

<pattern>是“/模式”

<that>你喜欢的电影“/的”

<template>你最喜欢的电影？“/模板”>

“/”类别下方>

这个类是当客户端激活说是。该机器人必须找出是什么，他说“是”。如果机器人问，“你喜欢电影吗？”这一类比赛，反应，“什么是你最喜欢的电影吗？”，继续沿着相同的路线的交谈。

一个有趣的应用<that>的类别，使机器人应对连锁反应敲笑话。类别：

<category>

<pattern>爆震爆震“/模式”

<template>是谁？“/模板”>

“/”类别下方>

<category>

<pattern> *“/模式”

<that>谁在那里“/的”

<template> <person/>谁？“/模板”>

“/”类别下方>

<category>

<pattern> *“/模式”

<that> *世卫组织“/的”

<template>哈哈很有趣，<get name="name"/>。“/模板”>

“/”类别下方>

产生以下的对话：

荤：爆震敲。

?：谁在那儿？

荤：香蕉。

?：香蕉是谁？

荤：爆震敲。

?：谁在那儿？

荤：香蕉。

?：香蕉是谁？

荤：爆震敲。

?：谁在那儿？

荤：橙色。

?：橙色是谁？

荤：橙色你高兴我没有说香蕉。

?：哈哈很有趣，南希。

内部的AIML翻译存储输入模式，这种模式和主题模式沿着单一的路径，如：输入<that>动议<topic>主题。当值<that>或<topic>未指定，程序隐式集的值相应的动议或主题模式通配符*.

路径的匹配第一部分是输入。如果超过一类有相同的输入模式，该程序可能会区分它们取决于价值<that>。如果两个或多个类别的具有相同的<pattern>和<that>，最后一步是选择的基础上答复<topic>。

这种结构表明，设计规则：不要使用<that>除非你写了相同的两类<pattern>，从不使用<topic>，除非你写两个拥有相同类别<pattern>和<that>。不过，其中一个<topic>是创建主题依赖“皮卡行，”我喜欢最有用的应用程序：

<topic name="CARS">

<category>

<pattern> *“/模式”

<template>

<random>

<li>你最喜欢什么车？“/李”

<li>什么样的车，你开车吗？“/李”

<li>你得到很多的停车票？“/李”

<li>我最喜欢的汽车是一辆带司机的。“/李”

“/随机”

“/模板”>

考虑到人们对事物的一套规模庞大，可以说是语法正确的或语义意义，事物的根本是说号码是出奇地低。史蒂芬平克，在他的著作思考方式写道：“假设你有10个单词的第一个选择，开始为第二个单词的句子，10选择（100高产两个单词的开始），第三字10选择（屈服1000三个词开始），等等。（10实际上是近似的文字可供选择，每个点可在聚集了语法和句子数量合理的几何平均数）。一个小算术表明，判刑人数20字以内（而不是一个不寻常的长度）的。约1020“

幸运的是，聊天机器人程序员，平克的计算路要走。我们的实验与A.L.I.C.E.表明，选择的“第一个字号码”是十多个，但只有大约2000。具体而言，约2000字涵盖95输入的第一句话％至爱丽丝。在选择第二个字，数目只有两个。当然，也有一些的第一句话（“我”和“你”为例）有许多可能的第二个字，但整体平均不到两个词。平均每个分支连续单词因素减少。

我们策划了香港雅丽一些美丽的图像此图所代表的大脑内容（http://alice.sunlitsurf.com/documentation/gallery/）。超过A.L.I.C.E.公正优雅图片大脑，这些螺旋图像（查看更多）概述了语言的领土已被有效地“征服了爱丽丝”和AIML。

没有自然语言处理别的理论能够更好地解释或复制在我国境内的结果。您不需要复杂的学习理论，神经网络，或认知模型解释如何在聊天Alice的25,000类别的限制。我们的刺激反应模型一样好这些案件的任何其他的理论，当然最简单的。如果是“左高”自然语言理论，任何房间里以外的爱丽丝地图谎言脑。

学者们炮制的谜语和语言的矛盾，很有可能显示如何恶劣，自然喜欢的语言问题。 “约翰看见山飞越苏黎世”或“水果如香蕉”揭示了语言的模糊性和爱丽丝的限制式的方法苍蝇（虽然不是这些具体的例子，当然，爱丽丝已经对他们知道）。在未来的日子里，我们只会进一步推动边境。在螺旋图的基本轮廓看起来是一样的，因为我们已经找到了“大树都”从“A *”到“你的*”。这些树木可能会变得更大，但除非语言本身的变化，我们不会找到更多的大树（除外国语言课程）。寻求解释的东西多复杂的条件刺激的反应自然语言将超出我们的边界的地方越来越多只珍稀的语言形式占领的腹地，这些部门的工作。我们的语言境内已经包含了句子使用人口最多的人。扩大边界更加我们将继续吸收掉队外，直到最后一人的批评，也没有想，一句“傻瓜”爱丽丝。
```
