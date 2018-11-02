# ___2018 - 10 - 24 Chatbot___
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
## 链接
  - [百度中文词法分析 LAC](https://github.com/baidu/lac)
  - [百度问答系统框架 AnyQ](https://github.com/baidu/AnyQ)
  - [结巴中文分词](https://github.com/fxsjy/jieba)
  - [百度情感识别系统](https://github.com/baidu/Senta)
## sklean 与 jieba 训练的简单模型
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
***

# AIML
## AIML guide
  - [AI Chat Bot in Python with AIML](https://www.devdungeon.com/content/ai-chat-bot-python-aiml)
  - [Github python-aiml](https://github.com/paulovn/python-aiml)
  - [Chatty Cathy](https://www.devdungeon.com/content/chatty-cathy)
  - [AIML Reference](https://pandorabots.com/docs/aiml/reference.html)
  - [Free-AIML by pandorabots](https://github.com/pandorabots/Free-AIML)
  - [Free AIML chat bot content by wikidot](http://alicebot.wikidot.com/aiml:en-us:archbotmaster:aaa)
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
  - **创建 AIML 对话文件 basic_chat.aiml** 使用自定义的 AIML 对话文件大约需要 10,000 条规则才能使对话比较真实，可以使用已有的 AIML 文件，如 `std-65-percent.xml` 包含了 65% 的常用场景
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
  - **Python AIML** 通过创建 AIML 对象，并加载定义的 `启动文件 std-startup.xml` 与 `对话文件 basic_chat.aiml`，训练后可以用于对话
    ```sh
    # python2
    pip install aiml
    # python3
    pip install python-aiml
    ```
    ```py
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
## AIML 标签元素
  | AIML Tag          | Description                                                                                       |
  | ----------------- | ------------------------------------------------------------------------------------------------- |
  | **`<aiml>`**      | Defines the beginning and end of a AIML document.                                                 |
  | **`<category>`**  | Defines the unit of knowledge in Alicebot's knowledge base.                                       |
  | **`<pattern>`**   | Defines the pattern to match what a user may input to an Alicebot.                                |
  | **`<template>`**  | Defines the response of an Alicebot to user's input.                                              |
  | **`<star>`**      | Used to match wild card * character(s) in the <pattern> Tag                                       |
  | **`<srai>`**      | Multipurpose tag, used to call/match the other categories                                         |
  | **`<random>`**    | Used <random> to get random responses                                                             |
  | **`<li>`**        | Used to represent multiple responses                                                              |
  | **`<set>`**       | Used to set value in an AIML variable                                                             |
  | **`<get>`**       | Used to get value stored in an AIML variable                                                      |
  | **`<that>`**      | Used in AIML to respond based on the context                                                      |
  | **`<topic>`**     | Used in AIML to store a context so that later conversation can be done based on that context      |
  | **`<think>`**     | Used in AIML to store a variable without notifying the user                                       |
  | **`<condition>`** | Similar to switch statements in programming language. It helps ALICE to respond to matching input |
## 基本标签元素 aiml category pattern template
  - **aiml** 标签定义文档起始 / 结束位置
    ```xml
    <aiml version="1.0.1" encoding="UTF-8">
        ...
    </aiml>
    ```
  - **Categories** 一条匹配规则的基本基本单元，至少包含一条 `pattern` 与 `template`
    ```xml
    <category>
        <pattern>WHAT IS YOUR NAME</pattern>
        <template>My name is Michael N.S Evanious.</template>
    </category>
    ```
  - **Patterns** 字符串表示的匹配规则，必须大写，匹配中忽略大小写，可以使用 `*` / `_` 匹配多个字符
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
## star
  - **`<star>`** 标签表示 `<pattern>` 中匹配到的 `*` 代表的内容
    ```xml
    <star index = "n"/>
    ```
    - **index** 表示匹配到的第 n 个内容
  - **示例**
    ```xml
    <category>
        <pattern>I LIKE *</pattern>
        <template>
            I like <star/> too.
        </template>
    </category>

    <category>
        <pattern> A * is a *. </pattern>
        <template>
            How <star index = "1"/> can not be a <star index = "2"/>?
        </template>
    </category>
    ```
## srai
  - **`<srai>`** 标签 Symbolic Reduction in Artificial Intelligence，用于将不同的匹配规则重定向到同一个目标模板
    ```xml
    <srai> pattern </srai>
    ```
  - **复杂语法约简 symbolic reduction** 使用简单的规则匹配更复杂的语法
    ```xml
    <category>
       <pattern>WHO IS ALBERT EINSTEIN?</pattern>
       <template>Albert Einstein was a German physicist.</template>
    </category>

    <category>
       <pattern> WHO IS Isaac NEWTON? </pattern>
       <template>Isaac Newton was a English physicist and mathematician.</template>
    </category>
    ```
    定义同义的 `<srai>` 标签
    ```xml
    <category>
       <pattern>DO YOU KNOW WHO * IS?</pattern>

       <template>
          <srai>WHO IS <star/></srai>
       </template>

    </category>
    ```
  - **分解 Divide and Conquer** 在一个回复中重用其他的子句
    ```xml
    <category>
       <pattern>BYE</pattern>
       <template>Good Bye!</template>
    </category>
    ```
    `<srai>` 标签定义更通用的匹配规则
    ```xml
    <category>
       <pattern>BYE *</pattern>
       <template>
          <srai>BYE</srai>
       </template>
    </category>
    ```
  - **同义词约简 Synonyms Resolution** 同义词重定向到一个统一的回复
    ```xml
    <category>
       <pattern>FACTORY</pattern>
       <template>Development Center!</template>
    </category>
    ```
    `<srai>` 标签定义同义词
    ```xml
    <category>
       <pattern>INDUSTRY</pattern>

       <template>
          <srai>FACTORY</srai>
       </template>

    </category>
    ```
  - **关键词解析 Keywords Detection** 包含特定关键词的匹配规则，重定向到同一个回复
    ```xml
    <category>
       <pattern>SCHOOL</pattern>
       <template>School is an important institution in a child's life.</template>
    </category>
    ```
    `<srai>` 标签定义关键词 `SCHOOL` 的规则
    ```xml
    <category>
       <pattern>_ SCHOOL</pattern>
       <template>
          <srai>SCHOOL</srai>
       </template>
    </category>

    <category>
       <pattern>SCHOOL *</pattern>
       <template>
          <srai>SCHOOL</srai>
       </template>
    </category>

    <category>
       <pattern>_ SCHOOL *</pattern>
       <template>
          <srai>SCHOOL</srai>
       </template>
    </category>
    ```
## random
  - **`<random>`** 标签定义随机回复内容，使用 `<li>` 标签定义多个随机回复的内容
    ```xml
    <random>
        <li> pattern1 </li>
        <li> pattern2 </li>
        ...
        <li> patternN </li>
    </random>
    ```
  - **示例**
    ```xml
    <aiml version = "1.0.1" encoding ="UTF-8"?>
      <category>
        <pattern>HI</pattern>
        <template>
          <random>
            <li> Hello! </li>
            <li> Hi! Nice to meet you! </li>
          </random>
        </template>
      <category>      
    </aiml>
    ```
## set get
  - **`<set>`** 与 **`<get>`** 标签用于设置 / 获取变量值，变量可以是预定义或程序创建的变量
    ```xml
    <set name = "variable-name"> variable-value </set>
    <get name = "variable-name"></get>
    ```
  - **示例**
    ```xml
    <aiml version = "1.0.1" encoding = "UTF-8"?>
      <category>
        <pattern>I am *</pattern>
        <template>
          Hello <set name = "username"> <star/>! </set>
        </template>  
      </category>  

      <category>
        <pattern>Good Night</pattern>
        <template>
          Hi <get name = "username"/> Thanks for the conversation!
        </template>  
      </category>
    </aiml>
    ```
## that
  - **`<that>`** 标签可以用来引用对话中的上下文
    ```xml
    <that> template </that>
    ```
  - **示例**
    ```xml
    <aiml version = "1.0.1" encoding = "UTF-8"?>
      <category>
        <pattern>WHAT ABOUT MOVIES</pattern>
        <template>Do you like comedy movies</template>  
      </category>

      <category>
        <pattern>YES</pattern>
        <that>Do you like comedy movies</that>
        <template>Nice, I like comedy movies too.</template>
      </category>

      <category>
        <pattern>NO</pattern>
        <that>Do you like comedy movies</that>
        <template>Ok! But I like comedy movies.</template>
      </category>
    </aiml>
    ```
  - 如果要获取更早的回复内容，可以使用 `index`
    ```xml
    <that index=”nx,ny”> </that>
    ```
    如使用 `<that index=”2,1”>` 表示取倒数第 2 句的回复，相当于 `<justbeforethat/>`
## topic
  - **`<topic>`** 标签用于指定一个后续对话的话题环境，使用 `<set>` 设置 topic
    ```xml
    <template>
       <set name = "topic"> topic-name </set>
    </template>
    ```
    定义 `<topic>` 话题环境
    ```xml
    <topic name = "topic-name">
       <category>
          ...
       </category>     
    </topic>
    ```
  - **示例**
    ```xml
    <aiml version = "1.0.1" encoding = "UTF-8"?>
      <category>
        <pattern>LET DISCUSS MOVIES</pattern>
        <template>Yes <set name = "topic">movies</set></template>  
      </category>

      <topic name = "movies">
        <category>
          <pattern> * </pattern>
          <template>Watching good movie refreshes our minds.</template>
        </category>

        <category>
          <pattern> I LIKE WATCHING COMEDY! </pattern>
          <template>I like comedy movies too.</template>
        </category>
      </topic>
    </aiml>
    ```
## think
  - **`<think>`** 标签可以配合 `<set>` 使用，设置变量后不回显给用户
    ```xml
    <think>
      <set name = "variable-name"> variable-value </set>
    </think>
    ```
  - **示例**
    ```xml
    <aiml version = "1.0.1" encoding = "UTF-8"?>
      <category>
        <pattern>My name is *</pattern>
        <template>
          Hello!<think><set name = "username"> <star/></set></think>
        </template>  
      </category>  

    <category>
      <pattern>Byeee</pattern>
      <template>
         Hi <get name = "username"/> Thanks for the conversation!
      </template>  
    </category>
    </aiml>
    ```
## condition
  - **`<condition>`** 标签类似 `switch` 语句，根据不同输入，回复不同内容
    ```xml
    <condition name = "variable-name" value = "variable-value"/>
    ```
  - **示例**
    ```xml
    <aiml version = "1.0.1" encoding = "UTF-8"?>
      <category>
        <pattern> HOW ARE YOU FEELING TODAY </pattern>

        <template>
          <think><set name = "state"> happy</set></think>
          <condition name = "state" value = "happy">
            I am happy!
          </condition>

          <condition name = "state" value = "sad">
            I am sad!
          </condition>
        </template>

      </category>
    </aiml>
    ```
    也可以使用 `<li>` 标签定义多个回复
    ```xml
    <category>
        <pattern>我 头 发 的 颜 色 是 蓝 色 *</pattern>
        <template>你很
            <condition>
                <li name="用户性别" value="女"> 漂亮阿！</li>
                <li name="用户性别" value="男"> 英俊阿！</li>
                <li>好看！</li>
            </condition>
        </template>
    </category>
    ```
## 其他标签
  - **`<gender>`** 标签，用于替换性别以及代名词
    ```xml
    <gender>She told him to take a hike.</gender>
    ```
    替换为
    ```xml
    He told her to take a hike
    ```
  - **`<gossip>`** 标签，用于把内容保存到 `gossip.log` 文件里
  - **`<if>`** 标签，用于判断
    ```py
    <if name="topic" value="cars"></if>
    <if name="topic" contains="cars"></if>
    <if name="topic" exists="true"></if>
    ```
    **示例**
    ```xml
    <template>
      <if name="用户名称" exists="true">
        你的名字叫 <get name=”用户名称”/>.
      <else/>
        你叫什么名字？
      </if>
    </template>
    ```
  - **`<input>`** 表示用户输入，`index` 指定获取倒数第 n 句的输入
    ```xml
    <input index="n"/>
    ```
    示例
    ```xml
    <category>
      <pattern>嘿 嘿</pattern>
      <template>
        <gossip>你刚才说：“<input index="2"/>”？</gossip>
      </template>
    </category>
    ```
  - **`<learn>`** 标签，指定学习某个 aiml 文件
    ```xml
    <learn filename="xxx.aiml"/>
    ```
  - **`<person>`** 与 **`<person2>`** 标签，用于将第一人称转化为 第三人称 / 第二人称
  - **`<sentence>`** 标签，用于格式化字符串，可以大写首字母 / 添加标点符号等
    ```xml
    <sentence>this is some kind of sentence test.</sentence>
    ```
    格式化成
    ```xml
    This is some kind of sentence test.
    ```
  - **`<system>`** 标签，表示调用系统函数
    ```xml
    <!-- 读取系统当前时间 -->
    <system>date</system>
    ```
  - **`<thatstar>`** 标签，可以使用 `index` 指定
    ```xml
    <thatstar index="n"> <thatstar/>
    <thatstar index="1"/>
    ```
    **示例**
    ```xml
    <category>
      <pattern>你好</pattern>
      <template>
        计算机 的 型 号 是 什 么
      </template>
    </category>

    <category>
      <pattern>*</pattern>
      <that>*  的 型 号 是 什 么</that>
      <template><star/>这个型号是<thatstar/>里面
        <random>
          <li>很好的商品</li>
          <li>很流行的商品</li>
          <li>很华丽的商品</li>
        <random>
      </template>
    </category>
    ```
    **对话**
    ```xml
    用户：你好
    机器人：计算机 的 型 号 是 什 么
    用户：p4
    机器人：p4这个型号是计算机里面很好的商品
    ```
***

# AIML 处理中文
## 判断中文字符
  ```py
  is_Chinese_char = lambda ch: u'\u4e00' <= ch <= u'\u9fff'

  def is_Chinese_string(check_str):
      if check_str == None:
          return False

      for ch in check_str:
          if is_Chinese_char(ch):
              return True
      return False

  def is_ascii_string(ss):
      for aa in ss:
          if aa > '~':
              return False
      return True
  ```
## 插入与移除空格
  - 原生的 AIML 解释器都是面向英文设计的，需要有空格来断字，并且在运行中会自动大写所有字母，并在回复中大写首字母
  - 对于中文的处理需要在编写 AIML 时在汉字之间加上空格，并在运行时同样先给汉字插上空格再交付 AIML 解释器解析
  - **插入与移除空格**
    ```py
    import jieba

    # Insert space by jieba split words
    # insert_space = lambda ss: ' '.join([ww for ww in jieba.lcut(ss) if not ww.startswith(' ')])
    def insert_space(ss):
        rr = ''
        for ww in jieba.lcut(ss):
            if is_Chinese_char(ww[0]):
                ww = ' ' + ww + ' '
            rr += ww

        while rr.find('  ') != -1:
            rr = rr.replace('  ', ' ')

        return rr.strip()

    insert_space('Python * 列表*迭代是* <get name=aaa></get>')
    # 'Python * 列表 * 迭代 是 * <get name=aaa></get>'

    # Remove space if it's between two Chinese characters
    # remove_space = lambda ss: (ss[0] + ''.join([aa for ii, aa in enumerate(ss[1:-1], 1) if not (aa == ' ' and is_Chinese_char(ss[ii-1]) and is_Chinese_char(ss[ii+1]))]) + ss[-1]).strip()
    def remove_space(ss):
        rr = ss[0]
        for ii, aa in enumerate(ss[1:-1], 1):
            if aa == ' ':
                if is_Chinese_char(ss[ii-1]) and is_Chinese_char(ss[ii+1]):
                    continue
            rr += aa
        return (rr + ss[-1]).strip()

    remove_space('Python * 列表 * 迭代 是 *')
    # 'Python * 列表 * 迭代是 *'
    ```
## 文本方式处理 aiml 文件
  - 需要处理的标签为 **需要文本匹配的内容**，包括 `pattern` / `srai`
  - 要求要处理的标签与文本在同一行，且是单独的一行
  - **文本方式处理 aiml 文件**
    ```py
    def fit_aiml_line_2_Chinese(ll):
        tags_need_convert = ['<pattern>', '<srai>']
        stripped = ll.strip()
        for pp in tags_need_convert:
            if stripped.startswith(pp):
                if is_Chinese_string(stripped):
                    contains = stripped[len(pp): -len(pp) - 1]
                    contains_conv = insert_space(contains)
                    ll = ll[:ll.find(pp) + len(pp)] + contains_conv + stripped[-len(pp) - 1:] + '\n'
                break
        return ll

    def fit_aiml_file_2_Chinese(aiml_file, target_file=None):
        is_inplace = False
        if target_file == aiml_file:
            is_inplace = True
            target_file = None

        if target_file == None:
            target_file = aiml_file[:aiml_file.rfind('.')] + '_convert.aiml'

        with open(aiml_file, 'r') as source_file:
            with open(target_file, 'w') as ff:
                for ll in source_file.readlines():
                    ll_fitted = fit_aiml_line_2_Chinese(ll)
                    ff.write(ll_fitted)

        if is_inplace:
            os.rename(target_file, aiml_file)
            target_file = aiml_file
        return target_file

    aiml_file = './basic_chat.aiml'
    fit_aiml_file_2_Chinese(aiml_file, aiml_file)
    ```
## xml 解析方式处理 aiml 文件
  - **xml 解析方式处理 aiml 文件**，xml 解析模块使用 `xml.etree.ElementTree`
    ```py
    # 添加空格
    import jieba
    insert_space = lambda ss: ' ' + (' '.join([ww for ww in jieba.lcut(ss) if not ww.startswith(' ')])).strip() + ' '

    from xml.etree import ElementTree

    # Fit all text data in element, including data in text / tail
    def fit_element_2_Chinese(elem):
        # print('elem string = {}'.format(ElementTree.tostring(elem)))
        if is_Chinese_string(elem.text):
            elem.text = insert_space(elem.text)
        for elem_tem_sub in elem:
            if is_Chinese_string(elem_tem_sub.tail):
                elem_tem_sub.tail = insert_space(elem_tem_sub.tail)

    def fit_aiml_root_2_Chinese(root, tags_need_fit=['pattern', 'srai']):
        for tt in tags_need_fit:
            for ee in root.iter(tt):
                fit_element_2_Chinese(ee)

    def fit_aiml_file_2_Chinese(aiml_file, target_file=None):
        if target_file == None:
            target_file = aiml_file

        tree = ElementTree.parse(aiml_file)
        root = tree.getroot()
        fit_aiml_root_2_Chinese(root)
        tree.write(target_file, encoding='utf-8')

    aiml_file = './topic.aiml'
    fit_aiml_file_2_Chinese(aiml_file, './foo.xml')
    ```
## 输入与输出处理
  - **输入与输出处理**，输入文本插入空格，输出文本移除空格
    ```py
    import aiml

    # Create the kernel and learn AIML files
    kernel = aiml.Kernel()
    # kernel.learn("std-startup.xml")
    # kernel.respond("load aiml b")
    kernel.bootstrap(learnFiles='./std-startup.xml', commands='load aiml a')

    # Press CTRL-C to break this loop
    while True:
        orig_input = input("Enter your message >> ")
        if orig_input == 'quit':
            break
        convert_input = insert_space(orig_input).strip()
        print('Converted: %s' % (convert_input))

        orig_response = kernel.respond(convert_input)
        if orig_response:
            print(remove_space(orig_response))
        else:
            print('hmmm....')
    ```
## PyAIML 项目
  - [PyAIML github](https://github.com/andelf/PyAIML.git)
  - **PyAIML 的源码组织**
    - **AimlParser.py** AIML XML 读取部分，使用 `xml.sax` 接口解析 XML 文件读入对话逻辑和机器人回复信息
    - **Kernel.py** 人机对话输入输出接口，提供运行期的 IO 接口，最主要的 IO 接口是 respond 函数
    - **PatternMgr.py** 利用 AIML 模式来匹配符合的输入字串，主要实现在 match 函数中
    - 其他支持模块
  - **中文空格断字** 中文化 PyAIML 主要需要实现的是自动的中文空格断字，接下来就是把语言支持加入到 PyAIML 中，主要需要添加中文切字的是 `<pattern>` 和 `<srai>` 标签的处理，以及 `<topic>` 和 `<that>` 标签的中文支持
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
***

# AIML python java topic
## 初始化启动 AIML xml 文件
  ```py
  import glob

  def add_topic_file_element_to_root(topic, file, root, format='multi_lines', space_width=4):
      new_cate = ElementTree.SubElement(root, 'category')
      new_cate.tail = '\n\n'
      new_pattern = ElementTree.SubElement(new_cate, 'pattern')
      new_pattern.text = 'LOAD AIML ' + topic
      new_template = ElementTree.SubElement(new_cate, 'template')
      new_learn = ElementTree.SubElement(new_template, 'learn')
      new_learn.text = file

      # Add line breaks and spaces if format as multi lines, or it's a single line
      if format == 'multi_lines':
          spaces = ' ' * space_width
          new_cate.text = '\n' + spaces
          new_pattern.tail = '\n' + spaces
          new_template.text = '\n' + spaces * 2
          new_learn.tail = '\n' + spaces
          new_template.tail = '\n'

  def update_topic_file_element_in_root(root, knowledge_dict):
      for cate in root.iter('category'):
          cate_topic = ' '.join(cate.find('pattern').text.split(' ')[2:])
          cate_file = cate.find('template').find('learn').text
          if cate_topic in knowledge_dict:
              new_topic_file = knowledge_dict.pop(cate_topic)
              if cate_file != new_topic_file:
                  print('Update topic {}, file: {}'.format(cate_topic, new_topic_file))
                  cate.find('template').find('learn').text = new_topic_file
      return knowledge_dict

  def init_knowledge_base(knowledge_path, startup_file, target_file=None, format='multi_lines', space_width=4):
      if not os.path.exists(startup_file):
          print('Start up file not exist, at least point the default dialog aiml')
          return

      # Find all aiml files in knowledge base path
      knowledge_files = glob.glob(os.path.join(knowledge_path, '*.aiml'))
      if len(knowledge_files) == 0:
          print('Empty knowledge base')
          return

      # Convert aiml files for Chinese language if still not
      fitted_flag_file = os.path.join(knowledge_path, 'IS_FITTED_FLAG')
      if not os.path.exists(fitted_flag_file):
          print("fit aiml files to Chinese:")
          for ff in knowledge_files:
              print("  Aiml file name: {}".format(ff))
              fit_aiml_file_2_Chinese(ff)
          with open(fitted_flag_file, 'w'):
              pass
          print("")

      # Knowledge topic is the aiml file name
      knowledge_topic = [os.path.basename(ii).split('.')[0].upper() for ii in knowledge_files]
      knowledge_dict = dict(zip(knowledge_topic, knowledge_files))

      # Parse atart up xml file
      tree = ElementTree.parse(startup_file)
      root = tree.getroot()

      # Update those same topics in both current start up xml file and knowledge base
      knowledge_dict = update_topic_file_element_in_root(root, knowledge_dict)

      # Add new topics to start up xml file if any
      for topic, file in knowledge_dict.items():
          print('Add new topic {}, file: {}'.format(topic, file))
          add_topic_file_element_to_root(topic, file, root, format, space_width)

      # Write to target file
      if target_file == None:
          target_file = startup_file
      tree.write(target_file, encoding='utf-8')

      return knowledge_topic

  knowledge_path = './knowledge_base'
  startup_file = './std-startup.xml'
  init_knowledge_base('./knowledge_base', './std-startup.xml')
  ```
## 对话
  ```py
  import aiml

  # kernel.learn("std-startup.xml")
  # kernel.respond("load aiml b")
  def start_conversation_loop(startup_file, topics=None):
      # Initiate topics from start up file if None
      if topics == None:
          tree = ElementTree.parse(startup_file)
          root = tree.getroot()
          topics = [' '.join(pp.text.split(' ')[2:]) for pp in root.iter('pattern')]

      # Create the kernel and learn AIML files
      kernel = aiml.Kernel()
      select_topic = lambda tt: kernel.bootstrap(learnFiles=startup_file, commands='load aiml {}'.format(tt))

      topic_in_use = topics[0]
      select_topic(topic_in_use)

      # Conversation loop
      while True:
          orig_input = input("Enter your message >> ")
          if orig_input == 'quit':
              break
          convert_input = insert_space(orig_input)
          # print('Converted: %s' % (convert_input))

          upper_input = convert_input.upper()
          for tt in topics:
              if tt in upper_input:
                  if topic_in_use != tt:
                      topic_in_use = tt
                      select_topic(topic_in_use)
                      break

          orig_response = kernel.respond(convert_input)
          if orig_response == None and topic_in_use != topics[0]:
              # Try again with basic topic
              topic_in_use = topics[0]
              select_topic(topic_in_use)
              orig_response = kernel.respond(convert_input)

          print('Response >> ', end='')
          if orig_response:
              print(remove_space(orig_response))
          else:
              print('hmmm....')
          print("")
  ```
## 组合
  ```py
  def aiml_talk(knowledge_path, startup_file, init_kb=False, topics=None, target_file=None, format='multi_lines', space_width=4):
      if target_file != None and init_kb == False:
          print('Error in parameter, target_file != None and init_kb == False')
          return

      if init_kb:
          init_knowledge_base(knowledge_path, startup_file, target_file, format, space_width)

      if target_file:
          startup_file = target_file

      start_conversation_loop(startup_file, topics)
  ```
***

# 中文分词词性对照表
  | 词性编码 | 词性名称 | 注解                                                                               |
  | -------- | -------- | ---------------------------------------------------------------------------------- |
  | Ag       | 形语素   | 形容词性语素，形容词代码为 a，语素代码 ｇ 前面置以 A                               |
  | a        | 形容词   | 取英语形容词 adjective 的第 1 个字母                                               |
  | ad       | 副形词   | 直接作状语的形容词，形容词代码 a 和副词代码 d 并在一起                             |
  | an       | 名形词   | 具有名词功能的形容词，形容词代码 a 和名词代码 n 并在一起                           |
  | b        | 区别词   | 取汉字“别”的声母                                                                   |
  | c        | 连词     | 取英语连词 conjunction 的第1个字母                                                 |
  | dg       | 副语素   | 副词性语素，副词代码为 d，语素代码 ｇ 前面置以 D                                   |
  | d        | 副词     | 取 adverb 的第 2 个字母，因其第 1 个字母已用于形容词                               |
  | e        | 叹词     | 取英语叹词 exclamation 的第1个字母                                                 |
  | f        | 方位词   | 取汉字“方”                                                                         |
  | g        | 语素     | 绝大多数语素都能作为合成词的“词根”，取汉字“根”的声母                               |
  | h        | 前接成分 | 取英语 head 的第1个字母                                                            |
  | i        | 成语     | 取英语成语 idiom 的第1个字母                                                       |
  | j        | 简称略语 | 取汉字“简”的声母                                                                   |
  | k        | 后接成分 |                                                                                    |
  | l        | 习用语   | 习用语尚未成为成语，有点“临时性”，取“临”的声母                                     |
  | m        | 数词     | 取英语 numeral 的第3个字母，n，u 已有他用                                          |
  | Ng       | 名语素   | 名词性语素，名词代码为 n，语素代码 ｇ 前面置以 N                                   |
  | n        | 名词     | 取英语名词 noun 的第1个字母                                                        |
  | nr       | 人名     | 名词代码 n 和“人(ren)”的声母并在一起                                               |
  | ns       | 地名     | 名词代码 n 和处所词代码 s 并在一起                                                 |
  | nt       | 机构团体 | “团”的声母为 t，名词代码 n 和 t 并在一起                                           |
  | nz       | 其他专名 | “专”的声母的第 1 个字母为 z，名词代码 n 和 z 并在一起                              |
  | o        | 拟声词   | 取英语拟声词 onomatopoeia 的第 1 个字母                                            |
  | p        | 介词     | 取英语介词 prepositional 的第 1 个字母                                             |
  | q        | 量词     | 取英语 quantity 的第 1 个字母                                                      |
  | r        | 代词     | 取英语代词 pronoun的 第 2 个字母，因 p 已用于介词                                  |
  | s        | 处所词   | 取英语 space 的第 1 个字母                                                         |
  | tg       | 时语素   | 时间词性语素，时间词代码为 t,在语素的代码 g 前面置以 T                             |
  | t        | 时间词   | 取英语 time 的第 1 个字母                                                          |
  | u        | 助词     | 取英语助词 auxiliar                                                                |
  | vg       | 动语素   | 动词性语素，动词代码为 v，在语素的代码 g 前面置以 V                                |
  | v        | 动词     | 取英语动词 verb 的第一个字母                                                       |
  | vd       | 副动词   | 直接作状语的动词，动词和副词的代码并在一起                                         |
  | vn       | 名动词   | 指具有名词功能的动词，动词和名词的代码并在一起                                     |
  | w        | 标点符号 |                                                                                    |
  | x        | 非语素字 | 非语素字只是一个符号，字母 x 通常用于代表未知数、符号                              |
  | y        | 语气词   | 取汉字“语”的声母                                                                   |
  | z        | 状态词   | 取汉字“状”的声母的前一个字母                                                       |
  | un       | 未知词   | 不可识别词及用户自定义词组，取英文 Unkonwn 首两个字母 (非北大标准，CSW 分词中定义) |
***

# jieba
## 简介
  - **三种分词模式**
    - **精确模式**，试图将句子最精确地切开，适合文本分析
    - **全模式**，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义
    - **搜索引擎模式**，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
  - **算法**
    - 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图 (DAG)
    - 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
    - 对于未登录词，采用了基于汉字成词能力的 HMM 模型，使用了 Viterbi 算法
  - [默认词库](https://raw.githubusercontent.com/fxsjy/jieba/master/jieba/dict.txt)
## 分词
  - `jieba.cut` 方法接受三个输入参数: 需要分词的字符串；cut_all 参数用来控制是否采用全模式；HMM 参数用来控制是否使用 HMM 模型
  - `jieba.cut_for_search` 方法接受两个参数：需要分词的字符串；是否使用 HMM 模型。该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细
  - 待分词的字符串可以是 unicode 或 UTF-8 字符串、GBK 字符串。注意：不建议直接输入 GBK 字符串，可能无法预料地错误解码成 UTF-8
  - `jieba.cut` 以及 `jieba.cut_for_search` 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)
  - `jieba.lcut` 以及 `jieba.lcut_for_search` 直接返回 list
  - `jieba.Tokenizer(dictionary=DEFAULT_DICT)` 新建自定义分词器，可用于同时使用不同词典。`jieba.dt` 为默认分词器，所有全局分词相关函数都是该分词器的映射
  - 分词时首先会按照 **概率连乘最大路径** 来切割，然后对于连续的单字中可能有词典中没有的新词，所以再用 **finalseg** 来切一遍，finalseg 是通过 **HMM** 模型来做的，简单来说就是给单字大上 B / M / E / S 四种标签以使得概率最大
    - **B** 开头
    - **E** 结尾
    - **M** 中间
    - **S** 独立成词的单字
    - 对于自定义的新词可以在词典中补充，并给予一个词频，增强歧义纠错能力
  - **示例**
    ```python
    # encoding=utf-8
    import jieba

    seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    print("Full Mode: " + "/ ".join(seg_list))  # 全模式

    seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

    seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
    print(", ".join(seg_list))

    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    print(", ".join(seg_list))
    ```
    **输出**
    ```py
    # 全模式
    Full Mode: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
    # 精确模式
    Default Mode: 我/ 来到/ 北京/ 清华大学
    # 新词识别
    他, 来到, 了, 网易, 杭研, 大厦 (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)
    # 搜索引擎模式
    小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, ，, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造
    ```
## 添加自定义词典
  - **载入词典** 开发者可以指定自己自定义的词典，以便包含 jieba 词库里没有的词，虽然 jieba 有新词识别能力，但是自行添加新词可以保证更高的正确率
    ```py
    load_userdict(f)
    ```
    - **f** 为文件类对象或自定义词典的路径，若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码
    - **词典格式** 一个词占一行，每一行分三部分 **word 词语** / **freq 词频**（可省略）/ **word_type 词性**（可省略），空格分隔，顺序不可颠倒，词频省略时使用自动计算的能保证分出该词的词频
    ```py
    创新办 3 i
    云计算 5
    凱特琳 nz
    台中
    ```
  - 更改分词器（默认为 `jieba.dt`）的 `tmp_dir` 和 `cache_file` 属性，可分别指定缓存文件所在的文件夹及其文件名，用于受限的文件系统
  - **示例**
    ```py
    import io
    import jieba
    import jieba.posseg as pseg

    test_sent = '云计算方面的专家，python easy_install 是好用的'
    print('/'.join(jieba.cut(test_sent)))
    # 云/计算/方面/的/专家/，/python/ /easy/_/install/ /是/好/用/的

    user_word_dict = '''
    easy_install 3 eng
    好用 300
    '''

    jieba.load_userdict(io.StringIO(user_word_dict))
    jieba.add_word('云计算')
    print('/'.join(jieba.cut(test_sent)))
    # 云计算/方面/的/专家/，/python/ /easy_install/ /是/好用/的

    result = pseg.cut(test_sent)
    for ww in result:
        print(ww.word, ':', ww.flag, '/', end=' ')
    # 云计算 : x / 方面 : n / 的 : uj / 专家 : n / ， : x / python : eng /   : x / easy_install : eng /   : x / 是 : v / 好用 : x / 的 : uj /
    ```
## 调整词典
  - 使用 `add_word(word, freq=None, tag=None)` 和 `del_word(word)` 可在程序中动态修改词典
  - 使用 `suggest_freq(segment, tune=True)` 可调节单个词语的词频，使其能（或不能）被分出来
  - 自动计算的词频在使用 HMM 新词发现功能时可能无效
  - **示例**
    ```py
    print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
    # 如果/放到/post/中将/出错/。
    jieba.suggest_freq(('中', '将'), True)
    # 494
    print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
    # 如果/放到/post/中/将/出错/。
    print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
    # 「/台/中/」/正确/应该/不会/被/切开
    jieba.suggest_freq('台中', True)
    # 69
    print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
    # 「/台中/」/正确/应该/不会/被/切开
    ```
## 基于 TF-IDF 算法的关键词抽取
  - **jieba.analyse.TFIDF 类** 新建 TFIDF 实例
    ```py
    class TFIDF(KeywordExtractor)
    __init__(self, idf_path=None)
    ```
    - **idf_path** 为 IDF 频率文件
  - **jieba.analyse.extract_tags** 使用 TF-IDF 算法提取句子中的关键词
    ```py
    extract_tags(sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False)
    ```
    - **sentence** 为待提取的文本
    - **topK** 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
    - **withWeight** 为是否一并返回关键词权重值，默认值为 False
    - **allowPOS** 仅包括指定词性的词，默认值为空，即不筛选
    ```py
    import jieba
    import jieba.analyse

    test_sent = '随便一句话吧，只是为了用来分析关键词用，just for fun, anything is ok'
    tags = jieba.analyse.extract_tags(test_sent, topK=5)
    print(",".join(tags))
    # just,fun,anything,ok,关键词

    # withWeight=True
    tags = jieba.analyse.extract_tags(test_sent, topK=5, withWeight=True)
    for tag in tags:
        print("tag: %-10s weight: %f" % (tag[0],tag[1]))
    # tag: just      weight: 1.086797
    # tag: fun       weight: 1.086797
    # tag: anything  weight: 1.086797
    # tag: ok        weight: 1.086797
    # tag: 关键词       weight: 0.809406
    ```
  - **jieba.analyse.set_idf_path** 自定义关键词提取所使用逆向文件频率 IDF 文本语料库
    ```py
    set_idf_path(idf_path)
    ```
    ```py
    # Download the IDF file
    import os
    idf_url = 'https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/idf.txt.big'
    idf_name = os.path.basename(idf_url)

    # Use urlretrieve
    from urllib.request import urlretrieve
    idf_path, _ = urlretrieve(idf_url, idf_name)

    # Or use keras
    from tensorflow import keras
    idf_path = keras.utils.get_file(idf_name, idf_url)

    !head -n 5 {idf_path}
    # 劳动防护 13.900677652
    # 勞動防護 13.900677652
    # 生化学 13.900677652
    # 生化學 13.900677652
    # 奥萨贝尔 13.900677652

    # Set user defined idf path
    jieba.analyse.set_idf_path(idf_path)

    tags = jieba.analyse.extract_tags(test_sent, topK=5)
    print(",".join(tags))
    # 一句,只是,用来,分析,just
    ```
  - **jieba.analyse.set_stop_words** 自定义关键词提取所使用停止词 Stop Words 文本语料库
    ```py
    set_stop_words(stop_words_path)
    ```
    ```py
    stw_url = 'https://raw.githubusercontent.com/dongxiexidian/Chinese/master/stopwords.dat'
    stw_path = keras.utils.get_file(os.path.basename(stw_url), stw_url)

    ll = [ii.strip() for ii in open(stw_path, 'r').readlines()]
    print(ll[195:200])  
    # ['不大', '不如', '不妨', '不定', '不对']
    print('只是' in ll, '一句' in ll)
    # True False

    jieba.analyse.set_stop_words(stw_path)
    jieba.analyse.set_idf_path(idf_path)

    tags = jieba.analyse.extract_tags(test_sent, topK=5)
    print(",".join(tags))
    # 一句,分析,just,fun,anything
    ```
## 基于 TextRank 算法的关键词抽取
  - [算法论文 TextRank: Bringing Order into Texts](http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
  - **基本思想**
    - 将待抽取关键词的文本进行分词
    - 以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
    - 计算图中节点的 PageRank，注意是无向带权图
  - **jieba.analyse.textrank** 使用 TextRank 算法州区关键词，注意默认过滤词性
    ```py
    textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False)
    ```
  - **jieba.analyse.TextRank** 新建自定义 TextRank 实例
    ```py
    class TextRank(jieba.analyse.tfidf.KeywordExtractor)
    __init__(self)
    ```
  - **示例**
    ```py
    import jieba
    import jieba.posseg
    import jieba.analyse

    test_sent = '随便一句话吧，只是为了用来分析关键词用，just for fun, anything is ok'
    for x, w in jieba.analyse.textrank(test_sent, withWeight=True):
        print('%s %s' % (x, w))
    # 用来 1.0
    # 分析 0.9966849915940917
    # 关键词 0.9929941828082526
    ```
## FOO
  ```py

  print('='*40)
  print('6. Tokenize: 返回词语在原文的起止位置')
  print('-'*40)
  print(' 默认模式')
  print('-'*40)

  result = jieba.tokenize('永和服装饰品有限公司')
  for tk in result:
      print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

  print('-'*40)
  print(' 搜索模式')
  print('-'*40)

  result = jieba.tokenize('永和服装饰品有限公司', mode='search')
  for tk in result:
  print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
  ```
## 词性标注
  - 标注句子分词后每个词的词性，采用和 ictclas 兼容的标记法
  - **jieba.posseg.POSTokenizer 类** 新建自定义分词器
    ```py
    class POSTokenizer(builtins.object)
    __init__(self, tokenizer=None)
    ```
    - **tokenizer** 可指定内部使用的 `jieba.Tokenizer` 分词器，`jieba.posseg.dt` 为默认词性标注分词器
  - **示例**
    ```py
    words = jieba.posseg.cut("我爱北京天安门")
    for word, flag in words:
        print('%s %s' % (word, flag))
    # 我 r
    # 爱 v
    # 北京 ns
    # 天安门 ns
    ```
## 并行分词
  - **原理** 将目标文本按行分隔后，把各行文本分配到多个 Python 进程并行分词，然后归并结果，从而获得分词速度的可观提升
  - 基于 python 自带的 **multiprocessing** 模块，目前暂不支持 Windows
  - **jieba.enable_parallel** / **jieba.disable_parallel** 开启 / 关闭并行分词模式，开启时参数可指定并行进程数
    ```py
    enable_parallel(processnum=None)
    disable_parallel()
    ```
    - 并行分词仅支持默认分词器 `jieba.dt` 和 `jieba.posseg.dt`
## Tokenize：返回词语在原文的起止位置
----------------------------------
* 注意，输入参数只接受 unicode
* 默认模式

```python
result = jieba.tokenize(u'永和服装饰品有限公司')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
```

```
word 永和                start: 0                end:2
word 服装                start: 2                end:4
word 饰品                start: 4                end:6
word 有限公司            start: 6                end:10

```

* 搜索模式

```python
result = jieba.tokenize(u'永和服装饰品有限公司', mode='search')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
```

```
word 永和                start: 0                end:2
word 服装                start: 2                end:4
word 饰品                start: 4                end:6
word 有限                start: 6                end:8
word 公司                start: 8                end:10
word 有限公司            start: 6                end:10
```


## ChineseAnalyzer for Whoosh 搜索引擎
--------------------------------------------
* 引用： `from jieba.analyse import ChineseAnalyzer`
* 用法示例：https://github.com/fxsjy/jieba/blob/master/test/test_whoosh.py

8. 命令行分词
-------------------

使用示例：`python -m jieba news.txt > cut_result.txt`

命令行选项（翻译）：

    使用: python -m jieba [options] filename

    结巴命令行界面。

    固定参数:
      filename              输入文件

    可选参数:
      -h, --help            显示此帮助信息并退出
      -d [DELIM], --delimiter [DELIM]
                            使用 DELIM 分隔词语，而不是用默认的' / '。
                            若不指定 DELIM，则使用一个空格分隔。
      -p [DELIM], --pos [DELIM]
                            启用词性标注；如果指定 DELIM，词语和词性之间
                            用它分隔，否则用 _ 分隔
      -D DICT, --dict DICT  使用 DICT 代替默认词典
      -u USER_DICT, --user-dict USER_DICT
                            使用 USER_DICT 作为附加词典，与默认词典或自定义词典配合使用
      -a, --cut-all         全模式分词（不支持词性标注）
      -n, --no-hmm          不使用隐含马尔可夫模型
      -q, --quiet           不输出载入信息到 STDERR
      -V, --version         显示版本信息并退出

    如果没有指定文件名，则使用标准输入。

`--help` 选项输出：
  ```py
  $> python -m jieba --help
  Jieba command line interface.

  positional arguments:
    filename              input file

  optional arguments:
    -h, --help            show this help message and exit
    -d [DELIM], --delimiter [DELIM]
                          use DELIM instead of ' / ' for word delimiter; or a
                          space if it is used without DELIM
    -p [DELIM], --pos [DELIM]
                          enable POS tagging; if DELIM is specified, use DELIM
                          instead of '_' for POS delimiter
    -D DICT, --dict DICT  use DICT as dictionary
    -u USER_DICT, --user-dict USER_DICT
                          use USER_DICT together with the default dictionary or
                          DICT (if specified)
    -a, --cut-all         full pattern cutting (ignored with POS tagging)
    -n, --no-hmm          don't use the Hidden Markov Model
    -q, --quiet           don't print loading messages to stderr
    -V, --version         show program's version number and exit

  If no filename specified, use STDIN instead.
  ```
延迟加载机制
------------

jieba 采用延迟加载，`import jieba` 和 `jieba.Tokenizer()` 不会立即触发词典的加载，一旦有必要才开始加载词典构建前缀字典。如果你想手工初始 jieba，也可以手动初始化。

    import jieba
    jieba.initialize()  # 手动初始化（可选）


在 0.28 之前的版本是不能指定主词典的路径的，有了延迟加载机制后，你可以改变主词典的路径:

    jieba.set_dictionary('data/dict.txt.big')

例子： https://github.com/fxsjy/jieba/blob/master/test/test_change_dictpath.py

其他词典
========
1. 占用内存较小的词典文件
https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.small

2. 支持繁体分词更好的词典文件
https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big

下载你所需要的词典，然后覆盖 jieba/dict.txt 即可；或者用 `jieba.set_dictionary('data/dict.txt.big')`


分词速度
=========
* 1.5 MB / Second in Full Mode
* 400 KB / Second in Default Mode
* 测试环境: Intel(R) Core(TM) i7-2600 CPU @ 3.4GHz；《围城》.txt

常见问题
=========

## 1. 模型的数据是如何生成的？

详见： https://github.com/fxsjy/jieba/issues/7

## 2. “台中”总是被切成“台 中”？（以及类似情况）

P(台中) ＜ P(台)×P(中)，“台中”词频不够导致其成词概率较低

解决方法：强制调高词频

`jieba.add_word('台中')` 或者 `jieba.suggest_freq('台中', True)`

## 3. “今天天气 不错”应该被切成“今天 天气 不错”？（以及类似情况）

解决方法：强制调低词频

`jieba.suggest_freq(('今天', '天气'), True)`

或者直接删除该词 `jieba.del_word('今天天气')`

## 4. 切出了词典中没有的词语，效果不理想？

解决方法：关闭新词发现

`jieba.cut('丰田太省了', HMM=False)`
`jieba.cut('我们中出了一个叛徒', HMM=False)`

**更多问题请点击**：https://github.com/fxsjy/jieba/issues?sort=updated&state=closed

修订历史
==========
https://github.com/fxsjy/jieba/blob/master/Changelog

# sklearn 文本聚类和文本分类
