___Markdown Grammar___
======================
*** *** ***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___Markdown Grammar___](#markdown-grammar)
  - [目录](#目录)
  - [标题 Headers](#标题-headers)
  - [区块引用 Blockquotes](#区块引用-blockquotes)
  - [列表 Lists](#列表-lists)
  	- [无序列表](#无序列表)
  	- [有序列表](#有序列表)
  	- [HTML](#html)
  	- [结合 markdown 其他格式](#结合-markdown-其他格式)
  - [代码区块 Code Blocks](#代码区块-code-blocks)
  - [分隔线 Horizontal Rules](#分隔线-horizontal-rules)
  - [区段元素 Span Elements](#区段元素-span-elements)
  	- [行内式链接 inline links](#行内式链接-inline-links)
  	- [参考式链接 reference links](#参考式链接-reference-links)
  	- [范例 example](#范例-example)
  - [强调 Emphasis](#强调-emphasis)
  - [代码 Code](#代码-code)
  - [图片 Images](#图片-images)
  - [其它 Miscellaneous](#其它-miscellaneous)
  - [HTML](#html)
  - [目录跳转](#目录跳转)

  <!-- /TOC -->
***

# 标题 Headers
  - Markdown 支持两种标题的语法，**类 Setext** 和 **类 atx** 形式
  - **类 Setext** 形式是用底线的形式，利用 **= 最高阶标题** 和 **- 第二阶标题**，任何数量的 = 和 - 都可以有效果
    ```md
    This is an H1
    =============
    This is an H2
    -------------
    ```
  - **类 Atx** 形式则是在行首插入 **1 到 6 个 #** ，对应到标题 1 到 6 阶，可以在行尾加上 #，而行尾的 # 数量不用和开头一样
    ```md
    # This is another `H1`
    ## This is another **H2** with additional sharp ######
    ### _This is an italic H3_
    #### __This is a bold H4__
    ##### ___This is a bold italic H5___ and a normal H5
    ```
***

# 区块引用 Blockquotes
  - Markdown 标记区块引用是使用类似 email 中用 **>** 的引用方式
  - 区块引用可以嵌套，只要根据层次加上不同数量的 >
    ```md
    > This is a blockquote with two paragraphs
    > The second line
    > > blockquote in blockquote
    > > > blockquote in blockquote in blockquote

    ```
  - 引用的区块内也可以使用其他的 Markdown 语法，包括标题、列表、代码区块等
    ```md
    > #### This is a blockquote H4 header
    > - Markdown grammar works here
    > > ```python
    > > print('hello')
    > > ```
    >     return shell_exec("echo $input | $markdown_script");

    ```
    实际显示

    > #### This is a blockquote H4 header
    > - Markdown grammar works here
    > > ```python
    > > print('hello')
    > > ```
    >     return shell_exec("echo $input | $markdown_script");
***

# 列表 Lists
## 无序列表
  - **无序列表** 使用 **星号** / **加号** / **减号** 作为列表标记
    ```md
    *   Red
    *   Green
    *   Blue
    ```
  - 等同于
    ```md
    +   Red
    +   Green
    +   Blue
    ```
  - 也等同于
    ```md
    -   Red
    -   Green
    -   Blue
    ```
## 有序列表
  - **有序列表** 则使用数字接着一个英文句点
    ```md
    1.  Bird
    2.  McHale
    3.  Parish
    ```
  - 列表标记上使用的数字并不会影响输出的结果，如果列表标记写成
    ```md
    1.  Bird
    1.  McHale
    1.  Parish
    ```
    或甚至是
    ```md
    3. Bird
    1. McHale
    8. Parish
    ```
    会得到完全相同的 markdown 输出
## HTML
  - 列表所产生的 HTML 标记
    ```html
    <ol>
    <li>Bird</li>
    <li>McHale</li>
    <li>Parish</li>
    </ol>
    ```
  - 如果列表项目间用空行分开，在输出 HTML 时 Markdown 就会将项目内容用 <p> 标签包起来
    ```md
    *   Bird

    *   Magic
    ```
  - 会被转换为
    ```html
    <ul>
    <li><p>Bird</p></li>
    <li><p>Magic</p></li>
    </ul>
    ```
## 结合 markdown 其他格式
  - 列表项目可以包含多个段落，每个项目下的段落都必须缩进 4 个空格或是 1 个制表符
    ```md
    1.  This is a list item with two paragraphs

        This is the second one
    2.  Another item in the same list
    ```
  - 在列表项目内放进引用，那 > 就需要缩进
    ```md
    - list id
        > This is a blockquote
        >
        > inside a list item.

    ```
  - 列表项包含一个列表区块，该区块就需要缩进两次，也就是 8 个空格或是 2 个制表符
    ```md
    * list id

          <code goes here>
    ```
***

# 代码区块 Code Blocks
  - Markdown 会用 `<pre>` 和 `<code>` 标签来把代码区块包起来，照原来的样子显示代码区块
  - 在 Markdown 中建立代码区块，只要简单地缩进 4 个空格或是 1 个制表符就可以
    ```md
        This is a normal paragraph

            This is a code block
                And another line
            And another line
    ```
  - Markdown 会转换成对应的 HTML
    ```html
    <p>This is a normal paragraph</p>

    <pre><code>This is a code block
        And another line
    And another line
    </code></pre>
    ```
  - 在代码区块里面， & 、 < 和 > 会自动转成 HTML 实体，这样的方式可以非常容易使用 Markdown 插入范例用的 HTML 原始码
    ```md
        <div class="footer">
          &copy; 2004 Foo Corporation
        </div>
    ```
  - 会被转换为
    ```html
    <pre><code>&lt;div class="footer"&gt;
        &amp;copy; 2004 Foo Corporation
    &lt;/div&gt;
    </code></pre>
    ```
  - 代码区块中，一般的 Markdown 语法不会被转换，像是星号便只是星号
***

# 分隔线 Horizontal Rules
  - 在一行中用三个以上的 **星号** / **减号** **底线** 来建立一个分隔线，中间可以插入空格
    ```md
    * * *
    *****
    ** * **
    --- --- ---
    ___
    ---------------------------------------
    ```
***

# 区段元素 Span Elements
## 行内式链接 inline links
  - 要建立一个 **行内式的链接**，只要在 **方块括号** 后面紧接着 **圆括号** 并插入 **网址链接** 即可，如果想要加上链接的 **title** 文字，只要在网址后面，用双引号把 title 文字包起来即可
    ```md
    This is [an example](http://example.com/ "Title") inline link

    [This link](http://example.net/) has no title attribute
    ```
  - 会产生
    ```html
    <p>This is <a href="http://example.com/" title="Title">
    an example</a> inline link.</p>

    <p><a href="http://example.net/">This link</a> has no
    title attribute.</p>
    ```
  - 如果要链接到同样主机的资源，可以使用相对路径
    ```md
    See my [About](/about/) page for details
    ```
## 参考式链接 reference links
  - **参考式的链接** 是在链接文字的括号后面再接上 **另一个方括号**，而在第二个方括号里面要填入 **用以辨识链接的标记**，链接辨别标记 可以有字母、数字、空白和标点符号，不区分大小写
    ```md
    This is [an example][Id] reference-style link.
    ```
  - 在文件的任意处，可以把这个标记的链接内容定义出来 `` [id]: url for the link "Title" ``
    ```md
    [id]: http://example.com/  "Optional Title Here"
    ```
  - **Title** 可以使用的形式
    ```md
    [foo]: http://example.com/  "Optional Title Here"
    [foo]: http://examMarkdownple.com/  'Optional Title Here'
    [foo]: http://example.com/  (Optional Title Here)
    [foo]: http://example.com/longish/path/to/resource/here
        "Optional Title Here"
    ```
  - **链接** 可以使用尖括号 `` <> ``
    ```md
    [id]: <http://example.com/>  "Optional Title Here"
    ```
  - **隐式链接标记** 链接标记会视为等同于链接文字
    ```md
    [Google][]

    定义链接

    [Google]: http://google.com/
    ```
## 范例 example
  - **行内式**
    ```md
    I get 10 times more traffic from [Google](http://google.com/ "Google")
    than from [Yahoo](http://search.yahoo.com/ "Yahoo Search") or
    [MSN](http://search.msn.com/ "MSN Search").
    ```
  - **参考式**
    ```md
    I get 10 times more traffic from [Google] [1] than from
    [Yahoo] [2] or [MSN] [3].

      [1]: http://google.com/        "Google"
      [2]: http://search.yahoo.com/  "Yahoo Search"
      [3]: http://search.msn.com/    "MSN Search"
    ```
  - **参考式隐式链接**
    ```md
    I get 10 times more traffic from [Google][] than from
    [Yahoo][] or [MSN][].

      [google]: http://google.com/        "Google"
      [yahoo]:  http://search.yahoo.com/  "Yahoo Search"
      [msn]:    http://search.msn.com/    "MSN Search"
    ```
***

# 强调 Emphasis
  - Markdown 使用星号 `*` 和底线 `_` 作为标记强调字词的符号，被 `*` 或 `_` 包围的字词会被转成用 `<em>` 标签包围，用两个 `*` 或 `_` 包起来的话，则会被转成 `<strong>`
    ```md
    *single asterisks*

    _single underscores_

    **double asterisks**

    __double underscores__

    ___triple underscores___

    __*mix asterisks and underscores*__
    ```
  - 会转成
    ```html
    <em>single asterisks</em>

    <em>single underscores</em>

    <strong>double asterisks</strong>

    <strong>double underscores</strong>

    <strong><em>triple underscores</em></strong>

    <strong><em>mix asterisks and underscores</em></strong>
    ```
  - 实际显示

    *single asterisks*

    _single underscores_

    **double asterisks**

    __double underscores__

    ___triple underscores___

    __*mix asterisks and underscores*__
***

# 代码 Code
  - 如果要标记一小段行内代码，可以用反引号 `` ` ``
    ```md
    Use the `printf()` function.
    ```
    会产生
    ```html
    <p>Use the <code>printf()</code> function.</p>
    ```
  - 如果要在代码区段内插入反引号，可以用多个反引号来开启和结束代码区段，代码区段的起始和结束端都可以放入一个空白
    ```md
    `` There is a literal backtick (`) here. ``

    ``
    There is another literal backtick (`) here.
    ``
    ```
  - 在代码区段内，& 和方括号都会被自动地转成 HTML 实体，这使得插入 HTML 原始码变得很容易
    ```md
    Please don't use any `<blink>` tags.
    ```
    转为
    ```html
    <p>Please don't use any <code>&lt;blink&gt;</code> tags.</p>
    ```
***

# 图片 Images
  - Markdown 使用一种和链接很相似的语法来标记图片，同样也允许两种样式 **行内式** 和 **参考式**
  - 行内式的图片语法 `` ! [图片的替代文字](图片的网址 "title") ``
    ```md
    ![Alt text](/path/to/img.jpg)

    ![Alt text](/path/to/img.jpg "Optional title")
    ```
  - 参考式的图片
    ```md
    ![Alt text][id]
    ```
    id 是图片参考的名称，图片参考的定义方式则和连结参考一样
    ```md
    [id]: url/to/image  "Optional title attribute"
    ```
***

# 其它 Miscellaneous
  - **自动链接 Automatic Links** Markdown 支持以比较简短的自动链接形式来处理网址和电子邮件信箱，只要是用方括号包起来
    ```md
    <http://example.com/>
    ```
    Markdown 会转为
    ```html
    <a href="http://example.com/">http://example.com/</a>
    ```
  - **反斜杠 Backslash Escapes** 转义插入普通字符
    ```md
    \*literal asterisks\*
    ```
***

# HTML
  - Markdown兼容 HTML，只对应 HTML 标记的一小部分，不在 Markdown 涵盖范围之内的标签，都可以直接在文档里面用 HTML 撰写
    ```html
    <table>
        <tr>
            <td>Foo</td>
        </tr>
    </table>
    ```
  - 在 HTML 文件中，有两个字符需要特殊处理： < 和 &，Markdown 可以自动转换 htnl 特殊字符
    - **<** 符号用于起始标签
    - **&** 符号则用于标记 HTML 实体
  - 如果使用的 ``&`` 字符是 HTML 字符实体的一部分，它会保留原状，否则它会被转换成 ``&amp;``，如要在文档中插入一个版权符号 ``©``，可以这样写
    ```md
    &copy;
    ```
    Markdown 会保留它不动，而若写
    ```md
    AT&T
    ```
    Markdown 就会将它转为
    ```md
    AT&amp;T
    ```
  - 类似的状况也会发生在 < 符号上
    ```md
    4 < 5
    ```
    Markdown 将会把它转换为
    ```md
    4 &lt; 5
    ```
***

# 目录跳转
  - **In Github**
    ```md
    ## My paragraph title
    ```
    will produce the following anchor `user-content-my-paragraph-title`, so you can reference it with
    ```md
    [Some text](#my-paragraph-title)
    ```
    However, I haven't found official documentation for this.
  - **span id testing area**
    * [1. ID 1 toggle point](#1)
    * [2. span id toggle here](#toggle-here)
    * [Some text](#my-paragraph-title)
    * [Some text 2](#2)
    * [Some text 3](#3)

    ### <span id="jump">Hello World</span>
    ## <span id="tith">This is another span id tith</span>
    ## <span id="toggle-here">span id toggle here

    - <span id="toggle-point-1">[span id toggle here 1](#toggle-point-2)

    <h2 id="1">1. ID 1 toggle point</h2>

    ## My paragraph title
    ```md
    Nothing, just place holder



    ```

    [跳转到Hello World](#jump)

    Take me to [tith](#tith)

    ## 2. Another my paragraph title
    ## 3. A third my paragraph title

    - <span id="toggle-point-2">[span id toggle here 2](#toggle-point-1)

    ```md
    Nothing, just place holder




















    ```
***
