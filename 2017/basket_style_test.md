# basket --> markdown --> calibre mobi
 - basket export html web page
 - 以下python程序 python3 basket_2_md.py 将html转化为markdown
 - atom 打开markdown，背景选择白色，save as html
 - calibre add books添加转化后的html
 - convert books，TOC指定自动生成标题，并使用
   - //h:h1
   - //h:h2
   - //h:h3

# <span style="color:#ff0000;">Basket Notebook
``` python
- 注释
- 输出
- tab / > &nbsp;&gt;&gt;&gt; type(3)
- &quot;
- &gt;
- &nbsp;
- &quot;  --> "
- &amp;   --> &


- %s/&gt;&gt;&gt; //g
- %s/&gt;/>/g
- %s/&nbsp;//g
- %s/&quot;/"/g
- %s/&lt;/</g

str_t = str_t.replace('&gt;&gt;&gt; ', '')
str_t = str_t.replace('&gt;', '>')
str_t = str_t.replace('&nbsp;', '')
str_t = str_t.replace('&quot;', '""')
str_t = str_t.replace('&lt;', '<')

标题，不缩进
内容，缩进2
区块，判断第一行开头
    不是空格，添加添加编号‘- ’，判断上一行是否有缩进，有的话插入新行```
    是空格，插入行```
    一个Tab键，8个空格，缩进 = 标题级别 + 2(区块级别) + 4缩进成块
```

```python
import re

# <span style="color:#ff0000;">ipython</span>
## <span style="color:#ff8000;">shell命令和别名</span>
### <span style ="color:#00ff00;">kmeans</span>
reg_h1 = re.compile(r'(<span style=") .*font-weight:600; (color:\#ff0000;">)(.*)</span></span>.*')
reg_h2 = re.compile(r'(<span style=") .*font-weight:600; (color:\#ff8000;">)(.*)</span></span>.*')
reg_h3 = re.compile(r'(<span style=") .*font-weight:600; (color:\#00ff00;">)(.*)</span></span>.*')

# 图片 <table class="note">.*<img src="02-06 python.html_files/data/Selection_017.png" width=
reg_img = re.compile(r'<table class="note">.*<img src="(.*)" width=')

# 块 <table class="note">
reg_table_mark = re.compile(r'<table class="note">')

# 空行 <br /></p>
reg_cont_blank = re.compile(r'<p style=".*;"><br /></p>')

# 内容 <p style=";">整数： 2</p>
# >'for the love of a princess'</span></td></tr></table></td>
reg_cont = re.compile(r'<p style=".*?;">(.*)</[span]*>')
reg_cont_end = re.compile(r'<p style=".*;">(.*)</span></td></tr></table></td>')

# 注释
reg_comment = re.compile(r'(.*)<span style=" .*;">(.*)')

# 链接
reg_link = re.compile(r'(.*)<a href=.*>(.*)</a>(.*)')

def basket_html2md(filename, output = 'foo.md', categories = True):
    ''' filename 输入文件名
        output 输出文件名
        categories 是否创建完整目录
    '''
    basic_f = open(filename, 'r')
    lines = basic_f.readlines()
    basic_f.close()
    f_md = open(output, 'w')

    level = 0           # 标题下面的内容缩进空格数
    table_flag = False  # 是否是一个区块的开始
    table_line_flag = False # 上一行是否是没有缩进的一行
    code_flag = False   # 是否是代码块
    block_flag = False  # 是否已添加区块标记'```'
    cont_table_flag = False # 是否包含表格

    if (categories == False):
        lev = [2, 3, 4]
    else:
        lev = [2, 2, 2]

    for l in lines:
        if (len(reg_h1.findall(l)) != 0):
            item = reg_h1.findall(l)
            level = lev[0]
            str_t = ('***' + '\n\n' + ' ' * (level-2) + '# ' + item[0][0] + item[0][1] + item[0][2])
        elif (len(reg_h2.findall(l)) != 0):
            item = reg_h2.findall(l)
            level = lev[1]
            str_t = (' ' * (level-2) + '## ' + item[0][0] + item[0][1] + item[0][2])
        elif (len(reg_h3.findall(l)) != 0):
            item = reg_h3.findall(l)
            level = lev[2]
            str_t = (' ' * (level-2) + '### ' + item[0][0] + item[0][1] + item[0][2])
        elif (len(reg_img.findall(l)) != 0):
            str_t = reg_img.findall(l)[0]
            str_t = str_t.rsplit('/')[-1]
            str_t = ('\n' + ' ' * level + '![](images/' + str_t + ')')
        elif (len(reg_table_mark.findall(l)) != 0):
            table_flag = True
            continue
        elif (len(reg_cont_blank.findall(l)) != 0):
            str_t = (' ')
        elif (len(reg_cont.findall(l)) != 0):
            str_t = reg_cont.findall(l)[0]

            str_t = str_t.replace('&gt;&gt;&gt; ', '')
            str_t = str_t.replace('&gt;', '>')
            str_t = str_t.replace('&nbsp;', '')
            str_t = str_t.replace('&quot;', '"')
            str_t = str_t.replace('&lt;', '<')
            # str_t = str_t.replace(' _', ' \_')
            # str_t = str_t.replace('\'_', '\'\\_')
            # str_t = str_t.replace(' *', ' \*')
            # str_t = str_t.replace('`', '\`')
            str_t = str_t.replace('</span>', '')
            str_t = str_t.replace('<span style=" color:#333333;">', '')

            str_t = str_t.replace('**kwargs', '** kwargs')

            print('str_t = ', str_t)
            while (len(reg_comment.findall(str_t)) != 0):  # 包含注释的行
                str_t = reg_comment.findall(str_t)[0]
                str_t = str_t[0] + str_t[1]

            if (len(reg_link.findall(str_t)) != 0):  # 包含链接
                str_t = reg_link.findall(str_t)[0]
                str_t = str_t[0] + str_t[1] + str_t[2]

            if (str_t.startswith('```python')): # 代码块
                code_flag = True

            if (code_flag == True):
                if (str_t.endswith('```')):
                    code_flag = False
                str_t = (' ' * level + str_t)
            else:
                if (str_t.strip().startswith('#') and str_t.startswith(' ' * 8) != True):
                    str_t = str_t.replace('#', '\#', 1)

                # 区块的首行有缩进，不添加项目编号
                if ((table_flag == True) and (str_t.startswith(' ') == True)):
                    table_flag = False
                    table_line_flag = False # 当作是有缩进的一行
                    f_md.write(' ' * level + '  ' + '```python\n')
                    block_flag = True

                if (table_flag == True):  # 没有缩进的区块的首行，添加项目编号
                    table_flag = False
                    table_line_flag = True
                    str_t = (' ' * level + '- ' + str_t)

                elif (str_t.startswith(' ' * 8)): # Tab键缩进的一行
                    if (table_line_flag == True): # 上一行没有缩进
                        table_line_flag = False
                        f_md.write(' ' * level + '  ' + '```python\n')
                        block_flag = True
                    str_t = (' ' * level + '  ' + str_t[8:])
                elif (str_t.startswith(' ' * 2)): # 少于一个Tab建缩进的一行
                    str_t = (' ' * level + '  '  + ' ' * 4 + str_t.strip())
                else: # 区块中没有缩进的一行
                    if (table_line_flag == True): # 上一行也是没有缩进的行
                        str_t = (' ' * level + '- ' + str_t)
                    else:
                        f_md.write(' ' * level + '  ' + '```\n')
                        block_flag = False
                        str_t = (' ' * level + '  ' + str_t.strip())
                    table_line_flag = True

        else:
            continue

        f_md.write(str_t + '\n')
        # 区块的结尾，且还没有添加标记'```'
        if (block_flag == True) and (len(reg_cont_end.findall(l)) != 0):
            f_md.write(' ' * level + '  ' + '```\n')
            block_flag = False
```
sed -i 's/^\(#\+\) \(.*\)$/\1 <span id="\2">\2/' 08-06_sql_table_test.md
