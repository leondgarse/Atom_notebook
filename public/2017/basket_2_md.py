#!/usr/bin/python3
#Filename basket_2_md.py

import sys, getopt
import re

opts, args = getopt.getopt(sys.argv[1:], "hcCi:o:")

print('sys.argv[0] = ', sys.argv[0])
print('opts = ', opts)
print('args = ', args)

filename = ""
output = 'foo.md'
categories = True

for op, value in opts:
    if op == '-i':
        filename = value
    elif op == '-o':
        output = value
    elif op == '-c':
        categories = True
    elif op == '-C':
        categories = False
    elif op == "-h":
        print('usage: %s filename output categories' %(sys.argv[0]))
        sys.exit()

# <span style="color:#ff0000;">ipython</span>
## <span style="color:#ff8000;">shell命令和别名</span>
### <span style ="color:#00ff00;">kmeans</span>
reg_h1 = re.compile(r'(<span style=") font-weight:600; (color:\#ff0000;">)(.*)</span></span>.*')
reg_h2 = re.compile(r'(<span style=") font-weight:600; (color:\#ff8000;">)(.*)</span></span>.*')
reg_h3 = re.compile(r'(<span style=") font-weight:600; (color:\#00ff00;">)(.*)</span></span>.*')

# 图片 <table class="note">.*<img src="02-06 python.html_files/data/Selection_017.png" width=
reg_img = re.compile(r'<table class="note">.*<img src="(.*)" width=')

# 块 <table class="note">
reg_table_mark = re.compile(r'<table class="note">')

# 空行 <br /></p>
reg_cont_blank = re.compile(r'<p style=".*;"><br /></p>')

# 内容 <p style=";">整数： 2</p>
# >'for the love of a princess'</span>
reg_cont = re.compile(r'<p style=".*?;">(.*)</[span]*>')
# reg_cont_end = re.compile(r'<p style=".*;">(.*)</span*>')

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

    if (categories == False):
        lev = [1, 2, 3]
    else:
        lev = [1, 1, 1]

    for l in lines:
        if (len(reg_h1.findall(l)) != 0):
            item = reg_h1.findall(l)
            level = lev[0]
            str_t = ('\n' + ' ' * (level-1) + '# ' + item[0][0] + item[0][1] + item[0][2])
        elif (len(reg_h2.findall(l)) != 0):
            item = reg_h2.findall(l)
            level = lev[1]
            str_t = ('\n' + ' ' * (level-1) + '## ' + item[0][0] + item[0][1] + item[0][2])
        elif (len(reg_h3.findall(l)) != 0):
            item = reg_h3.findall(l)
            level = lev[2]
            str_t = ('\n' + ' ' * (level-1) + '### ' + item[0][0] + item[0][1] + item[0][2])
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

            print('str_t = ', str_t)
            if (len(reg_comment.findall(str_t)) != 0):  # 包含注释的行
                str_t = reg_comment.findall(str_t)[0]
                str_t = str_t[0] + str_t[1]

            if (len(reg_comment.findall(str_t)) != 0):  # again 包含注释的行
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
                str_t = (' ' * level + '  ' + str_t)
            else:
                if (str_t.strip().startswith('#') and str_t.startswith(' ' * 8) != True):
                    str_t = str_t.replace('#', '\#', 1)

                if ((table_flag == True) and (str_t.startswith(' ') == True)):  # 区块的首行有缩进
                    table_flag = False
                    f_md.write('\n')

                if (table_flag == True):  # 没有缩进的区块的首行，添加项目编号
                    table_flag = False
                    table_line_flag = True
                    str_t = (' ' * level + '- ' + str_t)
                elif (str_t.startswith(' ' * 8)): # Tab键缩进的一行
                    if (table_line_flag == True): # 上一行没有缩进，行首添加一个换行符
                        table_line_flag = False
                        str_t = ('\n' + ' ' * level + '  '  + ' ' * 4 + str_t[8:])
                    else:
                        str_t = (' ' * level + '  '  + ' ' * 4 + str_t[8:])
                elif (str_t.startswith(' ' * 2)): # 少于一个Tab建缩进的一行
                    str_t = (' ' * level + '  '  + ' ' * 4 + str_t.strip())
                else: # 区块中没有缩进的一行
                    if (table_line_flag == True): # 上一行也是没有缩进的行
                        str_t = (' ' * level + '- ' + str_t)
                    else:
                        str_t = (' ' * level + '  ' + str_t.strip())
                    table_line_flag = True

        else:
            continue

        f_md.write(str_t + '\n')

basket_html2md(filename, output, categories)

