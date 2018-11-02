# ___2018 - 03 - 16 Python Modules___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2018 - 03 - 16 Python Modules___](#2018-03-16-python-modules)
  - [目录](#目录)
  - [argparse 解析参数](#argparse-解析参数)
  	- [argparse 典型格式](#argparse-典型格式)
  	- [parse known args 跳过不能识别的参数](#parse-known-args-跳过不能识别的参数)
  	- [ArgumentParser 初始化参数](#argumentparser-初始化参数)
  	- [add argument 位置参数与可选参数](#add-argument-位置参数与可选参数)
  	- [add argument 参数行为 action](#add-argument-参数行为-action)
  	- [add argument 参数类型 type](#add-argument-参数类型-type)
  	- [add argument 参数个数 nargs](#add-argument-参数个数-nargs)
  	- [add argument group 参数组](#add-argument-group-参数组)
  	- [register 注册参数解析方法](#register-注册参数解析方法)
  	- [根据文件大小合并多个文件夹](#根据文件大小合并多个文件夹)
  - [logging](#logging)
  - [zipfile](#zipfile)
  	- [解压缩](#解压缩)
  	- [下载 zip 文件并解压到指定目录](#下载-zip-文件并解压到指定目录)
  - [多线程 multiprocessing threading joblib](#多线程-multiprocessing-threading-joblib)
  	- [multiprocessing](#multiprocessing)
  	- [threading](#threading)
  	- [多线程下载](#多线程下载)
  	- [joblib](#joblib)
  - [xml 解析](#xml-解析)
  	- [xml 文本](#xml-文本)
  	- [minidom](#minidom)
  	- [lxml](#lxml)
  	- [ElementTree](#elementtree)

  <!-- /TOC -->
***

# argparse 解析参数
## argparse 典型格式
  - **parse_arguments 在脚本中典型格式**
    ```python
    import argparse

    def foo(arg1, arg2):
        print("arg1 = %s, arg2 = %s" % (arg1, arg2))

    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("arg1")
        parser.add_argument("arg2")

        return parser.parse_args()

    if __name__ == "__main__":
        args = parse_arguments()
        foo(args.arg1, args.arg2)
    ```
    **运行结果**
    ```shell
    $ python arg_parse.py -h
    usage: arg_parse.py [-h] arg1 arg2

    positional arguments:
      arg1
      arg2

    optional arguments:
      -h, --help  show this help message and exit

    $ python arg_parse.py foo goo
    arg1 = foo, arg2 = goo
    ```
  - **指定解析其他参数列表** 默认使用 `sys.argv[1:]` 用于解析，也可以指定其他参数列表用于解析
    ```py
    import argparse
    import sys
    parser = argparse.ArgumentParser()

    parser.add_argument("--pylab", action='store_true')
    parser.add_argument("--foo")

    print(sys.argv[1:])
    # ['--pylab']
    print(parser.parse_args(sys.argv[1:]))
    # Namespace(foo=None, pylab=True)

    print(parser.parse_args('--foo goo'.split(' ')))
    # Namespace(foo='goo', pylab=False)
    ```
## parse known args 跳过不能识别的参数
  - **parse_arguments** 解析参数列表时，如果列表包含没有添加的参数会报错 `unrecognized arguments`
    ```py
    # error: unrecognized arguments: --goo
    parser.parse_args(['--goo'])
    ```
  - **parse_known_args** 同时返回成功解析的参数与未识别的参数，而不是报错
    ```py
    FLAGS, unparsed = parser.parse_known_args(['--goo'])
    print(FLAGS, unparsed)
    # Namespace(foo=None, pylab=False) ['--goo']
    ```
## ArgumentParser 初始化参数
  - **prog / usage / description / epilog** ArgumentParser 初始化参数
    ```python
    import argparse

    parser = argparse.ArgumentParser(
        prog="This function",
        usage="%(prog)s [options]",
        description="descript the function",
        epilog="Message after help message")

    print(parser.prog)
    # 'This function'
    print(parser.usage)
    # '%(prog)s [options]'
    print(parser.description)
    # 'descript the function'
    print(parser.epilog)
    # 'Message after help message'
    parser.print_help()
    # usage: This function [options]

    # descript the function

    # optional arguments:
    #   -h, --help  show this help message and exit

    # Message after help message
    ```
  - **formatter_class** 字符串格式化方式
    - **argparse.ArgumentDefaultsHelpFormatter** help 信息中添加打印默认值
    - **argparse.HelpFormatter** 默认的格式化方式
    - **argparse.MetavarTypeHelpFormatter** help 信息中显示的是参数类型
    - **argparse.RawDescriptionHelpFormatter** description 部分按照指定格式显示
    - **argparse.RawTextHelpFormatter**
    ```python
    # formatter_class=argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "First line.\n"
            "Second line."
        )
    )
    parser.print_help()
    # usage: ipython [-h]

    # First line. Second line.
    #
    # optional arguments:
    #   -h, --help  show this help message and exit
    ```
    ```python
    # formatter_class=argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description=(
          "First line.\n"
          "Second line."
      )
    )
    parser.print_help()
    # usage: ipython [-h]
    #
    # First line.
    # Second line.
    #
    # optional arguments:
    #   -h, --help  show this help message and exit
    ```
## add argument 位置参数与可选参数
  - **位置参数 Positional argument** 为必须的参数，按照参数位置指定，不能单独指定
  - **可选参数 Optional argument** 使用 `-` / `--` 指定
  - **required=True** 将可选参数变为必须参数
  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument("foo", help="Positional argument")
  parser.add_argument("--goo", help="Optional argument 1")
  parser.add_argument("-j", help="Optional argument 2")
  parser.add_argument("-k", "--koo", help="Optional argument 3")

  parser.print_help()
  # usage: ipython [-h] [--goo GOO] [-j J] [-k KOO] foo
  #
  # positional arguments:
  #   foo                Positional argument
  #
  # optional arguments:
  #   -h, --help         show this help message and exit
  #   --goo GOO          Optional argument 1
  #   -j J               Optional argument 2
  #   -k KOO, --koo KOO  Optional argument 3

  parser.parse_args(["these", "--goo", "are", "-j", "the", "-k", "args"])
  # Out[146]: Namespace(foo='these', goo='are', j='the', koo='args')

  parser.parse_args("this --goo is -jan --koo=arg".split())
  # Out[157]: Namespace(foo='this', goo='is', j='an', koo='arg')

  parser.parse_args("this --g is -jan --ko=arg".split())
  # Out[159]: Namespace(foo='this', goo='is', j='an', koo='arg')
  ```
## add argument 参数行为 action
  - **store** 默认操作，存储这个参数值
  - **store_const** 存储 const 指定的值
  - **store_false** / **store_true** 分别对应存储 True 和 False 值
  - **append** 保存为列表格式，将每个参数的值添加到这个列表
  - **append_const** 保存为列表，但是值必须是 const 指定参数的值
  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument("--foo", action='store_const', const=42, help="store_const")
  parser.add_argument("--goo", action='store_false', help="store_false")
  parser.add_argument("--joo", action='store_true', help="store_true")
  parser.add_argument("--koo", action='append', help="append")
  parser.add_argument("--loo", action='append_const', const=72, help="append_const int")
  parser.add_argument("--moo", action='append_const', const='abc', help="append_const str")

  parser.parse_args('--foo --goo --joo --koo 32 --koo 52 --koo 62 --loo --loo --moo'.split())
  # Out[17]: Namespace(foo=42, goo=False, joo=True, koo=['32', '52', '62'], loo=[72, 72], moo=['abc'])
  ```
## add argument 参数类型 type
  - **string** 默认参数类型
  - **FileType** 文件参数，可以指定文件 读写 等
  - **choices** 提供参数范围，如果提供的参数值不在这个范围之内会报错
  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument("--foo", type=int)
  parser.add_argument("--goo", type=str)
  parser.add_argument("--joo", type=float)
  parser.parse_args("--foo 32 --goo 32 --joo 32".split())
  # Out[25]: Namespace(foo=32, goo='32', joo=32.0)
  ```
  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=argparse.FileType('w'))
  parser.add_argument("--str", type=int, choices=range(5, 10))
  parser.parse_args("--out foo --str 8".split())
  # Out[28]: Namespace(out=<_io.TextIOWrapper name='foo' mode='w' encoding='UTF-8'>, str=8)
  ```
## add argument 参数个数 nargs
  - **整数 N** N 个从命令行中获取的参数将会组成一个列表
  - **?** 从命令行参数中获取一个值
  - **\*** 支持多个参数值
  - **+** 一个或多个参数
  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)

  parser.add_argument('--foo', type=int, nargs=2)
  parser.add_argument('--goo', nargs='*')
  parser.add_argument('--joo', type=str, nargs='+')

  parser.parse_args('foo foo --foo 1 2 --goo a b cd e --joo fgh i'.split())
  # Out[23]: Namespace(
  #     infile=<_io.TextIOWrapper name='foo' mode='r' encoding='UTF-8'>,
  #     outfile=<_io.TextIOWrapper name='foo' mode='w' encoding='UTF-8'>,
  #     foo=[1, 2], goo=['a', 'b', 'cd', 'e'], joo=['fgh', 'i'])
  ```
## add argument group 参数组
  ```python
  parser = argparse.ArgumentParser(description="Argparse practice")

  group_1 = parser.add_argument_group("Required arguments")
  group_1.add_argument("--foo", type=int, required=True, help="foo int")
  group_1.add_argument("--goo", type=float, required=True, help="goo float")

  group_2 = parser.add_argument_group("List arguments")
  group_2.add_argument("--joo", type=int, nargs="+", choices=range(0, 5), help="joo (0, 5) int list")
  group_2.add_argument("--koo", type=str, nargs="+", help="koo string list")
  ```
  运行结果
  ```python
  In [37]: parser.print_help()
  usage: ipython [-h] --foo FOO --goo GOO [--joo {0,1,2,3,4} [{0,1,2,3,4} ...]]
                 [--koo KOO [KOO ...]]

  Argparse practice

  optional arguments:
    -h, --help            show this help message and exit

  Required arguments:
    --foo FOO             foo int
    --goo GOO             goo float

  List arguments:
    --joo {0,1,2,3,4} [{0,1,2,3,4} ...]
                          joo (0, 100) int list
    --koo KOO [KOO ...]   koo string list

  In [38]: parser.parse_args("--foo 32 --goo 0.2 --joo 1 2 3 --koo a b cd".split())
  Out[38]: Namespace(foo=32, goo=0.2, joo=[1, 2, 3], koo=['a', 'b', 'cd'])
  ```
## register 注册参数解析方法
  - 默认的 `bool` 类型参数只要有值就会是 `True`
    ```py
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", type=bool)
    print(parser.parse_args('--foo False'.split(' ')))
    # Namespace(foo=True)
    print(parser.parse_args('--foo '.split(' ')))
    # Namespace(foo=False)
    ```
    可以使用 `parser.register` 注册新的 `"bool"` 类型参数解析方法
    ```py
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--goo", type="bool")
    print(parser.parse_args('--foo False --goo False'.split(' ')))
    # Namespace(foo=True, goo=False)
    ```
## 根据文件大小合并多个文件夹
  - [merge_folder_by_size.py](merge_folder_by_size.py)
***

# logging
  - **初始化设置**
    ```python
    import logging
    logging.basicConfig(level=logging.DEBUG)

    logging.debug('This message should appear on the console')
    logging.info('So should this')
    logging.warning('And this, too')

    # 输出到文件，需要新的执行环境
    logging.basicConfig(filename='./foo', level=logging.DEBUG)
    ```
  - **log 级别**
    ```python
    logging.getLevelName(10)
    # Out[13]: 'DEBUG'
    ```
    | NOTSET | DEBUG | INFO | WARN | ERROR | CRITICAL | FATAL |
    | ------ | ----- | ---- | ---- | ----- | -------- | ----- |
    | 0      | 10    | 20   | 30   | 40    | 50       | 50    |
  - **使用 logger**
    ```python
    import logging

    # basicConfig 添加 stdout 到 root 的 handlers
    logging.basicConfig()
    ll = logging.getLogger()
    logger.handlers
    # Out[7]: [<logging.StreamHandler at 0x7f212f9f6d90>]

    ll.level
    # Out[4]: 30

    ll.setLevel(logging.DEBUG)
    # Out[6]: 10

    ll.debug('aaa')
    ll.info('bbb')
    ll.warning('ccc')
    ll.error('ddd')
    ```
  - **Handler** 输出到文件或其他 stream
    ```python
    # 指定文件
    fh = logging.FileHandler("app.log")
    fh.setLevel(logging.INFO)

    # 指定标准错误
    sth = logging.StreamHandler(sys.stderr)
    sth.setLevel(logging.ERROR)

    # app 中添加 handler
    ll.addHandler(fh)
    ll.addHandler(sth)

    # 查看
    ll.handlers
    # Out[6]:
    # [<logging.StreamHandler at 0x7f212f9f6d90>,
    #  <logging.FileHandler at 0x7f212fa03ad0>,
    #  <logging.StreamHandler at 0x7f212d658550>]

    # 删除
    ll.removeHandler(ll.handlers[0])
    ll.removeHandler(ll.handlers[1])
    ll.handlers
    # Out[10]: [<logging.FileHandler at 0x7f212fa03ad0>]
    ```
***

# zipfile
## 解压缩
  ```py
  import zipfile

  fn = 'spa-eng.zip'
  ff = zipfile.ZipFile(fn, 'r')
  print(ff.namelist())  # ['_about.txt', 'spa.txt']

  # 解压文件，默认解压到当前文件夹下
  f.extractall(fn[0:-4])
  ```
## 下载 zip 文件并解压到指定目录
  ```py
  def get_zip_file(origin, file_path=None, extract=True):
      import requests
      import zipfile
      import io
      import os

      fname = os.path.basename(origin)
      if file_path:
          fname = os.path.join(file_path, fname)
      else:
          fname = os.path.join(os.environ['HOME'], '.keras/datasets', fname)

      if not os.path.exists(fname):
          resp = requests.get(origin)
          # ff = zipfile.ZipFile(io.BytesIO(resp.content), 'r')
          open(fname, 'wb').write(resp.content)

      if extract:
          ff = zipfile.ZipFile(fname, 'r')
          ff.extractall(fname[0:-4])

      return os.path.abspath(fname)

  path_to_zip = get_zip_file('http://www.manythings.org/anki/spa-eng.zip')
  ```
***

# 多线程 multiprocessing threading joblib
## multiprocessing
  - [multiprocessing — Process-based “threading” interface](https://docs.python.org/2/library/multiprocessing.html)
  - **multiprocessing.Pool** 进程池
    ```py
    from multiprocessing import Pool

    def addd(args):
        x = args[0]
        y = args[1]
        return np.add(x, y)

    p = Pool(5)
    print(p.map(addd, [(1, 2), (2, 3), (3, 4)]))
    # [3, 5, 7]
    ```
  - **multiprocessing.Process** 进程
    ```py
    from multiprocessing import Process, Value, Array

    def f(n, a):
        n.value = 3.1415927
        for i in range(len(a)):
            a[i] = -a[i]

    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value, arr[:])
    # 3.1415927 [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
    ```
## threading
  - **threading**
    ```py
    from threading import Thread

    def foo(aa):
        print('aa = {}'.format(aa))

    Thread(target=foo, args=((1, 2), )).start()
    # aa = (1, 2)
    ```
## 多线程下载
  - **threading**
    ```py
    from threading import Thread
    from urllib.request import urlretrieve

    class Downloader(Thread):
        def __init__(self, file_url, save_path):
            super(Downloader, self).__init__()
            self.file_url = file_url
            self.save_path = save_path

        def run(self):
            urlretrieve(self.file_url, self.save_path)

    Downloader("https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg", './foo.jpg').start()
    Downloader("https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg", './goo.jpg').start()
    ```
  - **multiprocessing.Pool**
    ```py
    from urllib.request import urlretrieve

    def image_downloader(aa):
        return urlretrieve(aa[1], aa[0])

    def multi_download(download_dict, thread_num=50):
        import time
        from multiprocessing import Pool

        dd = list(download_dict.items())
        pp = Pool(thread_num)
        print("Images need to download: {}".format(len(dd)))
        for ii in range(0, len(dd), thread_num):
            start = time.time()
            print('Downloading images {} - {}'.format(ii, ii + thread_num), end=', ')
            tt = dd[ii: ii + thread_num]
            pp.map(image_downloader, tt)
            print ('Time taken for downloading {} images: {:.2f} sec'.format(thread_num, time.time() - start))
    ```
  - **threading 与 tf.train.Coordinator 多线程协同**
    ```py
    def multi_download(download_dict, thread_num=50):
        import time
        import threading
        from urllib.request import urlretrieve
        import tensorflow as tf

        dd = list(download_dict.items())
        coord = tf.train.Coordinator()
        for ii in range(0, len(dd), thread_num):
            start = time.time()
            print('Downloading images {} - {}'.format(ii, ii + thread_num))
            tt = dd[ii: ii + thread_num]
            threads = [threading.Thread(target=urlretrieve, args=(iuu, imm)) for imm, iuu in tt]
            for itt in threads:
                itt.start()

            coord.join(threads)
            print ('Time taken for downloading {} images: {:.2f} sec\n'.format(thread_num, time.time() - start))
    ```
## joblib
  ```py
  def task(aa):
      print(aa)

  def joblib_loop():
      Parallel(n_jobs=4)(delayed(task)(i) for i in range(20))
  ```
***

# xml 解析
## xml 文本
  ```py
  xx = '''<?xml version="1.0"?>
  <data>
      <country name="Liechtenstein">
          <rank updated="yes">2</rank>
          <year>2008</year>
          <gdppc>141100</gdppc>
          <neighbor name="Austria" direction="E"/>
          <neighbor name="Switzerland" direction="W"/>
      </country>
      <country name="Singapore">
          <rank updated="yes">5</rank>
          <year>2011</year>
          <gdppc>59900</gdppc>
          <neighbor name="Malaysia" direction="N"/>
      </country>
      <country name="Panama">
          <rank updated="yes">69</rank>
          <year>2011</year>
          <gdppc>13600</gdppc>
          <neighbor name="Costa Rica" direction="W"/>
          <neighbor name="Colombia" direction="E"/>
      </country>
  </data>
  '''

  with open('./foo.xml', 'w') as ff:
      ff.write(xx)
  ```
## minidom
  ```py
  from xml.dom import minidom

  domobj = minidom.parse("./foo.xml")
  elementobj = domobj.documentElement
  subElementObj = elementobj.getElementsByTagName('country')

  tt = subElementObj[0]
  tt.childNodes[0].data
  ```
## lxml
  - **lxml.etree**
    ```py
    from lxml import etree
    tree = etree.parse("./foo.xml")
    root = tree.getroot()

    def node_text(node):
       result = node.text.strip() if node.text else ''
       for child in node:
           child_text = node_text(child)
           if child_text:
               result = result + ' %s' % child_text if result else child_text
       return result

    def node_text(node):
        result = ""
        for text in node.itertext():
            result = result + text
        return result

    for article in root:
        print("Tag name",article.tag)
        for field in article:
            if field.tag == 'country':
                print(field.tag,":",node_text(field))
            else:
                print(field.tag,":",field.text)
            print("")

    tree.write('./foo.xml', encoding='utf-8')
    ```
## ElementTree
  - [The ElementTree XML API](https://docs.python.org/3/library/xml.etree.elementtree.html)
  - **xml.etree.ElementTree** 模块在应对恶意结构数据时并不安全 [XML vulnerabilities](https://docs.python.org/3/library/xml.html#xml-vulnerabilities)
  - **Element 类型** 是一种灵活的容器对象，用于在内存中存储结构化数据，具有以下属性
    - **tag** 标签名，表示数据种类，string 类型
    - **attrib** 属性，表示该数据附加的属性，dictionary 类型
    - **text** 文本内容，string 类型
    - **tail** 文本内容，表示 element 闭合之后的文本，string 类型
    - **child elements** 子元素
    ```xml
    <tag attrib_name="attrib_value">
      text
      <child_element_1>
        child_element_text
      </child_element_1>
      child_element_tail
    </tag>
    ```
  - **import**
    ```py
    import xml.etree.ElementTree as ET
    ```
  - **ET.parse** 解析文件，返回 `ElementTree`
    ```py
    tree = ET.parse("foo.xml")
    root = tree.getroot()
    ```
  - **ET.fromstring / ET.tostring** 解析字符串并返回根节点的 `Element` / 将 `Element` 解析成字符串
    ```py
    xx = open("foo.xml").read()
    root = ET.fromstring(xx)
    ET.tostring(root)
    ```
  - **ElementTree 类**
    ```py
    tree = ET.ElementTree(file="foo.xml")
    root = tree.getroot()

    tree = ET.ElementTree()
    root = tree.parse('foo.xml')
    ```
  - **ET.dump** 打印 ElementTree / Element 结构
    ```py
    ET.dump(tree)
    ET.dump(root[0])
    ```
  - **Element 属性** tag / attrib / text / tail，子元素可以通过下标方式访问
    ```py
    print(root.tag) # data
    print(root.attrib)  # {}

    for child in root:
        print(child.tag, child.attrib)
    # country {'name': 'Liechtenstein'}
    # country {'name': 'Singapore'}
    # country {'name': 'Panama'}

    print(root[0][1].text)  # 2008
    ```
    递归遍历
    ```py
    def print_tree(root, rank=0):
        for tt in root:
            print('  ' * rank, tt.tag, ':', tt.attrib)
            print_tree(tt, rank=rank+1)

    print_tree(root)
    ```
  - **Element 遍历方法** find / findall / findtext / iter / iterfind / itertext / text / get
    - **iter** 遍历当前元素与所有子元素下的指定标签
    - **findall** 查找当前元素下的指定标签
    - **find** 查找当前元素下指定标签的第一个匹配
    - **text** 获取当前标签的文本内容
    - **tail** 获取当前标签的结尾文本内容
    - **get** 获取属性
    ```py
    # iter
    for neighbor in root.iter("neighbor"):
        print(neighbor.tag, ":", neighbor.attrib)
    ```
    ```py
    # findall / find
    for country in root.findall('country'):
        rank = country.find('rank').text
        name = country.get('name')
        print(name, rank)
    ```
    ```py
    # itertext 遍历所有文本
    print(''.join(list(root.itertext())))

    # text / tail 遍历所有文本
    def print_text_tail(root, rank=0):
        if root.text:
            print('  ' * rank + ':'.join([root.tag, root.text]))
        for tt in root:
            print_text_tail(tt, rank + 1)
        if root.tail:
            print('  ' * rank + ':'.join([root.tag, root.tail]))
    ```
  - **Element.get / Element.set** 获取 / 修改 xml 属性
    ```py
    # 将所有的rank值加1,并添加属性updated为yes
    for rank in root.iter("rank"):
        new_rank = int(rank.text) + 1
        rank.text = str(new_rank) # string
        rank.set("updated", "yes")
    ```
  - **Element.remove** 删除子元素
    ```py
    # remove all countries with a rank higher than 50
    for country in root.iter("country"):
        rank = int(country.find("rank").text)
        if rank > 50:
            root.remove(country)
    ```
  - **Element.append** / **Element.extend** / **Element.insert** 附加 / 扩展 / 插入 添加新元素
    ```py
    a = ET.Element('a', {'name': 'a'}, foo='goo')
    b = ET.Element('b')
    c = ET.Element('c')
    d = ET.Element('d')
    e = ET.Element('e')

    a.append(b)
    a.extend([c, d])
    a.insert(2, e)
    ET.dump(a)
    # <a foo="goo" name="a"><b /><c /><e /><d /></a>
    ```
  - **ET.SubElement** 创建新的子元素，实际会调用 `append`
    ```py
    a = ET.Element('a')
    b = ET.SubElement(a, 'b')
    c = ET.SubElement(a, 'c')
    d = ET.SubElement(c, 'd')

    ET.dump(a)
    # <a><b /><c><d /></c></a>
    ```
  - **text / tail 添加格式化字符** `ET.SubElement` 等方法添加新元素时没有添加缩进与换行的格式，可以通过 text / tail 添加
    ```py
    def add_sub_element(root, tag, rank, text=None, space_width=2):
        if rank < 1:
            print("Rank should be greater than 1")
            return

        spaces = ' ' * space_width
        if len(root) != 0:
            root[-1].tail = '\n' + spaces * rank
        else:
            if root.text == None:
                root.text = '\n' + spaces * rank
            else:
                root.text += spaces

        ee = ET.SubElement(root, tag)
        ee.tail = '\n' + spaces * (rank - 1)
        ee.text = '\n' + spaces * rank

        if text:
            ee.text = ee.text + '  ' + text + ee.text

        return ee
    ```
    ```py
    a = ET.Element('a')
    b = add_sub_element(a, 'b', 1)
    c = add_sub_element(a, 'c', 1)
    d = add_sub_element(c, 'd', 2)
    e = add_sub_element(b, 'e', 2, text='foo')
    ET.dump(a)
    # <a>
    #   <b>
    #     <e>
    #       foo
    #     </e>
    #   </b>
    #   <c>
    #     <d>
    #     </d>
    #   </c>
    # </a>
    ```
  - **ElementTree.write** 将 `ElementTree` 写入到文件，`Element` 可以使用 `ET.ElementTree()` 创建出 `ElementTree`
    ```py
    tree.write("note.xml", encoding="utf-8", xml_declaration=True)

    a = ET.Element('aa', name='aa')
    b = add_sub_element(a, 'bb', rank=1, text='boo')
    b.set('name', 'bb')
    c = add_sub_element(b, 'cc', rank=2, text='coo')
    d = add_sub_element(a, 'dd', rank=1, text='doo')

    tree = ET.ElementTree(a)
    tree.write("foo.xml", encoding="utf-8", xml_declaration=True)
    ```
    ```py
    ! cat foo.xml
    # <?xml version='1.0' encoding='utf-8'?>
    # <aa name="aa">
    #   <bb name="bb">
    #     boo
    #     <cc>
    #       coo
    #     </cc>
    #   </bb>
    #   <dd>
    #     doo
    #   </dd>
    # </aa>
    ```
  - **xml.dom.minidom.writexml** 指定格式保存 xml 文件
    ```py
    writexml(writer, indent='', addindent='', newl='', encoding=None)
    ```
    - **indent** 指定整体的缩进
    - **addindent** 指定子元素的缩进
    - **newl** 指定换行符
    ```py
    from xml.dom import minidom

    a = ET.Element('a')
    a.text = 'aoo'
    a.set('name', 'aa')
    b = ET.SubElement(a, 'b')
    b.set('name', 'bb')
    c = ET.SubElement(a, 'c')
    d = ET.SubElement(c, 'd')
    d.text = 'doo'

    ET.dump(a)
    # <a name="aa">aoo<b name="bb" /><c><d>doo</d></c></a>

    def save_xml(root, file_name, addindent='  ', newl='\n', encoding='utf-8'):
        xx = ET.tostring(root)
        dom = minidom.parseString(xx)
        with open(file_name, 'w') as ff:
            dom.writexml(ff, '', addindent, newl, encoding)

    save_xml(a, 'foo.xml')
    ```
    ```py
    ! cat foo.xml
    # <?xml version="1.0" encoding="utf-8"?>
    # <a name="aa">
    #   aoo
    #   <b name="bb"/>
    #   <c>
    #     <d>doo</d>
    #   </c>
    # </a>
    ```
***
