# ___2018 - 08 - 01 Flask Quick Start___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2018 - 08 - 01 Flask___](#2018-08-01-flask)
  - [目录](#目录)
  - [链接](#链接)
  - [Flask Quick Start](#flask-quick-start)
  	- [Hello world](#hello-world)
  	- [Debug 模式](#debug-模式)
  	- [路由 Routing](#路由-routing)
  	- [路径 Variable Rules](#路径-variable-rules)
  	- [构造 URL Building](#构造-url-building)
  	- [HTTP 方法 methods](#http-方法-methods)
  	- [静态文件 Static Files](#静态文件-static-files)
  	- [模板生成 Rendering Templates](#模板生成-rendering-templates)
  	- [Request 对象](#request-对象)
  	- [request headers 指定不同请求类型](#request-headers-指定不同请求类型)
  	- [文件上传 File Uploads](#文件上传-file-uploads)
  	- [Cookies](#cookies)
  	- [重定向和错误 Redirects and Errors](#重定向和错误-redirects-and-errors)
  	- [errorhandler 错误请求处理](#errorhandler-错误请求处理)
  	- [响应处理 Response](#响应处理-response)
  	- [Response 返回值 jsonify](#response-返回值-jsonify)
  	- [用户会话 Sessions](#用户会话-sessions)
  	- [LOGGING](#logging)
  - [Jinja 模板简介](#jinja-模板简介)
  	- [模板标签](#模板标签)
  	- [继承](#继承)
  	- [控制流](#控制流)

  <!-- /TOC -->
***

# 链接
  - [Manual -- curl usage explained](https://curl.haxx.se/docs/manual.html)
  - [Welcome to Flask — Flask documentation](http://flask.pocoo.org/docs/)
  - [Flask 中文文档](http://dormousehole.readthedocs.io/)
  - [欢迎进入 Flask 大型教程项目](http://www.pythondoc.com/flask-mega-tutorial/)
***

# Flask Quick Start
## Hello world
  - **Flask 脚本** hello_world.py
    ```python
    import flask

    app = flask.Flask(__name__)

    @app.route("/")
    def hello_world():
        return "Hello world"

    if __name__ == "__main__":
        app.run()
    ```
  - **运行** shell 执行
    ```shell
    $ python hello_world.py
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```
  - **curl 测试**
    ```shell
    $ curl http://127.0.0.1:5000/
    Hello world
    ```
  - **requests 测试**
    ```python
    import requests
    requests.get("http://127.0.0.1:5000/").text
    # Out[8]: 'Hello world'
    ```
## Debug 模式
  - **Debug 模式** 修改脚本后，flask 自动重启
  - 通过环境变量指定打开 DEBUG 模式
    ```shell
    $ export FLASK_DEBUG=1
    $ python hello_world.py
    ```
  - 在 app 启动时指定 DEBUG 模式
    ```python
    app.run(debug=True)
    ```
## 路由 Routing
  - 通过 `@app.route(<route>)` 指定
    ```python
    @app.route("/")
    def index():
        return "Index page"

    @app.route("/hello")
    def hello():
        return "hello world"
    ```
## 路径 Variable Rules
  - URL 添加一个路径变量 `/path/<converter:varname>`，其中 converter 是可选的转化器

    | 转换器 | 作用                                     |
    | ------ | ---------------------------------------- |
    | string | 默认选项，接受除了斜杠之外的字符串       |
    | int    | 接受整数                                 |
    | float  | 接受浮点数                               |
    | path   | 和 string 类似，不过可以接受带斜杠的字符串 |
    | any    | 匹配任何一种转换器                       |
    | uuid   | 接受UUID字符串                           |
  - 示例
    ```python
    import flask
    app = flask.Flask(__name__)

    @app.route("/user/<username>")
    def shou_user_profile(username):
        return "User: %s" % (username)

    @app.route("/post/<int:post_id>")
    def show_post(post_id):
        return "Post: %d" % (post_id)

    if __name__ == "__main__":
        app.run()
    ```
    ```python
    requests.get("http://127.0.0.1:5000/user/aaa").text
    # Out[11]: 'User: aaa'

    requests.get("http://127.0.0.1:5000/post/345").text
    # Out[13]: 'Post: 345'
    ```
## 构造 URL Building
  - **url_for** 通过 `url_for(<function name>)` 获取对应函数的 URL 链接
    ```python
    from flask import Flask, url_for
    app = Flask(__name__)

    @app.route("/")
    def index(): pass

    @app.route("/login")
    def login(): pass

    @app.route("/user/<username>")
    def profile(username): pass

    with app.test_request_context():
        print(url_for("index"))
        print(url_for("login"))
        print(url_for("login", aaa="bbb"))
        print(url_for("profile", username="foo"))
    ```
    **运行结果**
    ```python
    /
    /login
    /login?aaa=bbb
    /user/foo
    ```
## HTTP 方法 methods
  - 默认的 HTTP 方法 是 GET
  - 通过指定 `app.route` 修饰符的 `methods` 参数，添加支持的 HTTP 方法
  - **常用方法**
    - **GET** 获取页面上存储的数据
    - **HEAD** 只获取 headers，不返回实际内容
    - **POST** 向 URL 推送数据，服务器应该存储该数据，并只存储一次
    - **PUT** 类似于 **POST**，但可以存储多次
    - **DELETE** 删除指定位置的数据
    - **OPTIONS** 获取 URL 支持的方法，自动实现
  - **Flask 脚本** echo_method.py
    ```python
    #!/usr/bin/env python
    from flask import Flask, request

    app = Flask(__name__)

    @app.route("/echo", methods=["GET", "POST", "DELETE"])
    def api_echo():
        if request.method == "GET":
            return "ECHO: GET\n"
        if request.method == "POST":
            return "ECHO: POST\n"
        if request.method == "DELETE":
            return "ECHO: DELETE\n"

    if __name__ == "__main__":
        app.run()
    ```
  - **运行** shell 执行
    ```shell
    $ python echo_method.py
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```
  - **测试**
    ```shell
    # curl 测试
    $ curl http://127.0.0.1:5000/echo
    ECHO: GET
    $ curl -X POST http://127.0.0.1:5000/echo
    ECHO: POST
    $ curl -X DELETE http://127.0.0.1:5000/echo
    ECHO: DELETE
    ```
    ```python
    # requests 测试
    import requests
    requests.get("http://127.0.0.1:5000/echo").text
    # Out[11]: 'ECHO: GET\n'

    requests.post("http://127.0.0.1:5000/echo").text
    # Out[12]: 'ECHO: POST\n'

    requests.delete("http://127.0.0.1:5000/echo").text
    # Out[15]: 'ECHO: DELETE\n'
    ```
  - **OPTIONS 获取支持的方法**
    ```python
    resp = requests.options("http://127.0.0.1:5000/echo")
    resp.headers['Allow']
    # Out[81]: 'HEAD, DELETE, OPTIONS, POST, GET'
    ```
## 静态文件 Static Files
  - **处理静态文件** 使用 `url_for` 并指定 `static 端点名` 和 `文件名`，实际的文件应放在 `static/` 文件夹下
    ```python
    url_for('static', filename='style.css')
    ```
## 模板生成 Rendering Templates
  - Flask 默认使用 **Jinja2** 作为模板，Flask 会自动配置 **Jinja** 模板
  - 默认情况下，模板文件需要放在 **templates** 文件夹下
  - **render_template 函数** 配置使用 Jinja 模板，需要传入 **模板文件名** 和 **模板需要的参数名**
    ```python
    from flask import Flask, render_template

    app = Flask(__name__)

    @app.route('/hello/')
    @app.route('/hello/<name>')
    def hello(name=None):
        return render_template('hello.html', name=name)

    if __name__ == "__main__":
        app.run()
    ```
    Flask 默认到 **templates 文件夹** 下寻找模板
    ```shell
    .
    ├── application.py
    └── templates
        └── hello.html
    ```
    **模板文件** hello.html
    ```html
    <!doctype html>
    <title>Hello from Flask</title>
    {% if name %}
      <h1>Hello {{ name }}!</h1>
    {% else %}
      <h1>Hello, World!</h1>
    {% endif %}
    ```
    **测试运行**
    ```shell
    $ python application.py
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

    $ curl http://127.0.0.1:5000/hello/aaa
    <!doctype html>
    <title>Hello from Flask</title>

     <h1>Hello aaa!</h1>

    $ curl http://127.0.0.1:5000/hello/
    <!doctype html>
    <title>Hello from Flask</title>

     <h1>Hello, World!</h1>
    ```
## Request 对象
  - **Request 对象** 是一个全局对象，利用它的属性和方法，可以方便的获取从页面传递过来的参数
  - **method 属性** 返回 HTTP 方法的类型，如 post / get
  - **form 属性** 是一个字典，获取 POST 类型的表单
    ```python
    @app.route('/login', methods=['POST', 'GET'])
    def login():
        error = None
        if request.method == 'POST':
            if valid_login(request.form['username'],
                           request.form['password']):
                return log_the_user_in(request.form['username'])
            else:
                error = 'Invalid username/password'
        # the code below is executed if the request method
        # was GET or the credentials were invalid
        return render_template('login.html', error=error)
    ```
  - **args 属性** 用于获取 URL 中 `?key=value` 的值
    ```python
    searchword = request.args.get('key', '')
    ```
    **Flask 脚本** hello_args.py
    ```python
    #!/usr/bin/env python
    from flask import Flask, request

    app = Flask(__name__)

    @app.route("/hello")
    def api_hello():
        if "name" in request.args:
            return "Hello " + request.args.get("name", "")
        else:
            return "Hello, No one"

    if __name__ == "__main__":
        app.run()
    ```
    **测试运行** shell 执行
    ```shell
    $ python hello_args.py
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

    # curl 测试
    $ curl http://127.0.0.1:5000/hello
    Hello, No one

    $ curl http://127.0.0.1:5000/hello?name=world
    Hello world

    $ curl -G http://127.0.0.1:5000/hello -d "name=world"
    Hello world
    ```
    ```python
    # requests 测试
    import requests
    requests.get("http://127.0.0.1:5000/hello").text
    # Out[5]: 'Hello, No one'

    requests.get("http://127.0.0.1:5000/hello?name=world").text
    # Out[6]: 'Hello world'

    requests.get("http://127.0.0.1:5000/hello", params={"name": "world"}).text
    # Out[7]: 'Hello world'
    ```
## request headers 指定不同请求类型
  - **Flask 脚本** headers.py
    ```python
    #!/usr/bin/env python
    from flask import Flask, request, json

    app = Flask(__name__)

    @app.route("/messages", methods=["POST"])
    def api_messages():
        if request.headers["content-type"] == "text/plain":
            return "Text Message: " + request.data.decode("ascii")
        elif request.headers["Content-type"] == "application/json":
            return "JSON Message: " + json.dumps(request.json)
        elif request.headers["content-Type"] == "application/octet-stream":
            with open("./binary", "wb") as ff:
                ff.write(request.data)
            return "Binary Message written to ./binary"
        else:
            return "415 Unsupported Media Type"


    if __name__ == "__main__":
        app.run()
    ```
  - **运行** shell 执行
    ```shell
    $ python headers.py
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```
  - **测试**
    ```shell
    # curl 测试
    $ curl -H "Content-type: text/plain" -X POST http://127.0.0.1:5000/messages -d '{"message":"Hello Data"}'
    Text Message: {"message":"Hello Data"}

    $ curl -H "Content-type: application/json" -X POST http://127.0.0.1:5000/messages -d '{"message":"Hello Data"}'
    JSON Message: {"message": "Hello Data"}

    $ curl -H "Content-type: application/octet-stream" -X POST http://127.0.0.1:5000/messages -d '{"message":"Hello Data"}'
    Binary Message written to ./binary

    $ cat binary
    {"message":"Hello Data"}
    ```
    ```python
    # requests 测试
    import requests
    requests.post("http://127.0.0.1:5000/messages", headers={"Content-type": "text/plain"}, data="aaa").text
    # Out[39]: 'Text Message: aaa'

    requests.post("http://127.0.0.1:5000/messages", headers={"Content-type": "application/json"}, data='{"message":"Hello Data"}').text
    # Out[42]: 'JSON Message: {"message": "Hello Data"}'

    requests.post("http://127.0.0.1:5000/messages", headers={"Content-type": "application/octet-stream"}, data='{"message":"Hello Data"}').text
    # Out[43]: 'Binary Message written to ./binary'

    open("./binary", "rb").read()
    # Out[46]: b'{"message":"Hello Data"}'
    ```
## GET 与 POST
  - **GET**
    - URL 中的参数使用明文
    - 参数在 `request` 对象的 `args` 参数里
    - 调用时 `requests.get` 方法中应使用 `params` 参数
    ```python
    resp = requests.get("http://127.0.0.1:5000/hello", params={"name": "world"})
    resp.url
    # Out[112]: 'http://127.0.0.1:5000/hello?name=world'
    ```
  - **POST**
    - URL 中不显示参数
    - 参数在 `request` 对象的 `from` 参数里
    - 支持 `headers` 中带 `Content-type`，此时可以使用 `request.json` / `request.data` 解析参数
    - 调用时 `requests.post` 方法中应使用 `data` 参数
    ```python
    resp = requests.post("http://127.0.0.1:5000/messages", headers={"Content-type": "application/json"}, data='{"message":"Hello Data"}')
    resp.url
    # Out[114]: 'http://127.0.0.1:5000/messages'
    ```
## 文件上传 File Uploads
  - 需要在 HTML 中设置 `enctype="multipart/form-data"`，否则文件不会上传
  - **request.files** 获取上传的文件
  - **werkzeug.utils.secure_filename** 更安全地获取文件名
    ```python
    from flask import request
    from werkzeug.utils import secure_filename

    @app.route('/upload', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            f = request.files['the_file']
            f.save('/var/www/uploads/' + secure_filename(f.filename))
    ```
## Cookies
  - **request.cookies** 用于处理 cookies
  - 读取 Reading cookies
    ```python
    from flask import request

    @app.route('/')
    def index():
        username = request.cookies.get('username')
        # use cookies.get(key) instead of cookies[key] to not get a
        # KeyError if the cookie is missing.
    ```
  - 存储 Storing cookies
    ```python
    from flask import make_response

    @app.route('/')
    def index():
        resp = make_response(render_template(...))
        resp.set_cookie('username', 'the username')
        return resp
    ```
## 重定向和错误 Redirects and Errors
  - **redirect 函数** 重定向错误页面
  - **abort 函数** 从错误页面返回
    ```python
    from flask import abort, redirect, url_for

    @app.route('/')
    def index():
        return redirect(url_for('login'))

    @app.route('/login')
    def login():
        abort(401)
        this_is_never_executed()
    ```
## errorhandler 错误请求处理
  - **@app.errorhandler** 用于自定义错误页面，默认的错误页面是一个空页面
    ```python
    from flask import render_template

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template('page_not_found.html'), 404
    ```
  - **Flask 脚本** error_404.py
    ```python
    #!/usr/bin/env python
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.errorhandler(404)
    def not_found(error=None):
        message = {"status": 404, "message": "Not found " + request.url, "error": str(error)}
        resp = jsonify(message)
        resp.status_code = 404

        return resp

    @app.route("/user/<user_id>", methods=["GET"])
    def api_user(user_id):
        users = {"1": "Foo", "2": "Goo", "3": "Koo"}
        if user_id in users:
            return jsonify({user_id: users[user_id]})
        else:
            return not_found("user id %s not found" % (user_id))

    if __name__ == "__main__":
        app.run()
    ```
  - **运行** shell 执行
    ```shell
    $ python error_404.py
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```
  - **测试**
    ```shell
    # curl 测试
    $ curl http://127.0.0.1:5000/user
    {"error":"404 Not Found: The requested URL was not found on the server.  If you entered the URL manually please check your spelling and try again.","message":"Not found http://127.0.0.1:5000/user","status":404}

    $ curl http://127.0.0.1:5000/user/0
    {"error":"user id 0 not found","message":"Not found http://127.0.0.1:5000/user/0","status":404}

    $ curl http://127.0.0.1:5000/user/1
    {"1":"Foo"}
    ```
    ```python
    # requests 测试
    import requests
    requests.get("http://127.0.0.1:5000/user").json()
    # Out[69]:
    # {'error': '404 Not Found: The requested URL was not found on the server.  If you entered the URL manually please check your spelling and try again.',
    #  'message': 'Not found http://127.0.0.1:5000/user',
    #  'status': 404}

    requests.get("http://127.0.0.1:5000/user/0").json()
    # Out[70]:
    # {'error': 'user id 0 not found',
    #  'message': 'Not found http://127.0.0.1:5000/user/0',
    #  'status': 404}

    requests.get("http://127.0.0.1:5000/user/1").json()
    # Out[71]: {'1': 'Foo'}
    ```
## 响应处理 Response
  - 默认情况下，Flask 会根据函数的返回值自动决定如何处理响应
  - **make_response 函数** 重新设置响应对象
    ```python
    @app.errorhandler(404)
    def not_found(error):
        resp = make_response(render_template('error.html'), 404)
        resp.headers['X-Something'] = 'A value'
        return resp
    ```
## Response 返回值 jsonify
  - **Flask 脚本** hello_jsonify.py
    ```python
    #!/usr/bin/env python
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route("/hello_json", methods=["GET"])
    def api_hello_json():
        data = {"hello": "world", "number": 3}

        resp = jsonify(data)
        resp.status_code = 200
        resp.headers["Link"] = "https://helloworld.com"

        return resp


    if __name__ == "__main__":
        app.run()
    ```
  - **运行** shell 执行
    ```shell
    $ python hello_jsonify.py
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```
  - **测试**
    ```shell
    # curl 测试
    $ curl -i http://127.0.0.1:5000/hello_json
    HTTP/1.0 200 OK
    Content-Type: application/json
    Content-Length: 29
    Link: https://helloworld.com
    Server: Werkzeug/0.14.1 Python/3.6.5
    Date: Thu, 02 Aug 2018 02:41:16 GMT

    {"hello":"world","number":3}
    ```
    ```python
    # requests 测试
    import requests
    resp = requests.get("http://127.0.0.1:5000/hello_json")

    resp.headers
    # Out[53]: {'Content-Type': 'application/json', 'Content-Length': '29', 'Link': 'http://helloworld.com', 'Server': 'Werkzeug/0.14.1 Python/3.6.5', 'Date': 'Thu, 02 Aug 2018 02:50:04 GMT'}

    resp.text
    # Out[54]: '{"hello":"world","number":3}\n'

    resp.json()
    # Out[56]: {'hello': 'world', 'number': 3}
    ```
## 用户会话 Sessions
  - **Sesison** 是建立在 Cookie 技术上的，可以用来管理用户会话
  - 在 Flask 中，可以为 Session 指定密钥，这样存储在 Cookie 中的信息就会被加密，从而更加安全
  ```python
  from flask import Flask, session, redirect, url_for, escape, request

  app = Flask(__name__)

  @app.route('/')
  def index():
      if 'username' in session:
          return 'Logged in as %s' % escape(session['username'])
      return 'You are not logged in'

  @app.route('/login', methods=['GET', 'POST'])
  def login():
      if request.method == 'POST':
          session['username'] = request.form['username']
          return redirect(url_for('index'))
      return '''
          <form method="post">
              <p><input type=text name=username>
              <p><input type=submit value=Login>
          </form>
      '''

  @app.route('/logout')
  def logout():
      # remove the username from the session if it's there
      session.pop('username', None)
      return redirect(url_for('index'))

  # set the secret key.  keep this really secret:
  app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
  ```
## LOGGING
  - **Loggging**
    ```python
    import logging
    import sys
    import flask

    app = flask.Flask('aa')

    # 指定文件
    fh = logging.FileHandler("app.log")

    # 指定标准错误
    sth = logging.StreamHandler(sys.stdout)

    # app 中添加 handler
    app.logger.addHandler(fh)
    app.logger.addHandler(sth)

    # 查看
    app.logger.handlers
    # Out[6]:
    # [<StreamHandler <stderr> (NOTSET)>,
    #  <FileHandler /home/leondgarse/practice_code/python_web/app.log (NOTSET)>,
    #  <StreamHandler <stdout> (NOTSET)>]

    # 删除
    app.logger.removeHandler(app.logger.handlers[0])
    app.logger.removeHandler(app.logger.handlers[1])
    app.logger.handlers
    # Out[10]: [<FileHandler /home/leondgarse/practice_code/python_web/app.log (NOTSET)>]
    ```
  - **Flask 脚本** debug_log.py
    ```python
    #!/usr/bin/env python

    from flask import Flask, request
    import logging

    app = Flask(__name__)

    file_handler = logging.FileHandler("app.log")
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

    @app.route("/hello", methods=["GET"])
    def api_hello():
        app.logger.debug("debug")
        app.logger.info("informing")
        app.logger.warning("warning")
        app.logger.error("error")

        return "Check your log in %s\n" % ("app.log")

    if __name__ == "__main__":
        app.run(debug=True)
    ```
  - **运行** shell 执行
    ```shell
    $ python debug_log.py
     * Debug mode: on
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```
  - **测试**
    ```shell
    $ curl http://127.0.0.1:5000/hello
    Check your log in app.log

    $ cat app.log
    informing
    warning
    error
    ```
***

# Jinja 模板简介
## 模板标签
  - **Jinja 模板** 和其他语言和框架的模板类似，通过某种语法将 HTML 文件中的特定元素替换为实际的值
  - **{\% \%}** 代码块
    ```html
    {% extends 'layout.html' %}
    {% block title %}主页{% endblock %}
    {% block body %}

        <div class="jumbotron">
            <h1>主页</h1>
        </div>

    {% endblock %}
    ```
  - **`{{ }}`** 中的内容不会被转义，所有内容都会原样输出，常常和其他辅助函数一起使用
    ```html
    <a class="navbar-brand" href={{ url_for('index') }}>Flask小例子</a>
    ```
## 继承
  - 模板可以继承其他模板，可以将布局设置为父模板，让其他模板继承，非常方便的控制整个程序的外观
  - **layout.html 模板** 作为整个程序的布局文件
    ```html
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{% block title %}{% endblock %}</title>
        <link rel="stylesheet" href="{{ url_for('static',filename='css/bootstrap.css') }}"/>
        <link rel="stylesheet" href="{{ url_for('static',filename='css/bootstrap-theme.css') }}"/>

    </head>
    <body>

    <div class="container body-content">
        {% block body %}{% endblock %}
    </div>

    <div class="container footer">
        <hr>
        <p>这是页脚</p>
    </div>

    <script src="{{ url_for('static',filename='js/jquery.js') }}"></script>
    <script src="{{ url_for('static',filename='js/bootstrap.js') }}"></script>

    </body>
    </html>
    ```
  - 其他模板继承 layout.html 模板
    ```html
    {% extends 'layout.html' %}
    {% block title %}主页{% endblock %}
    {% block body %}

        <div class="jumbotron">
            <h1>主页</h1>
            <p>本项目演示了Flask的简单使用方法，点击导航栏上的菜单条查看具体功能。</p>
        </div>

    {% endblock %}
    ```
## 控制流
  - **条件判断** 类似于 JSP 标签中的 Java 代码，`{% %}` 中也可以写 Python 代码
    ```html
    <div class=metanav>
    {% if not session.logged_in %}
      <a href="{{ url_for('login') }}">log in</a>
    {% else %}
      <a href="{{ url_for('logout') }}">log out</a>
    {% endif %}
    </div>
    ```
  - **循环** 类似 Python 中遍历
    ```html
    <tbody>
    {% for key,value in data.items() %}
        <tr>
            <td>{{ key }}</td>
            <td>{{ value }}</td>
        </tr>
    {% endfor %}
    <tr>
        <td>文件</td>
        <td></td>
    </tr>
    </tbody>
    ```
  - 不是所有的 Python 代码都可以写在模板里，如果希望从模板中引用其他文件的函数，需要显式将函数注册到模板中
***

# 部署 Deployment
## flask run
  ```shell
  flask --help

  export FLASK_APP=hello.py
  # export FLASK_ENV=development
  export FLASK_ENV=production
  flask run
  ```
## waitress
  - [Deploy to Production](flask.pocoo.org/docs/1.0/tutorial/deploy/)
  - 安装
    ```shell
    pip install waitress
    ```
  - 脚本中不再调用 `app.run()`，而是定义函数返回 `app`
    ```python
    # hello.py
    app = Flask(__name__)

    def create_app():
        return app

    if __name__ == '__main__':
        ...
        app.run()
    else:
        ...
    ```
  - 启动，调用模块中指定的方法，如果有 `argparse`，需定义 `--call`
    ```shell
    # 默认端口 8080
    waitress-serve --port 8041 --call 'hello:create_app'
    ```
***
