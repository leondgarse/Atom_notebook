# ___2018 - 05 - 16 Kubernetes___
***

# 安装
## 手动配置
  - **加载 kubernetes 需要的 docker images**
    - 下载 [kubernetes 1.10.1 docker images](https://pan.baidu.com/s/1ZJFRt_UNCQvwcu9UENr_gw#list/path=%2F)
    - 解压后得到 tar 文件
    - docker 加载
      ```shell
      ls *.tar | xargs -I {} sh -c 'sudo docker load< {}'
      ```
  - **apt 安装 kubelet kubeadm kubectl docker docker.io**
    ```shell
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    # 必须切换成 root 用户执行
    sudo su -c 'echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" > /etc/apt/sources.list.d/kubernetes.list'

    sudo apt-get update
    sudo apt-get install -y kubelet kubeadm kubectl
    sudo apt-get install -y docker docker.io
    ```
  - **初始化 master 节点**
    ```shell
    sudo systemctl start docker.service
    sudo swapoff -a

    # 需要代理，中间可能还需要下载其他 docker image
    sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.0.200

    # 初始化失败后需要 reset
    sudo kubeadm reset
    ```
  - 配置 master 节点，之后可以使用 kubectl 进行其他操作
    ```shell
    mkdir -p $HOME/.kube
    sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
    sudo chown $(id -u):$(id -g) $HOME/.kube/config
    ```
  - **Slave 节点加入集群**
    ```shell
    kubeadm join 192.168.0.200:6443 --token rw4enn.mvk547juq7qi2b5f --discovery-token-ca-cert-hash sha256:ba260d5191213382a806a9a7d92c9e6bb09061847c7914b1ac584d0c69471579
    ```
## minikube
  - [github minikube](https://github.com/kubernetes/minikube/releases/)
  - [minikube README.md](https://github.com/kubernetes/minikube/blob/v0.27.0/README.md)
  - **安装**
    ```shell
    # snap 安装，版本较低
    sudo snap install minikube
    # 安装指定版本
    curl -Lo minikube https://storage.googleapis.com/minikube/releases/v0.28.0/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/
    ```
  - **安装 kubectl**
    ```shell
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    # 必须切换成 root 用户执行
    sudo su -c 'echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" > /etc/apt/sources.list.d/kubernetes.list'

    sudo apt-get update
    sudo apt-get install -y kubelet kubeadm kubectl
    ```
  - **添加其他 image**，需要代理连接
    ```shell
    kubectl run hello-minikube --image=k8s.gcr.io/echoserver:1.4 --port=8080
    kubectl run kubernetes-bootcamp --image=gcr.io/google-samples/kubernetes-bootcamp:v1 --port=8080
    kubectl create -f https://k8s.io/docs/tasks/debug-application-cluster/shell-demo.yaml
    ```
  - **启动**
    ```shell
    minikube version
    # 需要代理连接
    minikube start

    kubectl version
    # 查看初始化是否完成，Verify that the Container is running
    kubectl get pods
    NAME                                   READY     STATUS    RESTARTS   AGE
    hello-minikube-6c47c66d8-qrvl6         1/1       Running   0          1h
    kubernetes-bootcamp-5c69669756-pttwm   1/1       Running   0          1d
    shell-demo                             1/1       Running   1          1h
    ```
  - Virtualbox 登录
    - [Boot2Docker ssh-into-vm](https://github.com/boot2docker/boot2docker#ssh-into-vm)
    - user: docker
    - pass: tcuser
## helm
  - [Linux](https://kubernetes-helm.storage.googleapis.com/helm-v2.9.1-linux-amd64.tar.gz)
  - [Quick Start Guide](https://docs.helm.sh/using_helm/#quickstart-guide)
  - Extract and copy `helm` to a PATH directory
    ```shell
    tar xvf helm-v2.9.1-linux-amd64.tar.gz
    cd linux-amd64
    cp helm ~/local_bin
    ```
  - **Init**
    ```shell
    # Find out which cluster Tiller would install to
    kubectl config current-context

    # Initialize the local CLI and also install Tiller into Kubernetes cluster in one step
    helm init

    # To install a chart using one of the official stable charts
    helm repo update
    helm install stable/mysql

    # See what has been released using Helm
    helm ls
    ```
***

# Kubernetes 核心技术概念
  - **API对象**
    - API对象是Kubernetes集群中的管理操作单元。Kubernetes集群系统每支持一项新功能，引入一项新技术，一定会新引入对应的API对象，支持对该功能的管理操作。例如副本集Replica Set对应的API对象是RS
    - 每个API对象都有3大类属性
      - 元数据metadata
      - 规范spec
      - 状态status
    - 元数据是用来标识API对象的
      - 每个对象都至少有3个元数据：namespace，name和uid
      - 除此以外还有各种各样的标签labels用来标识和匹配不同的对象
  - **Pod**
    - Pod 是在 Kubernetes 集群中运行部署应用或服务的最小单元
    - Pod 的设计理念是 **支持多个容器** 在一个Pod中共享网络地址和文件系统，可以通过进程间通信和文件共享这种简单高效的方式组合完成服务
    - Pod对多容器的支持是K8最基础的设计理念
    - Pod 是 Kubernetes 集群中所有业务类型的基础，不同类型的业务就需要不同类型的 pod 去执行
    - 目前 Kubernetes 中的业务主要可以分为
      - **长期伺服型 long-running** / **批处理型 batch** / **节点后台支撑型 node-daemon** / **有状态应用型 stateful application**
      - 分别对应的控制器为 **Deployment** / **Job** / **DaemonSet** / **StatefulSet**
  - **副本控制器 Replication Controller，RC**
    - RC是Kubernetes集群中最早的保证Pod高可用的API对象
    - 通过监控运行中的Pod来保证集群中运行指定数目的Pod副本。指定的数目可以是多个也可以是1个
    - 少于指定数目，RC就会启动运行新的Pod副本；多于指定数目，RC就会杀死多余的Pod副本
    - 即使在指定数目为1的情况下，通过RC运行Pod也比直接运行Pod更明智，因为RC也可以发挥它高可用的能力，保证永远有1个Pod在运行
    - RC是Kubernetes较早期的技术概念，只适用于长期伺服型的业务类型，比如控制小机器人提供高可用的Web服务
  - **副本集 Replica Set，RS**
    - RS是新一代RC，提供同样的高可用能力，区别主要在于RS后来居上，能支持更多种类的匹配模式
    - 副本集对象一般不单独使用，而是作为Deployment的理想状态参数使用
  - **部署 Deployment**
    - 部署表示用户对Kubernetes集群的一次更新操作
    - 部署是一个比RS应用模式更广的API对象，可以是创建一个新的服务，更新一个新的服务，也可以是滚动升级一个服务
    - 滚动升级一个服务，实际是创建一个新的RS，然后逐渐将新RS中副本数增加到理想状态，将旧RS中的副本数减小到0的复合操作
    - 这样一个复合操作用一个RS是不太好描述的，所以用一个更通用的Deployment来描述
    - 以Kubernetes的发展方向，未来对所有长期伺服型的的业务的管理，都会通过Deployment来管理
  - **服务 Service**
    - RC、RS和Deployment只是保证了支撑服务的微服务Pod的数量，但是没有解决如何访问这些服务的问题
    - 一个Pod只是一个运行服务的实例，随时可能在一个节点上停止，在另一个节点以一个新的IP启动一个新的Pod，因此不能以确定的IP和端口号提供服务
    - 要稳定地提供服务需要服务发现和负载均衡能力，服务发现完成的工作，是针对客户端访问的服务，找到对应的的后端服务实例
    - 在K8集群中，客户端需要访问的服务就是Service对象
    - 每个Service会对应一个集群内部有效的虚拟IP，集群内部通过虚拟IP访问一个服务
    - 在Kubernetes集群中微服务的负载均衡是由Kube-proxy实现的，Kube-proxy是Kubernetes集群内部的负载均衡器
  - **任务 Job**
    - Job是Kubernetes用来控制批处理型任务的API对象
    - 批处理业务与长期伺服业务的主要区别是批处理业务的运行有头有尾，而长期伺服业务在用户不停止的情况下永远运行，Job管理的Pod根据用户的设置把任务成功完成就自动退出了
    - 成功完成的标志根据不同的spec.completions策略而不同
    - 单Pod型任务有一个Pod成功就标志完成
    - 定数成功型任务保证有N个任务全部成功
    - 工作队列型任务根据应用确认的全局成功而标志成功
  - **后台支撑服务集 DaemonSet**
    - 长期伺服型和批处理型服务的核心在业务应用，可能有些节点运行多个同类业务的Pod，有些节点上又没有这类Pod运行
    - 而后台支撑型服务的核心关注点在Kubernetes集群中的节点（物理机或虚拟机），要保证每个节点上都有一个此类Pod运行
    - 节点可能是所有集群节点也可能是通过nodeSelector选定的一些特定节点
    - 典型的后台支撑型服务包括，存储，日志和监控等在每个节点上支持Kubernetes集群运行的服务
  - **有状态服务集 StatefulSet**
    - Kubernetes在1.3版本里发布了Alpha版的PetSet功能，在1.5版本里将PetSet功能升级到了Beta版本，并重新命名为StatefulSet，最终在1.9版本里成为正式GA版本
    - 在云原生应用的体系里，有下面两组近义词
      - 第一组是无状态（stateless）、牲畜（cattle）、无名（nameless）、可丢弃（disposable）
      - 第二组是有状态（stateful）、宠物（pet）、有名（having name）、不可丢弃（non-disposable）
    - RC和RS主要是控制提供无状态服务的，其所控制的Pod的名字是随机设置的，一个Pod出故障了就被丢弃掉，在另一个地方重启一个新的Pod，名字变了、名字和启动在哪儿都不重要，重要的只是Pod总数
    - 而StatefulSet是用来控制有状态服务，StatefulSet中的每个Pod的名字都是事先确定的，不能更改
    - 对于RC和RS中的Pod，一般不挂载存储或者挂载共享存储，保存的是所有Pod共享的状态，Pod像牲畜一样没有分别
    - 对于StatefulSet中的Pod，每个Pod挂载自己独立的存储，如果一个Pod出现故障，从其他节点启动一个同样名字的Pod，要挂载上原来Pod的存储继续以它的状态提供服务
    - 适合于StatefulSet的业务包括
      - 数据库服务MySQL和PostgreSQL
      - 集群化管理服务ZooKeeper、etcd等有状态服务
    - StatefulSet的另一种典型应用场景是作为一种比普通容器更稳定可靠的模拟虚拟机的机制，传统的虚拟机正是一种有状态的宠物，运维人员需要不断地维护它
    - 容器刚开始流行时，我们用容器来模拟虚拟机使用，所有状态都保存在容器里，而这已被证明是非常不安全、不可靠的
    - 使用StatefulSet，Pod仍然可以通过漂移到不同节点提供高可用，而存储也可以通过外挂的存储来提供高可靠性，StatefulSet做的只是将确定的Pod与确定的存储关联起来保证状态的连续性
***

# 操作
## 通用参数
  - **-o yaml** 以 yaml 格式输出完整信息
## kubectl get
  ```shell
  kubectl get all
  kubectl get nodes
  kubectl get deployments
  kubectl get pods
  ```
## kubectl describe
  ```shell
  kubectl describe pods
  ```
## kubectl proxy
  - 启动 proxy 后，可以通过 curl 命令 获取 / 操作
  ```shell
  kubectl proxy

  curl http://localhost:8001/version

  export POD_NAME=$(kubectl get pods -o go-template --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}')
  echo Name of the Pod: $POD_NAME
  curl http://localhost:8001/api/v1/namespaces/default/pods/$POD_NAME/proxy/
  kubectl logs existing-frog-mysql-8658bb7f7-4sv4l
  ```
## 其他命令
  ```shell
  kubectl cluster-info
  ```
***

# kubernetes client python
## kubectl get
  - **kubectl get pods --all-namespaces**
    ```python
    from kubernetes import client, config

    def k8s_list_pod_for_all_namespaces():
        # default configuration: ~/.kube/config
        config.load_kube_config()

        v1 = client.CoreV1Api()
        print("Listing pods with their IPs:")
        ret = v1.list_pod_for_all_namespaces(watch=False)
        for i in ret.items:
            print("%s\t%s\t%s" %
                  (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
    ```
  - **kubectl get pods -n default**
    ```python
    from kubernetes import client, config

    def k8s_list_namespaced_pod(namespace='default'):
        # default configuration: ~/.kube/config
        config.load_kube_config()

        v1 = client.CoreV1Api()
        print("Listing pods with their IPs:")
        ret = v1.list_namespaced_pod(namespace)
        for i in ret.items:
            print("%s\t%s\t%s" %
                  (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
    ```
## kubectl exec
  - **kubectl exec shell-demo date**
    ```python
    from kubernetes import client, config, stream
    from kubernetes.client.rest import ApiException

    def exec_namespaced_pod_stream(name, pod_name, exec_command):
        config.load_kube_config()
        # c = client.Configuration()
        # c.assert_hostname = False
        # client.Configuration.set_default(c)
        v1 = client.CoreV1Api()
        namespace = get_namespace(name)

        resp = None
        try:
            resp = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            if resp:
                resp = stream.stream(v1.connect_get_namespaced_pod_exec, pod_name, namespace,
                              command=exec_command,
                              stderr=True, stdin=False,
                              stdout=True, tty=False)
        except ApiException as e:
            print("Exception when calling CoreV1Api->connect_get_namespaced_pod_exec: %s\n" % e)
            return None

        return resp

    # Run
    pod_name = 'shell-demo'
    exec_command = ['/bin/sh', '-c', 'date']
    name = 'default'
    exec_namespaced_pod_stream(name, pod_name, exec_command)
    ```
***

# 控制器
## Deployment
  ```python
  # 创建 Deployment，--record参数可以记录命令
  kubectl create -f https://kubernetes.io/docs/user-guide/nginx-deployment.yaml --record
  # 查看与编辑
  kubectl get deployments
  kubectl describe deployments
  kubectl edit deployment/nginx-deployment

  # 显示 Replica Set（RS）和 Pod
  kubectl get rs
  kubectl get po
  kubectl get pods --show-labels

  # 更新Deployment
  kubectl set image deployment/nginx-deployment nginx=nginx:1.9.1

  # 查看 rollout 的状态
  kubectl rollout status deployment/nginx-deployment

  # 检查 Deployment 的 revision
  kubectl rollout history deployment/nginx-deployment
  kubectl rollout history deployment/nginx-deployment --revision=2

  # 回退到历史版本
  kubectl rollout undo deployment/nginx-deployment
  kubectl rollout undo deployment/nginx-deployment --to-revision=2

  # Deployment 扩容
  kubectl scale deployment nginx-deployment --replicas 10

  # 给 Deployment 设置一个 autoscaler，基于当前 Pod的 CPU 利用率选择最少和最多的 Pod 数
  kubectl autoscale deployment nginx-deployment --min=10 --max=15 --cpu-percent=80

  # 暂停 Deployment
  kubectl rollout pause deployment/nginx-deployment
  # 更新使用的资源
  kubectl set resources deployment nginx -c=nginx --limits=cpu=200m,memory=512Mi
  # 恢复这个 Deployment
  kubectl rollout resume deploy nginx
  KUBECTL get rs -w
  ```
## Job
  - Job 负责批处理任务，即仅执行一次的任务，它保证批处理任务的一个或多个Pod成功结束
  - Job 创建一个或多个 pods 保证一定数量的 pods 成功结束
  - Job 可以并行执行多个 pods，单个 Pod 时，默认 Pod 成功运行后 Job 即结束
  - **Job Spec** 格式
    - **需要的参数** apiVersion / kind / metadata / spec
    - **.spec.template** 是 spec 中必需的参数
    - **.spec.template** 指定 Pod template，格式与 pod 相同
    - **RestartPolicy** 仅支持 Never 或 OnFailure，默认值是 Always，因此需要指定成 Job 支持的值
    - **.spec.selector** 可选，通常不需要指定
    - **.spec.completions** 标志Job结束需要成功运行的Pod个数，默认为1
    - **.spec.parallelism** 标志并行运行的Pod的个数，默认为1
    - **.spec.activeDeadlineSeconds** 标志失败Pod的重试最大时间，超过这个时间不会继续重试
  - **示例**
    ```yaml
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: pi
    spec:
      template:
        spec:
          containers:
          - name: pi
            image: perl
            command: ["perl",  "-Mbignum=bpi", "-wle", "print bpi(2000)"]
          restartPolicy: Never
      backoffLimit: 4
    ```
    ```bash
    $ kubectl create -f ./job.yaml
    job "pi" created
    $ kubectl describe jobs/pi
    $ pods=$(kubectl get pods --selector=job-name=pi --output=jsonpath={.items..metadata.name})
    $ echo $pods
    $ kubectl logs $pods
    3.141592653589793238462643383279502...
    ```
## CronJob
  - **典型的用法**
    - 在给定的时间点调度 Job 运行
    - 创建周期性运行的 Job，例如：数据库备份、发送邮件
  - **CronJob Spec**
    - `.spec.schedule`：**调度**，必需字段，指定任务运行周期，格式同 [Cron](https://en.wikipedia.org/wiki/Cron)
    - `.spec.jobTemplate`：**Job 模板**，必需字段，指定需要运行的任务，格式同 [Job](job.md)
    - `.spec.startingDeadlineSeconds` ：**启动 Job 的期限（秒级别）**，该字段是可选的。如果因为任何原因而错过了被调度的时间，那么错过执行时间的 Job 将被认为是失败的。如果没有指定，则没有期限
    - `.spec.concurrencyPolicy`：**并发策略**，该字段也是可选的。它指定了如何处理被 Cron Job 创建的 Job 的并发执行。只允许指定下面策略中的一种：
      - `Allow`（默认）：允许并发运行 Job
      - `Forbid`：禁止并发运行，如果前一个还没有完成，则直接跳过下一个
      - `Replace`：取消当前正在运行的 Job，用一个新的来替换
    - `.spec.suspend` ：**挂起**，该字段也是可选的。如果设置为 `true`，后续所有执行都会被挂起。它对已经开始执行的 Job 不起作用。默认值为 `false`
    - `.spec.successfulJobsHistoryLimit` 和 `.spec.failedJobsHistoryLimit` ：**历史限制**，是可选的字段。它们指定了可以保留多少完成和失败的 Job
      - 默认没有限制，所有成功和失败的 Job 都会被保留
      - 然而，当运行一个 Cron Job 时，Job 可以很快就堆积很多，推荐设置这两个字段的值。设置限制的值为 `0`，相关类型的 Job 完成后将不会被保留
  - **示例**
    ```yaml
    apiVersion: batch/v2alpha1
    kind: CronJob
    metadata:
      name: hello
    spec:
      schedule: "*/1 * * * *"
      jobTemplate:
        spec:
          template:
            spec:
              containers:
              - name: hello
                image: busybox
                args:
                - /bin/sh
                - -c
                - date; echo Hello from the Kubernetes cluster
              restartPolicy: OnFailure
    ```

    ```Bash
    $ kubectl create -f cronjob.yaml
    cronjob "hello" created
    ```
  - **`kubectl run` 创建 CronJob**
    ```bash
    kubectl run hello --schedule="*/1 * * * *" --restart=OnFailure --image=busybox -- /bin/sh -c "date; echo Hello from the Kubernetes cluster"
    ```

    ```bash
    $ kubectl get cronjob
    NAME      SCHEDULE      SUSPEND   ACTIVE    LAST-SCHEDULE
    hello     */1 * * * *   False     0         <none>
    $ kubectl get jobs
    NAME               DESIRED   SUCCESSFUL   AGE
    hello-1202039034   1         1            49s
    $ pods=$(kubectl get pods --selector=job-name=hello-1202039034 --output=jsonpath={.items..metadata.name} -a)
    $ kubectl logs $pods
    Mon Aug 29 21:34:09 UTC 2016
    Hello from the Kubernetes cluster

    # 注意，删除cronjob的时候不会自动删除job，这些job可以用kubectl delete job来删除
    $ kubectl delete cronjob hello
    cronjob "hello" deleted
    ```
  - **删除 Cron Job**
    ```shell
    $ kubectl delete cronjob hello
    cronjob "hello" deleted
    ```
    这将会终止正在创建的 Job。然而，运行中的 Job 将不会被终止，不会删除 Job 或 它们的 Pod。为了清理那些 Job 和 Pod，需要列出该 Cron Job 创建的全部 Job，然后删除它们：
    ```shell
    $ kubectl get jobs
    NAME               DESIRED   SUCCESSFUL   AGE
    hello-1201907962   1         1            11m
    hello-1202039034   1         1            8m
    ...

    $ kubectl delete jobs hello-1201907962 hello-1202039034 ...
    job "hello-1201907962" deleted
    job "hello-1202039034" deleted
    ...

    ```
***

# Docker
  ```shell
  sudo docker build -t my-rest-client .
  sudo docker run -it my-rest-client

  sudo docker image ls

  sudo docker image save my-rest-client -o my-rest-client.tar
  sudo docker image load -i my-rest-client.tar

  sudo docker images
  sudo docker ps -a

  sudo docker run ubuntu:16.04
  ```
  ```shell
  docker run
  kubectl run teste-test-e --image=cbur-test-2 --image-pull-policy=Never
  ```
- kubernetes uses Docker images by creating local registry
  ```shell
  # Use a local registry:
  docker run -d -p 5000:5000 --restart=always --name registry registry:2

  # Now tag your image properly:
  # Note that localhost should be changed to dns name of the machine running registry container.
  docker tag ubuntu localhost:5000/ubuntu

  # Now push your image to local registry:
  docker push localhost:5000/ubuntu

  # You should be pull it back:
  docker pull localhost:5000/ubuntu

  # Now change your yaml file to use local registry.
  ```
- kubernetes uses Docker images for minikube
  ```shell
  # Start minikube
  minikube start

  # Set docker env
  eval $(minikube docker-env)

  # Build image
  docker build -t foo:0.0.1 .

  # Run in minikube
  kubectl run hello-foo --image=foo:0.0.1 --image-pull-policy=Never

  # Check that it's running
  kubectl get pods
  ```
***
