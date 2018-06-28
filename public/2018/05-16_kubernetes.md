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
