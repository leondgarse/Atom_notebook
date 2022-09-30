# ___2019 - 08 - 12 Caffe___
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2019 - 08 - 12 Caffe___](#2019-08-12-caffe)
  - [Install python caffe](#install-python-caffe)
  	- [links](#links)
  	- [git clone](#git-clone)
  	- [apt and pip install](#apt-and-pip-install)
  	- [Makefile.config](#makefileconfig)
  	- [Makefile](#makefile)
  	- [Protobuf](#protobuf)
  	- [libboost_python](#libboostpython)
  	- [make](#make)
  	- [Python test](#python-test)
  - [Q / A](#q-a)
  - [Docker](#docker)

  <!-- /TOC -->
***

# Install python caffe
## links
  - [Caffe Installation](http://caffe.berkeleyvision.org/installation.html)
## git clone
  ```sh
  $ git clone https://github.com/BVLC/caffe --depth=10
  $ cd caffe/
  $ git pull --depth=100000
  ```
## apt and pip install
  ```sh
  sudo apt install \
          gcc-8 g++-8 libboost-all-dev \
          libgflags-dev libgoogle-glog-dev libleveldb-dev liblmdb-dev libopencv-dev \
          libsnappy-dev libhdf5-serial-dev libopenblas-dev libatlas-base-dev

  # sudo apt install protobuf-c-compiler protobuf-compiler libprotobuf-dev
  ```
  ```sh
  pip install opencv six numpy scikit-image pydot

  pip install ipython pysqlite3
  ```
## Makefile.config
  ```sh
  $ cp Makefile.config.example Makefile.config
  ```
  ```sh
  $ diff Makefile.config.example Makefile.config
  5c5
  < # USE_CUDNN := 1
  ---
  > USE_CUDNN := 1
  27c27
  < # CUSTOM_CXX := g++
  ---
  > CUSTOM_CXX := g++-8
  39,40c39,40
  < CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
  <               -gencode arch=compute_20,code=sm_21 \
  ---
  > CUDA_ARCH := # -gencode arch=compute_20,code=sm_20 \
  >               # -gencode arch=compute_20,code=sm_21 \
  53c53
  < BLAS := atlas
  ---
  > BLAS := open
  71,72c71,72
  < PYTHON_INCLUDE := /usr/include/python2.7 \
  <               /usr/lib/python2.7/dist-packages/numpy/core/include
  ---
  > # PYTHON_INCLUDE := /usr/include/python2.7 \
  > #             /usr/lib/python2.7/dist-packages/numpy/core/include
  81,83c81,83
  < # PYTHON_LIBRARIES := boost_python3 python3.5m
  < # PYTHON_INCLUDE := /usr/include/python3.5m \
  < #                 /usr/lib/python3.5/dist-packages/numpy/core/include
  ---
  > PYTHON_LIBRARIES := boost_python38 python3.8
  > PYTHON_INCLUDE := /usr/include/python3.8 \
  >                 /usr/lib/python3/dist-packages/numpy/core/include
  97,98c97,98
  < INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
  < LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
  ---
  > INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial /usr/include/opencv4
  > LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
  ```
## Makefile
  ```sh
  $ git diff Makefile
  diff --git a/Makefile b/Makefile
  old mode 100644
  new mode 100755
  index b7660e85..119f59d7
  --- a/Makefile
  +++ b/Makefile
  @@ -198,14 +198,14 @@ ifeq ($(USE_HDF5), 1)
          LIBRARIES += hdf5_hl hdf5
   endif
   ifeq ($(USE_OPENCV), 1)
  -       LIBRARIES += opencv_core opencv_highgui opencv_imgproc
  +       LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs

          ifeq ($(OPENCV_VERSION), 3)
  -               LIBRARIES += opencv_imgcodecs
  +               LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
          endif

   endif
  -PYTHON_LIBRARIES ?= boost_python python2.7
  +PYTHON_LIBRARIES ?= boost_python38
   WARNINGS := -Wall -Wno-sign-compare

   ##############################
  @@ -385,6 +385,7 @@ ifeq ($(BLAS), mkl)
          BLAS_LIB ?= $(MKLROOT)/lib $(MKLROOT)/lib/intel64
   else ifeq ($(BLAS), open)
          # OpenBLAS
  +       LDFLAGS += -L/opt/OpenBLAS/lib -lopenblas
          LIBRARIES += openblas
   else
          # ATLAS
  ```
## Protobuf
  - When set using `anaconda` in `Makefile.config`, `libprotobuf` version in system should match with `anaconda` one
    ```sh
    $ locate libprotobuf  # List only those matters
    # /opt/anaconda3/pkgs/libprotobuf-3.11.4-hd408876_0/lib/libprotobuf.so
    # /opt/anaconda3/lib/libprotobuf.so.22
    # /usr/local/lib/libprotobuf.so
    # /usr/local/lib/libprotobuf.so.22

    $ ls /usr/local/lib/libprotobuf.so -l
    # /usr/local/lib/libprotobuf.so -> libprotobuf.so.22.0.4
    ```
    Or may lead to errors like
    ```sh
    .build_release/lib/libcaffe.so: undefined reference to google::protobuf::internal::UnknownFieldParse
    .build_release/lib/libcaffe.so: undefined reference to google::protobuf::RepeatedPtrField
    ```
  - Download the matched protobuf release `protobuf-cpp-xxx.tar.gz` from [protobuf releases](https://github.com/protocolbuffers/protobuf/releases)
  - Extract and compile:
    ```sh
    cd protobuf-3.11.4
    ./autogen.sh
    ./configure
    make
    make check
    sudo make install
    sudo ldconfig
    pkg-config --cflags --libs protobuf
    ```
## libboost_python
  - When build caffe with different python version from system, may meet error `Segmentation failt (core dump)` when `import caffe`. Need to build `libboost_python` from source with the new python version.
  - [Boost C++ librarie.](https://www.boost.org)
  ```sh
  wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz
  mv boost_1_80_0.tar.gz /opt
  cd /opt
  tar xvf boost_1_80_0.tar.gz
  cd boost_1_80_0
  ./booststrap.sh --prefix=/opt/boost/ --with-python=$(which python) --with-libraries=python inlcude="$HOME/local_bin/python-3.9.13/include/python3.9/"
  CPLUS_INCLUDE_PATH="$HOME/local_bin/python-3.9.13/include/python3.9/" ./b2 install

  # Check /opt/boost/lib/ contains libboost_python39.so

  export LD_LIBRARY_PATH="/opt/boost/lib/:$LD_LIBRARY_PATH"
  ```
## make
  ```sh
  make all
  make test
  make runtest
  make pytest  # --> python/caffe/_caffe.so
  export PYTHONPATH=$PYTHONPATH:$HOME/workspace/caffe/python
  ```
## Python test
  ```py
  import sys
  sys.path.append("/opt/caffe/python")
  import caffe
  caffe.__version__
  # 1.0.0
  ```
***

# Q / A
  - **Q: libboost import error: undefined symbol**
    ```py
    import caffe
    undefined symbol: _ZN5boost6python6detail11init_moduleER11PyModuleDefPFvvE
    ```
    A: Installed `boost_python` version is `python 3.8` / `python 3.6`, not matching with Makefile which is using `2.7`
    ```sh
    $ locate libboost_python
    # /usr/lib/x86_64-linux-gnu/libboost_python38.so

    $ ls /usr/lib/x86_64-linux-gnu/libboost_python* -l
    # /usr/lib/x86_64-linux-gnu/libboost_python38.so -> libboost_python38.so.1.71.0

    $ vi Makefile +208
    PYTHON_LIBRARIES ?= boost_python38
    # Or: PYTHON_LIBRARIES ?= boost_python-py36
    ```
    Re-compile `.so`
    ```sh
    make pytest
    ```
  - **Q: Build error: undefined reference to cv::imread**
    ```sh
    build_release/lib/libcaffe.so: undefined reference to cv::imread(cv::String const&, int)
    ```
    A: Add `opencv` lib in Makefile：
    ```sh
    $ vi Makefile +201
    ifeq ($(USE_OPENCV), 2)
        LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
    endif  
    ```
  - **Q: include error: hdf5.h: No such file or directory**
    ```sh
    ./include/caffe/util/hdf5.hpp:6:18: fatal error: hdf5.h: No such file or directory
    ```
    A: PYTHON_INCLUDE / PYTHON_LIB NOT including `hdf5` lib
    ```sh
    $ vi Makefile.config +97
    < INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
    < LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
    ```
  - **Q: Build error: Unsupported gpu architecture 'compute_20'**
    ```sh
    NVCC src/caffe/layers/reduction_layer.cu
    nvcc fatal   : Unsupported gpu architecture 'compute_20'
    Makefile:588: recipe for target '.build_release/cuda/src/caffe/layers/reduction_layer.o' failed
    make: *** [.build_release/cuda/src/caffe/layers/reduction_layer.o] Error 1
    ```
    A: Modify CUDA architecture setting
    ```sh
    $ vi Makefile.config +39
    CUDA_ARCH := # -gencode arch=compute_20,code=sm_20 \
    		# -gencode arch=compute_20,code=sm_21 \
    ```
  - **Q: Build error: undefined reference to caffe::cudnn::dataType<float>::one**
    ```sh
    .build_release/lib/libcaffe.so: undefined reference to caffe::cudnn::dataType<float>::one
    collect2: error: ld returned 1 exit status
    make: *** [.build_release/tools/upgrade_net_proto_text.bin] Error 1
    ```
    A: Use `openblas` instead of `atlas blas`
    ```sh
    $ vi Makefile.config +53
    BLAS := open
    ```
  - **Q: Build error: ‘CV_LOAD_IMAGE_COLOR’ was not declared in this scope**
    ```sh
    src/caffe/util/io.cpp:76:34: error: ‘CV_LOAD_IMAGE_COLOR’ was not declared in this scope
       76 |   int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
          |                                  ^~~~~~~~~~~~~~~~~~~
    src/caffe/util/io.cpp:77:5: error: ‘CV_LOAD_IMAGE_GRAYSCALE’ was not declared in this scope
       77 |     CV_LOAD_IMAGE_GRAYSCALE);
    ```
    A: The error caused by the OpenCV module in version 3 and 4 has change, change `src/caffe/util/io.cpp` `src/caffe/layers/window_data_layer.cpp` `src/caffe/test/test_io.cpp`
    ```sh
    # From:
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
        CV_LOAD_IMAGE_GRAYSCALE);

    # To:
    int cv_read_flag = (is_color ? cv::IMREAD_COLOR :
        cv::IMREAD_GRAYSCALE);
    ```
  - **Q: CUDA error: unsupported GNU version! gcc versions later than 8 are not supported!**
    ```sh
    In file included from /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime.h:83:0,
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/host_config.h:138:2: error: #error — unsupported GNU version! gcc versions later than 8 are not supported!
    ```
    A: Install `g++-8` and set to `Makefile.config` `CUSTOM_CXX`
    ```sh
    $ sudo apt install gcc-8 g++-8

    $ vi Makefile.config +27
    CUSTOM_CXX := g++-8
    ```
  - **Q: Protobuf error: undefined reference to google::protobuf::xxx** [Undefined reference to google protobuf](https://github.com/BVLC/caffe/issues/3046)
    ```sh
    .build_release/lib/libcaffe.so: undefined reference to google::protobuf::internal::UnknownFieldParse
    .build_release/lib/libcaffe.so: undefined reference to google::protobuf::RepeatedPtrField
    ```
    A: [Protobuf](#protobuf)
  - **Q: Couldn't build proto file into descriptor pool: duplicate file name**
    ```sh
    pip install --no-binary protobuf
    ```
  - **Q: Error while building libboost_python `pyconfig.h: No such file or directory`** [Cannot build boost python library (fatal error: pyconfig.h: No such file or directory)](https://stackoverflow.com/questions/57244655/cannot-build-boost-python-library-fatal-error-pyconfig-h-no-such-file-or-dire)
    ```sh
    ./boost/python/detail/wrap_python.hpp:57:11: fatal error: pyconfig.h: No such file or directory
     # include <pyconfig.h>
    ```
    A: export `CPLUS_INCLUDE_PATH` inluding python source include
    ```sh
    CPLUS_INCLUDE_PATH="$HOME/local_bin/python-3.9.13/include/python3.9/" ./b2 install
    ```
***

# Docker
  - [BVLC/caffe/docker](https://github.com/BVLC/caffe/tree/master/docker)
  ```sh
  docker run --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) bvlc/caffe:cpu caffe train --solver=example_solver.prototxt
  ```
***
