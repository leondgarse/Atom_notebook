# ___2017 - 07 - 25 scikit-learn___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2017 - 07 - 25 scikit-learn___](#2017-07-25-scikit-learn)
  - [目录](#目录)
  	- [scikit-learn 介绍](#scikit-learn-介绍)
  	- [sklearn 约定规则](#sklearn-约定规则)
  	- [sklearn 载入数据集](#sklearn-载入数据集)
  	- [sklearn学习和预测一般流程](#sklearn学习和预测一般流程)
  	- [数据预处理](#数据预处理)
  	- [sklearn 库中的算法](#sklearn-库中的算法)
  	- [优化算法的参数](#优化算法的参数)
  	- [sklearn 中的评价尺度](#sklearn-中的评价尺度)
  - [sklearn 中的监督学习算法](#sklearn-中的监督学习算法)
  	- [基本概念](#基本概念)
  	- [分类算法 KNN](#分类算法-knn)
  	- [维数灾难 The curse of dimensionality](#维数灾难-the-curse-of-dimensionality)
  	- [线型回归模型 Linear regression model](#线型回归模型-linear-regression-model)
  	- [岭回归 Ridge 缩减 shrinkage 与过拟合](#岭回归-ridge-缩减-shrinkage-与过拟合)
  	- [Lasso 缩减与稀疏 Sparsity 降低模型复杂度](#lasso-缩减与稀疏-sparsity-降低模型复杂度)
  	- [Logistic 回归与sigmoid函数，回归算法用于分类](#logistic-回归与sigmoid函数回归算法用于分类)
  	- [支持向量机 SVM Support vector machines](#支持向量机-svm-support-vector-machines)
  - [交叉验证与模型参数选择](#交叉验证与模型参数选择)
  	- [score 方法与交叉验证 cross-validated scores](#score-方法与交叉验证-cross-validated-scores)
  	- [sklearn 库中的交叉验证生成器使用 Cross-validation generators](#sklearn-库中的交叉验证生成器使用-cross-validation-generators)
  	- [sklearn 库中的交叉验证生成器类别](#sklearn-库中的交叉验证生成器类别)
  	- [网格搜索 Grid-search 寻找模型的最佳参数](#网格搜索-grid-search-寻找模型的最佳参数)
  	- [自动使用交叉验证选择参数的估计模型 Cross-validated estimators](#自动使用交叉验证选择参数的估计模型-cross-validated-estimators)
  - [sklearn 中的无监督学习算法](#sklearn-中的无监督学习算法)
  	- [聚类 Clustering 将数据分成离散的组](#聚类-clustering-将数据分成离散的组)

  <!-- /TOC -->
***

## scikit-learn 介绍
  - scikit-learn是Python的一个开源机器学习模块，建立在NumPy，SciPy和matplotlib模块之上
## sklearn 约定规则
  - 除非专门指定，输入数据被转化为float64类型
    ```python
    import numpy as np
    from sklearn import random_projection

    rng = np.random.RandomState(0)
    x = rng.rand(10, 2000)
    x = np.array(x, dtype='float32')
    x.dtype
    # Out[38]: dtype('float32')

    # cast to float64 by fit_transform(x)
    transformer = random_projection.GaussianRandomProjection()
    x_new = transformer.fit_transform(x)
    x_new.dtype
    # Out[41]: dtype('float64')
    ```
  - 回归的目标预测值会转化成float64，分类的目标值类型不变
    ```python
    from sklearn import datasets
    from sklearn.svm import SVC
    iris = datasets.load_iris()
    clf = SVC()
    # iris.target is an integer array, predict() returns an integer array
    clf.fit(iris.data, iris.target)
    list(clf.predict(iris.data[:3]))
    Out[47]: [0, 0, 0]

    # iris.target_names was for fitting, predict() returns a string array
    clf.fit(iris.data, iris.target_names[iris.target])
    list(clf.predict(iris.data[:3]))
    Out[49]: ['setosa', 'setosa', 'setosa']
    ```
  - 模型的高级参数 (Hyper-parameters) 在创建后可以通过 **sklearn.pipeline.Pipeline.set_params** 方法修改，通过调用 **fit()** 覆盖之前的参数
    ```python
    import numpy as np
    from sklearn.svm import SVC
    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    y = rng.binomial(1, 0.5, 100)
    X_test = rng.rand(5, 10)

    # default kernel is first changed to linear after the estimator has been constructed via SVC()
    clf = SVC()
    clf.set_params(kernel='linear').fit(X, y)
    Out[12]:
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    clf.predict(X_test)
    Out[13]: array([1, 0, 1, 1, 0])

    # changed kernel to rbf to refit the estimator
    clf.set_params(kernel='rbf').fit(X, y)
    Out[14]:
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    clf.predict(X_test)
    Out[15]: array([0, 0, 0, 1, 0])
    ```
  - Multiclass 与 multilabel fitting
    - 当使用 [multiclass classifiers](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass) 时训练与预测的结果取决于目标值的形式
      ```python
      from sklearn.svm import SVC
      from sklearn.multiclass import OneVsRestClassifier
      from sklearn.preprocessing import LabelBinarizer

      # 一维目标值
      X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
      y = [0, 0, 1, 1, 2]
      classif = OneVsRestClassifier(estimator=SVC(random_state=0))
      classif.fit(X, y).predict(X)
      Out[23]: array([0, 0, 1, 1, 2])

      # 二维二进制目标值
      y = LabelBinarizer().fit_transform(y)
      y
      Out[25]:
      array([[1, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 1, 0],
             [0, 0, 1]])

      classif.fit(X, y).predict(X)
      Out[26]:
      array([[1, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]])
      ```
    - 当使用多组标签 **multiple labels** 时，预测结果中可能有全0值，表示不符合任何一个标签，或者多个1值，表示符合多个分组
      ```python
      from sklearn.preprocessing import MultiLabelBinarizer
      y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
      y = MultiLabelBinarizer().fit_transform(y)
      y
      Out[31]:
      array([[1, 1, 0, 0, 0],
             [1, 0, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 1, 1, 0],
             [0, 0, 1, 0, 1]])

      classif.fit(X, y).predict(X)
      Out[32]:
      array([[1, 1, 0, 0, 0],
             [1, 0, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 1, 0, 0],
             [1, 0, 1, 0, 0]])
      ```
## sklearn 载入数据集
  - scikit-learn 处理的数据集是二维的，其中行向量表示多个采样值 **samples axis**，列向量表示特征值 **features axis**
  - scikit-learn 内包含了常用的机器学习数据集，比如做分类的 iris 和 digit 数据集，用于回归的经典数据集 Boston house prices
  - scikit-learn 载入的数据集是以类似于 **字典的形式** 存放的，该对象中包含了所有有关该数据的数据信息 (甚至还有参考文献)
  - **鸢尾花 iris 数据集**，是一类多重变量分析的数据集，通过花瓣petal 与 萼片sepal 的长宽，划分鸢尾花的三个种类 山鸢尾Setosa / 杂色鸢尾Versicolour / 维吉尼亚鸢尾Virginica
    ```python
    from sklearn import datasets
    iris = datasets.load_iris()
    ```
  - **数据值统一存放在.data的成员中**，iris数据中每个实例有4维特征，分别为：sepal length、sepal width、petal length和petal width
    ```python
    type(iris.data)
    Out[34]: numpy.ndarray

    iris.data.shape
    Out[35]: (150, 4)

    iris.data[:3]
    Out[36]:
    array([[ 5.1,  3.5,  1.4,  0.2],
           [ 4.9,  3. ,  1.4,  0.2],
           [ 4.7,  3.2,  1.3,  0.2]])
    ```
  - 对于监督学习，比如分类问题，**数据对应的分类结果存在.target成员中**
    ```python
    np.unique(iris.target)
    Out[37]: array([0, 1, 2])

    iris.target[45:55]
    Out[46]: array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    ```
  - 如果数据集的形式不是(n_samples, n_features)，需要进行 **预处理**
    ```python
    # digits 数据集的 image 数据是 1797 x 8 x 8 的形式
    digits = datasets.load_digits()
    digits.data.shape
    Out[12]: (1797, 64)

    digits.images.shape
    Out[13]: (1797, 8, 8)
    # 转化为 1797 * 64 的数据集
    data = digits.images.reshape(digits.images.shape[0], -1)
    data.shape
    Out[15]: (1797, 64)
    ```
## sklearn学习和预测一般流程
  - scikit-learn 实现的主要API就是各种估计模型，提供了各种机器学习算法的接口，每个算法的调用就像一个黑箱，只需要根据自己的需求，设置相应的参数
  - 模型的所有 **参数** 都可以在初始化时指定，或者通过相应的属性修改
  - scikit-learn 每个模型都提供一个 **fit(X, Y)** 接口函数，可以接受一个二维数据集参数，用于 **模型训练**，模型通过 fit() 函数估计出的参数在模型的属性中以下划线 `_` 结尾
    ```python
    estimator.fit(data)
    estimator.estimated_param_
    ```
  - 模型预测使用 **predict(T)** 函数
  - **示例** digits手写数字数据集 与 支持向量机SVM
    ```python
    # 调用最常用的支撑向量分类机（SVC）
    from sklearn import svm
    # 不使用默认参数，使用用户自己给定的参数
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.gamma
    Out[48]: 0.001

    # 分类器的具体信息和参数
    clf
    Out[49]:
    SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    ```
    分类器的学习和预测可以分别利用 **fit(X,Y)** 和 **predict(T)** 来实现
    ```python
    # 将digit数据划分为训练集和测试集，前n-1个实例为训练集，最后一个为测试集
    from sklearn import datasets
    from sklearn import svm
    clf = svm.SVC(gamma=0.001, C=100.)
    digits = datasets.load_digits()
    # 模型训练
    clf.fit(digits.data[:-1], digits.target[:-1])
    # 模型训练后的参数
    clf.classes_
    Out[65]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # 预测结果，使用列向量
    clf.predict(digits.data[-1:])
    # Out[66]: array([8])
    # 真实结果
    digits.target[-1]
    # Out[67]: 8

    # 绘制digits图形
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(3, 3))
    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    ```
    ![](images/sikit-learn_digit8.png)
  - 模型存储
    - 可以使用pickle存储模型
    - 对于scikit，可以使用joblib，在大数据集上更有效，但只能存储到文件中，随后可以在其他程序中使用存储的模型
    ```python
    from sklearn.externals import joblib
    joblib.dump(clf, 'foo.pkl')
    # Out[30]: ['foo.pkl']

    clf2 = joblib.load('foo.pkl')
    clf2.predict(digits.data[-1:])
    # Out[32]: array([8])
    ```
## 数据预处理
  - 大多数的梯度方法（几乎所有的机器学习算法都基于此）对于数据的缩放很敏感，因此在运行算法之前，应该进行 **标准化或规格化**
    - 标准化包括替换所有特征的名义值，让它们每一个的值在0和1之间
    - 规格化包括数据的预处理，使得每个特征的值有0和1的离差
    ```python
    # 数据获取
    import numpy as np
    import urllib.request
    # url with dataset
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    # download the file
    raw_data = urllib.request.urlopen(url)
    # load the CSV file as a numpy matrix
    dataset = np.loadtxt(raw_data, delimiter=",")
    # separate the data from the target attributes
    x = dataset[:,:-1]
    y = dataset[:,-1]

    # 标准化与规格化
    from sklearn import preprocessing
    # normalize the data attributes
    normalized_x = preprocessing.normalize(x)
    # standardize the data attributes
    standardized_x = preprocessing.scale(x)
    ```
    运行结果
    ```python
    x[1]
    Out[6]: array([ 1., 85., 66., 29., 0., 26.6, 0.351, 31. ])

    normalized_x[1]
    Out[7]:
    array([ 0.008424, 0.71604034, 0.55598426, 0.24429612,
            0., 0.22407851, 0.00295683, 0.26114412])

    standardized_x[1]
    Out[8]:
    array([-0.84488505, -1.12339636, -0.16054575, 0.53090156,
           -0.69289057, -0.68442195, -0.36506078, -0.19067191])

    y[1]
    Out[9]: 0.0
    ```
  - **特征选取和特征工程** 解决一个问题最重要的是恰当选取特征、甚至创造特征的能力，特征工程是一个相当有创造性的过程，有时候更多的是靠直觉和专业的知识，但对于特征的选取，已经有很多的算法可供直接使用，如树算法就可以计算特征的信息量
    ```python
    from sklearn import metrics
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(x, y)
    Out[15]:
    ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
               max_depth=None, max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
               verbose=0, warm_start=False)
    # display the relative importance of each attribute
    model.feature_importances_
    Out[16]:
    array([ 0.10275952,  0.25440925,  0.09016066,  0.07965089,
            0.0757741 , 0.13128523,  0.11951687,  0.14644348])
    ```
  - 其他所有的方法都是基于对 **特征子集的高效搜索**，从而找到最好的子集，意味着演化了的模型在这个子集上有最好的质量，**递归特征消除算法**（RFE）是这些搜索算法的其中之一
    ```python
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(model, 3)
    rfe = rfe.fit(x, y)
    # summarize the selection of the attributes
    rfe.support_
    Out[20]: array([ True, False, False, False, False,  True,  True, False], dtype=bool)

    rfe.ranking_
    Out[21]: array([1, 2, 3, 5, 6, 1, 1, 4])
    ```
## sklearn 库中的算法
  - 除了分类和回归问题，Scikit-Learn还有海量的更复杂的算法，包括了聚类，以及建立混合算法的实现技术，如 Bagging 和 Boosting
  - **逻辑回归** 大多数情况下被用来解决分类问题（二元分类），但多类的分类（所谓的一对多方法）也适用，优点是对于每一个输出的对象都有一个对应类别的概率
    ```python
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(x, y)
    Out[27]:
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)

    # make predictions
    expected = y
    predicted = model.predict(x)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
                 precision    recall  f1-score   support

            0.0       0.79      0.90      0.84       500
            1.0       0.74      0.55      0.63       268

    avg / total       0.77      0.77      0.77       768

    metrics.confusion_matrix(expected, predicted)
    Out[33]:
    array([[448,  52],
           [121, 147]])
    ```
  - **朴素贝叶斯** 主要任务是恢复训练样本的数据分布密度，这个方法通常在多类的分类问题上表现的很好
    ```python
    from sklearn import metrics
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(x, y)
    Out[36]: GaussianNB(priors=None)

    # make predictions
    expected = y
    predicted = model.predict(x)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
                 precision    recall  f1-score   support

            0.0       0.80      0.84      0.82       500
            1.0       0.68      0.62      0.64       268

    avg / total       0.76      0.76      0.76       768

    metrics.confusion_matrix(expected, predicted)
    Out[40]:
    array([[421,  79],
           [103, 165]])
    ```
  - **k-最近邻 KNN** 通常用于一个更复杂分类算法的一部分，例如，可以用它的估计值做为一个对象的特征，有时候，一个简单的kNN算法在良好选择的特征上会有很出色的表现，当参数（主要是metrics）被设置得当，这个算法在回归问题中通常表现出最好的质量
    ```python
    from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier
    # fit a k-nearest neighbor model to the data
    model = KNeighborsClassifier()
    model.fit(x, y)
    Out[43]:
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')

    # make predictions
    expected = y
    predicted = model.predict(x)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
                 precision    recall  f1-score   support

            0.0       0.83      0.88      0.85       500
            1.0       0.75      0.65      0.70       268

    avg / total       0.80      0.80      0.80       768

    metrics.confusion_matrix(expected, predicted)
    Out[46]:
    array([[442,  58],
           [ 93, 175]])
    ```
  - **决策树 分类和回归树（CART）** 适用的分类问题中对象有可分类的特征，且被用于回归和分类，决策树很适用于多类分类
    ```python
    from sklearn import metrics
    from sklearn.tree import DecisionTreeClassifier
    # fit a CART model to the data
    model = DecisionTreeClassifier()
    model.fit(x, y)
    Out[49]:
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')

    # make predictions
    expected = y
    predicted = model.predict(x)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
                 precision    recall  f1-score   support

            0.0       1.00      1.00      1.00       500
            1.0       1.00      1.00      1.00       268

    avg / total       1.00      1.00      1.00       768

    metrics.confusion_matrix(expected, predicted)
    Out[52]:
    array([[500,   0],
           [  0, 268]])
    ```
  - **支持向量机 SVM** 是最流行的机器学习算法之一，主要用于分类问题，同样也用于逻辑回归，SVM在一对多方法的帮助下可以实现多类分类
    ```python
    from sklearn import metrics
    from sklearn.svm import SVC
    # fit a SVM model to the data
    model = SVC()
    model.fit(x, y)
    Out[54]:
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    # make predictions
    expected = y
    predicted = model.predict(x)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
                   precision    recall  f1-score   support

            0.0       1.00      1.00      1.00       500
            1.0       1.00      1.00      1.00       268

    avg / total       1.00      1.00      1.00       768

    metrics.confusion_matrix(expected, predicted)
    Out[56]:
    array([[500,   0],
           [  0, 268]])
    ```
## 优化算法的参数
  - 在编写高效的算法的过程中最难的步骤之一就是正确参数的选择，Scikit-Learn提供了很多函数来帮助解决这个问题
    ```python
    # 规则化参数的选择，在其中不少数值被相继搜索了
    import numpy as np
    from sklearn.linear_model import Ridge
    # This module will be removed in 0.20
    from sklearn.grid_search import GridSearchCV
    # prepare a range of alpha values to test
    alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    # create and fit a ridge regression model, testing each alpha
    model = Ridge()
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    grid.fit(x, y)
    Out[66]:
    GridSearchCV(cv=None, error_score='raise',
           estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'alpha': array([  1.00000e+00,   1.00000e-01,   1.00000e-02,   1.00000e-03,
             1.00000e-04,   0.00000e+00])},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)

    # summarize the results of the grid search
    grid.best_score_
    Out[68]: 0.27961755931297216

    grid.best_estimator_.alpha
    Out[69]: 1.0
    ```
  - 随机地从既定的范围内选取一个参数有时候更为高效，估计在这个参数下算法的质量，然后选出最好的
    ```python
    import numpy as np
    from scipy.stats import uniform as sp_rand
    from sklearn.linear_model import Ridge
    from sklearn.grid_search import RandomizedSearchCV
    # prepare a uniform distribution to sample for the alpha parameter
    param_grid = {'alpha': sp_rand()}
    # create and fit a ridge regression model, testing random alpha values
    model = Ridge()
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
    rsearch.fit(x, y)
    Out[7]:
    RandomizedSearchCV(cv=None, error_score='raise',
              estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001),
              fit_params={}, iid=True, n_iter=100, n_jobs=1,
              param_distributions={'alpha': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7f3aff8469e8>},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              scoring=None, verbose=0)

    # summarize the results of the random parameter search
    rsearch.best_score_
    Out[8]: 0.27961752547790575

    rsearch.best_estimator_.alpha
    Out[9]: 0.99827013993379388
    ```
## sklearn 中的评价尺度
  - 在sklearn中包含四种评价尺度
    - explained_variance_score()
    - mean_absolute_error()
    - mean_squared_error()
    - r2_score()
  - 均方差 mean-squared-error
    ```python
    MSE(y, yp) = 1 / N * Σ(1, N)(y - yp)^2
    ```
  - 平均绝对值误差 mean_absolute_error
    ```python
    MAE(y, yp) = 1 / N * Σ(1, N)|y - yp|
    ```
  - 可释方差得分 explained_variance_score
    ```python
    EVS(y, yp) = 1 - var(y - yp) / var(y)
    ```
    最大值是1，表示模型的拟合程度最好，值越小则效果越差
  - 中值绝对误差 Median absolute error
    ```python
    MedAE(y, yp) = median(|y1 - yp1|, ... , |yN - ypN|)
    ```
    适应含有离群点的数据集
  - R2 决定系数（拟合优度）
    ```python
    R2(y, yp) = 1 - Σ(1, N)(y - yp)^2 / Σ(1, N)(y - mean(y))^2
    ```
    表征回归方程在多大程度上解释了因变量的变化，或者说方程对观测值的拟合程度
  - **参数 multioutput**
    - 用来指定在多目标回归问题中，若干单个目标变量的损失或得分以什么样的方式被平均起来
    - 默认值 **uniform_average**，将所有预测目标值的损失以等权重的方式平均起来
    - 指定一个 **shape 为（n_oupputs,）的ndarray**，那么数组内的数将被视为是对每个输出预测损失（或得分）的加权值，最终的损失按照指定的加权方式来计算
    - 指定为 **raw_values**，那么所有的回归目标的预测损失或预测得分都会被单独返回一个shape是（n_output）的数组中
***

# sklearn 中的监督学习算法
## 基本概念
  - **监督学习算法** 一般用于学习两个数据集之间的关系，**观测集X** 与 **目标集 Y**，预测结果通常称为 **target** 或 **labels**，通常情况下，Y是一个一维向量
  - scikit-learn 中所有的监督学习算法都实现了 **fit(X, y)** 方法用于模型训练，以及 **predict(X)** 方法用于预测未分组(unlabeled)数据 X 的标签值(labels) Y
  - **分类算法** 预测的目标值是离散的，即将观测值划分成有限多个目标值，分类算法中的目标值y是一个数字或字符串组成的向量
  - **回归算法** 预测的目标值是连续的
  - **训练集与测试集** 在实验任何机器学习算法时，应避免使用训练模型的数据来测试预测结果，这无法反应模型在新数据上的预测效果
## 分类算法 KNN
  - [KNN](http://scikit-learn.org/stable/modules/neighbors.html#neighbors) k-Nearest neighbors classifier k-近邻，是最简单的分类算法
  - 对于新的预测数据，在已分类数据的训练集中寻找距离最近的数据，将其对应的分类标签作为新数据的预测分类
  - iris 数据集上的 KNN 示例
    ```python
    import numpy as np
    from sklearn import datasets

    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target

    # Split iris data in train and test data randomly
    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    iris_X_train = iris_X[indices[:-10]]
    iris_y_train = iris_y[indices[:-10]]
    iris_X_test  = iris_X[indices[-10:]]
    iris_y_test  = iris_y[indices[-10:]]

    # Create and fit a nearest-neighbor classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)
    knn.predict(iris_X_test)
    # Out[32]: array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])

    iris_y_test
    # Out[33]: array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])
    ```
## 维数灾难 The curse of dimensionality
  - 当维数增大时，**高维空间数据会变得更稀疏**
    - 维数 p=1 时，N 个样本数据间的平均距离是 1 / N
    - 维数 p=2 时，N 个样本数据间的平均距离是 (1 / N) ^ (1 / 2)，即需要 N ^ 2 个点才能维持距离为 1 / N
    - 维数 p=p 时，N 个样本数据间的平均距离是 (1 / N) ^ (1 / p)，即需要 N ^ p 个点才能维持距离为 1 / N
  - 在以距离作为预测依据的机器学习算法(如 KNN)中，当维数增大时，空间数据会变得更稀疏，各个分组间的界限会变小，算法预测的效率会降低
## 线型回归模型 Linear regression model
  - **diabetes 糖尿病数据集** 包含442个病人的10个生理特征数据 (age, sex, weight, blood pressure)，以及一年后的病情指标
    ```python
    diabetes = datasets.load_diabetes()
    diabetes.data.shape
    Out[35]: (442, 10)

    diabetes_X_train = diabetes.data[:-20]
    diabetes_X_test  = diabetes.data[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test  = diabetes.target[-20:]
    ```
  - 线型回归模型 Linear models，将数据集拟合成一个一阶模型，使得预测的总方差最小
    ```python
    y = Xβ + ε
    其中
        X: 数据集
        y: 目标向量
        β: 预测系数
        ε: 观测噪声
    ```
  - 使用示例
    ```python
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    # Out[42]: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

    # 线型回归模型的预测系数
    print(regr.coef_)

    # 预测错误均方差
    np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)
    # Out[46]: 2004.5676026898223

    # Explained variance score: 1表示完美拟合，0表示没有线型关系
    regr.score(diabetes_X_test, diabetes_y_test)
    # Out[50]: 0.58507530226905713
    ```
## 岭回归 Ridge 缩减 shrinkage 与过拟合
  - 如果数据集中的观测值过少，会使模型预测中产生较大的方差，即模型之间的差异会变大
    ```python
    X = np.c_[ .5, 1].T
    y = [.5, 1]
    test = np.c_[ 0, 2].T
    regr = linear_model.LinearRegression()

    import matplotlib.pyplot as plt
    plt.figure()

    np.random.seed(0)
    for _ in range(6):
        # 随机添加噪声
        this_X = .1 * np.random.normal(size=(2, 1)) + X
        regr.fit(this_X, y)
        plt.plot(test, regr.predict(test))
        plt.scatter(this_X, y, s=3)
    ```
    ![](images/sk_linear_model.png)
  - **岭回归 Ridge** 在高维数据中的一种解决方法是 **将一部分回归系数 β 缩减到0**，减少了模型的复杂度，但同时增大了模型偏差
    ```python
    # 使用岭回归 Ridge 模型
    regr = linear_model.Ridge(alpha=.1)
    plt.figure()

    np.random.seed(0)
    for _ in range(6):
        this_X = .1*np.random.normal(size=(2, 1)) + X
        regr.fit(this_X, y)
        plt.plot(test, regr.predict(test))
        plt.scatter(this_X, y, s=3)
    ```
    ![](images/sk_ridge.png)
  - **权衡偏差与方差 bias/variance tradeoff** 岭回归模型中的 alpha 参数增大，会导致更大的偏差与更小的方差，调整 alpha 参数可以使得模型的效果最好
    ```python
    # 调整alpha值，在 diabetes 数据集上测试模型拟合效果
    alphas = np.logspace(-4, -1, 6)
    from __future__ import print_function
    [regr.set_params(alpha=alpha
               ).fit(diabetes_X_train, diabetes_y_train,
               ).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]
    Out[77]:
    [0.58511106838835292, 0.58520730154446765,
     0.58546775406984908, 0.58555120365039159,
     0.58307170855541623, 0.57058999437280111]
    ```
  - **过拟合 overfitting** 模型过拟合，对新数据的预测效果变差
  - **正则化 regularization** 岭回归中引入的偏差称为正则化 regularization，降低模型的过拟合
## Lasso 缩减与稀疏 Sparsity 降低模型复杂度
  - **Lasso 缩减** least absolute shrinkage and selection operator，只选取与预测目标关联度高的特征，而将不重要的特征系数缩减到0，lasso estimate 具有 shrinkage 和 selection 两种功能
  - 岭回归会减小数据集中不重要特征的系数，但不会缩减到0，lasso缩减会将某些系数缩减到0，即特征选择 selection
  - 减小问题的复杂度，防止过拟合，是一种 **稀疏方法 sparse method**
  - 稀疏 Sparsity 可以看作是奥卡姆剃刀原则的应用
    ```
    Occam’s razor: prefer simpler models
    ```
  - lasso 回归示例
    ```python
    # 使用 lasso 回归模型
    regr = linear_model.Lasso()
    scores = [regr.set_params(alpha=alpha
               ).fit(diabetes_X_train, diabetes_y_train
               ).score(diabetes_X_test, diabetes_y_test)
            for alpha in alphas]
    best_alpha = alphas[scores.index(max(scores))]
    best_alpha
    # Out[86]: 0.025118864315095794

    regr.alpha = best_alpha
    regr.fit(diabetes_X_train, diabetes_y_train)
    # 某些系数缩减到了0
    regr.coef_
    Out[88]:
    array([   0.        , -212.43764548,  517.19478111,  313.77959962,
           -160.8303982 ,   -0.        , -187.19554705,   69.38229038,
            508.66011217,   71.84239008])
    ```
  - **不同的算法可以用于解决同样的数学问题**
    - scikit-learn 中的 **Lasso 对象** 使用坐标下降的方法 coordinate descent method 解决 lasso 回归问题，这在大数据集上很有效
    - scikit-learn 中同样提供了 **LassoLars 对象**，使用 **LARS 算法** ((Least Angle Regression 最小角回归)，在估计的权重向量非常稀疏，如观测值很少的数据集中很有效
## Logistic 回归与sigmoid函数，回归算法用于分类
  - 在分类预测中，线型回归模型通常并不适用，因为模型会给远离决策边界的数据更大的权重
  - **sigmoid or logistic** 将线型回归模型的结果转化成分类结果，类似与阶跃函数
    ```
    sigmoid(z) = 1 / (1 + e^(-z))
    ```
  - iris数据集上 Logistic 回归用于分类示例
    ```python
    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(iris_X_train, iris_y_train)
    Out[90]:
    LogisticRegression(C=100000.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

    logistic.predict(iris_X_test)
    Out[91]: array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])

    iris_y_test
    Out[92]: array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])
    ```
  - Logistic 回归中的 **缩减 shrinkage 与稀疏 sparsity**
    - **参数 C**，默认1.0，指定数据正则化的程度，值越小正则化越低
    - **参数 penalty**，默认'l2'，指定惩罚的基准 the norm used in the penalization，'l2'指定缩减 Shrinkage，'l1'指定稀疏 Sparsity
## 支持向量机 SVM Support vector machines
  - SVM 包含 **回归模型 SVR** Support Vector Regression，以及 **分类模型 SVC** Support Vector Classification
  - **SVM 线性模型** SVM 模型试图找到一组样本值，来建立两个分组间的分隔超平面，使得该组样本值与分隔超平面的间隔最大，正则化程度通过 **参数 C** 设定
    - C 值越小，正则化程度高，使用分隔超平面附近更多或全部的点来计算间隔
    - C 值越大，正则化程度低，使用分隔超平面最近的点来计算间隔
    - [Plot different SVM classifiers in the iris dataset](http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py)
  - iris数据集上 SVM 示例
    ```python
    from sklearn import svm
    svc = svm.SVC(kernel='linear')
    svc.fit(iris_X_train, iris_y_train)
    Out[96]:
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    svc.predict(iris_X_test)
    Out[97]: array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])
    ```
  - **核函数 kernels**，数据集的特征并不总是线性可分的，可以使用 **核技巧 kernel trick** 在特征空间上应用一个决策函数 decision function，将数据映射到另一个特征空间，通常会将 **低维特征空间映射到高维空间**
  - 示例 [SVM-Kernels](http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html)
    ```python
    # Linear kernel
    svc = svm.SVC(kernel='linear')
    # Polynomial kernel
    svc = svm.SVC(kernel='poly', degree=3)
    # RBF kernel (Radial Basis Function)
    svc = svm.SVC(kernel='rbf')
    ```
    ![](images/svm_kernels.png)
  - 其他链接
    - [plot iris exercise](http://scikit-learn.org/stable/_downloads/plot_iris_exercise.py)
    - [Libsvm GUI]( http://scikit-learn.org/stable/auto_examples/applications/svm_gui.html#sphx-glr-auto-examples-applications-svm-gui-py)
***

# 交叉验证与模型参数选择
## score 方法与交叉验证 cross-validated scores
  - 每个模型都有一个 **score()方法** 用于评估模型在新数据上的预测质量，值越大模型估计越好
    ```python
    from sklearn import datasets, svm
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target
    svc = svm.SVC(C=1, kernel='linear')
    svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])
    Out[12]: 0.97999999999999998
    ```
  - **kfold 交叉验证** 将数据的特征与目标划分成连续的k个部分 fold，其中 K-1 个子集作为训练数据，另一个作为测试数据，可以更好的估计模型效果
    ```python
    X_fold = np.split(X_digits, 3)
    y_fold = np.split(y_digits, 3)
    scores = lsit()
    scores = list()
    for k in range(3):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_fold)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_fold)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)
        scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
    scores
    Out[22]: [0.93489148580968284, 0.95659432387312182, 0.93989983305509184]
    ```
## sklearn 库中的交叉验证生成器使用 Cross-validation generators
  - 对于流行的交叉验证策略，scikit-learn 中有几个类可以用于生成训练集 / 测试集的索引列表
  - **split 方法** 接受一个待划分的数据集，产生(yields 返回)一个训练 / 测试数据索引的列表生成器
    ```python
    from sklearn.model_selection import KFold, cross_val_score
    X = ["a", "a", "b", "c", "c", "c"]
    k_fold = KFold(n_splits=3)
    for train_indice, test_indices in k_fold.split(X):
        print('Train: %s | test: %s' % (train_indices, test_indices))
    Out[]
    Train: [2 3 4 5] | Test [0 1]
    Train: [0 1 4 5] | Test [2 3]
    Train: [0 1 2 3] | Test [4 5]

    # The cross-validation can then be performed easily:
    [svc.fit(X_digits[test], y_digits[test]).score(X_digits[train], y_digits[train])
    for test, train in k_fold.split(X_digits)]
    Out[38]: [0.93489148580968284, 0.95659432387312182, 0.93989983305509184]
    ```
  - **cross_val_score 方法** 计算交叉验证的得分 cross-validation score，参数指定一个 **估计模型**，**交叉验证生成器** 与 **待验证数据集**，cross_val_score 方法会自动使用每个训练数据集训练模型，在测试集上测试并返回得分 score
    ```python
    cross_val_score(estimator, X, y=None, groups=None, scoring=None,
        cv=None, n_jobs=1, verbose=0, fit_params=None,
        pre_dispatch='2*n_jobs')
    ```
    **参数 n_jobs** -1 表示使用当前计算机上的所有 CPU
    ```python
    cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
    Out[39]: array([ 0.93489149,  0.95659432,  0.93989983])
    ```

    **参数 scoring** 默认计算独立的得分 individual scores，可以指定其他可以选择的方法，模块 metrics 中获取更多 scoring 方法
    ```python
    cross_val_score(svc, X_digits, y_digits, cv=k_fold, scoring='precision_macro')
    Out[40]: array([ 0.93969761,  0.95911415,  0.94041254])
    ```
## sklearn 库中的交叉验证生成器类别
  - **KFold (n_splits, shuffle, random_state)** 将数据集划分成 K 个子集 folds，其中 K-1 个子集作为训练数据，另一个作为测试数据
  - **StratifiedKFold (n_splits, shuffle, random_state)** 类似于 KFold，但在每个子集中尽量包含每一个分类
    ```python
    from sklearn.model_selection import StratifiedKFold
    sk_fold = StratifiedKFold(n_splits=2)
    # 参数必须有目标分类
    for train_indice, test_indices in sk_fold.split(X, [1, 1, 2, 3, 3, 2]):
        print('Train: %s | test: %s' % (train_indices, test_indices))

    Out[]
    Train: [0 1 2 3] | test: [0 2 3]
    Train: [0 1 2 3] | test: [1 4 5]
    ```
  - **GroupKFold (n_splits)** 可以指定一个groups参数，相同分组的数据不会分在同一个子集中
    ```python
    from sklearn.model_selection import GroupKFold
    gk_fold = GroupKFold(n_splits=3)
    # 参数指定 groups
    for train_indice, test_indices in gk_fold.split(X, groups=[1, 1, 2, 3, 3, 2]):
        print('Train: %s | test: %s' % (train_indices, test_indices))

    Out[]
    Train: [0 1 2 3] | test: [3 4]
    Train: [0 1 2 3] | test: [2 5]
    Train: [0 1 2 3] | test: [0 1]
    ```
  - **ShuffleSplit (n_splits, test_size, train_size, random_state)** 随机产生训练 / 测试集索引
  - **StratifiedShuffleSplit** 类似于 ShuffleSplit，但在每个子集中尽量包含每一个分类
  - **GroupShuffleSplit** 可以指定一个groups参数，相同分组的数据不会分在同一个子集中
  - **LeaveOneGroupOut ()** 根据goups参数提供的分组划分数据
    ```python
    from sklearn.model_selection import LeaveOneGroupOut
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 1, 2])
    groups = np.array([1, 1, 2, 2])
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, y, groups)
    # Out[98]: 2

    for tr, te in logo.split(X, y, groups):
        print('TRAIN: ', tr, 'TEST: ', te)

    Out[]
    TRAIN:  [2 3] TEST:  [0 1]
    TRAIN:  [0 1] TEST:  [2 3]
    ```
  - **LeavePGroupsOut (n_groups)** 测试集中包含 P 个分组，n_groups参数指定p，groups参数指定分组
    ```python
    from sklearn.model_selection import LeavePGroupsOut
    lpgo = LeavePGroupsOut(n_groups=2)
    lpgo.get_n_splits(X, y, groups)
    # Out[103]: 1
    # goups中的分组类别数量必须大于n_groups
    for tr, te in lpgo.split(X, y, [1, 1, 2, 3]):
         print('TRAIN: ', tr, 'TEST: ', te)
    Out[]
    TRAIN:  [3] TEST:  [0 1 2]
    TRAIN:  [2] TEST:  [0 1 3]
    TRAIN:  [0 1] TEST:  [2 3]
    ```
  - **LeaveOneOut ()** 测试集使用一个样本值，对于大数据集效率很低
    ```python
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    # Out[110]: 4

    for tr, te in loo.split(X, y, groups):
        print('TRAIN: ', tr, 'TEST: ', te)

    Out[]
    TRAIN:  [1 2 3] TEST:  [0]
    TRAIN:  [0 2 3] TEST:  [1]
    TRAIN:  [0 1 3] TEST:  [2]
    TRAIN:  [0 1 2] TEST:  [3]
    ```
  - **LeavePOut (p)** 测试集使用 p 个样本值，对于大数据集效率很低
  - **PredefinedSplit** 通过参数 test_fold 指定预定义的划分方式，不使用原数据集划分数据
    ```python
    from sklearn.model_selection import PredefinedSplit
    test_fold = [0, 1, -1, 1]
    ps = PredefinedSplit(test_fold)
    ps.get_n_splits()
    # Out[115]: 2

    for tr, te in ps.split():
        print('TRAIN: ', tr, 'TEST: ', te)

    Out[]
    TRAIN:  [1 2 3] TEST:  [0]
    TRAIN:  [0 2] TEST:  [1 3]
    ```
  - **使用示例**
    - [Cross-validation on Digits Dataset Exercise]( http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_digits.html#sphx-glr-auto-examples-exercises-plot-cv-digits-py)
## 网格搜索 Grid-search 寻找模型的最佳参数
  - scikit-learn 提供的对象，在指定的数据集与估计模型上，通过 **参数 param_grid** 指定估计模型某个参数的一组数据，寻找使得交叉验证得分 cross-validation score 最大的参数值
    ```python
    from sklearn.model_selection import GridSearchCV, cross_val_score
    # 生成10个随机数，作为svc的参数C
    Cs = np.logspace(-6, -1, 10)
    # 通过 param_grid 将 svc 的参数 C 指定成一个列表
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)
    clf.fit(X_digits[:1000], y_digits[:1000])
    clf.best_score_
    # Out[124]: 0.92500000000000004

    clf.best_estimator_.C
    # Out[125]: 0.0077426368268112772

    # 在测试集上的预测结果，可能没有训练集上的效果好
    clf.score(X_digits[1000:], y_digits[1000:])
    # Out[126]: 0.94353826850690092
    ```
  - GridSearchCV 默认使用 3-fold (KFold, k = 3) 交叉验证，在分类任务中会自动使用 stratified 3-fold
  - **嵌套的交叉验证**
    ```python
    cross_val_score(clf, X_digits, y_digits)
    Out[127]: array([ 0.93853821,  0.96327212,  0.94463087])
    ```
    两个交叉验证的循环并行运行，GridSearchCV 用交叉验证获得最佳参数，cross_val_score 检验模型的预测效果，可以很好的估计出模型在新数据上的表现
## 自动使用交叉验证选择参数的估计模型 Cross-validated estimators
  - 交叉验证选择参数的实现可以是基于算法的，因此对于一些估计模型，scikit-learn 提供了可以自动根据交叉验证选择参数的版本，通常是 **以 CV 结尾的**
    ```python
    from sklearn import linear_model, datasets
    lasso = linear_model.LassoCV()
    diabetes = datasets.load_diabetes()
    lasso.fit(diabetes.data, diabetes.target)
    # 模型自动选择参数
    lasso.alpha_
    Out[134]: 0.012291895087486173
    ```
  - 使用示例
    - [Cross-validation: evaluating estimator performance](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)
    - [Cross-validation on diabetes Dataset Exercise](http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#sphx-glr-auto-examples-exercises-plot-cv-diabetes-py)
***

# sklearn 中的无监督学习算法
## 聚类 Clustering 将数据分成离散的组
  - 聚类的结果不能保证完全恢复原数据的分类，首先合适的聚类数目很难确定，算法对初始化的参数也很敏感
  - the algorithm is sensitive to initialization, and can fall into local minima, although scikit-learn employs several tricks to mitigate this issue
  - K-means 最简单的聚类算法
    ```python
    from sklearn import cluster, datasets
    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target
    k_means = cluster.KMeans(n_clusters=3)
    k_means.fit(X_iris)
    k_means.labels_[::10]
    # Out[144]: array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=int32)

    y_iris[::10]
    # Out[145]: array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    ```
## 矢量量化 VQ vector quantization
  - 一种数据压缩技术，指从N维实空间RN到RN中L个离散矢量的映射，也可称为分组量化，将若干个标量数据组构成一个矢量，然后在矢量空间给以整体量化，从而压缩了数据而不损失多少信息，标量量化是矢量量化在维数为1时的特例
  - 聚类可以看作选取一小部分样本来压缩整体的信息，即矢量量化 VQ
    ```python
    # 聚类用于图像处理
    # 不同的 python 版本中，face 可能位于不同的库中
    import scipy as sp
    try:
      face = sp.face(gray=True)
    except AttributeError:
      from scipy import misc
      face = misc.face(gray=True)

    X = face.reshape((-1, 1))
    k_means = cluster.KMeans(n_clusters=5, n_init=1)
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    face_compressed = np.choose(labels, values)
    face_compressed.shape = face.shape
    ```
  - VQ 示例
    - [Vector Quantization Example]( http://scikit-learn.org/stable/auto_examples/cluster/plot_face_compress.html)
## 层次聚类 Hierarchical clustering Ward
  - **层次聚类 Hierarchical clustering** 是层次化的聚类，得出来的结构是一棵树
    - **Agglomerative 自底向上的方法**，一开始所有的样本值属于单独的的分类，随后根据相关程度合并分类，直到最后只剩下一个类别，完成一棵树的构造，当样本较少，分类数目大时，计算会比 k-means 更有效
    - **Divisive 自顶向下的方法**，初始所有的样本位于同一类，随着分级将数据迭代划分成更多分类，通常在分类数目不多时考虑这种方法，如 **二分k均值**（bisecting Ｋ-means）算法
  - **Connectivity-constrained clustering** Agglomerative 算法中可以根据一个连接图 connectivity graph 来指定哪些样本可以划分成同一类，通常会使用一个稀疏矩阵作为连接图，如用于图片处理
    ```python
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.image import grid_to_graph
    from sklearn.cluster import AgglomerativeClustering

    # Generate data
    try:  # SciPy >= 0.16 have face in misc
        from scipy.misc import face
        face = face(gray=True)
    except ImportError:
        face = sp.face(gray=True)

    # Resize it to 10% of the original size to speed up the processing
    face = sp.misc.imresize(face, 0.10) / 255.
    X = np.reshape(face, (-1, 1))

    # Define the structure A of the data. Pixels connected to their neighbors.
    # * 表示解包，将一个元组解开成单独的元素
    connectivity = grid_to_graph(*face.shape)
    ```
## 特征合并 Feature agglomeration
  - 在样本值与特征值相比数量不足时，**稀疏方法** 可以用于减小 **维数灾难** 的影响，另一种方法是 **合并相似的特征**，这可以通过在特征值轴向上使用聚类实现，即 **在转置的数据集上聚类**
    ```python
    # 使用手写数字数据集
    digits = datasets.load_digits()
    images = digits.images
    # 将 images 数据转化为二维
    X = np.reshape(images, (len(images), -1))
    # * 表示解包，将一个元组解开成单独的元素
    connectivity = grid_to_graph(*images[0].shape)
    agglo = cluster.FeatureAgglomeration(connectivity=connectivity, n_clusters=32)
    agglo.fit(X)

    # transform / inverse_transform 方法
    X_reduced = agglo.transform(X)
    X_approx = agglo.inverse_transform(X_reduced)
    images_approx = np.reshape(X_approx, images.shape)
    ```
    **transform / inverse_transform 方法** 有些估计模型提供的方法，可以用于减小数据集的维度
## 降维分解 Decompositions
  Components and loadings

  If X is our multivariate data, then the problem that we are trying to solve is to rewrite it on a different observational basis: we want to learn loadings L and a set of components C such that X = L C. Different criteria exist to choose the components
  - 主成分分析 Principal component analysis PCA selects the successive components that explain the maximum variance in the signal.
  - The point cloud spanned by the observations above is very flat in one direction: one of the three univariate features can almost be exactly computed using the other two. PCA finds the directions in which the data is not flat

  When used to transform data, PCA can reduce the dimensionality of the data by projecting on a principal subspace.
  ```python
  # Create a signal with only 2 useful dimensions
  x1 = np.random.normal(size=100)
  x2 = np.random.normal(size=100)
  x3 = x1 + x2
  X = np.c_[x1, x2, x3]
  from sklearn import decomposition
  pca = decomposition.PCA()
  pca.fit(X)
  print(pca.explained_variance_)  
  # As we can see, only the 2 first components are useful
  pca.n_components = 2
  X_reduced = pca.fit_transform(X)
  X_reduced.shape
  ```
  [Principal components analysis (PCA)](http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_3d.html)
  - 独立成分分析 Independent Component Analysis ICA  selects components so that the distribution of their loadings carries a maximum amount of independent information. It is able to recover non-Gaussian 高斯 independent signals
  ```python
  # Generate sample data
  import numpy as np
  from scipy import signal
  time = np.linspace(0, 10, 2000)
  s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
  s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
  s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
  S = np.c_[s1, s2, s3]
  S += 0.2 * np.random.normal(size=S.shape)  # Add noise
  S /= S.std(axis=0)  # Standardize data
  # Mix data
  A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])  # Mixing matrix
  X = np.dot(S, A.T)  # Generate observations
  # Compute ICA
  ica = decomposition.FastICA()
  S_ = ica.fit_transform(X)  # Get the estimated sources
  A_ = ica.mixing_.T
  np.allclose(X,  np.dot(S_, A_) + ica.mean_)
  ```
  [Blind source separation using FastICA](http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html)
***

# Pipelining
***

# Choosing the right estimator

  Often the hardest part of solving a machine learning problem can be finding the right estimator for the job.

  Different estimators are better suited for different types of data and different problems.

  The flowchart below is designed to give users a bit of a rough guide on how to approach problems with regard to which estimators to try on your data.

  Click on any estimator in the chart below to see its documentation.
  [Choosing the right estimator](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
  ![](images/ml_map.png)
