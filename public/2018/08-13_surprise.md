# ___2018 - 08 - 13 Surprise___

- [Welcome to Surprise’ documentation!](http://surprise.readthedocs.io/en/stable/index.html)
## Practice
  ```shell
  # user item rating timestamp
  $ head -n 10 ~/.surprise_data/ml-100k/ml-100k/u.data
  196	242	3	881250949
  186	302	3	891717742
  22	377	1	878887116
  244	51	2	880606923
  166	346	1	886397596
  298	474	4	884182806
  115	265	2	881171488
  253	465	5	891628467
  305	451	3	886324817
  6	86	3	883603013
  ```
  ```python
  from surprise import Dataset

  data = Dataset.load_builtin('ml-100k')
  # Retrieve the trainset.
  trainset = data.build_full_trainset()

  aa = list(trainset.all_ratings())
  aa[:5]
  # Out[69]: [(0, 0, 3.0), (0, 528, 4.0), (0, 377, 4.0), (0, 522, 3.0), (0, 431, 5.0)]

  [ii for ii in aa if ii[0]==196]
  [tt for tt in [ii for ii in aa if ii[0]==196] if tt[1] == 302]

  get_user_item_rating = lambda ds, uid, iid: [tt for tt in [ii for ii in ds.all_ratings() if ii[0] == uid] if tt[1] == iid]
  get_user_item_rating(trainset, 196, 302)
  # Out[73]: [(196, 302, 3.0)]

  df = DataFrame(list(trainset.all_ratings()), columns=['user', 'item', 'rating'])
  df[df.user == 196][df.item == 302]
  df[np.logical_and(df.user == 196, df.item == 302)]
  # Out[74]:
  #        user  item  rating
  # 28241   196   302     3.0

  trainset.knows_user(196)
  # Out[56]: True

  trainset.knows_item(302)
  # Out[57]: True

  trainset.n_users
  # Out[58]: 943

  trainset.n_items
  # Out[59]: 1682

  trainset.n_ratings
  # Out[60]: 100000

  trainset.ur.get(196)
  [ii for ii in trainset.ur.get(196) if ii[0] == 302]
  dict(trainset.ur.get(196)).get(302)
  # Out[79]: 3.0

  trainset.ir.get(302)
  [ii for ii in trainset.ir.get(302) if ii[0] == 196]

  uid = trainset.to_raw_uid(196)
  uid
  # Out[22]: '164'

  trainset.to_inner_uid(uid)
  # Out[23]: 196
  ```
  ```python
  from surprise import KNNBasic
  from surprise import Dataset

  # Load the movielens-100k dataset
  data = Dataset.load_builtin('ml-100k')

  # Retrieve the trainset.
  trainset = data.build_full_trainset()

  # Build an algorithm, and train it.
  algo = KNNBasic()
  algo.fit(trainset)

  algo.predict(uid=196, iid=302, r_ui=4, verbose=True)
  # Out[26]: Prediction(uid=196, iid=302, r_ui=4, est=3.52986, details={'was_impossible': True, 'reason': 'User and/or item is unkown.'})

  algo.predict(uid=str(196), iid=str(302), r_ui=4, verbose=True)
  # Out[28]: Prediction(uid='196', iid='302', r_ui=4, est=4.06292421377939, details={'actual_k': 40, 'was_impossible': False})

  algo.predict(uid=trainset.to_raw_uid(196), iid=trainset.to_raw_iid(302), r_ui=4, verbose=True)
  # Out[30]: Prediction(uid='164', iid='845', r_ui=4, est=3.772560784013482, details={'actual_k': 40, 'was_impossible': False})
  ```
  ```python
  # Compute the rating prediction for given user and item.
  predict(uid, iid, r_ui=None, clip=True, verbose=False)
  ```
  - uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`
  - iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`
  - r_ui(float): The true rating :math:`r_{ui}`. Optional, default is ``None``
***

# Getting Started
## Basic usage
  - **交叉验证 cross-validation**
    ```python
    from surprise import SVD
    from surprise import Dataset
    from surprise.model_selection import cross_validate

    # Load the movielens-100k dataset (download it if needed),
    data = Dataset.load_builtin('ml-100k')

    # We'll use the famous SVD algorithm.
    algo = SVD()

    # Run 5-fold cross-validation and print results
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    ```
    - `load_builtin 函数` 会下载 `movielens-100k` 数据集到 `~/.surprise_data` 文件夹
    - `cv 参数` 指定交叉验证中数据划分方式 `5-fold`
  - **不使用交叉验证的训练过程**
    - 训练 / 测试 数据集划分方式 Train-test split
    - fit 训练模型
    - accuracy 评估正确率
    - test 测试模型效果
    ```python
    from surprise import SVD
    from surprise import Dataset
    from surprise import accuracy
    from surprise.model_selection import train_test_split

    # Load the movielens-100k dataset (download it if needed),
    data = Dataset.load_builtin('ml-100k')

    # sample random trainset and testset
    # test set is made of 25% of the ratings.
    trainset, testset = train_test_split(data, test_size=.25)

    # We'll use the famous SVD algorithm.
    algo = SVD()

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    accuracy.rmse(predictions)
    # RMSE: 0.9411

    # train and test an algorithm with the following one-line
    predictions = algo.fit(trainset).test(testset)
    ```
  - **build_full_trainset 在整个训练集上训练** / **predict 方法预测**
    ```python
    from surprise import KNNBasic
    from surprise import Dataset

    # Load the movielens-100k dataset
    data = Dataset.load_builtin('ml-100k')

    # Retrieve the trainset.
    trainset = data.build_full_trainset()

    # Build an algorithm, and train it.
    algo = KNNBasic()
    algo.fit(trainset)
    ```
    预测用户 `user_id = 196` 对物品 `item_id = 302` 的得分 `rating`
    ```python
    # the true rating
    dict(trainset.ur.get(196)).get(302)
    # Out[80]: 3.0

    # Predicting
    uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
    iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

    # get a prediction for specific users and items.
    pred = algo.predict(uid, iid, r_ui=4, verbose=True)
    # user: 196        item: 302        r_ui = 4.00   est = 4.06   {'actual_k': 40, 'was_impossible': False}
    ```
    **predict** 方法使用的是 `raw ids`
    ```python
    predict(uid, iid, r_ui=None, clip=True, verbose=False)

    uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
    iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
    ```
## 使用自定义的数据集
  - 自定义数据集可以使用 csv 文件 / dataframe
  - 需要定义 Reader 来解析数据
  - **load_from_file** 加载 csv 文件
    ```python
    from surprise import BaselineOnly
    from surprise import Dataset
    from surprise import Reader
    from surprise.model_selection import cross_validate

    # path to dataset file
    file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')

    # As we're loading a custom dataset, we need to define a reader. In the
    # movielens-100k dataset, each line has the following format:
    # 'user item rating timestamp', separated by '\t' characters.
    reader = Reader(line_format='user item rating timestamp', sep='\t')

    data = Dataset.load_from_file(file_path, reader=reader)

    # We can now use this dataset as we please, e.g. calling cross_validate
    cross_validate(BaselineOnly(), data, verbose=True)
    ```
  - **load_from_df** 加载 dataframe，在定义 reader 时，只有 **rating_scale 参数** 是必须的
    ```python
    import pandas as pd

    from surprise import NormalPredictor
    from surprise import Dataset
    from surprise import Reader
    from surprise.model_selection import cross_validate


    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                    'userID': [9, 32, 2, 45, 'user_foo'],
                    'rating': [3, 2, 4, 3, 1]}
    df = pd.DataFrame(ratings_dict)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    # We can now use this dataset as we please, e.g. calling cross_validate
    cross_validate(NormalPredictor(), data, cv=2)
    ```
## 交叉验证迭代器 cross-validation iterators
  - **KFold** 定义一个 K-fold 交叉验证迭代器，**split** 方法将数据划分成 trainset / testset，**test** 测试
    ```python
    from surprise import SVD
    from surprise import Dataset
    from surprise import accuracy
    from surprise.model_selection import KFold

    # Load the movielens-100k dataset
    data = Dataset.load_builtin('ml-100k')

    # define a cross-validation iterator
    kf = KFold(n_splits=3)

    algo = SVD()

    for trainset, testset in kf.split(data):

        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        accuracy.rmse(predictions, verbose=True)
    ```
    **运行结果**
    ```python
    RMSE: 0.9502
    RMSE: 0.9472
    RMSE: 0.9434
    ```
  - 其他的交叉验证迭代器 **LeaveOneOut** / **ShuffleSplit**
  - **surprise.model_selection.split.PredefinedKFold** 处理数据集已经按照文件划分好的情况，如 movielens-100K 数据集已经划分了 5 个训练 / 测试集
    ```shell
    ls ~/.surprise_data/ml-100k/ml-100k
    u1.base  u1.test  u2.base  u2.test  u3.base  u3.test  u4.base  u4.test  u5.base  u5.test
    ```
    ```python
    from surprise import SVD
    from surprise import Dataset
    from surprise import Reader
    from surprise import accuracy
    from surprise.model_selection import PredefinedKFold

    # path to dataset folder
    files_dir = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/')

    # This time, we'll use the built-in reader.
    reader = Reader('ml-100k')

    # folds_files is a list of tuples containing file paths:
    # [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
    train_file = files_dir + 'u%d.base'
    test_file = files_dir + 'u%d.test'
    folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()

    algo = SVD()

    for trainset, testset in pkf.split(data):

        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        accuracy.rmse(predictions, verbose=True)
    ```
## GridSearchCV 模型调参 Tune algorithm
  - **GridSearchCV** 组合参数，并选出正确率最高的一组
    ```python
    from surprise import SVD
    from surprise import Dataset
    from surprise.model_selection import GridSearchCV

    # Use movielens-100K
    data = Dataset.load_builtin('ml-100k')

    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

    gs.fit(data)

    # best RMSE score
    print(gs.best_score['rmse'])
    # 0.9635414530341425

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])
    # {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}

    # We can now use the algorithm that yields the best rmse:
    algo = gs.best_estimator['rmse']
    algo.fit(data.build_full_trainset())
    ```
  - **cv_results** 包含测试的中间结果，可以导入到 dataframe 用于分析
    ```python
    results_df = pd.DataFrame.from_dict(gs.cv_results)
    ```
## 命令行使用
  - Surprise 可以直接从命令行使用
    ```python
    surprise -h

    surprise -algo SVD -params "{'n_epochs': 5, 'verbose': True}" -load-builtin ml-100k -n-folds 3
    ```
***

# 预定义的预测算法 prediction algorithms
## AlgoBase
  - Surprise 的算法都是继承自 `AlgoBase`，在 `prediction_algorithms` 包中，位于 `Surprise namespace` 中，可以用 `from surprise import` 导入
    ```python
    help(surprise.prediction_algorithms)

    surprise.prediction_algorithms.AlgoBase
    surprise.prediction_algorithms.BaselineOnly
    surprise.prediction_algorithms.CoClustering
    surprise.prediction_algorithms.KNNBaseline
    surprise.prediction_algorithms.KNNBasic
    surprise.prediction_algorithms.KNNWithMeans
    surprise.prediction_algorithms.KNNWithZScore
    surprise.prediction_algorithms.NMF
    surprise.prediction_algorithms.NormalPredictor
    surprise.prediction_algorithms.SVD
    surprise.prediction_algorithms.SVDpp
    surprise.prediction_algorithms.SlopeOne
    ```
## 基线法估计配置 Baselines estimates configuration
  - **正则化平方差 regularized squared error**
    ```
    ∑(r<ui> ∈ R<train>) (r<ui> − (μ + b<u> + b<i>)) ^ 2 + λ(b<u> ^ 2 + b<i> ^ 2)
    ```
  - **Baselines** 有两种估计方法，算法标准 [Kor10](http://surprise.readthedocs.io/en/stable/notation_standards.html#koren-2010)
    - **SGD** 随机梯度下降 Stochastic Gradient Descent
    - **ALS** 交替最小二乘 Alternating Least Squares
  - **算法的 bsl_options 参数** 字典形式，创建算法时指定 Baselines 算法参数，`method` 指定 als / sgd，默认 als
  - **bsl_options 中的 ALS 参数**
    - `reg_i` items 的正则化参数，对应 λ2，默认值 10
    - `reg_u` users 的正则化参数，对应 λ3，默认值 15
    - `n_epochs` ALS 算法的迭代次数，默认 10
  - **bsl_options 中的 SGD 参数**
    - `reg` 损失函数 cost function 的正则化参数，对应 λ1 / λ5，默认值 0.02
    - `learning_rate` SGD 的学习率 learning rate，对应 γ，默认值 0.005
    - `n_epochs` The number of iteration of the SGD procedure. Default is 20.
  - **使用示例**
  ```python
  # ALS
  print('Using ALS')
  bsl_options = {'method': 'als',
                 'n_epochs': 5,
                 'reg_u': 12,
                 'reg_i': 5
                 }
  algo = BaselineOnly(bsl_options=bsl_options)

  # SGD
  print('Using SGD')
  bsl_options = {'method': 'sgd',
                 'learning_rate': .00005,
                 }
  algo = BaselineOnly(bsl_options=bsl_options)
  ```
## 相似度度量配置 Similarity measure configuration
  - **surprise.similarities** 模块
    ```python
    help(surprise.similarities)

    cosine
    msd
    pearson
    pearson_baseline
    ```
  - **算法的 sim_options 参数** 字典形式，，创建算法时指定 similarity measure 算法参数
    - `name`  相似度算法名称，默认 `MSD`
    - `user_based` 指定相似度是基于用户还是基于物品，默认 True
    - `min_support` 最小支持度
    - `shrinkage` 缩减，仅用于 `pearson_baseline`，默认值 100
  - **使用示例**
    ```python
    # cosine
    sim_options = {'name': 'cosine',
                   'user_based': False  # compute  similarities between items
                   }
    algo = KNNBasic(sim_options=sim_options)

    # pearson_baseline
    sim_options = {'name': 'pearson_baseline',
                   'shrinkage': 0  # no shrinkage
                   }
    algo = KNNBasic(sim_options=sim_options)
    ```
  - **pearson_baseline** 同时指定 基线法估计配置 与 相似度度量配置
    ```python
    bsl_options = {'method': 'als',
                   'n_epochs': 20,
                   }
    sim_options = {'name': 'pearson_baseline'}
    algo = KNNBasic(bsl_options=bsl_options, sim_options=sim_options)
    ```
## GridSearchCV 中指定 bsl_options sim_options
  ```python
  param_grid = {'k': [10, 20],
                'sim_options': {'name': ['msd', 'cosine'],
                                'min_support': [1, 5],
                                'user_based': [False]}
                }
  ```
  ```python
  param_grid = {'bsl_options': {'method': ['als', 'sgd'],
                                'reg': [1, 2]},
                'k': [2, 3],
                'sim_options': {'name': ['msd', 'cosine'],
                                'min_support': [1, 5],
                                'user_based': [False]}
                }
  ```
***

# 自定义算法 custom prediction algorithm
## The basics
  - **AlgoBase** 算法继承的基类
  - **estimate 方法** predict 调用时使用的方法，根据 **user id** / **item id** 参数，返回 **rating**
  - **示例** 只返回 rating 3
    ```python
    from surprise import AlgoBase
    from surprise import Dataset
    from surprise.model_selection import cross_validate

    class MyOwnAlgorithm(AlgoBase):
        def __init__(self):
            # Always call base method before doing anything.
            AlgoBase.__init__(self)

        def estimate(self, u, i):
            return 3

    data = Dataset.load_builtin('ml-100k')
    algo = MyOwnAlgorithm()

    cross_validate(algo, data, verbose=True)
    ```
  - **details** 返回的字典值，预测时返回更多信息，prediction 调用时会存储这些数据
    ```python
    def estimate(self, u, i):
        details = {'info1' : 'That was',
                   'info2' : 'easy stuff :)'}
        return 3, details
    ```
## The fit method
  - **fit** 适配数据，通常由 cross_validate 在每次的验证进程中调用，返回类本身，支持链式调用 `algo.fit(trainset).test(testset)`
    ```python
    class MyOwnAlgorithm(AlgoBase):
        def __init__(self):
            # Always call base method before doing anything.
            AlgoBase.__init__(self)

        def fit(self, trainset):
            # Here again: call base method before doing anything.
            AlgoBase.fit(self, trainset)

            # Compute the average rating. We might as well use the
            # trainset.global_mean attribute ;)
            self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])
            return self

        def estimate(self, u, i):
            return self.the_mean
    ```
## The trainset attribute
  - **self.trainset** fit 适配之后，训练数据集的信息保存在 Trainset 对象中，**ur** 用户评分，**ir** 物品评分
    ```python
    def estimate(self, u, i):
        sum_means = self.trainset.global_mean
        div = 1

        if self.trainset.knows_user(u):
            sum_means += np.mean([r for (_, r) in self.trainset.ur[u]])
            div += 1
        if self.trainset.knows_item(i):
            sum_means += np.mean([r for (_, r) in self.trainset.ir[i]])
            div += 1

        return sum_means / div
    ```
    平均分的计算最好放在 fit 中，避免每次重复计算
## When the prediction is impossible
    - **PredictionImpossible exception** 如果预测不能得出结果，可以抛出 PredictionImpossible 异常
      ```python
      from surprise import PredictionImpossible
      ```
      该异常会被 **predict** 方法捕获，并由 **default_prediction** 得出结果，默认的返回值是 trainset 中所有评分的平均值
## Using similarities and baselines
  - 如果自定义的算法需要 **similarity measure** / **baseline estimates**，定义的 `__init__` 方法应该接受 **bsl_options** / **sim_options** 参数，并传递给基类
  - **compute_baselines** / **compute_similarities** 用于在 fit 方法中适配数据
    ```python
    class MyOwnAlgorithm(AlgoBase):
        def __init__(self, sim_options={}, bsl_options={}):
            AlgoBase.__init__(self, sim_options=sim_options, bsl_options=bsl_options)

        def fit(self, trainset):
            AlgoBase.fit(self, trainset)

            # Compute baselines and similarities
            self.bu, self.bi = self.compute_baselines()
            self.sim = self.compute_similarities()

            return self

        def estimate(self, u, i):
            if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
                raise PredictionImpossible('User and/or item is unkown.')

            # Compute similarities between u and v, where v describes all other
            # users that have also rated item i.
            neighbors = [(v, self.sim[u, v]) for (v, r) in self.trainset.ir[i]]
            # Sort these neighbors by similarity
            neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

            print('The 3 nearest neighbors of user', str(u), 'are:')
            for v, sim_uv in neighbors[:3]:
                print('user {0:} with sim {1:1.2f}'.format(v, sim_uv))

            # ... Aaaaand return the baseline estimate anyway ;)
    ```
***
