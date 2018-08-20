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
## What are raw and inner ids
  - **Raw ids** 从文件 / dataframe 读取到的 id，可以是字符串 / 数字，从文件中读取到的会转化为字符串
  - **Inner ids** 创建 trainset 时，每个 raw id 会被转化成唯一的整数型 inner id，用于 surprise 的计算
  - **转化方法** trainset 的 id 转化方法 **to_inner_uid** / **to_inner_iid** / **to_raw_uid** / **to_raw_iid**
## Where are datasets stored and how to change it?
  By default, datasets downloaded by Surprise will be saved in the '~/.surprise_data' directory. This is also where dump files will be stored. You can change the default directory by setting the 'SURPRISE_DATA_FOLDER' environment variable.
## prediction_algorithms
  random_pred.NormalPredictor 	Algorithm predicting a random rating based on the distribution of the training set, which is assumed to be normal.
  baseline_only.BaselineOnly 	Algorithm predicting the baseline estimate for given user and item.
  knns.KNNBasic 	A basic collaborative filtering algorithm.
  knns.KNNWithMeans 	A basic collaborative filtering algorithm, taking into account the mean ratings of each user.
  knns.KNNWithZScore 	A basic collaborative filtering algorithm, taking into account the z-score normalization of each user.
  knns.KNNBaseline 	A basic collaborative filtering algorithm taking into account a baseline rating.
  matrix_factorization.SVD 	The famous SVD algorithm, as popularized by Simon Funk during the Netflix Prize.
  matrix_factorization.SVDpp 	The SVD++ algorithm, an extension of SVD taking into account implicit ratings.
  matrix_factorization.NMF 	A collaborative filtering algorithm based on Non-negative Matrix Factorization.
  slope_one.SlopeOne 	A simple yet accurate collaborative filtering algorithm.
  co_clustering.CoClustering 	A collaborative filtering algorithm based on co-clustering.
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

# 应用示例
## 获取每个用户 top-N 的推荐
  - 获取 MovieLens-100k 数据集中每个用户 top-10 的物品推荐
  - **在整个数据集上训练 SVD 算法**
    ```python
    from collections import defaultdict
    from surprise import SVD
    from surprise import Dataset

    # First train an SVD algorithm on the movielens dataset.
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    ```
  - **预测所有未出现在数据集中的 `(user, item)` 组合的得分**
    ```python
    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    ```
  - **获取每个用户得分最高的 top-10**
    ```python
    # Dataframe
    dd = pd.DataFrame(predictions)
    dd[:3]
    # Out[77]:
    #    uid  iid     r_ui       est                    details
    # 0  196  302  3.52986  3.845034  {'was_impossible': False}
    # 1  196  377  3.52986  2.665580  {'was_impossible': False}
    # 2  196   51  3.52986  3.370517  {'was_impossible': False}

    def get_top_n(ddf, n=10, column='est'):
        return ddf.sort_values(by=column, ascending=False)[:n]

    top_n = dd.groupby('uid').apply(get_top_n, n=10)
    top_n.loc['196'][:3]
    # Out[84]:
    #      uid  iid     r_ui       est                    details
    # 232  196   64  3.52986  4.695576  {'was_impossible': False}
    # 174  196  408  3.52986  4.546361  {'was_impossible': False}
    # 194  196  318  3.52986  4.531698  {'was_impossible': False}
    ```
    ```python
    # dict
    def get_top_n(predictions, n=10):
        '''Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        '''

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    top_n = get_top_n(predictions, n=10)

    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])

    # 196 ['64', '408', '318', '114', '169', '197', '603', '357', '427', '208']
    ```
## 计算每个用户推荐的精确率 precision 与召回率 recall
  - **relevant item 有关联的物品** 物品真实的得分 true rating 大于某个阈值
  - **recommended item 推荐的物品** 物品预测的得分 estimated rating 大于某个阈值，并且物品位于推荐的 top k 中
  - **精确率 precision** 为 **判断为正确的样本** 中，预测为正且实际为正的样本 的比例
    ```python
    P = 正例正确判断为正例 TP / (正例正确判断为正例 TP + 反例错误判断为正确 FP)
    Precision@k = |{推荐的物品为有关联的}| / |{所有推荐的物品}|
    ```
  - **召回率 recall** 为 **所有正例样本** 中，预测为正且实际为正的样本 的比例
    ```python
    TPR = 正例正确判断为正例 TP / (正例正确判断为正例 TP + 正例错误判断为反例 FN)
    Recall@k = |{推荐的物品为有关联的}| / |{有关联的物品}|
    ```
  - **python 示例**
    ```python
    from collections import defaultdict

    from surprise import Dataset
    from surprise import SVD
    from surprise.model_selection import KFold


    def precision_recall_at_k(predictions, k=10, threshold=3.5):
        '''Return precision and recall at k metrics for each user.'''

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k0
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls


    data = Dataset.load_builtin('ml-100k')
    kf = KFold(n_splits=5)
    algo = SVD()

    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

        # Precision and recall can then be averaged over all users
        print(sum(prec for prec in precisions.values()) / len(precisions))
        print(sum(rec for rec in recalls.values()) / len(recalls))
    ```
## 获取用户或物品的 k 个最近邻
  - **get_neighbors** 基于 similarity measure 获取相似的 用户 / 物品，如 **k-NN**，获取到的是用户还是物品，取决于 `sim_options` 的 `user_based`
    ```python
    get_neighbors(iid, k)
    ```
    - **iid(int)** 用户 / 物品的 inner id
    - **k(int)** 获取的相似数量
  - **python 示例**
    ```python
    import io  # needed because of weird encoding of u.item file

    from surprise import KNNBaseline
    from surprise import Dataset
    from surprise import get_dataset_dir


    def read_item_names():
        """Read the u.item file from MovieLens 100-k dataset and return two
        mappings to convert raw ids into movie names and movie names into raw ids.
        """

        file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
        rid_to_name = {}
        name_to_rid = {}
        with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                line = line.split('|')
                rid_to_name[line[0]] = line[1]
                name_to_rid[line[1]] = line[0]

        return rid_to_name, name_to_rid


    # First, train the algortihm to compute the similarities between items
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    algo = KNNBaseline(sim_options=sim_options)
    algo.fit(trainset)

    # Read the mappings raw id <-> movie name
    rid_to_name, name_to_rid = read_item_names()

    # Retrieve inner id of the movie Toy Story
    toy_story_raw_id = name_to_rid['Toy Story (1995)']
    toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

    # Retrieve inner ids of the nearest neighbors of Toy Story.
    toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)

    # Convert inner ids of the neighbors into names.
    toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
                           for inner_id in toy_story_neighbors)
    toy_story_neighbors = (rid_to_name[rid]
                           for rid in toy_story_neighbors)

    print()
    print('The 10 nearest neighbors of Toy Story are:')
    for movie in toy_story_neighbors:
        print(movie)
    ```
    **运行结果**
    ```python
    The 10 nearest neighbors of Toy Story are:
    Beauty and the Beast (1991)
    Raiders of the Lost Ark (1981)
    That Thing You Do! (1996)
    Lion King, The (1994)
    Craft, The (1996)
    Liar Liar (1997)
    Aladdin (1992)
    Cool Hand Luke (1967)
    Winnie the Pooh and the Blustery Day (1968)
    Indiana Jones and the Last Crusade (1989)
    ```
## 算法持久化 serialize an algorithm
  - **dump / load** 算法持久化 serialized 与重加载 loaded back，属于模块 `dump <surprise.dump>`
    ```python
    dump(file_name, predictions=None, algo=None, verbose=0)
    load(file_name) # Returns a tuple ``(predictions, algo)``
    ```
  - **python 示例**
    ```python
    import os

    from surprise import SVD
    from surprise import Dataset
    from surprise import dump


    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    # Compute predictions of the 'original' algorithm.
    predictions = algo.test(trainset.build_testset())

    # Dump algorithm and reload it.
    file_name = os.path.expanduser('~/dump_file')
    dump.dump(file_name, algo=algo)
    _, loaded_algo = dump.load(file_name)

    # We now ensure that the algo is still the same by checking the predictions.
    predictions_loaded_algo = loaded_algo.test(trainset.build_testset())
    assert predictions == predictions_loaded_algo
    print('Predictions are the same')
    ```
  - [Dumping and analysis of the KNNBasic algorithm](http://nbviewer.jupyter.org/github/NicolasHug/Surprise/blob/master/examples/notebooks/KNNBasic_analysis.ipynb)
  - [Comparison of two algorithms](http://nbviewer.jupyter.org/github/NicolasHug/Surprise/blob/master/examples/notebooks/Compare.ipynb)
## 获取训练数据集上的准确率
  - **build_testset** 创建测试数据集，用于 `test` 方法
    ```python
    from surprise import Dataset
    from surprise import SVD
    from surprise import accuracy
    from surprise.model_selection import KFold


    data = Dataset.load_builtin('ml-100k')

    algo = SVD()

    trainset = data.build_full_trainset()
    algo.fit(trainset)

    testset = trainset.build_testset()
    predictions = algo.test(testset)
    # RMSE should be low as we are biased
    accuracy.rmse(predictions, verbose=True)  # ~ 0.68 (which is low)
    ```
## 从数据集中获取数据用于无偏差的正确率估计 unbiased accuracy estimation
  - 将数据集划分成两部分
    - sets A 用于 GridSearchCV 调参
    - sets B 用于无偏差的正确率估计 unbiased accuracy estimation
  - **python 示例**
    ```python
    import random
    from surprise import SVD
    from surprise import Dataset
    from surprise import accuracy
    from surprise.model_selection import GridSearchCV

    # Load the full dataset.
    data = Dataset.load_builtin('ml-100k')
    raw_ratings = data.raw_ratings

    # shuffle ratings if you want
    random.shuffle(raw_ratings)

    # A = 90% of the data, B = 10% of the data
    threshold = int(.9 * len(raw_ratings))
    A_raw_ratings = raw_ratings[:threshold]
    B_raw_ratings = raw_ratings[threshold:]

    data.raw_ratings = A_raw_ratings  # data is now the set A

    # Select your best algo with grid search.
    print('Grid Search...')
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
    grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(data)

    algo = grid_search.best_estimator['rmse']

    # retrain on the whole set A
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # Compute biased accuracy on A
    predictions = algo.test(trainset.build_testset())
    print('Biased accuracy on A,', end='   ')
    accuracy.rmse(predictions)

    # Compute unbiased accuracy on B
    testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
    predictions = algo.test(testset)
    print('Unbiased accuracy on B,', end=' ')
    accuracy.rmse(predictions)
    ```
## 创建可复现的测试 reproducible experiments
  - 算法每次会随机初始化参数，cross-validation 划分的数据集每次也是随机产生的
  - **seed of the RNG** 在算法开始时指定 seed，使每次获得相同的随机数
    ```python
    import random
    import numpy as np

    my_seed = 0
    random.seed(my_seed)
    numpy.random.seed(my_seed)
    ```
***

# prediction_algorithms 模块
## 算法基类 AlgoBase
  - **AlgoBase** 位于 `surprise.prediction_algorithms.algo_base.AlgoBase`
    ```python
    help(surprise.AlgoBase)
    ```
  - **支持的方法**
    - **compute_baselines** 基于参数 **bsl_options** 指定的方式计算用户 / 物品 基线 baselines，适用于 Pearson baseline similarty 与 BaselineOnly 算法
    - **compute_similarities** 基于参数 **sim_options** 指定的方式建立相似度矩阵，适用于相似度估计算法，如 k-NN
    - **default_prediction** 当 predict 预测不出结果 PredictionImpossible 时调用的方法，默认返回所有评分的平均值
    - **fit** 在训练数据集上训练算法
    - **get_neighbors** 基于 similarity measure 获取相似的 用户 / 物品，获取到的是用户还是物品，取决于 `sim_options` 的 `user_based`
    - **predict** 使用 `raw uid` 与 `raw iid` 计算得分
    - **test** 在测试数据集上测试算法
## predictions 模块
  - **Prediction 类** 位于 `surprise.prediction_algorithms.predictions`
    ```python
    help(surprise.prediction_algorithms.predictions)
    ```
  - **PredictionImpossible** 异常，用于发生不能预测的情况，抛出该异常
## 基础算法
  - **NormalPredictor** 位于 `surprise.prediction_algorithms.random_pred.NormalPredictor`
    ```python
    help(surprise.NormalPredictor)
    ```
    根据训练数据集上的得分，给出一个符合正泰分布的随机得分值
  - **BaselineOnly** 位于 `surprise.prediction_algorithms.baseline_only.BaselineOnly`
    ```python
    help(surprise.BaselineOnly)
    ```
    计算给定用户 user 与物品 item 的基线 baseline
    ```python
    r<ui> = b<ui> = μ + b<u> + b<i>
    ```
