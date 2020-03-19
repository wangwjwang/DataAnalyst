# Intro to Machine Learning #
Learn the core ideas in machine learning, and build your first models.

课程地址：
[https://www.kaggle.com/learn/intro-to-machine-learning](https://www.kaggle.com/learn/intro-to-machine-learning)
##一. basic-data-exploration##

    import pandas as pd
	
 定义数据存放路径

    melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

1.读取数据表格内容

    melbourne_data = pd.read_csv(melbourne_file_path)

2.查看数据前5行

    melbourne_data.head() 

3.查看数据整体情况与大致分布

    melbourne_data.describe()

其中的count是有值数据的个数求和，可以判断哪些列存在缺失值。

mean是平均值，std是标准差。标准差用于反映一维数据集中数据的离散程度，标准差越小，离散程度越小，各数据更集中。两个数据集即使平均值相同，标准差也未必相同。

25%，50%，75%分数数用于表示：数据由小到大排序后，位置第四分之一，四分之二，四分之三的数据值。一定程度也可以观察数据的分布情况。例如单位员工工资，如果50%值小于mean值，则说明少数高管的工资拉高了平均值。

4.绘制直方图，查看数据分布情况

    age = 2020 - melbourne_data['YearBuilt']
    age.hist()
绘制房屋建造年龄直方图

## 二.Selecting Data for Modeling ##

1.查看数据集列信息

    melbourne_data.columns

2.删除空数据的行

    melbourne_data = melbourne_data.dropna(axis=0)

axis的理解：axis=0 是指数据在纵轴上的垂直变化，axis=1 是指数据在横轴上的水平变化。这里的drop axis=0，指减少行。

3.选择待预测目标

    y = melbourne_data.Price

price是该数据集的一个列名，可以作为属性直接调用

4.选择特征值

    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
这里创建的是list。

常用数据结构list列表、array数组、dataframe。

**list**是python的数据结构，元素保存数据存放的地址即指针，元素的数据类型可以不同，因此不能进行整体数值运算

**array**是numpy封装的数据结构，元素类型必须相同

**dataframe**是pandas封装的类似表格的数据结构

numpy与pandas都是python的数据包，numpy用于数值操作与计算，pandas用于数据处理与分析

5.定义特征值

    X = melbourne_data[melbourne_features]

6.创建模型
通常创建和使用模型需要经历四个步骤：Define、fit、predict、evaluate。
这里的例子是使用决策树模型

    from sklearn.tree import DecisionTreeRegressor
    melbourne_model = DecisionTreeRegressor(random_state=1)
    melbourne_model.fit(X, y)
7.执行预测，预测前5例的价格

    print("Making predictions for the following 5 houses:")
    print(X.head())
    print("The predictions are")
    print(melbourne_model.predict(X.head()))

这里使用训练集的数据进行验证，会发现准确率非常高。因为模型是基于训练集数据进行学习的，所以使用模型预测的结果非常贴近训练集中的实际结果
## 三.Model Validation ##

1.评估模型预测准确度的时候，容易犯的错误是使用训练集数据进行预测后，用预测结果与训练集数据中的目标值进行比对。
衡量模型预测结果的准确度时，需要总结为单个度量值，以便比较效果。
衡量模型质量有很多种度量值，其中一种是MAE（Mean Absolute Error ），平均绝对误差。直白说就是所有预测值与真实值的平均偏离量。计算方法：每个预测值减去真实值的差求绝对值，将这些绝对值按所有预测值个数求平均。
2.可以使用现成模型来计算MAE


    from sklearn.metrics import mean_absolute_error
    
    predicted_home_prices = melbourne_model.predict(X)
    mean_absolute_error(y, predicted_home_prices)

3.样本内分数问题。以上的计算方法的到的度量值只能叫做样本内成绩（"In-Sample" Scores），因为我们使用同一组数据（训练集training data）进行建模与模型评估。这样可能导致这个模型计算训练集的数据时很准确，但预测从未见过的新数据时非常不准。
解决这个问题的比较直接的方法是从原训练集中分理出一部分数据作为验证集（validation data），这部分数据不参与建模，仅用于测试模型的准确度。

4.scikit-learn库中有函数train_test_split用于分离训练集与验证集。

    from sklearn.model_selection import train_test_split
    
    # split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    # Define model
    melbourne_model = DecisionTreeRegressor()
    # Fit model
    melbourne_model.fit(train_X, train_y)
    
    # get predicted prices on validation data
    val_predictions = melbourne_model.predict(val_X)
    print(mean_absolute_error(val_y, val_predictions))
    
    
分离训练集和验证集后，得到这个模型的MAE是25万刀以上，拆分前MAE为5百刀多点。拆分后更能准确评估该模型的准确性。验证集中的平均房价是110万刀，也就是说该模型预测得到的房价平均偏离度达到了25%，准确率非常低。

5.可以查看预测完全正确值的个数

    sum((val_predictions-val_y).abs() == 0)

abs()函数用来求绝对值，sum()函数用来求和


## 四.underfitting and overfitting ##
1.欠拟合与过拟合。以回归决策树（DecisionTreeRegressor）为例，决策树的的深度depth越大，决策树的叶子节点leaf越多。如果每个分支都有2个分叉，则10层深的决策树的叶子节点就有2的10次方个。

在用训练集建模时，叶子越多，叶子中的数据越少，针对训练集的预测就会越准确。但这样的模型对于新数据的预测就越不准，这种现象就叫做**过拟合**。

如果一个决策树层数过少，叶子过少，假如只有1层，只分2个叶子，那么无论对训练集还是验证集的数据都无法进行有效预测。这种现象就叫做**欠拟合**。

2.随着决策树的叶子数量增加，训练集的预测MAE值越低，但验证集的预测MAE值会呈现下降再上升的图形，这个曲线的MAE最低点对应的叶子节点数就是我们建模时要找到的甜区。

3.回归决策树模型中可以以参数形式指定最大叶子数。我们可以通过计算不同叶子数的MAE值来找到模型的甜区。先定义一个函数用来计算mae值

    from sklearn.metrics import mean_absolute_error
    from sklearn.tree import DecisionTreeRegressor
    
    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

这里的train_X, val_X, train_y, val_y是之前已经拆分好的训练集与验证集。

4.利用for循环，计算不同max_leaf对应的mae值

    for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

从结果我们可以看到max_leaf为500时，mae最小。可以得出结论，针对该模型，max_leaf设为500，预测值更准确。

5.进一步，我们也可以遍历50到5000找到该模型更好的max_leaf值
    best_tree_size = 500
    best_mae = get_mae(500, train_X, val_X, train_y, val_y)
    
    for max_leaf_nodes in range(50,5000):
    cur_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    if cur_mae < best_mae:
    best_mae = cur_mae
    best_tree_size = max_leaf_nodes
    print(best_tree_size)


## 五.Random Forests ##

不同于单一决策树模型，随机森林模型拥有许多树，最终预测结果有每个树的预测值均衡得出。随机森林模型在处理欠拟合和过拟合方面能够提供更好的效果，而且可以不用额外调节树深度、叶子节点数等参数，该模型使用缺省参数即可得到很好的预测值。使用方法与决策树模型类似。
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    melb_preds = forest_model.predict(val_X)
    print(mean_absolute_error(val_y, melb_preds))

这个例子中算得mae为20万刀左右，答复优于之前使用单一决策树模型的25万刀mae值。之后将介绍XGBoost模型，预测效果会更好，但需要惊醒调节参数值。

## 六.Exercise ##
Housing Prices Competition for Kaggle Learn Users

使用RandomForestRegressor进行预测

1.使用练习中给出的7个特征值，利用所有train数据进行fit，预测得分21839.72

2.使用练习中给出的7个特征值，将train数据进行分割，利用新train数据集进行fit，预测得分22649.89。成绩更差了，理解因为模型参数没变，但训练集数据减少了，所以成绩降了。

3.考虑增加特征值。需要注意的是使用RandomForestRegressor预测时，特征值必须是非空数值型。

4.查看data_description文件，分析字段定义。尝试将MSSubClass、OverallQual、OverallCond、YearRemodAdd、FullBath、HalfBath、KitchenAbvGr、TotRmsAbvGrd、MiscVal
等列添加作为特征值。TotalBsmtSF、BsmtFullBath、BsmtHalfBath、GarageYrBlt有空值，无法直接作为特征值。增加以上9个特征值后，成绩为19758.24，有明显提升
