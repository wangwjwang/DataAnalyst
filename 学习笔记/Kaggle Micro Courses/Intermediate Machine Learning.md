# Intermediate Machine Learning #
Learn to handle missing values, non-numeric values, data leakage and more. Your models will be more accurate and useful.

课程地址：

[https://www.kaggle.com/learn/intermediate-machine-learning](http://https://www.kaggle.com/learn/intermediate-machine-learning)
##一. Introduction#

    
 直接Exercise：
仍然利用RandomForest预测房价。参数仍然是最早的7个，但进行了参数调整。

n_estimators：对原始数据集进行有放回抽样生成的子数据集个数，即决策树的个数。若n_estimators太小容易欠拟合，太大计算速度增加却不能显著的提升模型，默认值是100。

criterion： 即CART树做划分时对特征的评价标准。分类RF对应的CART分类树默认是基尼系数gini,另一个可选择的标准是信息增益。回归RF对应的CART回归树默认是均方差mse，另一个可以选择的标准是绝对值差mae。

max_depth:决策树最大深度。若等于None,表示决策树在构建最优模型的时候不会限制子树的深度。如果模型样本量多，特征也多的情况下，推荐限制最大深度；若样本量少或者特征少，则不限制最大深度。

min_samples_leaf:叶子节点含有的最少样本。若叶子节点样本数小于min_samples_leaf，则对该叶子节点和兄弟叶子节点进行剪枝，只留下该叶子节点的父节点。整数型表示个数，浮点型表示取大于等于（样本数 * min_samples_leaf)的最小整数。min_samples_leaf默认值是1。

min_samples_split:节点可分的最小样本数，默认值是2。

##二. Missing Values#

造成数据集中存在空值的原因有很多，例如按逻辑无值（例如一个房子只有1间卧室，那第二间卧室的面积值就是空的）或者未收集到（例如受访者不愿意提供收入数据）。
大多数的机器学习模型都不接受含缺失数据的数据集。

通常由三种方法来处理数据集中的缺失值。

1.将含缺失值的列直接抛掉

这种方法最简单，但极有可能将包含重要数据的列全部删除了，影响预测效果。

2.简单填充

例如使用本列数据的均值对缺失值进行填充。

3.扩展填充

例子：
drop object类型列，选择所有数字化列作特征值

    # Load the data
    data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
    
    # Select target
    y = data.Price
    
    # To keep things simple, we'll use only numerical predictors
    melb_predictors = data.drop(['Price'], axis=1)
    X = melb_predictors.select_dtypes(exclude=['object'])
    
    # Divide data into training and validation subsets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
      random_state=0)
8.2分将原始数据集拆分为训练集与验证集

定义固定模型下，根据不同数据集计算score的函数
    
    > #Function for comparing different approaches
    > def score_dataset(X_train, X_valid, y_train, y_valid):
    > model = RandomForestRegressor(n_estimators=10, random_state=0)
    > model.fit(X_train, y_train)
    > preds = model.predict(X_valid)
    > return mean_absolute_error(y_valid, preds)

方法一：


    cols_with_missing = [col for col in X_train.columns
	 if X_train[col].isnull().any()]
    
    # Drop columns in training and validation data
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
    
    print("MAE from Approach 1 (Drop columns with missing values):")
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

方法二：

使用SimpleImputer去填充缺失值，填平均值。有的时候平均值填充效果就很好。虽然有时候会用到更复杂的方法去计算填充值（例如回归），但更复杂的填充值在复杂的机器学习模型中并不一定有更多收益。


    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFram(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
    
    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns
    
    print("MAE from Approach 2 (Imputation):")
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
    
SimpleImputer的常用参数：



- missing_values,也就是缺失值是什么，一般情况下缺失值当然就是空值，也就是np.nan



- strategy:也就是你采取什么样的策略去填充空值，总共有4种选择。分别是**mean,median, most_frequent,以及constant**，这是对于每一列来说的，如果是mean，则该列则由该列的均值填充。而median,则是中位数，most_frequent则是众数。需要注意的是，如果是constant,则可以将空值填充为自定义的值，这就要涉及到后面一个参数了，也就是fill_value。如果strategy='constant',则填充fill_value的值。

SimpleImputer也要先fit再transform数据，同一模型fit一次即可。这里的fit是为了得到数据的特征（例如平均值），在transform时可以根据fit得到的特征进行填充。

transform返回的是ndarryN维数组，所以需要转换格式为DataFrame。但是转换完的DataFrame列名为0开始的序号，所有还需要添加列名。

第三种方法：

    # Make copy to avoid changing original data (when imputing)
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()
    
    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
    
    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
    
    # Imputation removed column names; put them back
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns
    
    print("MAE from Approach 3 (An Extension to Imputation):")
    print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

为甚麽填充方法的效果要比简单drop空值列的方法效果好，在这组实验中，数据集中有10864行，12列。其中3列的数据存在缺失值，但缺失值在每列占比不超过50%。所以简单drop会丢掉很多有用的信息，正是这些信息帮助填充方法提高预测效果

可以查看空值列的信息，列名和该列空值数量

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

接下来我们把所有数值型列全部列为特征值，为缺失值填充平均值这两个提升点应用到房价预测中，提交后得到score为16608.59，有较大提升

**课后练习：**

    X_full = pd.read_csv('../input/train.csv', index_col='Id')
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    
dataframe的dropna方法用于删除缺失值，参数说明如下：



- axis

0表示对包含缺失值的行进行删除

1表示对包含缺失值的列进行删除；


- how

any表示有任何NA存在就删除所在行或列

all表示该行或列必须都是NA才删除


- thresh 缺失值数量大于该值的会执行删除

int整数数据类型

optional随意数据类型


- subset 在指定列名的列上查找缺失值

array-like选定列

potional所有列


- inplace

True在原表上进行修改

False不在原表上进行修改

    missing_val_count_by_column = (X_train.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

DataFrame的isnull()方法返回元素是否为空对应的bool值矩阵

DataFrame的any()方法返回列中是否有元素为真

DataFrame的sum()方法返回含有列小计的Series，传入axis=1将会按行进行求和运算

    X_train.shape[0]
    X_train.shape[1]

可以使用shape得到DataFrame的行数和列数


    X_train.isnull().sum().sum()

可以得到数据集的空值总数

练习中比较了删除缺失值列的方法与用平均值填充缺失值的方法，练习题中的数据计算后发现直接删除缺失值列的方法得到的预测值略好一些。这与初步想象不同，之前查看缺失值的数量可以看到缺失值在列中占比较少，按一般理解直接删除该列会丢失有用的信息，预测应该更不准才对。造成这种反差的原因可能有两个，一个是数据集中的噪音数据导致，还有一个可能就是采用的填充方法不合适（平均值填充，0值填充，众数填充，中值填充等），例如“车库建造年代”数据，该数据为空很有可能意味着没有车库，直接用平均值填充非常不合理，用0值填充缺失值更合理。

接下来又试了0值填充、众数填充、中值填充，中值填充验证集得分更高。采用0值填充缺失值的方法，最终提交后得到score为16384.47。采用众数填充缺失值的方法，最终提交后得到score为16463.94。采用中值填充缺失值的方法，最终提交后得到score为16359.57，得分更高。

##二. Categorical Variables#

dtype属性返回的数据类型可以判断是否为数值型变量还是分类变量。如果dtype返回object类型，证明该变量为分类变量。

    
    # Get list of categorical variables
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)

在大多数的基于Python的机器学习模型中，直接处理非数字型的分类变量都会报错。下面介绍三种处理分类变量的方法。

方法一：Drop

对于列中无有用信息的，可以直接drop。


    drop_X_train = X_train.select_dtypes(exclude=['object'])


方法二：Lable Encoding标签编码

对于定序变量（Ordinal variables）可以进行Lable Encoding

Scikit-learn的LableEncoder类可以直接使用。

    from sklearn.preprocessing import LabelEncoder    
    label_encoder = LabelEncoder()
    for col in object_cols:
        label_X_train[col] = label_encoder.fit_transform(X_train[col])
        label_X_valid[col] = label_encoder.transform(X_valid[col])
    
LableEncoder类的fit方法用于数据编码，transform方法用于将非数值变量值按照fit得到的对应关系进行转化。

课后练习中提到一个问题，某定序变量所在列在训练集中的值的集合与在验证集中的值的集合不一样，验证集中该列有某个值，但该值在训练集中并未出现。这样直接fit然后transform验证集时就会报错，因为fit时未对该未出现的值进行label。

方法三：One-Hot Encoding

对于无序的定类变量（nominal variables）可以进行One-Hot Encoding，一般用于变量类别种类不太多的列，如果变量有超过15个不同值，通常效果不好。如果变量种类过多可以考虑使用稀疏矩阵

Scikit-learn的OneHotEncoder类可以直接使用。

    from sklearn.preprocessing import OneHotEncoder    
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))


##三. Pipellines#

1.管道的优点：

更整洁的代码、更少的bug、更容易产品化、更多验证选择

2.房价预测的例子中逐步建立管道。

第一步 定义预处理步骤

使用ColumnTransformer类将以下不同预处理步骤捆绑到一起

对于数字类型数据，采取填充空缺值的方法处理

对于分类变量，采取填充空值与one-hot编码两种方法处理

第二步 定义模型

第三步 建立并评估pipeline

使用pipeline类将数据预处理与模型进行绑定。使用pipeline可以只用一条语句就实现数据处理和模型建立。使用中可以直接使用未作数据预处理的验证集进行预测，pipeline可以自动进行数据预处理。

