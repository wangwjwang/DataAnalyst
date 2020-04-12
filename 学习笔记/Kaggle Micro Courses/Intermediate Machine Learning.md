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

##四. Cross-Validation#

机器学习是一个迭代过程，我们需要面对各种选择，选择哪些变量，采用哪种模型，为模型赋予哪些参数等等。通过验证集可以对模型进行评估。

但单一验证集有个缺点，通常只分离较少部门数据作为验证集，这样根据验证结果得出的评估结果就具备了一定随机性。最极端的就是验证集只有1条数据，那么根据这个验证集来验证模型的效果就只能靠运气了。

总的来说，验证集越大，模型评估结果的随机性（噪音）越小，验证集就更可靠。但是如果在抽取验证集时使用了过多数据，我们的训练集就会更小，这样训练出的模型必定会很糟糕。

**交叉验证。**我们可以在数据的不同子集上运行模型，以期得到模型的复合评估值。

例如我们将数据集分为5分，命名为5个**folds**，然后在每个fold上运行模型。第一次将第一个fold作为验证集，其余4个fold做训练集，得到一个模型评分。然后再将第二个fold做验证集，重复以上步骤得到第二个模型评分。逐个执行结合后，所有数据都依次做过了验证集，虽然不是一次全使用。

如何决定是否使用交叉验证。交叉验证千般好，尤其当我们在进行很多模型选优时。但交叉验证会耗费更长时间，因为会将模型运行多次。那么如何选择。
- 如果数据集比较小，交叉验证带来的额外计算时间不多，可以选用交叉验证
- 如果数据集比较大，那么单次验证就足够了，因为按比例提取得到的验证集本身很大，可以很大程度较少随机性。

这里的多和少并没有严格值，通常如果运行一次模型也就花费几分钟，那就非常值得我们采用交叉验证的方法。

还有一种方法，我们在完成交叉验证后可以观察每个fold得到的成绩，如果相差不大，那么单次验证可能就够了。

示例，不用pipeline也可以实现交叉验证，但很难，如果使用pipeline，编码会相当简单。

使用scikit-learn库的cross_val_score()函数，可以获得交叉验证的各阶段分值。

scores = -1 * cross_val_score(my_pipeline, X, y,
  cv=5,
  scoring='neg_mean_absolute_error')

这里scoring评分参数选择的'neg_mean_absolute_error'，这里的mae再取负值，因为scikit-learn有个惯例，为统一度量，约定数值越大越好。mae值越大说明越不好，所以要取负值。为对不同模型进行评估，可以将不同阶段的分支取平均来评估。

##五.XGBoost#

梯度提升（gradinet boosting）用于建立并优化模型是kaggle竞赛常用方法。

1.介绍 

之前的课程中我们建立RandomForest模型，这种模型平均了多个决策树的预测结果，所以比单一决策树模型性能更好。这个RandomForest模型就是一种集成方法（ensemble methods）。集成方法就是结合多个模型的预测结果。

2.Gradient Boosting

梯度提升算法通过迭代的方式将模型加入到现有集成中。

一开始集成中只有一个模型，它的预测结果可能很不准确，带之后逐步加入集成中的模型会逐步优化这些错误。迭代步骤如下：

第一，我们使用现有集成为所有数据进行预测，然后将集成中每个模型产生的每个预测值合起来来产生预测。
第二，挑选一个损失函数如均方差，来计算上边的预测值。
第三，我们使用损失函数拟合将加入集成的新模型。具体来说，我们决定模型参数以便该模型加入集成能够减少"损失"。（梯度提升中的梯度就是使用梯度下降的方法来决定使用哪些参数）
最后，将该新模型加入集成中

3.example

XGBoost作为极端梯度提升算法，是一种通过多个额外特征提高性能与速度的算法。接下来我们学习XGBRegressor类的多个可调节参数，调参能够显著影响准确度与速度。

    from xgboost import XGBRegressor
    
    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
    my_model.fit(X_train, y_train, 
     early_stopping_rounds=5, 
     eval_set=[(X_valid, y_valid)], 
     verbose=False)

**n_estimators**参数指定了模型加入的迭代次数，也代表集成模型中共有几个子模型。这个参数过小会造成欠拟合，训练集与测试集的预测都不准确。这个参数过大会造成过拟合，训练集预测非常准确，但真正关心的测试集预测效果反而不准确。这个值通常取100-1000，更具体的需要依赖于learning-rate参数。

**early-stopping-rounds**提供了一种自动找到理想n_estimators值的方法。在模型成绩停止提高的时候，能够提前终止模型的迭代。所以比较聪明的方式是设置一个高n_estimators并用early-stopping-rounds找到合适的实际停止迭代。由于随机的原因，有的时候第一轮开始模型成绩就不提高了，这是就还需要定义最少还得走几轮，避免区域极值造成提前终止迭代。通常我们设置early-stopping-rounds为5.

**eval_set**参数，使用early-stopping-round参数时，需要使用eval_set指定验证集来计算验证成绩。如果最后想用全部数据集进行模型训练，可以用算得的最优n_estimators值来重新训练模型

**learning-rate**参数，不同于将所有子模型的预测成绩求和，我们可以将每个模型的预测结果乘以一个小数值（即learning-rate），然后再求和，这样我们加入到集成模型中的每个树的影响都很小，这样我们就可以在避免过拟合的同时设置更高的n_estimatos值。总的来说，小learning-tate和大estimators能够使得SGBoost模型更准确。当然模型训练时间也会更长，因为迭代了更多次。缺省情况下，SGBoost设置learing-rate为0.1

**n_jobs**参数，当数据集非常巨大时，就应该开始考虑运行时间了，并行机制能够更快建模。通常设置n_jobs参数与电脑的核数相同。如果数据集较小，这个参数没什么效果。这个参数本身仅能提高大数据集的模型训练速度，但无法直接提高模型性能。所以小数据操心这个值就不值当了。

4.结论

SGBoost是一个杰出的库，用于处理标准表格类数据，如我们在Pandas DataFrames类型，但无法处理类似图像视频一类的特殊数据类型。
 
课后Exercis中的数据处理手法：



**挑选数值列与低cardinality的非数值列作为特征值列**

    low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
     X_train_full[cname].dtype == "object"] 
    
    numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    my_cols = low_cardinality_cols + numeric_cols
    


**使用pandas的get_dummies进行one-hot encod**

	X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)

**由于X_train,X_valid,X_test中非数值特征列中的值有可能不一样，在做完one-hot encoding后体现为列数的不一样。这时就需要进行补充对齐。**

    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    X_train, X_test = X_train.align(X_test, join='left', axis=1)

利用XGBoost方法提交房价预测competition

    XGBRegressor(n_estimators=1000,  
                 learning_rate=0.05, 
                 early_stopping_rounds=5）    
最终成绩为14794.29，排名1924             


##六. DataLeakage#

当训练集中包含了目标的信息时，就会发生数据泄露，而模型应用于实际预测时并不可能得到类似的数据。这样的状况导致训练集（甚至验证集）的行能非常好，但在实际工作中性能却很差。数据泄露包括两种类型：**target leakage**（目标泄露）与**train-test contamination**（训练测试污染）

**Target leakage**。在挑选特征时，需要更多考虑时间或时间顺序以避免目标信息泄露。例如在预测个人是否患肺炎时，原始数据集中有“是否服用消炎药”的特征列。这个特征列与待预测结果有极强的相关性，通常患肺炎的人会服用消炎药已利于病情恢复。这个特征列的值在患肺炎后会发生变化，造成了target leakage。

训练集与验证集中都有这个特征值，所以模型会得到非常好的验证成绩，即使是交叉验证。但模型在预测现实世界数据时准确性会非常差，因为现实生活中待预测人群通常不会服用消炎药。要避免此类data leakage，需要在训练模型时将该种值会随target值变化的特征剔除。

Train-Test contamination。验证集适用于检验模型成绩的，所以验证集中的数据不应出现在训练集中。如果不小心将验证集的数据用于预测，就会发生Train-Test contamination。例如在进行训练测试数据分离前进行了填充缺失值的fit等数据预处理操作。这种情况尤其容易发生于进行复杂特征工程时。

使用scikit-learn的pipeline封装数据预处理步骤比较容易避免此类leakage，尤其是使用交叉验证时。

例子：信用卡申请通过与否的预测。使用简单的随机森林分类模型，利用交叉验证计算模型成绩高达0.98。依靠经验判断，应该很难找到一个准确率达到98%的模型，所以要怀疑发生了数据泄露。接下来分析特征列的描述，expenditure变量描述平均每月信用卡消费金额，这里就要再看看这个消费是基于之前的信用卡消费还是在申请的这张卡的消费。

    expenditures_cardholders = X.expenditure[y]
    expenditures_noncardholders = X.expenditure[~y]
    
    print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
    print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))

这里可以看出来申请这张卡失败的人的expenditure值100%为0，而申请这张卡成功的人的expenditure值只有2%为0，所以可以判断有极大的可能，这个expenditure值是指在申请这张卡的月平均消费。所以这个值是申请该卡成功与否之后产生的值，与待预测目标强相关，如果用于模型训练，会造成data leakage。变量share与expenditure相关，所以也会造成data leakage。这里还有两个变量active和majorcards也很可疑，安全比抱歉强，所以最好也把这两个值剔除出去。当把这四个特征值剔除以后再建模后进行交叉验证得到成绩为0.83。虽然这个成绩不太好，但用于实际预测时成绩也是可预期的，而不会想之前存在数据泄露的模型会造成出乎意料的坏成绩。

结论。数据泄露是可能造成数百万刀的错误。小心谨慎的分离训练集与验证集以及使用pipeline能够帮助解决training-test污染。同样，谨慎、理性以及数据探索能够帮助鉴别目标泄露。














