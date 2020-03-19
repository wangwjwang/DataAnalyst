# 跟bilibili视频学习 #
视频地址：
[https://www.bilibili.com/video/av64604811?p=2](https://www.bilibili.com/video/av64604811?p=2 "Bilibili上的视频-从零开始学python数据分析（罗攀）")
##一. Numpy 介绍##

Numpy库是开源Python库，是数据分析包的基础包，主要提供了高性能的数组与矩阵的运算处理能力

# 二. 数组创建#

导入库：`import numpy as np`

一维数组构造：`arr1 = np.array([2,3,4])`

多维数组构造：`arr2 = np.array([[2,3,4],[5,6,7]])`

创建全1数组,2行3列：`arr3=np.ons(2,3)`

同上创建全0数组，全空数组`np.zeros()``np.empty`

创建随机整数数组，randint：`arr5 = np.random.randint(0,10,size(3,4))`用0到10的随机数创建3X4的数组

创建正态分布的样本值：`randn`
# 三.数组属性 #

查看数组形状，X行X列：`arr1.shape`

查看数组元素数据类型：`arr1.dtype`

查看数组元素个数：`arr1.size`

查看数组元素字节大小：`arr1.itemsize`

# 四.数组变换 #


数组元素类型变换，astype：`arr3 = arr1.astype('int')`

数组形状变换，reshape：`arr4 = arr2.reshape(3,2)`

数组展开为一维数组，并返回引用，ravel：`arr4.ravel()`

flatten功能也是降为一维，但仅返回拷贝，修改不会影响原数组

多数组合并堆叠，`concatenate`，由axis参数指定横叠还是竖叠

`vstack`和`hstack`可以直接实现横叠和竖叠
    
数组拆分`split`

数组转制，行列翻转，transpose，T：`arr3.Transpose`简写`arr3.T`

# 五.数组索引与切片 #

索引为正数，意味着由0开始从前往后数。索引为负数，意味着由-1开始从后往前数。

索引与切片都是返回原数组的索引，会改变原数组值

# 五.数组运算#

对数组直接+-* /，就是对每个元素+- */
求绝对值，`abs()`
对数组某行或某列求和，求平均，求标准差`sum，mean，std`
求最小最大值`min，max`
