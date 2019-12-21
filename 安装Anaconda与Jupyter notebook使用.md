# 跟bilibili视频安装基础环境 #
视频地址：
[https://www.bilibili.com/video/av64604811?p=2](https://www.bilibili.com/video/av64604811?p=2 "Bilibili上的视频-从零开始学python数据分析（罗攀）")
##一. Anaconda安装 ##
Anaconda是集成安装了很多数据分析三方库的Python集成环境

清华开源站下载：[https://mirror.tuna.tsinghua.edu.cn/help/anaconda/](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)
## 二.Anaconda配置 ##
打开Anaconda Prompt

查看已安装的库包,输入`Conda list`

安装具备anaconda包的新环境XXX,输入`conda create --name XXX python=3 anaconda`

查看已安装环境`conda info --envs`

激活环境`activate XXX`

包的安装`conda install XXX`或`pip install XXX`
# 三.Jupyter notebook使用 #

更改Jupyter notebook的编写空间，Prompt中切换到对应目录，然后输入`jupter notebook`启动程序


