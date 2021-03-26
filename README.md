# recommendation-system-model
&emsp;This repo is created to store the classic Recommendation System model here during the period of datawhale deep recommendation system learing.

## 忏悔&致谢
&emsp;忘记给optimzer添加step导致大量的时间被花费在找其他不存在的bug上，此处感谢耿大佬的帮助！
&emsp;在3/23日前clone的代码没有添加step，对关注了这个库的大家表示深深的歉意😭

## DeepCrossing
&emsp;目前无bug，但是训练出的分类器变成了
```python
def net(x):
    return 0
```
这种情况。
这个bug已经解决了，解决方案为：减小学习率和减小batchsize（当数据集很小时）

2021.3.17 19：31
&emsp;将batch_size调整成10，且损失函数选择交叉熵可以正确进行网络的训练。
&emsp;有两个问题：1.为什么学习代码中的tf代码的batch_size可以是64。2.为什么将损失函数改为MSE又会变成弱(智商)分类器。

## W&D
&emsp;bug to fix yet.

## DeepFM
&emsp;将数据换为完整数据集。


## NFM
&emsp;将数据换为完整数据集并保留了迭代结果，具体可见NFM文件夹，可直接运行main文件。

## DIN
&emsp;可直接运行main.py文件