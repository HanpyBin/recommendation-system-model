# recommendation-system-model
This repo is created to store the classic Recommendation System model here during the period of datawhale deep recommendation system learing.

## DeepCrossing
目前无bug，但是训练出的分类器变成了
```python
def net(x):
    return 0
```
这种情况。

2021.3.17 19：31
&emsp;将batch_size调整成10，且损失函数选择交叉熵可以正确进行网络的训练。
&emsp;有两个问题：1.为什么学习代码中的tf代码的batch_size可以是64。2.为什么将损失函数改为MSE又会变成弱(智商)分类器。

## W&D
&emsp;准确率不上升，修改batch_size和lr也无济于事。。。I need help!!!

## DeepFM
&emsp;准确率不上升。。。