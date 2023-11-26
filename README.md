# Gesture_recognition

结合注意力机制的卷积神经网络,多维度特征融合的手势识别算法。

参考文献：李楚杨. 基于毫米波雷达的手势识别算法研究[D].电子科技大学,2020.DOI:10.27005/d.cnki.gdzku.2020.003496.

本部分主要实现参考文献的第三章——结合注意力机制的 CNN 多维度特征融合识别算法。

## 网络结构

|操作|核长|步长|输出维度|备注|
|---|---|---|---|---|
|输入|/|/|224x224x3|/|
|特征提取模块|
|conv*2|3|1|224x224x32|block_1|
|maxpool|2|2|112x112x32|block_1|
|conv*2|3|1|112x112x64|block_2|
|maxpool|2|2|56x56x64|block_2|
|conv*3|3|1|56x56x128|block_3|
|maxpool|2|2|28x28x128|block_3|
|conv*3|3|1|28x28x256|block_4|
|maxpool|2|2|14x14x256|block_4|
|conv*3|3|1|14x14x512|block_5|
|maxpool|2|2|7x7x512|block_5|
|特征融合|
|reshape|/|/|49x512|/|
|Reshape&transpose|/|/|512x49|/|
|multiply&Softmax|/|/|512x512|相关性信息|
|multiply|/|/|49x512|注意力谱|
|plus&reshape|/|/|7x7x512|残差结构|
|识别结果输出|
|conv*3|3|1|7x7x512|/|
|FC*3|/|/|8|/|
|Softmax|/|/|8|类别预测概率|

## 算法训练相关设置

|名称|具体内容|
|---|---|
|最大训练迭代次数|500|
|初始学习率|0.001|
|Batch 大小|8|
|损失函数|交叉熵损失|
|优化算法|Adam|

因在算法训练过程中，为达到更好的训练效果，学习率应随算法训练进程而改
变，训练进程越深入，学习率应越小，从而得到更加精细的参数优化，本章算法训
练的初始学习率设置为 0.001，设计每 30 个训练迭代次数后，学习率变为上一阶
段的 0.5 倍。


## 安装依赖
pip install -r requirements.txt

## 文件说明
1. data
所有与模型相关的数据集，包括训练集、验证集和测试集。
2. history
模型训练过程可视化的结果。
3. model
保存模型结构及其权重。
4. test_img
用于生成测试集的图片集。
5. train_img
用于生成训练集的图片集。
6. valid_img
用于生成验证集的图片集。
7. config.json
程序运行的配置文件。
8. DataGet.py
由图片集生成数据集，从数据集加载数据和标签。
9. Gesture_recognition.py
程序运行的入口，评估测试集。
10. ImgSave.py
模型训练可视化，绘制混淆矩阵。
11. ModelCreate.py
创建模型，训练模型。

## 配置

loadmodel: 0: 重新训练模型并保存权重 1: 加载已训练过的模型权重

dataupdate: 0: 不更新数据集 1: 更新数据集(图片集发生变化)

showmodel: 0: 不展示网络结构 1: 展示网络结构

showhistory: 0: 不呈现训练过程的可视化 1: 训练过程可视化并保存图片

learning_rate: 学习率

batch_size: 一次训练所抓取的数据样本数量

epochs: 循环次数

train_img_nums: 训练图片的总数

valid_img_nums: 验证图片的总数

test_img_nums: 测试图片的总数

 
