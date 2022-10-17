# Gesture_recognition
结合注意力机制的卷积神经网络,多维度特征融合的手势识别算法。

## 安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
为方便起见，已将深度学习相关库tensorflow拷贝至本文件所在文件夹下。
如有需要，可自行安装深度学习相关库。

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
config
    loadmodel: 
        0: 重新训练模型并保存权重
        1: 加载已训练过的模型权重
    dataupdate: 
        0: 不更新数据集
        1: 更新数据集(图片集发生变化)
    showmodel: 
        0: 不展示网络结构
        1: 展示网络结构
    showhistory: 
        0: 不呈现训练过程的可视化
        1: 训练过程可视化并保存图片
    learning_rate: 
        学习率
    batch_size:
        一次训练所抓取的数据样本数量
    epochs:
        循环次数
    train_img_nums: 
        训练图片的总数
    valid_img_nums: 
        验证图片的总数
    test_img_nums: 
        测试图片的总数

## 运行
双击run.bat

 