import os
import json

from ModelCreate import create_model, model_train, load_model
from DataGet import jpg2array, load_data
from ImgSave import historysave, confusionsave


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_dir = os.path.dirname(__file__)
model_file = os.path.join(_dir,'model\\model.h5')
train_dir = os.path.join(_dir,'train_img')
valid_dir = os.path.join(_dir,'valid_img')
test_dir = os.path.join(_dir,'test_img')
train_file = os.path.join(_dir,'data\\train_data.pkl')
valid_file = os.path.join(_dir,'data\\valid_data.pkl')
test_file = os.path.join(_dir,'data\\test_data.pkl')
confusion_file = os.path.join(_dir,'history\\confusion_matrix.jpg')
config_file = os.path.join(_dir,'config.json')

with open(config_file, "r", encoding="utf-8") as f:
     config = json.loads(f.read())
loadmodel = config["loadmodel"]
dataupdate = config["dataupdate"]
showmodel = config["showmodel"]
showhistory = config["showhistory"]
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
epochs = config["epochs"]
train_img_nums = config["train_img_nums"]
valid_img_nums = config["valid_img_nums"]
test_img_nums = config["test_img_nums"]
print("配置加载完成")

if dataupdate:
    jpg2array(train_dir, train_file, train_img_nums)
    jpg2array(valid_dir, valid_file, valid_img_nums)
    jpg2array(test_dir, test_file, test_img_nums)
    print("数据更新成功")

if loadmodel:
    model = load_model(model_file)
    print("模型加载成功")
else:
    model = create_model(learning_rate)
    print("模型创建成功")
    history = model_train(model, train_file, valid_file, 
                          epochs, batch_size) 
    model.save(model_file)
    if showhistory:
        history=history.history
        accuracy_file = os.path.join(_dir,'history\\accuracy.jpg')
        loss_file = os.path.join(_dir,'history\\loss.jpg')
        historysave(history, accuracy_file, loss_file)

if showmodel:
    model.summary()

test_data, test_labels = load_data(test_file)
print("测试集加载成功")
print("模型测试结果")
model.evaluate(test_data, test_labels)

pred_labels = model.predict(test_data)
confusionsave(pred_labels, test_labels, confusion_file)