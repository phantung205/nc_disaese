import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#-----------------------
# path data
#-----------------------
data_dir = os.path.join(base_dir,"data")
# path data raw
dir_data_raw = os.path.join(data_dir,"raw","three")
# path data processed
data_processed_dir = os.path.join(data_dir,"processed")



splits = ["train","val"]
# categorys = ["No_DR","Early_DR","Dangerous_DR"]
categorys = ["No_DR","DR"]
batch_size = 16
image_size = 224
train_ratio = 0.8


learning_rate = 1e-4
weight_decay = 1e-4
momentom = 0.9
epochs = 100

report_dir = os.path.join(base_dir,"reports")
path_tensorboard = os.path.join(report_dir,"tensorboard")

model_dir = os.path.join(base_dir,"trained_models")