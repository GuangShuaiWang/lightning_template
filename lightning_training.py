'''
@File    :   lightning_training.py
@Time    :   2025/03/17 16:29
@Author  :   wangsh
@Contact :   hiwangguangshuai@gmail.com
@Brief   :   lightning teampalet
'''
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

from utils import get_cosine_schedule_with_warmup
from datasets import load_from_disk,concatenate_datasets
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import pandas as pd
import os


class EATemaPLmoduel(L.LightningModule):
    def __init__(self,
                 num_training_steps,
                 num_warmup_steps,
                 lr,
                 ):
        super().__init__()
        self.save_hyperparameters() 
        self.num_training_steps = num_training_steps 
        self.num_warmup_steps = num_warmup_steps 
        self.lr = lr

        self.model = None
        self.loss = torch.nn.MSELoss()
        ###########
        # 保存超参数
        # 后三个都是用来对学习率进行操作，定义好warm up和cosine的学习率
        # 最后两个是定义模型，和损失函数。


        ###########


    def training_step(self, batch, batch_idx):

        #定于训练过程中，一个step是如何进行前向传播的。batch是一个字典。发挥的是一个loss


        for data_loader in batch:
            meg_raw_data,layout_pos =  data_loader["sample_raw"],data_loader["position"]
            
            cls_feature,teacher_layer_average_mean_pooling,student_decoder,teacher_layer_average = self.EAT(meg_raw_data,layout_pos)

            cls_loss = self.loss(cls_feature,teacher_layer_average_mean_pooling.detach())
            frame_loss = self.loss(student_decoder,teacher_layer_average.detach())

            loss = cls_loss + frame_loss
            self.log("cls loss",cls_loss,on_step=True,prog_bar=True)
            self.log("Frame_loss",frame_loss,on_step=True,prog_bar=True)

            self.log("train_loss", loss, on_step=True,prog_bar=True)
            self.log_gradients()
            return loss

    def on_train_batch_end(self,outputs, batch, batch_idx):
        self.scheduler.step()
        current_lr = self.optimizers().param_groups[0]['lr']
        self.lr_history.append(current_lr)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'])

        # 这个勾子函数定义在一个batch结束后，需要做什么，需要做的是通过scheduler更新lr

    def on_after_backward(self):
        # 定义在反向传播之后干什么，我这里是记录了一些东西。
        # 获取当前 step 和 epoch
        current_step = self.global_step
        current_epoch = self.current_epoch

        current_version = f"version_{self.logger.version}"
        version_log_dir = os.path.join(self.logger.save_dir, self.logger.name, current_version)
        #定义文件存储路径

        grad_info = []
        if self.trainer.is_global_zero: #在多卡中，如果不设置只在主进程上保存的话，会出现文件夹不存在的情况。该命令是用于区分主进程。
            if current_step <= 30:
                for name, param in self.named_parameters():
                    if "bias" not in name:  
                        if param.requires_grad: 
                            if param.grad is not None: 
                                grad_norm = param.grad.norm().item()
                                grad_info.append({"param_name": name, "grad_norm": grad_norm})
                            else:
                                grad_info.append({"param_name": name, "grad_norm": None})  
                        else:
                            grad_info.append({"param_name": name, "grad_norm": "no require grad"})

                grad_df = pd.DataFrame(grad_info)
                filename = os.path.join(version_log_dir,f"gradient_epoch_{current_epoch}_step_{current_step}.csv")
                grad_df.to_csv(filename, index=False) 


    
    def validation_step(self, batch, batch_idx,dataloader_idx=0):
        #跟training_step一样，返回的是val_loss
        pass
        
    def log_gradients(self):
        #记录模型的梯度变化，有利于查找问题。
        total_grad_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                total_grad_norm += torch.norm(param.grad, p=2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5  
        self.log('grad_norm', total_grad_norm, prog_bar=True)

    def configure_optimizers(self):
        #这一步非常重要，定义优化器和学习率变化的scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=0.01)
        self.scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps = self.num_training_steps, num_warmup_steps=self.num_warmup_steps)
        return [optimizer], [self.scheduler]

    
    
class DataModule(L.LightningDataModule):
    # 这里实现了多个dataloader要怎么训练，因为我的数据集纬度不一样，不能塞到同一个dataloader里
    def __init__(self, file_path,dataset_name: list,batch_size=32,num_workers = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        all_dataset = []
        for i in dataset_name:
            path = []
            import os
            for root, dirs, _ in os.walk(file_path):
                for dir_name in dirs:
                    if dir_name.startswith(i):
                            path.append(os.path.join(root,dir_name))
            dataset_list = [ load_from_disk(x).with_format('torch') for x in path]
            data = concatenate_datasets(dataset_list)
            
            all_dataset.append(data)

        self.all_dataset_split = [ i.train_test_split(test_size=0.2,load_from_cache_file = False) for i in all_dataset]
        self.all_dataset_train = [i["train"] for i in self.all_dataset_split]
        self.all_dataset_test = [i["test"] for i in self.all_dataset_split]
    def train_dataloader(self):

        return [DataLoader(i, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers,pin_memory=True) for i in self.all_dataset_train]

    def val_dataloader(self):

        return [DataLoader(i, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers,pin_memory=True) for i in self.all_dataset_test]
    
    def __len__(self):
        return sum(len(i) for i in self.train_dataloader())
