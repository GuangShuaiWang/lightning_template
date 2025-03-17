'''
@File    :   run_EAT.py
@Time    :   2024/12/10 11:41
@Author  :   wangsh
@Contact :   hiwangguangshuai@gmail.com
@Brief   :   
'''

from pytorch_lightning_template.lightning_training import EATemaPLmoduel,DataModule
import yaml
from lightning.pytorch.loggers import CSVLogger,WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
import lightning as L
import argparse

parser = argparse.ArgumentParser(description='EAT_training config')
parser.add_argument('--multiple_gpu_nodes', type=bool, help='use multipel gpu nodes',default=False)
parser.add_argument('--model_size', type=str, help='use multipel gpu nodes',default="small")
args = parser.parse_args()
print(args)
with open("/train_config.yaml","r") as f:
        config = yaml.safe_load(f)

model_parameters = config["model_config"][args.model_size]
args_dict = vars(args)
combined_config = {**config, **args_dict}

data_module = DataModule(file_path="/hf_dataset_clamping",
                         dataset_name=["camcan_batch_","HCP_batch_"],
                         batch_size=config["batch_size"],num_workers=config["num_workers"])
print("num_train",len(data_module))


#这里去确定好总共的training step数目，这样才能对lr进行周期变化。
if args.multiple_gpu_nodes :
    num_training_steps = (len(data_module) * config["max_epochs"] // (config["num_gpu"]*config["nodes"]) ) + 1
else: 
    num_training_steps = (len(data_module) * config["max_epochs"] // config["num_gpu"] ) + 1
print("number step",num_training_steps)

#logger = CSVLogger("/home/bingxing2/ailab/scxlab0023/BrainCoding/MEG2/lightning_logs", name="EAT_training_multigpu")
logger = WandbLogger(
    project="BrainDecoding",
    name="clamping-MSEloss-model_size_{}-batch_size_{}-lr_{}".format(args.model_size,config["batch_size"],config["lr"]),
    save_dir="/home/bingxing2/ailab/scxlab0023/BrainCoding/MEG2/lightning_logs",
    tags=["pretiran"],
)
#这里定义了logger，可以是wandb的，也可以只是本地CSV的log。
logger.log_hyperparams(combined_config) #而外保存超参数


checkpoint_callback = ModelCheckpoint(monitor='val_loss/dataloader_idx_0',
                                    save_top_k=-1,
                                    save_last = True,
                                    filename="{epoch:02d}-{step}")
early_stopping = EarlyStopping('val_loss/dataloader_idx_0',patience=5)
#几个有用的call_back函数。

if not args.multiple_gpu_nodes:
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],default_root_dir=config["root_dir"],
        devices=config["num_gpu"],strategy='ddp_find_unused_parameters_true',
        #val_check_interval=config["val_step"],
        logger=logger,callbacks=[checkpoint_callback],
        log_every_n_steps=1
    )
else:
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],default_root_dir=config["root_dir"],
        devices=config["num_gpu"],strategy='ddp',
        #val_check_interval=config["val_step"],
        logger=logger,callbacks=[checkpoint_callback,early_stopping],num_nodes=config["nodes"],
        log_every_n_steps=1,
    )
#定义trainer，两个不同的点就只在于有无多机运行。

num_warmup_steps = 0.1 * num_training_steps

model = EATemaPLmoduel(                 
            num_training_steps = num_training_steps,
            num_warmup_steps = num_warmup_steps,
            lr = config["lr"],
            encoder_nhead=model_parameters["encoder_nhead"],encoder_dmodel=model_parameters["encoder_dmodel"],encoder_nlayer=model_parameters["encoder_nlayer"],
            decoder_nhead = model_parameters["decoder_nhead"],decoder_dmodel =model_parameters["decoder_dmodel"],decoder_nlayer = model_parameters["decoder_nlayer"],
            )

trainer.fit(model=model, datamodule=data_module)