# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> parameter.py
@Author : yge
@Date : 2024/7/7 10:11
@Desc :

==============================================================
'''
import os

LOGGING_ENABLED  = False
SAVE_MODEL = False

epoch_num = 50
lr = 0.01
batch_size=256

in_channels_28 = 1
out_features_28 = 10

in_channels_56 = 3
out_features_56 = 2

tensorboard_path = os.path.join("..", "logs","augment")