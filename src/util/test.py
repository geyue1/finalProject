# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> test.py
@Author : yge
@Date : 2024/7/23 21:34
@Desc :

==============================================================
'''
import os

import torch
import torchvision.datasets as dt
from PIL import Image

csv = "/Users/yge/Downloads/batch1_test_annotation_G10.csv"

f = open(file=csv,mode="r",encoding="utf-8")
index = 5
for line in f:
    print(line)
    if not "jpg" in line:
        continue
    line = line.rstrip()
    file_name = line.split(",")[0].replace("/content/drive/My Drive/AAR","/Users/yge/Downloads")
    x_min = line.split(",")[1]
    y_min = line.split(",")[2]
    x_max = line.split(",")[3]
    y_max = line.split(",")[4]
    label = line.split(",")[5]

    img = Image.open(file_name).crop((int(x_min),int(y_min),int(x_max),int(y_max)))
    img = img.resize((56,56))
    img.save(fp="/Users/yge/Downloads/"+label+str(index)+".png",format="PNG")
    index+=1





