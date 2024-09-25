# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> process_seedsplit.py
@Author : yge
@Date : 2023/7/13 15:32
@Desc :

==============================================================
'''
import os
from PIL import Image
from tqdm import tqdm

RE_SIZE=(56,56)
LOCAL = "/Users/yge/Downloads/"
REMOTE = "/content/drive/My Drive/AAR/"
SAVE = "../../data/AAR/"

def process_seedsplit(source_path,targent_path):
    if not os.path.isdir(targent_path):
        os.makedirs(targent_path)

    for file_name in os.listdir(source_path):
        if file_name.endswith(".jpg"):
            img = Image.open(os.path.join(source_path,file_name))
            img = img.resize(RE_SIZE)
            img.save(fp=os.path.join(targent_path,file_name[0:-4]+".png"),format="PNG")

def process(csv_files,local, remote, save, re_size):
    """
        Process a list of CSV files to crop and resize images based on bounding box coordinates and save them.

        Args:
        csv_files (list): List of CSV file names to process.
        local (str): Local directory path to read images from.
        remote (str): Remote directory path to be replaced in the CSV file.
        save (str): Directory path to save processed images.
        re_size (tuple): Desired size to resize the cropped images (width, height).
        """
    index = 0
    for csv_file in csv_files:
        with open(file=os.path.join(local,csv_file),mode="r",encoding="utf-8") as f:
            lines = f.readlines()
            with tqdm(lines, desc=f"processing {csv_file}",total=len(lines), unit="line") as pbar:
                for line in lines:
                    if not "jpg" in line.lower():
                        continue
                    line = line.rstrip()
                    file_name = line.split(",")[0].replace(remote, local)
                    x_min = line.split(",")[1]
                    y_min = line.split(",")[2]
                    x_max = line.split(",")[3]
                    y_max = line.split(",")[4]
                    label = line.split(",")[5]

                    img = Image.open(file_name).crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                    img = img.resize(re_size)
                    save_path = save+label
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    img.save(fp=os.path.join(save_path,label+str(index)+".png"), format="PNG")
                    index += 1
                    pbar.update()
                pbar.close()
    return index





if __name__ == '__main__':
    # process_seedsplit("/Users/yge/Downloads/seedsplit/train/GoodSeed","../../data/AAR/train/GoodSeed")
    # process_seedsplit("/Users/yge/Downloads/seedsplit/train/BadSeed","../../data/AAR/train/BadSeed")
    # process_seedsplit("/Users/yge/Downloads/seedsplit/test/GoodSeed", "../../data/AAR/test/GoodSeed")
    # process_seedsplit("/Users/yge/Downloads/seedsplit/test/GoodSeed","../../data/AAR/test/GoodSeed")
    csv_files = ["batch1_test_annotation_G10.csv","batch1_train_annotation_G10.csv",
                 "LightBox_annotation.csv","NormalRoomLight_annotation.csv"]
    index = process(csv_files,local=LOCAL,remote=REMOTE,save=SAVE,re_size=RE_SIZE)
    print(index)