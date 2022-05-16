import os

import shutil

path = "PaintMe_Debug_Backup"

if __name__ == "__main__":
    with open("0295843_ids.txt", "r") as f:
        contents = f.readlines()
        ids = [content.strip() for content in contents]
    
    num_removed_files = 0
    for sample in os.listdir(path):
        if not str(sample) in ids:
            print("{} does not exist!".format(sample))
            num_removed_files += 1
            shutil.rmtree(os.path.join(path, sample))