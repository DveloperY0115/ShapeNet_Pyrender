import os

path = "PaintMe_Images_Backup"

if __name__ == "__main__":
    ids = sorted(os.listdir(path))

    with open("0295843_ids.txt", "w") as f:
        for id in ids:
            f.write(id + "\n")