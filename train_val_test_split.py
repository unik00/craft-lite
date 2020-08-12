import os
from pathlib import Path
import random

if __name__ == "__main__":
    train_loc = '/home/huny/Documents/gmo/craft_keras_hakaru/data/train'
    val_loc = '/home/huny/Documents/gmo/craft_keras_hakaru/data/val'

    img_exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    img_paths = list([])

    for img_ext in img_exts:
        img_paths += list(Path(train_loc).glob(img_ext))

    random.shuffle(img_paths)

    datas = []

    if not os.path.exists(val_loc):
        os.mkdir(val_loc)
        print("Created validate folder.")

    thresh = 0.2
    thresh_int = thresh * len(img_paths)
    print(thresh_int)
    for i, img_path in enumerate(img_paths):
        img_path = str(img_path)
        anno_path = '.'.join(img_path.split('.')[:-1]) + ".xml"

        if i <= thresh_int:
            dest = img_path.replace("/data/train", "/data/val")
            print(img_path, dest)
            os.rename(img_path, dest)

            if not os.path.exists(anno_path):
                print("WARNING: not found {}".format(anno_path))
            else:
                dest = anno_path.replace("/data/train", "/data/val")
                os.rename(anno_path, dest)

        else:
            break