import os
import shutil
import random
from glob import glob

BASE_DIR = './YOLOv8'
SPLIT_RATIO = 0.1  # 10%

sets = ['images', 'labels_hbb', 'labels_obb']
cases = os.listdir(os.path.join(BASE_DIR, 'train/images'))

for case in cases:
    print(f"Processing {case}...")

    # 각 set(images, labels_hbb, labels_obb) 별로 처리
    for set_name in sets:
        train_dir = os.path.join(BASE_DIR, 'train', set_name, case)
        valid_dir = os.path.join(BASE_DIR, 'valid', set_name, case)
        os.makedirs(valid_dir, exist_ok=True)

        files = glob(os.path.join(train_dir, '*.jpg' if set_name == 'images' else '*.txt'))
        # num_valid = max(1, int(len(files) * SPLIT_RATIO))
        num_valid = max(1, int(len(files) * SPLIT_RATIO))
        num_valid = min(num_valid, len(files))  # 파일 수보다 많지 않게 조정

        valid_files = random.sample(files, num_valid)

        for f in valid_files:
            shutil.move(f, valid_dir)

print("Split complete!")
