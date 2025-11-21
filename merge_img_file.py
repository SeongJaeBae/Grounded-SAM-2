import os
import shutil
from glob import glob

# ✅ 기본 디렉토리
base_dir = '/home/bsj/workspace/Grounded-SAM-2/YOLOv8/train'
images_root = os.path.join(base_dir, 'images')
labels_root = os.path.join(base_dir, 'labels_obb')

# ✅ 출력 디렉토리
merged_images_dir = os.path.join(base_dir, 'merged_images')
merged_labels_dir = os.path.join(base_dir, 'merged_labels_obb')
os.makedirs(merged_images_dir, exist_ok=True)
os.makedirs(merged_labels_dir, exist_ok=True)

# ✅ 하위 폴더들 (공통 기준)
sub_dirs = [d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
count = 0

print(sub_dirs)
for sub in sorted(sub_dirs):
    print(sub)    
    img_dir = os.path.join(images_root, sub)
    label_dir = os.path.join(labels_root, sub)

    img_files = sorted(glob(os.path.join(img_dir, '*.jpg')))
    for img_path in img_files:
        # 이미지 파일 이름 저장
        new_img_name = f'img_dataframe_{count:04d}.jpg'
        new_img_path = os.path.join(merged_images_dir, new_img_name)
        shutil.copy2(img_path, new_img_path)

        # 라벨 파일 복사 (같은 이름, 확장자만 .txt)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{basename}.txt")
        new_label_name = f'img_dataframe_{count:04d}.txt'
        new_label_path = os.path.join(merged_labels_dir, new_label_name)

        if os.path.exists(label_path):
            shutil.copy2(label_path, new_label_path)
        else:
            # 라벨 파일이 없는 경우 빈 라벨 파일 생성
            open(new_label_path, 'w').close()

        count += 1

print(f"✅ 병합 완료: 총 {count}개의 이미지와 라벨이 저장되었습니다.")
print(f"📁 이미지 → {merged_images_dir}")
print(f"📁 라벨   → {merged_labels_dir}")
