import os
import shutil
import random

# ✅ 디렉토리 경로 설정
base_dir = '/home/bsj/workspace/Grounded-SAM-2/YOLOv8/train'
image_dir = os.path.join(base_dir, 'merged_images')
label_dir = os.path.join(base_dir, 'merged_labels_obb')
pcd_dir = os.path.join(base_dir, 'merged_pcd')

output_image_dir = os.path.join(base_dir, 'final_images')
output_label_dir = os.path.join(base_dir, 'final_labels_obb')
output_pcd_dir = os.path.join(base_dir, 'final_pcd')

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_pcd_dir, exist_ok=True)

# ✅ 총 파일 수 (쌍 기준)
total = len(os.listdir(image_dir))
indices = list(range(total))
random.seed(42)
random.shuffle(indices)

# ✅ 복제 수 설정
target_count = 5000
current_count = 0
idx = 0

print(f"🔁 총 {total}개의 원본 쌍을 기준으로 {target_count}개까지 확장 시작...")

while current_count < target_count:
    index = indices[idx % total]

    img_src = os.path.join(image_dir, f'img_dataframe_{index:04d}.jpg')
    label_src = os.path.join(label_dir, f'img_dataframe_{index:04d}.txt')
    pcd_src = os.path.join(pcd_dir, f'cloud_{index:04d}.pcd')

    img_dst = os.path.join(output_image_dir, f'data_{current_count:04d}.jpg')
    label_dst = os.path.join(output_label_dir, f'data_{current_count:04d}.txt')
    pcd_dst = os.path.join(output_pcd_dir, f'data_{current_count:04d}.pcd')

    shutil.copyfile(img_src, img_dst)
    shutil.copyfile(label_src, label_dst)
    shutil.copyfile(pcd_src, pcd_dst)

    current_count += 1
    idx += 1

    if current_count % 100 == 0 or current_count == target_count:
        print(f"✅ 진행 중... {current_count}/{target_count}")

print(f"\n🎉 최종 완료: {target_count}개의 쌍이 생성되어 저장되었습니다.")
