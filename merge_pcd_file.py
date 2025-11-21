import os
import shutil
from glob import glob

# ✅ 기본 디렉토리
base_dir = '/home/bsj/workspace/Grounded-SAM-2/YOLOv8/train'
pcd_root = os.path.join(base_dir, 'pcd')

# ✅ 출력 디렉토리
merged_pcd_dir = os.path.join(base_dir, 'merged_pcd')
os.makedirs(merged_pcd_dir, exist_ok=True)

# ✅ 하위 폴더 목록 (pcd_root 안의 폴더들)
sub_dirs = [d for d in os.listdir(pcd_root) if os.path.isdir(os.path.join(pcd_root, d))]
count = 0

for sub in sorted(sub_dirs):
    pcd_dir = os.path.join(pcd_root, sub)
    pcd_files = sorted(glob(os.path.join(pcd_dir, '*.pcd')))
    for pcd_path in pcd_files:
        new_pcd_name = f'cloud_{count:04d}.pcd'
        new_pcd_path = os.path.join(merged_pcd_dir, new_pcd_name)
        shutil.copy2(pcd_path, new_pcd_path)
        count += 1

print(f"✅ 병합 완료: 총 {count}개의 PCD 파일이 {merged_pcd_dir}에 저장되었습니다.")
