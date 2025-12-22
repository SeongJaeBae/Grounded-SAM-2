import os
import cv2
import re
import shutil
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images


"""
===========================================================
  USER SETTINGS
===========================================================
"""

# 원본 이미지가 들어있는 폴더 (data_00001.jpg 같은 이름)
IMAGE_INPUT_DIR = "./input_images_2"  # 여기에 data_00001.jpg, data_00002.jpg ...

TEXT_PROMPT = "packet."
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT_TYPE_FOR_VIDEO = "mask"  # ["point", "box", "mask", "obb"]

# YOLO 저장 디렉토리
IMAGE_DIR = './YOLOv8/train/images/from_imagefolder_2'
LABEL_DIR_HBB = './YOLOv8/train/labels_hbb/from_imagefolder_2'
LABEL_DIR_OBB = './YOLOv8/train/labels_obb/from_imagefolder_2'

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR_HBB, exist_ok=True)
os.makedirs(LABEL_DIR_OBB, exist_ok=True)

# SAM2용 임시 프레임 폴더 (여기에 00000.jpg, 00001.jpg 형태로 복사)
TEMP_FRAME_DIR = "./temp_numeric_frames"
os.makedirs(TEMP_FRAME_DIR, exist_ok=True)


"""
===========================================================
  SAM2 + GroundingDINO 초기화
===========================================================
"""

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# SAM2 image / video predictor
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# Grounding DINO
processor = AutoProcessor.from_pretrained(MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to("cuda")


"""
===========================================================
 Step 1 — 원본 이미지 파일 리스트 + 숫자 정렬 + 임시 폴더로 복사
===========================================================
"""

def extract_number(filename: str) -> int:
    """
    파일명 안의 숫자만 추출해서 정렬용 key로 사용.
    ex) 'data_00586.jpg' -> 586
    """
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1

assert os.path.exists(IMAGE_INPUT_DIR), f"[ERROR] 입력 폴더 없음: {IMAGE_INPUT_DIR}"

# 원본 폴더에서 이미지 파일 목록 가져오기
orig_frame_names = [
    p for p in os.listdir(IMAGE_INPUT_DIR)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
]

# 숫자 기준으로 정렬 (data_00001, frame_0001 등 다 커버)
orig_frame_names.sort(key=lambda p: extract_number(p))

print(f"[INFO] 총 {len(orig_frame_names)}장의 이미지 로드됨 from {IMAGE_INPUT_DIR}")

# SAM2가 요구하는 '순수 숫자 이름' 프레임 생성
rename_map = {}  # SAM2 frame index -> 원본 파일명 매핑

print(f"[INFO] 임시 폴더({TEMP_FRAME_DIR})에 숫자 파일명으로 복사 중...")
for i, fname in enumerate(orig_frame_names):
    src_path = os.path.join(IMAGE_INPUT_DIR, fname)
    newname = f"{i:05d}.jpg"
    dst_path = os.path.join(TEMP_FRAME_DIR, newname)
    shutil.copy2(src_path, dst_path)
    rename_map[i] = fname  # 나중에 결과 저장할 때 원본 이름으로 되돌리기

print(f"[INFO] 복사 완료. 총 {len(rename_map)}장 숫자 프레임 생성됨 in {TEMP_FRAME_DIR}")

# SAM2 VideoPredictor 초기화 (이제 TEMP_FRAME_DIR에는 00000.jpg, 00001.jpg ... 형식)
inference_state = video_predictor.init_state(video_path=TEMP_FRAME_DIR)


"""
===========================================================
 Step 2 — 첫 프레임에서 GroundingDINO → box detection
===========================================================
"""

ann_frame_idx = 0  # 첫 프레임 기준
first_frame_path = os.path.join(TEMP_FRAME_DIR, f"{ann_frame_idx:05d}.jpg")
first_image = Image.open(first_frame_path)

inputs = processor(images=first_image, text=TEXT_PROMPT, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[first_image.size[::-1]]
)

input_boxes = results[0]["boxes"].cpu().numpy()
class_names = results[0]["labels"]
OBJECTS = class_names

print("[INFO] GroundingDINO boxes:", input_boxes)
print("[INFO] 객체 이름:", OBJECTS)


"""
===========================================================
 Step 3 — 첫 프레임에서 SAM2 Image Predictor로 Mask 추출
===========================================================
"""

image_predictor.set_image(np.array(first_image.convert("RGB")))

masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

if masks.ndim == 4:
    masks = masks.squeeze(1)


"""
===========================================================
 Step 4 — 첫 프레임 mask를 SAM2 VideoPredictor에 등록
===========================================================
"""

if PROMPT_TYPE_FOR_VIDEO == "point":
    all_sample_points = sample_points_from_masks(masks, 10)
    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        labels = np.ones(points.shape[0], dtype=np.int32)
        video_predictor.add_new_points_or_box(
            inference_state, ann_frame_idx, object_id, points=points, labels=labels
        )

elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        video_predictor.add_new_points_or_box(
            inference_state, ann_frame_idx, object_id, box=box
        )

elif PROMPT_TYPE_FOR_VIDEO in ["mask", "obb"]:
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        video_predictor.add_new_mask(
            inference_state, ann_frame_idx, object_id, mask
        )

else:
    raise NotImplementedError(f"Unsupported PROMPT_TYPE_FOR_VIDEO: {PROMPT_TYPE_FOR_VIDEO}")


"""
===========================================================
 Step 5 — 전체 프레임에 대해 SAM2 tracking (propagate)
===========================================================
"""

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

print(f"[INFO] SAM2 tracking 완료. 결과 frame 수: {len(video_segments)}")


"""
===========================================================
 Step 6 — YOLO HBB/OBB 라벨 생성 + 이미지 저장
===========================================================
"""

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

for frame_idx, segments in video_segments.items():

    # SAM2에서 나온 frame_idx -> 원본 파일명으로 복원
    if frame_idx not in rename_map:
        # 이론상 없어야 하지만, 안전하게 체크
        print(f"[WARN] rename_map에 없는 frame_idx: {frame_idx}, 스킵")
        continue

    orig_filename = rename_map[frame_idx]           # 예: "data_00001.jpg"
    orig_basename = os.path.splitext(orig_filename)[0]  # 예: "data_00001"

    # 원본 이미지 읽기 (input_images에서 읽음)
    img_path = os.path.join(IMAGE_INPUT_DIR, orig_filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 이미지를 읽을 수 없음: {img_path}, 스킵")
        continue

    h, w, _ = img.shape

    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)

    label_lines_hbb = []
    label_lines_obb = []

    for idx, mask in enumerate(masks):
        combined = mask.astype(np.uint8)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)

            # ------ HBB (수평 Bounding Box, YOLO 형식) ------
            x, y, bw, bh = cv2.boundingRect(largest)
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            ww = bw / w
            hh = bh / h
            # class_id는 일단 0으로 고정 (원하면 OBJECTS 기반으로 바꿀 수 있음)
            label_lines_hbb.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

            # ------ OBB (회전 Bounding Box, 4점) ------
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect).astype(np.int32)
            normalized = [(px / w, py / h) for px, py in box]
            flat = [c for pt in normalized for c in pt]
            label_lines_obb.append("0 " + " ".join(f"{pt:.6f}" for pt in flat) + "\n")

    # HBB 라벨 저장: data_00001.txt
    hbb_path = os.path.join(LABEL_DIR_HBB, f"{orig_basename}.txt")
    with open(hbb_path, "w") as f:
        f.writelines(label_lines_hbb)

    # OBB 라벨 저장: data_00001.txt
    obb_path = os.path.join(LABEL_DIR_OBB, f"{orig_basename}.txt")
    with open(obb_path, "w") as f:
        f.writelines(label_lines_obb)

    # 이미지도 동일 이름으로 저장: data_00001.jpg
    out_img_path = os.path.join(IMAGE_DIR, f"{orig_basename}.jpg")
    cv2.imwrite(out_img_path, img)

print("[DONE] 모든 프레임에 대해 이미지 + HBB/OBB 라벨 생성 완료!")
