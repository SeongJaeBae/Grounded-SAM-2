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

IMAGE_INPUT_DIR = "./input_images_2"

TEXT_PROMPT = "packet."
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT_TYPE_FOR_VIDEO = "mask"  # 그대로 유지

CHUNK_SIZE = 250   # ⭐ 추가된 부분 (batch size 개념)

IMAGE_DIR = './YOLOv8/train/images/from_imagefolder_2'
LABEL_DIR_HBB = './YOLOv8/train/labels_hbb/from_imagefolder_2'
LABEL_DIR_OBB = './YOLOv8/train/labels_obb/from_imagefolder_2'

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR_HBB, exist_ok=True)
os.makedirs(LABEL_DIR_OBB, exist_ok=True)

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

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

processor = AutoProcessor.from_pretrained(MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to("cuda")


"""
===========================================================
  Utils
===========================================================
"""

def extract_number(filename: str) -> int:
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1


"""
===========================================================
  Main (CHUNK LOOP)
===========================================================
"""

assert os.path.exists(IMAGE_INPUT_DIR)

orig_frame_names = [
    p for p in os.listdir(IMAGE_INPUT_DIR)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
]

# orig_frame_names.sort(key=lambda p: extract_number)
orig_frame_names.sort(key=lambda p: extract_number(p))

num_frames = len(orig_frame_names)
print(f"[INFO] Total frames: {num_frames}")

for chunk_start in range(0, num_frames, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, num_frames)
    print(f"\n[INFO] Processing chunk: {chunk_start} ~ {chunk_end - 1}")

    # ----------------------------
    # TEMP_FRAME_DIR 재생성
    # ----------------------------
    if os.path.exists(TEMP_FRAME_DIR):
        shutil.rmtree(TEMP_FRAME_DIR)
    os.makedirs(TEMP_FRAME_DIR, exist_ok=True)

    rename_map = {}

    for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
        fname = orig_frame_names[global_idx]
        src = os.path.join(IMAGE_INPUT_DIR, fname)
        dst = os.path.join(TEMP_FRAME_DIR, f"{local_idx:05d}.jpg")
        shutil.copy2(src, dst)
        rename_map[local_idx] = fname

    # ----------------------------
    # SAM2 VideoPredictor init
    # ----------------------------
    inference_state = video_predictor.init_state(video_path=TEMP_FRAME_DIR)

    """
    Step 2 — GroundingDINO (첫 프레임)
    """
    ann_frame_idx = 0
    first_frame_path = os.path.join(TEMP_FRAME_DIR, "00000.jpg")
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
    OBJECTS = results[0]["labels"]

    """
    Step 3 — SAM2 Image Predictor (mask)
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
    Step 4 — mask 등록
    """
    for object_id, mask in enumerate(masks, start=1):
        video_predictor.add_new_mask(
            inference_state, ann_frame_idx, object_id, mask
        )

    """
    Step 5 — Tracking
    """
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in \
            video_predictor.propagate_in_video(inference_state):

        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Step 6 — YOLO HBB / OBB 생성
    """
    for frame_idx, segments in video_segments.items():
        if frame_idx not in rename_map:
            continue

        orig_filename = rename_map[frame_idx]
        orig_basename = os.path.splitext(orig_filename)[0]

        img_path = os.path.join(IMAGE_INPUT_DIR, orig_filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape

        masks = np.concatenate(list(segments.values()), axis=0)

        label_lines_hbb = []
        label_lines_obb = []

        for mask in masks:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)

            # HBB
            x, y, bw, bh = cv2.boundingRect(largest)
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            ww = bw / w
            hh = bh / h
            label_lines_hbb.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

            # OBB
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            flat = []
            for px, py in box:
                flat.extend([px / w, py / h])
            label_lines_obb.append(
                "0 " + " ".join(f"{v:.6f}" for v in flat) + "\n"
            )

        with open(os.path.join(LABEL_DIR_HBB, f"{orig_basename}.txt"), "w") as f:
            f.writelines(label_lines_hbb)

        with open(os.path.join(LABEL_DIR_OBB, f"{orig_basename}.txt"), "w") as f:
            f.writelines(label_lines_obb)

        cv2.imwrite(os.path.join(IMAGE_DIR, f"{orig_basename}.jpg"), img)

    # ----------------------------
    # 메모리 정리 (핵심)
    # ----------------------------
    del inference_state
    torch.cuda.empty_cache()

print("[DONE] All chunks processed successfully.")
