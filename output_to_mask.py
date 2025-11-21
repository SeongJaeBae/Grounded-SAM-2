from pycocotools import mask as mask_utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 예시 output
output = {
    "image_path": "mealkit3.jpg",
    "annotations": [
        {
            "class_name": "packet",
            "bbox": [
                734.0021362304688,
                345.00823974609375,
                959.8800048828125,
                628.2254028320312
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "bk\\h09]Q14L4N3L5L3L4M3M2M3N4L4L3L5L3M4K4M3M3L4M3M2M4M3M3L3N3L4M3M2M5L3M3L4M3M3M2M4M3L3N5K2N3L4M2N3M4K4M3L5L3M4K4M3M4K4M2N2N3L3N2N3L6K4L4K4M2N2M4M2N2N2M3N2N0O2O0O101O0O101OO10O010O01000O010O10O10O10O01000O0100O010O10O100O010O10O1000O010O010O10O01000O010O1000O0100000O010O1000O0100000O010O10O10O0100O0100000O101N1PNVVORLji0l3]VOoKdi0n3aVOnK`i0P4dVOnK]i0o3hVOmKZi0R4jVOjKVi0U4oVOgKRi0W4RWOeKPi0Y4RWOfKoh0Y4RWOeKQi0X4QWOgKQi0W4QWOgKUi0R4oVOjKWi0Q4kVOlKYi0Q4hVOnK[i0n3iVOnKZi0P4iVOlKZi0R4iVOjK]i0P4W2O2N4K4M1N4L3N2N2M4M2M3N2M3M4M2M3N4K3M3N2M3N1N5L3L4M2M3N2M3N2N3L3M3N1N3N2M3N2M4L4M3L3M4L4K3KQVgo0"
            },
            "score": [
                0.984375
            ]
        },
        {
            "class_name": "packet",
            "bbox": [
                768.7409057617188,
                141.86239624023438,
                1042.2877197265625,
                423.4853210449219
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "U^^i06`Q14M3M4L3M2O0O2O2M5L2M2O0O3N1N2N4M3L3M3N4K3M3M2O2M3N1N3N1N4L3N2M2N3N1N2N2O2M3N2M3N1O1N2N2O2M3N1N2O1N2N2O2M3N1N2O2M2O1N2O2M2N3N1N3N2M2O2M2O1N101N2O1N2O0O2O1N2O1O0O1O100O1O10000O100O100O10O0100O10000O10000O01O0010O010O1000O0100O100O10O02O0O100O101N101N101O1N2O1N2O1O3L5L4K5L3L3N5J4M3L3N2M4M2M4M3L3N1N4M2M3N2N1N3N1O0O2O001N100000O2N101N2O002M3N3L4M3L<E8G4M2M3N1N3N1N4M2M6KO1O001000O100O3N3L4M2M6K2M3N1O3L5K3N2M4M3M1N4M2M3M3N2N1N3N1O2M3N2M3N1N2O1N3N2M2O2M3N1N3N2M2O2M3N2M3M2O2M2O2N2M3N2N2M3N1N3N1N3N2M3M3N2M3N3L4M2M3N2N1N3N3L4L3N2M3M4M2M4L4L4K\\eml0"
            },
            "score": [
                0.98046875
            ]
        },
        {
            "class_name": "packet",
            "bbox": [
                996.8162231445312,
                40.50284194946289,
                1290.335205078125,
                342.40826416015625
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "\\hmP12aQ1;I3M3M2N3M3M3M3M2N3M3M3N3L2N3M3M3M3M2O2M3M2N3M3M3M2N3M3M3M3N2M3M2N3N2M3M3M4M2M3M2O2M3M3M4M2M3M2N3N3L3N2M2N3N2M2O2M4L3N2M4L3M3N2M3M4M2M3M3N1N4M2M3M3M3M3N3L2O3L2N2O1N2O1N2N3N1N3M4M3L3N1N2O2M3M3N2M3N3L3N1N3N2N2M3N2N1N1000001O000O1000O1000000O10O10O1O00100O1O010O1O010O001O001O100O1O1O1O1O1O1O2O1N3M2O1N2N3M3M2N4L3M2N2O1N2N3N1N2N3N2M2O1N3N2M4L2O2M2N2O3L3M3M3M3M2N2N2N2N3M3M3M2N2N1O3N1N3M2N3M2O1N2N3N2M2O1N101O1N2O2M3M2O1N2N2N3N2M3N1N3N2M3N1N3N2M2O2N2M2O1N3N2N1N3N2N2M2O1O2M4M2N2N2N2N2M3N2N2N1O2N2M3N1O1O0O101O00001N10000000O10O10000O1000000O10000O10000O100O1O1O1M3L4M4M2O1O100O4J8^OmYid0"
            },
            "score": [
                0.984375
            ]
        },
        {
            "class_name": "packet",
            "bbox": [
                890.4722900390625,
                426.75909423828125,
                1178.0677490234375,
                683.5663452148438
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "\\]_m02_Q1;K4M2M4M3L4L4L3M3M2O3L3N2N2M3N3M2M4M3M2M3M2N3N1N3N2N2M3N4K3M4M2N2M3N2N2M3M3N2M3N2N3L3M3N2M4L3N2M3N1N4M2M3M2N3N3M3L3N2M3M3N1N2O1N2O000O2O0O100O01000O10O01000O010O010O0100O001O01O010O01000O0100O00100O10O10O010O1O01000O01O01O010O10O0100O010O00100O010O10O01O100O01000O010O10O0100O010O010O00100O010O10O01O10O01000O010O10O010O1O010O0010O01O10O0100O010O010O001000O0100O0100O010O010O0100O010O10O0100O10O1O010O10O10O100O101N10001O0O3N2M2O3M2M3N2M3N3M2M3N2M4M2N2M3N2M3N3L4M2M4M2N2M3N3M3L3N2M3N2N2M2O2N3L3N1O2M3N2M3N3M2M4M2N2M3N3L4M2M3N1O2M3M3N2M3N2M3N2N2M3N4L3L3N1N3N3L3M4K5KWP^h0"
            },
            "score": [
                0.98828125
            ]
        },
        {
            "class_name": "packet",
            "bbox": [
                1129.4149169921875,
                253.03665161132812,
                1308.1737060546875,
                448.1913757324219
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "QYZU1b0dP1k0iNT1H9G9G6K5J5L3M2N1O2N1O100O1O1O1O100O1N2O100O1O00100O1O100O1O1O00100O10O0100O100O001O1N2O100O001O1O1O1N10100O1O100O00100O1O100O100O01000O100O1O1O00100O1O1O10O0100O100O010O10000O100O1000000O01000000000O1000O01000000O100000O10O10000O1000O10O10000000O1000O010000000O01000O100O10O10000O101O001N5L6PKXSO1=o3cm0B7I5K3M5K3M3L3N3M2N5J4M4L4L4K4M4L3L4M3M3M5J4L5K8FUVWd0"
            },
            "score": [
                0.98046875
            ]
        }
    ],
    "box_format": "xyxy",
    "img_width": 1920,
    "img_height": 1080
}

save_dir = "./outputs"
os.makedirs(save_dir, exist_ok=True)

# 원본 이미지
image_all = cv2.imread(output["image_path"])
image_all = cv2.cvtColor(image_all, cv2.COLOR_BGR2RGB)

for i, ann in enumerate(output["annotations"]):
    rle = ann["segmentation"]
    decoded_mask = mask_utils.decode(rle).astype(np.uint8)

    image = image_all.copy()
    mask_color = np.zeros_like(image)
    mask_color[:, :, 0] = 255
    mask_area = decoded_mask.astype(bool)
    image[mask_area] = cv2.addWeighted(image, 0.5, mask_color, 0.5, 0)[mask_area]
    image_all[mask_area] = cv2.addWeighted(image_all, 0.5, mask_color, 0.5, 0)[mask_area]

    contours, _ = cv2.findContours(decoded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect).astype(np.intp)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        cv2.drawContours(image_all, [box], 0, (0, 255, 0), 2)

        # 중심점 1: 무게중심 (moments) => 노란색
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(image_all, (cx, cy), 4, (255, 255, 0), -1)

        # 중심점 2: 사각형 중심 (minAreaRect) => 하늘색
        center = tuple(map(int, rect[0]))  # rect[0] = (cx, cy)
        cv2.circle(image, center, 4, (0, 255, 255), -1)
        cv2.circle(image_all, center, 4, (0, 255, 255), -1)

        # 클래스명 표시
        class_name = ann.get("class_name", "unknown")
        cv2.putText(image, class_name, (box[0][0], box[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(image_all, class_name, (box[0][0], box[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # 개별 저장
    save_path = os.path.join(save_dir, f"segmented_result_{i}_with_rect.png")
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved individual: {save_path}")

# 전체 결과 저장
merged_path = os.path.join(save_dir, "merged_result_all.png")
cv2.imwrite(merged_path, cv2.cvtColor(image_all, cv2.COLOR_RGB2BGR))
print(f"Saved merged: {merged_path}")