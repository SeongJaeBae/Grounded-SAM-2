from pycocotools import mask as mask_utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 예시 output
output = {
    "image_path": "notebooks/images/mealkit.jpg",
    "annotations": [
        {
            "class_name": "packet",
            "bbox": [
                737.4977416992188,
                187.5142822265625,
                956.485595703125,
                455.14501953125
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "ok\\h07TQ1KQoN>dP1a0E:I7G8J6K6J5G9J6H8J7F9J6G9F:G:J5I7H8H8G:I6F:G9K6J5K5L4J6N3M200O101N100001O01O0000001O0001O0001O0000010O0000010O000000010O00000000010O00000000010O0000000010O0000000010O00000001O0001O000001O0001O00000000010O0000000010O000000001O01O000001O000000010O000000001O00000010O000000000001O00000001O01O00000000001O00000001O0001O000000001N1000001N1O1O1N3L3J6M3G9K5N3L3I7M3L4K5L4M3M4J5I7L4J6K5M3M4I6K5K5L4J6K5L5M2G9K5K5K6K4I7L4I7H9I6I8I8E:JT`ho0"
            },
            "score": [
                0.98828125
            ]
        },
        {
            "class_name": "packet",
            "bbox": [
                1108.88818359375,
                512.9212036132812,
                1304.175048828125,
                765.4379272460938
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "h^dT15Xo0e2\\Na1WOc0_Ob0^Ob0@a0_O`0J5M3M200O2O000O100000000000000000000000000000000000000000000001O000000000000000000000000000001O0001O0000000000000000000000000001O000001O00000000000000000001O00000000000001O000000000001O000000000000000000000001O01O00000000000000001O000000000001O000001O000000000000000000001O000001O0000000000000001O000000000001O00000001O00O101O0O10000O1M3XOh0G9\\OdTO_Jik0Q5VTOnJ\\l0m4;G8M3E;G9G:C<YOh0C=^Ob0F;Af0VOeUYd0"
            },
            "score": [
                0.98828125
            ]
        },
        {
            "class_name": "packet",
            "bbox": [
                671.43408203125,
                523.55224609375,
                874.4692993164062,
                782.8746948242188
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "TcWf03TQ1e0[Oc0I6G9F9D<F;C<B>F:C=B>E;F;@?K5G9E;F:I7N3L3N2O1O1O1O2O00000000000O20O000000000001O01O000000010O000000001O01O000000000001O01O00000001O0000000001O01O0000000000001O0001O000000000001O000000010O000000001O0000001O000000001O00001O0000001O000000001O01O00000001O000000000000001O01O00000000000000001O0000000000010O000000O1000O1000O100O1M3K5J6B>M2L5L4H8J5H9K5I7K4N3K5J6K5J6L4J6L4H8I7H8J6I7I7L4L4F:K6H7O1M4F:H9BSd]R1"
            },
            "score": [
                0.98828125
            ]
        },
        {
            "class_name": "packet",
            "bbox": [
                1112.6158447265625,
                177.01719665527344,
                1331.5848388671875,
                438.1352844238281
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "fWfT1:XQ1;G7I6L3N2M2O2N1N3N1O1O2M2O1O2N1nN\\NnQOf1em0^1F9H8G9H9G8J6I8J5L4I8K4G9L4J7I6L4L5L3M4M2O1N3M2O1O2O0O1010O000001O0001O01O0001O01O0000010O000000010O000010O0000010O00000010O0001O000010O0001O01O0001O0010O0001O000010O0001O001O01O0001O0000000010O0000010O00000001O000001O0001O000000001O01O00000001O000000001O0001O01O0000001O000000000010O000000000001O00000000001O01O0000000001O00000O100O2M2O1M3K5^Ob0H8G:K4I7D<E;H8E<H7H8L4H8H8M4K4L4N2M4L3M3M3K6L3M4J6L4G9EUP]c0"
            },
            "score": [
                0.98828125
            ]
        },
        {
            "class_name": "packet",
            "bbox": [
                887.3059692382812,
                529.2437744140625,
                1107.89697265625,
                778.9519653320312
            ],
            "segmentation": {
                "size": [
                    1080,
                    1920
                ],
                "counts": "fXYm02aQ17J6J5L4K4M4J5N3N1O2M2N3N1N2N3O0O1O2M2O2N1O1O2N100O1O2N1O101N1O101N1N2ZOg0C<K5L4`M[LeTONm1l3Xi0eLaVO`3[i0eL]VOa3ai0cLZVO`3fi0aLSVOf3ki0]LPVOf3Pj0[LlUOh3Tj0YLgUOl3Xj0VLcUOm3]j0TL`UOn3`j0SL[UOR4dj0a10000010O0000010O01O00010O00001O001O00001O1O00001O000O101O00001O001O00001O0O101O001O001O00001O0000001O00001O0010O0001O0001O01O01O01O0010O01O0001O01O00010O0001O01O000001O00000O100O1O1O1N2N2N2N2M3L5M2N2L4L4L4H9ROYTOYK0Lnk0g4j0O1O1M4L3N2O1O1N3N1O1N2N2O2M2N2N2N2N3M2O1N2O2N1O1O1N3M2M3N3M2N2N2M4L3N3M2N2O2M2M3M4L3N2M4L3N3M2N2O2M2O1N3M2O1O2L4L3L5L4L3L5M4K^kgj0"
            },
            "score": [
                0.98828125
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