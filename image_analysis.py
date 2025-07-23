# -*- coding: utf-8 -*-
# 두 이미지의 차이점 분석
import cv2
import numpy as np

def analyze_image_properties(image_path):
    """이미지의 속성을 분석하는 함수"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return
    
    h, w, c = image.shape
    
    # 이미지 기본 정보
    print(f"\n=== {image_path.split('\\')[-1]} ===")
    print(f"해상도: {w} x {h}")
    print(f"채널 수: {c}")
    print(f"파일 크기: {w * h * c} bytes")
    
    # 밝기 분석
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    print(f"평균 밝기: {mean_brightness:.2f}")
    print(f"밝기 표준편차: {std_brightness:.2f}")
    
    # 대비 분석
    contrast = std_brightness / mean_brightness if mean_brightness > 0 else 0
    print(f"대비도: {contrast:.3f}")
    
    # 노이즈 분석
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_level = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
    print(f"노이즈 레벨: {noise_level:.2f}")
    
    return {
        'resolution': (w, h),
        'brightness': mean_brightness,
        'contrast': contrast,
        'noise': noise_level
    }

# 두 이미지 비교
image1 = r"C:\Users\user\Desktop\Github\cosrec\KakaoTalk_20250706_161302093.jpg"  # 감지 실패
image2 = r"C:\Users\user\Desktop\Github\cosrec\KakaoTalk_20250715_141326156.jpg"  # 감지 성공

props1 = analyze_image_properties(image1)
props2 = analyze_image_properties(image2)

print("\n=== 비교 결과 ===")
if props1 and props2:
    print(f"해상도 차이: {props1['resolution']} vs {props2['resolution']}")
    print(f"밝기 차이: {props1['brightness']:.2f} vs {props2['brightness']:.2f}")
    print(f"대비도 차이: {props1['contrast']:.3f} vs {props2['contrast']:.3f}")
    print(f"노이즈 차이: {props1['noise']:.2f} vs {props2['noise']:.2f}")
