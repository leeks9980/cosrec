# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np

# 1. 얼굴 추출 및 피부 마스크 생성 함수
def extract_face_with_skin_mask(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지를 불러올 수 없습니다: {image_path}")
            return None, None
        
        # 이미지 품질 개선 (대비도 낮고 노이즈 많은 이미지 대응)
        def enhance_image(img):
            # 1. 노이즈 제거
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            # 2. 대비도 개선 (CLAHE)
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        
        # MediaPipe 초기화
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        # 얼굴 감지 먼저 수행
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3) as face_detection:
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 원본에서 감지 실패 시 향상된 이미지로 재시도
            if not results.detections:
                print("🔄 원본 이미지에서 얼굴 감지 실패. 이미지 품질 개선 후 재시도...")
                enhanced_image = enhance_image(image)
                results = face_detection.process(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
                
                if not results.detections:
                    print("🔄 신뢰도를 낮춰서 재시도...")
                    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1) as face_detection_low:
                        results = face_detection_low.process(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
                        if results.detections:
                            print("✅ 낮은 신뢰도에서 얼굴 감지 성공!")
                            image = enhanced_image
                        else:
                            print("❌ 모든 시도 후에도 얼굴이 감지되지 않았습니다.")
                            return None, None
                else:
                    print("✅ 품질 개선된 이미지에서 얼굴 감지 성공!")
                    image = enhanced_image
            
            if not results.detections:
                return None, None

            print(f"✅ {len(results.detections)}개의 얼굴을 감지했습니다.")
            
            # 가장 큰 얼굴 선택
            largest_detection = max(results.detections, 
                                  key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            
            bboxC = largest_detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            
            # 바운딩 박스 계산
            margin = 0.1
            x1 = max(0, int((bboxC.xmin - margin * bboxC.width) * w))
            y1 = max(0, int((bboxC.ymin - margin * bboxC.height) * h))
            x2 = min(w, int((bboxC.xmin + bboxC.width + margin * bboxC.width) * w))
            y2 = min(h, int((bboxC.ymin + bboxC.height + margin * bboxC.height) * h))
            
            face_img = image[y1:y2, x1:x2]
            
            # 얼굴 랜드마크로 피부 마스크 생성
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                                     refine_landmarks=True, min_detection_confidence=0.3) as face_mesh:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(face_rgb)
                
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0]
                    
                    # MediaPipe Face Mesh의 얼굴 윤곽선 (외곽선)
                    FACE_OVAL = [
                        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
                    ]
                    
                    # 제외할 영역 (눈동자, 눈썹, 입술만 정확히)
                    # 왼쪽 눈 (눈동자와 속눈썹)
                    LEFT_EYE_EXCLUDE = [
                        # 눈꺼풀 안쪽
                        159, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
                    ]
                    
                    # 오른쪽 눈 (눈동자와 속눈썹)
                    RIGHT_EYE_EXCLUDE = [
                        # 눈꺼풀 안쪽
                        385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382
                    ]
                    
                    # 눈썹 영역 (정확한 눈썹만)
                    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                    RIGHT_EYEBROW = [296, 334, 293, 300, 276, 283, 282, 295, 285, 336]
                    
                    # 입술 영역 (정확한 입술 경계만, 최소한으로)
                    LIPS = [
                        # 입술 윤곽선만 (최소한의 영역)
                        61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
                        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
                        # 입술 내부
                        13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
                    ]
                    
                    # 피부 영역 마스크 생성 (전체 얼굴에서 시작)
                    mask = np.zeros(face_img.shape[:2], dtype=np.uint8)
                    face_h, face_w = face_img.shape[:2]
                    
                    # 1. 전체 얼굴 윤곽선으로 기본 마스크 생성
                    face_points = []
                    for idx in FACE_OVAL:
                        if idx < len(landmarks.landmark):
                            x = int(landmarks.landmark[idx].x * face_w)
                            y = int(landmarks.landmark[idx].y * face_h)
                            face_points.append([x, y])
                    
                    if len(face_points) > 0:
                        face_points = np.array(face_points)
                        cv2.fillPoly(mask, [face_points], 255)
                        
                        # 2. 제외할 영역들을 작은 크기로 정밀하게 마스킹
                        exclude_regions = [
                            ("왼쪽 눈", LEFT_EYE_EXCLUDE),
                            ("오른쪽 눈", RIGHT_EYE_EXCLUDE),
                            ("왼쪽 눈썹", LEFT_EYEBROW),
                            ("오른쪽 눈썹", RIGHT_EYEBROW),
                            ("입술", LIPS)
                        ]
                        
                        for region_name, indices in exclude_regions:
                            exclude_points = []
                            for idx in indices:
                                if idx < len(landmarks.landmark):
                                    x = int(landmarks.landmark[idx].x * face_w)
                                    y = int(landmarks.landmark[idx].y * face_h)
                                    exclude_points.append([x, y])
                            
                            if len(exclude_points) > 2:
                                exclude_points = np.array(exclude_points)
                                
                                # 입술은 더 정밀하게 처리
                                if region_name == "입술":
                                    # 입술 영역을 더 작게 만들기
                                    hull = cv2.convexHull(exclude_points)
                                    # 입술 마스크를 약간 축소
                                    kernel_small = np.ones((3,3), np.uint8)
                                    temp_mask = np.zeros(face_img.shape[:2], dtype=np.uint8)
                                    cv2.fillPoly(temp_mask, [hull], 255)
                                    temp_mask = cv2.erode(temp_mask, kernel_small, iterations=1)
                                    mask = cv2.bitwise_and(mask, cv2.bitwise_not(temp_mask))
                                else:
                                    # 다른 영역은 기본 처리
                                    hull = cv2.convexHull(exclude_points)
                                    cv2.fillPoly(mask, [hull], 0)  # 검은색으로 제외
                        
                        # 3. 마스크 후처리 (부드럽게)
                        kernel = np.ones((3,3), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        
                        # 가장자리 부드럽게 처리
                        mask = cv2.GaussianBlur(mask, (3,3), 0)
                        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                        
                        print("✅ 정밀한 피부 영역 마스크 생성 완료 (코 포함)")
                        return face_img, mask
                    else:
                        print("⚠️ 얼굴 윤곽선 포인트 부족")
                        return face_img, None
                else:
                    print("⚠️ 얼굴 랜드마크 감지 실패. 전체 얼굴 영역 사용")
                    return face_img, None
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None, None

# 이전 extract_face 함수를 호환성을 위해 유지
def extract_face(image_path):
    face_img, _ = extract_face_with_skin_mask(image_path)
    return face_img

# 2. 주름 탐지 함수 (피부 마스크 적용)
def wrinkle_detector(face_img, skin_mask=None):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)  # 대비 증가
    blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # 피부 마스크가 있으면 적용
    if skin_mask is not None:
        edges = cv2.bitwise_and(edges, skin_mask)
        print("✅ 피부 영역에만 주름 탐지 적용")
    
    # 마스크 영역 기준으로 비율 계산
    if skin_mask is not None:
        mask_area = (skin_mask > 0).sum()
        if mask_area > 0:
            wrinkle_ratio = (edges > 0).sum() / mask_area
        else:
            wrinkle_ratio = 0
    else:
        wrinkle_ratio = (edges > 0).sum() / edges.size
    
    return wrinkle_ratio, edges

# 3. 모공 탐지 함수 (피부 마스크 적용)
def pore_detector(face_img, skin_mask=None):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    high_pass = cv2.subtract(gray, cv2.GaussianBlur(gray, (9, 9), 0))
    _, binarized = cv2.threshold(high_pass, 15, 255, cv2.THRESH_BINARY)
    
    # 피부 마스크가 있으면 적용
    if skin_mask is not None:
        binarized = cv2.bitwise_and(binarized, skin_mask)
        print("✅ 피부 영역에만 모공 탐지 적용")
    
    # 마스크 영역 기준으로 비율 계산
    if skin_mask is not None:
        mask_area = (skin_mask > 0).sum()
        if mask_area > 0:
            pore_ratio = (binarized > 0).sum() / mask_area
        else:
            pore_ratio = 0
    else:
        pore_ratio = (binarized > 0).sum() / binarized.size
    
    return pore_ratio, binarized

# 4. 실행 함수 (피부 마스크 적용 + 이미지 저장)
def analyze_skin(image_path, save_results=True, output_dir="processed_images"):
    face_img, skin_mask = extract_face_with_skin_mask(image_path)

    if face_img is None:
        return None

    wrinkle_score, wrinkle_img = wrinkle_detector(face_img, skin_mask)
    pore_score, pore_img = pore_detector(face_img, skin_mask)

    print(f"\n🔬 피부 분석 결과:")
    print(f"🧵 주름 점수 (0~1): {wrinkle_score:.3f}")
    print(f"🕳️ 모공 점수 (0~1): {pore_score:.3f}")
    
    # 점수 해석
    def interpret_score(score, score_type):
        if score_type == "wrinkle":
            if score < 0.01:
                return "매우 좋음 (주름이 거의 없음)"
            elif score < 0.02:
                return "좋음 (약간의 주름)"
            elif score < 0.04:
                return "보통 (중간 정도의 주름)"
            else:
                return "주의 (주름이 많음)"
        else:  # pore
            if score < 0.005:
                return "매우 좋음 (모공이 거의 없음)"
            elif score < 0.01:
                return "좋음 (약간의 모공)"
            elif score < 0.02:
                return "보통 (중간 정도의 모공)"
            else:
                return "주의 (모공이 많음)"
    
    print(f"📊 주름 상태: {interpret_score(wrinkle_score, 'wrinkle')}")
    print(f"📊 모공 상태: {interpret_score(pore_score, 'pore')}")

    # 결과 저장
    saved_files = {}
    if save_results:
        saved_files = save_analysis_results(
            face_img, skin_mask, wrinkle_img, pore_img, 
            wrinkle_score, pore_score, output_dir
        )
    
    # 결과 시각화
    cv2.imshow("원본 얼굴", face_img)
    cv2.imshow("주름 탐지 (Canny)", wrinkle_img)
    cv2.imshow("모공 탐지 (High-pass)", pore_img)
    
    # 피부 마스크가 있으면 표시
    if skin_mask is not None:
        cv2.imshow("피부 영역 마스크", skin_mask)
        
        # 마스크 적용된 원본 이미지 표시
        masked_face = cv2.bitwise_and(face_img, face_img, mask=skin_mask)
        cv2.imshow("피부 영역만 표시", masked_face)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return saved_files

# 이미지 저장 함수 추가
def save_analysis_results(face_img, skin_mask, wrinkle_img, pore_img, wrinkle_score, pore_score, output_dir="processed_images"):
    """
    피부 분석 결과를 파일로 저장하는 함수
    멀티모달 모델에 입력으로 사용할 수 있도록 전처리된 이미지들을 저장
    """
    import os
    from datetime import datetime
    
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 출력 디렉토리 생성: {output_dir}")
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    try:
        # 1. 원본 얼굴 이미지 저장
        face_path = os.path.join(output_dir, f"face_{timestamp}.jpg")
        cv2.imwrite(face_path, face_img)
        saved_files['face'] = face_path
        print(f"💾 원본 얼굴 저장: {face_path}")
        
        # 2. 피부 영역만 추출한 이미지 저장 (가장 중요!)
        if skin_mask is not None:
            # 피부 영역만 표시한 이미지 (멀티모달 모델용)
            masked_face = cv2.bitwise_and(face_img, face_img, mask=skin_mask)
            skin_only_path = os.path.join(output_dir, f"skin_only_{timestamp}.jpg")
            cv2.imwrite(skin_only_path, masked_face)
            saved_files['skin_only'] = skin_only_path
            print(f"🎯 피부 영역만 저장: {skin_only_path}")
            
            # 피부 마스크 자체도 저장
            mask_path = os.path.join(output_dir, f"skin_mask_{timestamp}.jpg")
            cv2.imwrite(mask_path, skin_mask)
            saved_files['skin_mask'] = mask_path
            print(f"🎭 피부 마스크 저장: {mask_path}")
            
            # 깔끔한 피부 이미지 생성 (배경 제거)
            clean_skin_path = os.path.join(output_dir, f"clean_skin_{timestamp}.jpg")
            # 배경을 흰색으로 만든 깔끔한 이미지
            clean_skin = face_img.copy()
            clean_skin[skin_mask == 0] = [255, 255, 255]  # 비피부 영역을 흰색으로
            cv2.imwrite(clean_skin_path, clean_skin)
            saved_files['clean_skin'] = clean_skin_path
            print(f"✨ 깔끔한 피부 이미지 저장: {clean_skin_path}")
        
        # 3. 주름 분석 결과 저장
        wrinkle_path = os.path.join(output_dir, f"wrinkle_analysis_{timestamp}.jpg")
        cv2.imwrite(wrinkle_path, wrinkle_img)
        saved_files['wrinkle'] = wrinkle_path
        print(f"🧵 주름 분석 저장: {wrinkle_path}")
        
        # 4. 모공 분석 결과 저장
        pore_path = os.path.join(output_dir, f"pore_analysis_{timestamp}.jpg")
        cv2.imwrite(pore_path, pore_img)
        saved_files['pore'] = pore_path
        print(f"🕳️ 모공 분석 저장: {pore_path}")
        
        # 5. 분석 결과 텍스트 파일 저장
        report_path = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"🔬 피부 분석 결과 리포트\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"📊 정량적 분석 결과:\n")
            f.write(f"🧵 주름 점수: {wrinkle_score:.3f}\n")
            f.write(f"🕳️ 모공 점수: {pore_score:.3f}\n\n")
            
            # 점수 해석
            def interpret_score(score, score_type):
                if score_type == "wrinkle":
                    if score < 0.01:
                        return "매우 좋음 (주름이 거의 없음)"
                    elif score < 0.02:
                        return "좋음 (약간의 주름)"
                    elif score < 0.04:
                        return "보통 (중간 정도의 주름)"
                    else:
                        return "주의 (주름이 많음)"
                else:  # pore
                    if score < 0.005:
                        return "매우 좋음 (모공이 거의 없음)"
                    elif score < 0.01:
                        return "좋음 (약간의 모공)"
                    elif score < 0.02:
                        return "보통 (중간 정도의 모공)"
                    else:
                        return "주의 (모공이 많음)"
            
            f.write(f"📈 상태 평가:\n")
            f.write(f"주름 상태: {interpret_score(wrinkle_score, 'wrinkle')}\n")
            f.write(f"모공 상태: {interpret_score(pore_score, 'pore')}\n\n")
            
            f.write(f"📁 저장된 파일들:\n")
            for key, path in saved_files.items():
                f.write(f"- {key}: {path}\n")
            
            f.write(f"\n💡 멀티모달 모델 추천 입력 파일:\n")
            if 'clean_skin' in saved_files:
                f.write(f"- 주요 이미지: {saved_files['clean_skin']}\n")
            elif 'skin_only' in saved_files:
                f.write(f"- 주요 이미지: {saved_files['skin_only']}\n")
            f.write(f"- 분석 데이터: 주름 점수 {wrinkle_score:.3f}, 모공 점수 {pore_score:.3f}\n")
        
        saved_files['report'] = report_path
        print(f"📋 분석 리포트 저장: {report_path}")
        
        print(f"\n✅ 모든 분석 결과가 '{output_dir}' 폴더에 저장되었습니다!")
        print(f"🎯 멀티모달 모델용 주요 이미지: {saved_files.get('clean_skin', saved_files.get('skin_only', '없음'))}")
        
        return saved_files
        
    except Exception as e:
        print(f"❌ 저장 중 오류 발생: {e}")
        return saved_files

# 멀티모달 모델용 데이터 준비 함수
def prepare_multimodal_data(image_path, user_description="", save_dir="multimodal_input"):
    """
    멀티모달 모델에 입력할 데이터를 준비하는 함수
    """
    print("🚀 멀티모달 모델용 데이터 준비 시작...")
    
    # 사용자 설명을 직접 딕셔너리로 파싱
    user_info = parse_user_description(user_description)
    
    # 피부 분석 실행
    face_img, skin_mask = extract_face_with_skin_mask(image_path)
    if face_img is None:
        print("❌ 얼굴 감지 실패")
        return None
    
    wrinkle_score, wrinkle_img = wrinkle_detector(face_img, skin_mask)
    pore_score, pore_img = pore_detector(face_img, skin_mask)
    
    # 결과 저장
    saved_files = save_analysis_results(
        face_img, skin_mask, wrinkle_img, pore_img, 
        wrinkle_score, pore_score, save_dir
    )
    
    # 멀티모달 모델용 데이터 구성
    multimodal_data = {
        'image_path': saved_files.get('clean_skin', saved_files.get('skin_only')),
        'skin_analysis': {
            'wrinkle_score': wrinkle_score,
            'pore_score': pore_score,
            'wrinkle_status': interpret_score_simple(wrinkle_score, 'wrinkle'),
            'pore_status': interpret_score_simple(pore_score, 'pore')
        },
        'user_info': user_info,
        'analysis_prompt': generate_analysis_prompt_structured(wrinkle_score, pore_score, user_info)
    }
    
    # 멀티모달 입력 데이터 JSON 저장
    import json
    from datetime import datetime
    import os
    
    # 저장 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(save_dir, f"multimodal_input_{timestamp}.json")
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(multimodal_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 멀티모달 입력 데이터 저장: {json_path}")
        
        # 저장된 파일 확인
        with open(json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            print(f"✅ JSON 파일 저장 및 한글 인코딩 검증 완료")
            
    except Exception as e:
        print(f"❌ JSON 저장 중 오류: {e}")
        return None
    
    return multimodal_data

def parse_user_description(user_description):
    """사용자 설명을 구조화된 딕셔너리로 파싱"""
    user_info = {
        "age": "",
        "gender": "",
        "skin_type": "",
        "concerns": "",
        "additional_info": ""
    }
    
    lines = user_description.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("나이:"):
            user_info["age"] = line.replace("나이:", "").strip()
        elif line.startswith("성별:"):
            user_info["gender"] = line.replace("성별:", "").strip()
        elif line.startswith("피부 타입:"):
            user_info["skin_type"] = line.replace("피부 타입:", "").strip()
        elif line.startswith("주요 고민:"):
            user_info["concerns"] = line.replace("주요 고민:", "").strip()
        elif line and not any(line.startswith(prefix) for prefix in ["나이:", "성별:", "피부 타입:", "주요 고민:"]):
            user_info["additional_info"] += line + " "
    
    return user_info

def generate_analysis_prompt_structured(wrinkle_score, pore_score, user_info):
    """구조화된 멀티모달 모델용 분석 프롬프트 생성"""
    prompt = f"""다음 피부 이미지를 분석하고 개인 맞춤형 코스메틱 제품을 추천해주세요.

📊 컴퓨터 비전 분석 결과:
- 주름 지수: {wrinkle_score:.3f} ({interpret_score_simple(wrinkle_score, 'wrinkle')})
- 모공 지수: {pore_score:.3f} ({interpret_score_simple(pore_score, 'pore')})

👤 사용자 정보:
- 나이: {user_info.get('age', '미제공')}
- 성별: {user_info.get('gender', '미제공')}
- 피부 타입: {user_info.get('skin_type', '미제공')}
- 주요 고민: {user_info.get('concerns', '미제공')}

🎯 분석 요청사항:
1. 이미지에서 관찰되는 피부 특징 분석
2. 주름과 모공 상태에 대한 전문적 평가
3. 피부 타입 및 상태 진단
4. 개인 맞춤형 스킨케어 루틴 제안
5. 구체적인 제품 추천 (성분, 브랜드 포함)
6. 단계별 케어 방법 제시

전문가 관점에서 상세하고 실용적인 조언을 해주세요."""
    
    return prompt

def interpret_score_simple(score, score_type):
    """점수 해석을 간단한 문자열로 반환"""
    if score_type == "wrinkle":
        if score < 0.01:
            return "매우 좋음"
        elif score < 0.02:
            return "좋음"
        elif score < 0.04:
            return "보통"
        else:
            return "주의"
    else:  # pore
        if score < 0.005:
            return "매우 좋음"
        elif score < 0.01:
            return "좋음"
        elif score < 0.02:
            return "보통"
        else:
            return "주의"

# 5. 테스트 실행
if __name__ == "__main__":
    # 기본 테스트 이미지 경로
    image_path = r"C:\Users\user\Desktop\Github\cosrec\KakaoTalk_20250715_141326156.jpg"
    
    # 이미지 파일 존재 여부 확인
    import os
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        print("다른 이미지 파일을 사용하거나 경로를 확인해주세요.")
        
        # 대체 이미지 경로 제안
        alternative_paths = [
            r"C:\Users\user\Desktop\Github\cosrec\test_face.jpg",
            r"C:\Users\user\Desktop\test_image.jpg",
            r"C:\Users\user\Desktop\face_test.jpg"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"대체 이미지 발견: {alt_path}")
                image_path = alt_path
                break
        else:
            print("사용 가능한 이미지를 찾을 수 없습니다.")
            print("이미지 파일을 추가하거나 경로를 수정해주세요.")
            exit()
    else:
        print(f"이미지 파일 발견: {image_path}")
    
    # 사용자 선택
    print("\n🎯 실행 모드 선택:")
    print("1. 기본 피부 분석 + 이미지 저장")
    print("2. 멀티모달 모델용 데이터 준비")
    
    choice = input("선택하세요 (1 또는 2): ").strip()
    
    if choice == "1":
        print("\n🔬 기본 피부 분석 실행...")
        saved_files = analyze_skin(image_path, save_results=True)
        if saved_files:
            print(f"\n🎯 멀티모달 모델 추천 이미지: {saved_files.get('clean_skin', saved_files.get('skin_only', '없음'))}")
    
    elif choice == "2":
        print("\n🤖 멀티모달 모델용 데이터 준비...")
        
        # 사용자 정보 입력
        print("\n👤 사용자 정보를 입력해주세요:")
        age = input("나이: ").strip()
        gender = input("성별: ").strip()
        skin_type = input("피부 타입 (건성/지성/복합성/민감성): ").strip()
        concerns = input("주요 피부 고민: ").strip()
        
        user_description = f"""
        나이: {age}
        성별: {gender}
        피부 타입: {skin_type}
        주요 고민: {concerns}
        """.strip()
        
        multimodal_data = prepare_multimodal_data(image_path, user_description)
        
        if multimodal_data:
            print("\n✅ 멀티모달 모델용 데이터 준비 완료!")
            print(f"📁 주요 이미지: {multimodal_data['image_path']}")
            print(f"📊 분석 결과: 주름 {multimodal_data['skin_analysis']['wrinkle_score']:.3f}, 모공 {multimodal_data['skin_analysis']['pore_score']:.3f}")
            print(f"💬 생성된 프롬프트 미리보기:")
            print(multimodal_data['analysis_prompt'][:200] + "...")
            
            # 간단한 분석 화면 표시
            face_img, skin_mask = extract_face_with_skin_mask(image_path)
            if face_img is not None and skin_mask is not None:
                clean_skin = face_img.copy()
                clean_skin[skin_mask == 0] = [255, 255, 255]
                cv2.imshow("멀티모달 모델용 피부 이미지", clean_skin)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    else:
        print("잘못된 선택입니다. 기본 분석을 실행합니다.")
        analyze_skin(image_path)
