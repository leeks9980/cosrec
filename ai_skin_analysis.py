# -*- coding: utf-8 -*-
"""
Google AI Studio 통합 피부 분석 시스템
기존 컴퓨터 비전 분석 + Google Gemini 멀티모달 AI 분석
"""

import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
from PIL import Image
import os
import tempfile
import base64
from datetime import datetime

class SkinAnalysisSystem:
    def __init__(self, api_key=None):
        """
        피부 분석 시스템 초기화
        
        Args:
            api_key (str): Google AI Studio API 키
        """
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            print("?? API 키가 설정되지 않았습니다. 컴퓨터 비전 분석만 가능합니다.")
    
    def extract_face_with_skin_mask(self, image_path):
        """기존 얼굴 추출 및 피부 마스크 생성 함수"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"? 이미지를 불러올 수 없습니다: {image_path}")
                return None, None
            
            # 이미지 품질 개선
            def enhance_image(img):
                denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
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
            
            # 얼굴 감지
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3) as face_detection:
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                if not results.detections:
                    print("? 이미지 품질 개선 후 재시도...")
                    enhanced_image = enhance_image(image)
                    results = face_detection.process(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
                    
                    if not results.detections:
                        print("? 신뢰도를 낮춰서 재시도...")
                        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1) as face_detection_low:
                            results = face_detection_low.process(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
                            if results.detections:
                                print("? 얼굴 감지 성공!")
                                image = enhanced_image
                            else:
                                print("? 얼굴을 감지할 수 없습니다.")
                                return None, None
                    else:
                        print("? 얼굴 감지 성공!")
                        image = enhanced_image
                
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
                
                # 피부 마스크 생성
                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                                         refine_landmarks=True, min_detection_confidence=0.3) as face_mesh:
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    mesh_results = face_mesh.process(face_rgb)
                    
                    if mesh_results.multi_face_landmarks:
                        landmarks = mesh_results.multi_face_landmarks[0]
                        mask = self._create_skin_mask(face_img, landmarks)
                        return face_img, mask
                    else:
                        print("?? 피부 마스크 생성 실패")
                        return face_img, None
                        
        except Exception as e:
            print(f"? 오류 발생: {e}")
            return None, None
    
    def _create_skin_mask(self, face_img, landmarks):
        """피부 마스크 생성"""
        face_h, face_w = face_img.shape[:2]
        mask = np.zeros((face_h, face_w), dtype=np.uint8)
        
        # 얼굴 윤곽선
        FACE_OVAL = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        # 제외할 영역들
        exclude_regions = {
            "left_eye": [159, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            "right_eye": [385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
            "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            "right_eyebrow": [296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
            "lips": [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        }
        
        # 얼굴 기본 마스크 생성
        face_points = []
        for idx in FACE_OVAL:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * face_w)
                y = int(landmarks.landmark[idx].y * face_h)
                face_points.append([x, y])
        
        if len(face_points) > 0:
            face_points = np.array(face_points)
            cv2.fillPoly(mask, [face_points], 255)
            
            # 제외 영역 마스킹
            for region_name, indices in exclude_regions.items():
                exclude_points = []
                for idx in indices:
                    if idx < len(landmarks.landmark):
                        x = int(landmarks.landmark[idx].x * face_w)
                        y = int(landmarks.landmark[idx].y * face_h)
                        exclude_points.append([x, y])
                
                if len(exclude_points) > 2:
                    exclude_points = np.array(exclude_points)
                    hull = cv2.convexHull(exclude_points)
                    
                    if region_name == "lips":
                        # 입술은 더 정밀하게 처리
                        kernel_small = np.ones((3,3), np.uint8)
                        temp_mask = np.zeros((face_h, face_w), dtype=np.uint8)
                        cv2.fillPoly(temp_mask, [hull], 255)
                        temp_mask = cv2.erode(temp_mask, kernel_small, iterations=1)
                        mask = cv2.bitwise_and(mask, cv2.bitwise_not(temp_mask))
                    else:
                        cv2.fillPoly(mask, [hull], 0)
            
            # 마스크 후처리
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (3,3), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
        return mask
    
    def computer_vision_analysis(self, face_img, skin_mask):
        """컴퓨터 비전 기반 피부 분석"""
        if face_img is None:
            return None
        
        # 주름 탐지
        def detect_wrinkles(img, mask):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            if mask is not None:
                edges = cv2.bitwise_and(edges, mask)
                mask_area = (mask > 0).sum()
                wrinkle_ratio = (edges > 0).sum() / mask_area if mask_area > 0 else 0
            else:
                wrinkle_ratio = (edges > 0).sum() / edges.size
            
            return wrinkle_ratio, edges
        
        # 모공 탐지
        def detect_pores(img, mask):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            high_pass = cv2.subtract(gray, cv2.GaussianBlur(gray, (9, 9), 0))
            _, binarized = cv2.threshold(high_pass, 15, 255, cv2.THRESH_BINARY)
            
            if mask is not None:
                binarized = cv2.bitwise_and(binarized, mask)
                mask_area = (mask > 0).sum()
                pore_ratio = (binarized > 0).sum() / mask_area if mask_area > 0 else 0
            else:
                pore_ratio = (binarized > 0).sum() / binarized.size
            
            return pore_ratio, binarized
        
        wrinkle_score, wrinkle_img = detect_wrinkles(face_img, skin_mask)
        pore_score, pore_img = detect_pores(face_img, skin_mask)
        
        return {
            'wrinkle_score': wrinkle_score,
            'pore_score': pore_score,
            'wrinkle_img': wrinkle_img,
            'pore_img': pore_img
        }
    
    def gemini_analysis(self, face_img, user_description="", cv_results=None):
        """Google Gemini를 활용한 멀티모달 AI 피부 분석"""
        if self.model is None:
            return "? API 키가 설정되지 않았습니다."
        
        try:
            # 이미지를 PIL 형태로 변환
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            
            # 컴퓨터 비전 결과 포함한 프롬프트 생성
            cv_info = ""
            if cv_results:
                cv_info = f"""
                
                **컴퓨터 비전 분석 결과:**
                - 주름 지수: {cv_results['wrinkle_score']:.3f}
                - 모공 지수: {cv_results['pore_score']:.3f}
                """
            
            prompt = f"""
            안녕하세요! 첨부된 얼굴 이미지를 피부과 전문의 관점에서 전문적으로 분석해주세요.
            
            **사용자 정보:**
            {user_description}
            {cv_info}
            
            **분석 요청 사항:**
            
            1. ? **피부 상태 종합 평가** (10점 만점)
               - 전반적인 피부 건강도
               - 피부 톤의 균일성
               - 피부 질감과 탄력성
               - 종합 점수와 근거
            
            2. ? **주름 분석**
               - 주름의 유형 (표정 주름, 나이 주름, 잔주름)
               - 주름의 깊이와 분포
               - 주요 발생 부위 (이마, 눈가, 입가 등)
               - 심각도 평가 (1-5단계)
            
            3. ?? **모공 분석**
               - 모공의 크기와 가시성
               - 모공 확장 부위 (코, 볼, 이마)
               - 블랙헤드/화이트헤드 여부
               - 모공 상태 평가 (1-5단계)
            
            4. ? **피부 톤 & 색소 분석**
               - 피부 톤의 균일성
               - 색소 침착이나 반점 여부
               - 홍조나 염증 징후
               - 피부 투명도와 윤기
            
            5. ? **수분/유분 균형**
               - 추정 피부 타입 (건성/지성/복합성/민감성)
               - 수분 부족 징후
               - T존과 U존 상태 차이
               - 피부 장벽 상태
            
            6. ? **전문의 조언**
               - 우선 개선 필요 사항
               - 일상 스킨케어 루틴 제안
               - 주의해야 할 피부 문제
               - 피부과 치료 권장 사항
            
            7. ? **단계별 개선 계획**
               - 1-2주 단기 집중 케어
               - 1-3개월 중기 개선 방안
               - 6개월+ 장기 관리 계획
            
            **분석 시 고려사항:**
            - 조명과 각도의 영향을 고려하여 분석
            - 나이와 성별에 따른 일반적 피부 특성 반영
            - 개인차와 유전적 요인 고려
            - 실용적이고 구체적인 조언 제공
            - 한국인 피부 특성 고려
            
            가능한 한 구체적이고 실용적인 분석과 조언을 한국어로 제공해주세요.
            """
            
            response = self.model.generate_content([prompt, pil_image])
            return response.text
            
        except Exception as e:
            return f"? AI 분석 중 오류 발생: {e}"
    
    def comprehensive_analysis(self, image_path, user_description=""):
        """종합적인 피부 분석 (컴퓨터 비전 + AI)"""
        print("? 얼굴 감지 및 피부 영역 추출 중...")
        face_img, skin_mask = self.extract_face_with_skin_mask(image_path)
        
        if face_img is None:
            return None
        
        print("? 컴퓨터 비전 분석 중...")
        cv_results = self.computer_vision_analysis(face_img, skin_mask)
        
        print("? AI 전문가 분석 중...")
        ai_analysis = self.gemini_analysis(face_img, user_description, cv_results)
        
        # 결과 통합
        report = self._generate_comprehensive_report(cv_results, ai_analysis, user_description)
        
        return {
            'face_img': face_img,
            'skin_mask': skin_mask,
            'cv_results': cv_results,
            'ai_analysis': ai_analysis,
            'comprehensive_report': report
        }
    
    def _generate_comprehensive_report(self, cv_results, ai_analysis, user_description):
        """종합 리포트 생성"""
        
        def interpret_cv_score(score, score_type):
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
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
        ???????????????????????????????????????????????????????????
        ? **AI 피부 분석 종합 리포트**
        ???????????????????????????????????????????????????????????
        
        ? **분석 일시:** {timestamp}
        ? **사용자 정보:** {user_description}
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ? **정량적 분석 결과 (Computer Vision)**
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        ? **주름 분석**
        ? 주름 지수: {cv_results['wrinkle_score']:.3f}
        ? 상태 평가: {interpret_cv_score(cv_results['wrinkle_score'], 'wrinkle')}
        
        ?? **모공 분석**
        ? 모공 지수: {cv_results['pore_score']:.3f}
        ? 상태 평가: {interpret_cv_score(cv_results['pore_score'], 'pore')}
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ? **AI 전문가 분석 결과 (Google Gemini)**
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        {ai_analysis}
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ? **종합 결론**
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        이 분석은 컴퓨터 비전의 정량적 분석과 AI의 정성적 분석을 결합한 결과입니다.
        
        ? **분석 방법론:**
        ? 컴퓨터 비전: 객관적 수치 기반 정량 분석
        ? AI 전문가: 맥락적 해석과 종합적 평가
        ? 통합 결과: 상호 보완적 종합 진단
        
        ?? **주의사항:**
        ? 본 분석은 참고용이며 의학적 진단을 대체하지 않습니다
        ? 심각한 피부 문제 시 전문의 상담을 권장합니다
        ? 개인차와 환경 요인을 고려하여 해석하시기 바랍니다
        
        ???????????????????????????????????????????????????????????
        """
        
        return report
    
    def save_analysis_results(self, results, output_dir="analysis_results"):
        """분석 결과 저장"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 이미지 결과 저장
        cv2.imwrite(f"{output_dir}/face_{timestamp}.jpg", results['face_img'])
        cv2.imwrite(f"{output_dir}/wrinkle_{timestamp}.jpg", results['cv_results']['wrinkle_img'])
        cv2.imwrite(f"{output_dir}/pore_{timestamp}.jpg", results['cv_results']['pore_img'])
        
        if results['skin_mask'] is not None:
            cv2.imwrite(f"{output_dir}/skin_mask_{timestamp}.jpg", results['skin_mask'])
        
        # 리포트 저장
        with open(f"{output_dir}/report_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(results['comprehensive_report'])
        
        print(f"? 분석 결과가 '{output_dir}' 폴더에 저장되었습니다.")
        
        return f"{output_dir}/report_{timestamp}.txt"


def interactive_skin_analysis():
    """대화형 피부 분석 인터페이스"""
    print("? AI 피부 분석 시스템 v2.0")
    print("=" * 60)
    
    # API 키 입력
    api_key = input("? Google AI Studio API 키를 입력하세요 (선택사항): ").strip()
    if not api_key:
        print("?? API 키 없이 컴퓨터 비전 분석만 진행합니다.")
    
    # 시스템 초기화
    system = SkinAnalysisSystem(api_key if api_key else None)
    
    # 이미지 경로 입력
    image_path = input("? 분석할 이미지 파일 경로를 입력하세요: ").strip()
    
    if not os.path.exists(image_path):
        print("? 파일을 찾을 수 없습니다.")
        return
    
    # 사용자 정보 입력
    print("\n? 추가 정보를 입력해주세요 (선택사항):")
    age = input("나이: ").strip()
    gender = input("성별: ").strip()
    skin_type = input("피부 타입 (건성/지성/복합성/민감성): ").strip()
    concerns = input("주요 피부 고민: ").strip()
    routine = input("현재 스킨케어 루틴: ").strip()
    
    user_description = f"""
    나이: {age}
    성별: {gender}
    피부 타입: {skin_type}
    주요 고민: {concerns}
    현재 루틴: {routine}
    """.strip()
    
    # 분석 실행
    print("\n? 분석을 시작합니다...")
    print("=" * 60)
    
    results = system.comprehensive_analysis(image_path, user_description)
    
    if results is None:
        print("? 분석에 실패했습니다.")
        return
    
    # 결과 출력
    print(results['comprehensive_report'])
    
    # 이미지 결과 표시
    print("\n? 분석 결과 이미지를 표시합니다...")
    cv2.imshow("원본 얼굴", results['face_img'])
    cv2.imshow("주름 분석", results['cv_results']['wrinkle_img'])
    cv2.imshow("모공 분석", results['cv_results']['pore_img'])
    
    if results['skin_mask'] is not None:
        cv2.imshow("피부 마스크", results['skin_mask'])
        masked_face = cv2.bitwise_and(results['face_img'], results['face_img'], mask=results['skin_mask'])
        cv2.imshow("피부 영역만", masked_face)
    
    # 결과 저장 여부 확인
    print("\n? 분석 결과를 저장하시겠습니까? (y/n): ", end="")
    if input().lower() == 'y':
        report_path = system.save_analysis_results(results)
        print(f"? 상세 리포트: {report_path}")
    
    print("\n아무 키나 누르면 창이 닫힙니다...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    interactive_skin_analysis()
