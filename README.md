# AI 피부 분석 시스템 (cosrec)

Google AI Studio와 컴퓨터 비전을 활용한 고급 피부 분석 및 코스메틱 추천 시스템

## ? 개요

이 시스템은 최신 AI 기술을 활용하여 사용자의 피부 상태를 종합적으로 분석하고 맞춤형 코스메틱 제품을 추천합니다.

### ? 주요 기술
- **컴퓨터 비전**: MediaPipe + OpenCV 기반 정밀 얼굴 분석
- **멀티모달 AI**: Google Gemini를 활용한 전문가 수준 피부 분석
- **실시간 처리**: 빠르고 정확한 실시간 분석
- **개인 맞춤형**: 사용자 정보를 반영한 맞춤 조언

## ? 주요 기능

### 1. 정밀 얼굴 분석
- 468개 랜드마크 기반 정확한 얼굴 인식
- 피부 영역만 추출하여 정밀 분석
- 눈, 입술, 눈썹 등 비피부 영역 자동 제외

### 2. 컴퓨터 비전 분석
- **주름 분석**: Canny 엣지 검출로 주름 정량화
- **모공 분석**: 고주파 필터링으로 모공 검출
- **객관적 지표**: 수치 기반 정량적 평가

### 3. AI 전문가 분석
- **멀티모달 분석**: 이미지 + 사용자 설명 통합
- **전문의 관점**: 피부과 전문의 수준 종합 평가
- **개인 맞춤형**: 나이, 성별, 피부 타입별 맞춤 조언

### 4. 종합 리포트
- **정량적 + 정성적**: 수치와 해석의 조화
- **단계별 계획**: 단기/중기/장기 개선 방안
- **실용적 조언**: 구체적인 스킨케어 방법 제안

## ?? 설치 및 설정

### 1. 라이브러리 설치
```bash
pip install opencv-python mediapipe numpy google-generativeai pillow
```

### 2. Google AI Studio API 키 설정
1. [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키 발급
2. 환경 변수 설정:
```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
```

### 3. 실행
```bash
python ai_skin_analysis.py
```

## ? 사용법

### 대화형 모드
```bash
python ai_skin_analysis.py
```

### 프로그래밍 모드
```python
from ai_skin_analysis import SkinAnalysisSystem

# 시스템 초기화
system = SkinAnalysisSystem(api_key="YOUR_API_KEY")

# 종합 분석 실행
results = system.comprehensive_analysis(
    image_path="path/to/your/image.jpg",
    user_description="나이: 25, 성별: 여성, 피부 타입: 복합성"
)

# 결과 출력
print(results['comprehensive_report'])

# 결과 저장
system.save_analysis_results(results)
```

## ? 분석 결과 해석

### 주름 지수
- **0.000 - 0.010**: 매우 좋음 (거의 주름 없음)
- **0.010 - 0.020**: 좋음 (약간의 미세 주름)
- **0.020 - 0.040**: 보통 (보통 수준의 주름)
- **0.040 이상**: 주의 (뚜렷한 주름 존재)

### 모공 지수
- **0.000 - 0.005**: 매우 좋음 (모공 거의 보이지 않음)
- **0.005 - 0.010**: 좋음 (작은 모공)
- **0.010 - 0.020**: 보통 (보통 크기 모공)
- **0.020 이상**: 주의 (확장된 모공)

## ? 분석 방법론

### 1. 컴퓨터 비전 분석
- **얼굴 감지**: MediaPipe Face Detection
- **피부 마스킹**: 468개 랜드마크 기반 정밀 마스킹
- **주름 검출**: Canny 엣지 검출 + 히스토그램 균등화
- **모공 검출**: 고주파 필터링 + 이진화

### 2. AI 전문가 분석
- **멀티모달 입력**: 이미지 + 사용자 설명 + CV 결과
- **전문가 프롬프트**: 피부과 전문의 관점 분석
- **종합 평가**: 10개 항목별 상세 분석

## ? 파일 구조

```
cosrec/
├── ai_skin_analysis.py      # 메인 분석 시스템
├── face_detect.py           # 기존 얼굴 감지 모듈
├── README.md                # 사용 설명서
└── analysis_results/        # 분석 결과 저장 폴더
    ├── face_YYYYMMDD_HHMMSS.jpg
    ├── wrinkle_YYYYMMDD_HHMMSS.jpg
    ├── pore_YYYYMMDD_HHMMSS.jpg
    └── report_YYYYMMDD_HHMMSS.txt
```

## ?? 주의사항

- 본 시스템은 참고용이며 의학적 진단을 대체하지 않습니다
- 심각한 피부 문제 시 전문의 상담을 권장합니다
- API 키와 개인정보는 안전하게 보관하세요
- 조명, 각도, 이미지 품질에 따라 결과가 달라질 수 있습니다

## ? 최적화 팁

### 이미지 품질
- 해상도: 1280x720 이상
- 조명: 자연광 또는 균일한 실내 조명
- 각도: 정면 촬영
- 표정: 자연스러운 무표정

### 사용자 정보
- 나이, 성별, 피부 타입 등 상세 정보 제공
- 현재 피부 고민과 스킨케어 루틴 명시
- 정기적 분석으로 피부 변화 추적

## ? 업데이트 내역

### v2.0 (2024-01-15)
- Google AI Studio 통합
- 멀티모달 AI 분석 추가
- 종합 리포트 생성
- 사용자 맞춤형 조언

### v1.0 (2024-01-10)
- 기본 컴퓨터 비전 분석
- 얼굴 감지 및 피부 마스킹
- 주름/모공 정량화

## ? 문의사항

이슈나 문의사항이 있으시면 GitHub Issues를 통해 연락 주세요.
- Python
- PyTorch
- CLIP 모델
- Selenium
- SQLite
