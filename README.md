# Bike Sharing Demand Prediction (AI Core Engineering)

UCI Bike Sharing Dataset(워싱턴 D.C. Capital Bikeshare)을 기반으로 시간대별 자전거 대여량(`cnt`)을 예측하는 회귀(Regression) 베이스라인 머신러닝 파이프라인 프로젝트입니다.

---

## 1. 프로젝트 주요 특징

- **시계열 누수 방지(Temporal Split)**: 무작위 분할 대신 일자(`dteday`) 기준 순차 분할(Train / Validation / Test)을 적용하여 미래 데이터 누수를 차단했습니다.
- **타깃 누수(Target Leakage) 차단**: 타깃(`cnt`)의 구성 성분인 회원(`registered`) 및 비회원(`casual`) 변수를 입력에서 배제했습니다.
- **Scikit-Learn 파이프라인**: `ColumnTransformer` 기반의 수치형 스케일링 및 범주형 원핫인코딩을 파이프라인으로 캡슐화하여 검증/테스트 세트의 오염을 방지했습니다.
- **체계적 모델 평가**: Dummy, Linear Regression, Random Forest를 검증 세트(RMSE) 기준으로 비교 후, 최적 모델을 전체 Train 데이터로 재학습하여 Test 세트 최종 일반화 성능 및 잔차를 평가합니다.

---

## 2. 환경 설정 및 실행 방법

### 가상환경 생성 및 의존성 설치
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 파이프라인 실행
```bash
python main.py
```

실행 시 데이터 다운로드부터 분할, EDA 차트 생성, 모델 학습 및 평가, 모델 아티팩트(`outputs/`) 저장이 일괄 진행됩니다.
