"""Bike Sharing Demand Regression Baseline Pipeline.

UCI Bike Sharing Dataset(hour.csv)을 활용하여 시간별 자전거 대여량(cnt)을 예측하는
회귀 모델 학습, 검증, 최종 평가 및 아티팩트 저장을 수행하는 실행 스크립트입니다.
"""

from pathlib import Path
import urllib.request
import zipfile
import json
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Constants
RANDOM_STATE = 42
TARGET = "cnt"
NUMERIC_FEATURES = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL_FEATURES = [
    "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
UCI_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def download_and_load_data(data_dir: Path) -> pd.DataFrame:
    """UCI 원격 저장소에서 데이터셋을 다운로드하고 DataFrame으로 로드합니다."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "Bike-Sharing-Dataset.zip"
    csv_path = raw_dir / "hour.csv"

    if not csv_path.exists():
        if not zip_path.exists():
            print(f"[*] UCI 데이터셋 다운로드 시작: {UCI_URL}")
            urllib.request.urlretrieve(UCI_URL, zip_path)
            print("[+] 다운로드 완료")

        with zipfile.ZipFile(zip_path, "r") as zf:
            member = next(name for name in zf.namelist() if name.endswith("hour.csv"))
            with zf.open(member) as source, open(csv_path, "wb") as target:
                target.write(source.read())
        print(f"[+] 압축 해제 완료: {csv_path}")

    df = pd.read_csv(csv_path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    print(f"[+] 데이터 로드 성공: {df.shape[0]:,}행 x {df.shape[1]}열")
    return df


def split_data_by_date(df: pd.DataFrame, test_size: float = 0.20, val_size_within_train: float = 0.20):
    """시계열 누수를 방지하기 위해 날짜 단위로 Train / Validation / Test를 순차 분할합니다."""
    dates = np.sort(df["dteday"].unique())
    test_idx = int(len(dates) * (1 - test_size))
    test_start = pd.Timestamp(dates[test_idx])

    development_df = df[df["dteday"] < test_start].copy()
    test_df = df[df["dteday"] >= test_start].copy()

    development_dates = np.sort(development_df["dteday"].unique())
    val_idx = int(len(development_dates) * (1 - val_size_within_train))
    val_start = pd.Timestamp(development_dates[val_idx])

    fit_df = development_df[development_df["dteday"] < val_start].copy()
    val_df = development_df[development_df["dteday"] >= val_start].copy()

    # 데이터 순서 및 무결성 검증
    assert fit_df["dteday"].max() < val_df["dteday"].min(), "Fit 세트와 Validation 세트 간 날짜 역전 오류"
    assert val_df["dteday"].max() < test_df["dteday"].min(), "Validation 세트와 Test 세트 간 날짜 역전 오류"

    print("\n[데이터 분할 요약]")
    print(f"- Train/Fit  : {fit_df['dteday'].min().date()} ~ {fit_df['dteday'].max().date()} | {len(fit_df):,}행 | {fit_df['dteday'].nunique():,}일")
    print(f"- Validation : {val_df['dteday'].min().date()} ~ {val_df['dteday'].max().date()} | {len(val_df):,}행 | {val_df['dteday'].nunique():,}일")
    print(f"- Test       : {test_df['dteday'].min().date()} ~ {test_df['dteday'].max().date()} | {len(test_df):,}행 | {test_df['dteday'].nunique():,}일")

    return development_df, fit_df, val_df, test_df


def run_eda_and_save_figures(fit_df: pd.DataFrame, figures_dir: Path):
    """Train/Fit 데이터 기반 EDA 시각화 결과물을 저장합니다."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. 타깃 분포
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(fit_df[TARGET], bins=40, kde=True, ax=axes[0])
    axes[0].set_title("Distribution of hourly rentals (cnt)")
    axes[0].set_xlabel("cnt")
    sns.boxplot(x=fit_df[TARGET], ax=axes[1])
    axes[1].set_title("Boxplot of cnt")
    plt.tight_layout()
    fig.savefig(figures_dir / "target_distribution.png", dpi=150)
    plt.close(fig)

    # 2. 수치형 변수 분포
    fig = plt.figure(figsize=(10, 7))
    fit_df[NUMERIC_FEATURES].hist(figsize=(10, 7), bins=30, edgecolor="black")
    plt.tight_layout()
    plt.savefig(figures_dir / "numeric_features_hist.png", dpi=150)
    plt.close("all")

    # 3. 시간대/요일/날씨별 평균 대여량
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    hourly = fit_df.groupby("hr", as_index=False)[TARGET].mean()
    sns.lineplot(data=hourly, x="hr", y=TARGET, marker="o", ax=axes[0])
    axes[0].set_title("Average rentals by hour")
    axes[0].set_xticks(range(0, 24, 2))

    weekday = fit_df.groupby("weekday", as_index=False)[TARGET].mean()
    sns.barplot(data=weekday, x="weekday", y=TARGET, ax=axes[1])
    axes[1].set_title("Average rentals by weekday")

    weather = fit_df.groupby("weathersit", as_index=False)[TARGET].mean()
    sns.barplot(data=weather, x="weathersit", y=TARGET, ax=axes[2])
    axes[2].set_title("Average rentals by weather")
    plt.tight_layout()
    fig.savefig(figures_dir / "rentals_by_patterns.png", dpi=150)
    plt.close(fig)

    # 4. 상관관계 행렬
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_cols = NUMERIC_FEATURES + [TARGET]
    sns.heatmap(fit_df[corr_cols].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation matrix")
    plt.tight_layout()
    fig.savefig(figures_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)

    print(f"[+] EDA 시각화 차트 저장 완료: {figures_dir}")


def build_preprocessor() -> ColumnTransformer:
    """결측치 대체 및 인코딩/스케일링 전처리기를 생성합니다."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_model_pipeline(model) -> Pipeline:
    """전처리기와 모델을 하나의 파이프라인으로 연결합니다."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def calculate_metrics(y_true, y_pred) -> dict:
    """회귀 평가 지표(MAE, RMSE, R2)를 산출합니다."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def compare_baselines(X_fit, y_fit, X_val, y_val):
    """Validation 세트를 기반으로 후보 베이스라인 모델 성능을 비교합니다."""
    models = {
        "Dummy Mean": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    results = []
    print("\n[베이스라인 모델 검증 시작]")
    for model_name, model in models.items():
        pipeline = build_model_pipeline(model)
        pipeline.fit(X_fit, y_fit)
        val_pred = pipeline.predict(X_val)
        metrics = calculate_metrics(y_val, val_pred)
        results.append({"Model": model_name, **metrics})
        print(f" - {model_name:18s} | MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | R2: {metrics['R2']:.4f}")

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = results_df.loc[0, "Model"]
    print(f"\n[+] Validation RMSE 기준 최적 모델: {best_model_name}")
    return results_df, best_model_name


def instantiate_model(model_name: str):
    """모델 이름에 대응하는 Scikit-Learn 추정기를 반환합니다."""
    if model_name == "Dummy Mean":
        return DummyRegressor(strategy="mean")
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    raise ValueError(f"지원하지 않는 모델 이름: {model_name}")


def evaluate_and_plot_test(final_model, X_test, y_test, best_model_name: str, figures_dir: Path):
    """최종 테스트 세트 평가 및 잔차 시각화를 수행합니다."""
    test_pred = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_pred)

    print("\n[최종 Test 세트 평가 결과]")
    print(f"- Model : {best_model_name}")
    print(f"- MAE   : {test_metrics['MAE']:.2f}")
    print(f"- RMSE  : {test_metrics['RMSE']:.2f}")
    print(f"- R2    : {test_metrics['R2']:.4f}")

    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. 실제값 vs 예측값 산점도
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=test_pred, alpha=0.35, ax=ax)
    min_val = min(y_test.min(), float(np.min(test_pred)))
    max_val = max(y_test.max(), float(np.max(test_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")
    ax.set_xlabel("Actual rentals")
    ax.set_ylabel("Predicted rentals")
    ax.set_title(f"Actual vs Predicted: {best_model_name}")
    plt.tight_layout()
    fig.savefig(figures_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # 2. 잔차 분석
    residuals = y_test.to_numpy() - test_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.scatterplot(x=test_pred, y=residuals, alpha=0.35, ax=axes[0])
    axes[0].axhline(0, linestyle="--", color="red")
    axes[0].set_xlabel("Predicted rentals")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs Predicted")

    sns.histplot(residuals, bins=40, kde=True, ax=axes[1])
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    plt.tight_layout()
    fig.savefig(figures_dir / "residuals_analysis.png", dpi=150)
    plt.close(fig)

    test_results_df = pd.DataFrame([{"Model": best_model_name, **test_metrics}])
    return test_results_df, test_metrics


def save_artifacts(
    output_dir: Path,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    final_model: Pipeline,
    best_model_name: str,
    test_metrics: dict,
    development_df: pd.DataFrame,
    test_dataset_df: pd.DataFrame
):
    """모델 파이프라인, 지표 테이블 및 메타데이터를 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    val_csv_path = output_dir / "validation_model_comparison.csv"
    test_csv_path = output_dir / "final_test_metrics.csv"
    model_joblib_path = output_dir / "best_model_pipeline.joblib"
    info_json_path = output_dir / "best_model_info.json"

    validation_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    joblib.dump(final_model, model_joblib_path)

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": "Validation RMSE",
        "validation_rmse": float(validation_df.loc[0, "RMSE"]),
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "target": TARGET,
        "features": FEATURES,
        "train_start": str(development_df["dteday"].min().date()),
        "train_end": str(development_df["dteday"].max().date()),
        "test_start": str(test_dataset_df["dteday"].min().date()),
        "test_end": str(test_dataset_df["dteday"].max().date()),
    }

    with open(info_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 모든 파이프라인 아티팩트 저장 완료: {output_dir}")


def main():
    warnings.filterwarnings("ignore")
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    figures_dir = output_dir / "figures"

    # 1. 데이터 로드
    df = download_and_load_data(data_dir)

    # 2. 날짜 기준 분할
    development_df, fit_df, val_df, test_df = split_data_by_date(df)

    # 3. EDA 시각화 저장
    run_eda_and_save_figures(fit_df, figures_dir)

    # 4. Feature / Target 준비
    X_fit, y_fit = fit_df[FEATURES], fit_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    X_train, y_train = development_df[FEATURES], development_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # 5. 검증 세트 기반 모델 비교 및 최적 모델 선택
    validation_results_df, best_model_name = compare_baselines(X_fit, y_fit, X_val, y_val)

    # 6. 전체 Train 세트(Fit + Validation)로 최종 모델 재학습
    print(f"\n[*] 최적 모델({best_model_name}) 전체 Train 데이터로 재학습 시작...")
    final_estimator = instantiate_model(best_model_name)
    final_pipeline = build_model_pipeline(final_estimator)
    final_pipeline.fit(X_train, y_train)
    print("[+] 최종 모델 재학습 완료")

    # 7. Test 세트 최종 평가
    test_results_df, test_metrics = evaluate_and_plot_test(
        final_pipeline, X_test, y_test, best_model_name, figures_dir
    )

    # 8. 아티팩트 저장
    save_artifacts(
        output_dir=output_dir,
        validation_df=validation_results_df,
        test_df=test_results_df,
        final_model=final_pipeline,
        best_model_name=best_model_name,
        test_metrics=test_metrics,
        development_df=development_df,
        test_dataset_df=test_df
    )


if __name__ == "__main__":
    main()
