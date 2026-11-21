import argparse
import json
import re
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "linear_regression.pkl"
META_PATH = BASE_DIR / "models" / "meta.json"
IMG_DIR = BASE_DIR / "static" / "images"
RANDOM_STATE = 42
TEST_SIZE = 0.2

#Cấu hình các cột
ALIASES = {"price": ["price", "gia", "gia_nha", "price_billion"],
            "area": ["area", "dien_tich", "area_m2"],
            "frontage": ["frontage", "mat_tien", "frontage_area"],
            "access_road": ["access_road", "duong_vao"],
            "floors": ["floors", "so_tang", "floor_count"],
            "bedrooms": ["bedrooms", "so_phong_ngu", "bed_rooms"],
            "bathrooms": ["bathrooms", "so_phong_tam", "bath_rooms"],
            "legal_status": ["legal_status", "phap_ly"],
            "furniture_state": ["furniture_state", "furniture", "noi_that"],
            "house_direction": ["house_direction", "huong_nha"],
            "balcony_direction": ["balcony_direction", "huong_ban_cong"],
            "address": ["address", "dia_chi"],}
#Các đặc trưng tự tạo
NUMERIC_COLS = ["area", "log_area", "frontage", "access_road", "floors", "bedrooms", "bathrooms"]
CATEGORICAL_COLS = ["district_key", "ward_key", "legal_status", "furniture_state", "house_direction", "balcony_direction",]
MISSING_LABEL = "Không rõ"
STATS = {} 
#Cách viết khác nhau của cùng một tỉnh
CITY_ALIASES = {"hn": "Hà Nội","hcm": "Hồ Chí Minh", "tphcm": "Hồ Chí Minh", "tphồchíminh": "Hồ Chí Minh",}

def normalize_name(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")
def to_number(series: pd.Series) -> pd.Series: # Chuyển cột dữ liệu về kiểu số
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    extracted = series.astype(str).str.extract(r"(\d+(?:[.,]\d+)?)")[0]
    return pd.to_numeric(extracted.str.replace(",", ".", regex=False), errors="coerce")
def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [normalize_name(c) for c in df.columns]
    print(f"Đọc {len(df)} dòng, các cột: {list(df.columns)}")

    #Đổi tên cột về tên chuẩn
    rename = {}
    for std, candidates in ALIASES.items():
        for c in candidates:
            if c in df.columns:
                rename[c] = std
                break
    df = df.rename(columns=rename)

    #Thống kê dữ liệu gốc
    STATS["n_raw"] = int(len(df))
    STATS["columns"] = [{"col": c,"kind": "số" if pd.api.types.is_numeric_dtype(df[c]) else "chữ","missing_pct": round(float(df[c].isna().mean() * 100), 2),}
        for c in df.columns]
    for required in ("price", "area"):
        if required not in df.columns:
            raise SystemExit(f"Không tìm thấy cột '{required}'. Hãy sửa ALIASES trong train_model.py "
                             f"cho khớp tên cột thật: {list(df.columns)}")

    #Tách địa chỉ thành 3 cấp.
    if "address" in df.columns:
        def clean(s):
            return str(s).strip().rstrip(". ").strip() #Bỏ dấu chấm cuối và khoảng trắng dư thừa
        def cap(s):  #Viết hoa chữ cái đầu
            return s[:1].upper() + s[1:] if isinstance(s, str) else s
        def norm_city(s):
            key = re.sub(r"[\s.]", "", s.lower())
            return CITY_ALIASES.get(key, cap(s))
        def norm_district(s):
            return cap(re.sub(r"^(quận|huyện)\s+(?=\D)", "", s, flags=re.IGNORECASE))

        parts = df["address"].astype(str).str.split(",")
        df["city"] = parts.apply(lambda p: norm_city(clean(p[-1])) if len(p) >= 2 else np.nan)
        df["district"] = parts.apply(lambda p: norm_district(clean(p[-2])) if len(p) >= 2 else np.nan)
        df["ward"] = parts.apply(lambda p: cap(clean(p[-3])) if len(p) >= 3 else np.nan)
        city_counts = df["city"].value_counts()
        df = df[df["city"].isin(city_counts[city_counts >= 5].index)].copy()
        STATS["n_after_city"] = int(len(df))
        df["district_key"] = df["city"] + "|" + df["district"]
        df["ward_key"] = df["district_key"] + "|" + df["ward"]

    for col in ["price"] + NUMERIC_COLS:
        if col in df.columns:
            df[col] = to_number(df[col])

    #Thống nhất đơn vị giá tiền
    if df["price"].median() > 1e6:
        df["price"] = df["price"] / 1e9
        print("Giá có vẻ tính bằng VNĐ -> đã đổi sang tỷ VNĐ")

    before = len(df)
    df = df[(df["price"] > 0) & (df["area"] > 0)].dropna(subset=["price", "area"])
    STATS.setdefault("n_after_city", int(before))
    for col in ("price", "area"):
        lo, hi = df[col].quantile([0.01, 0.99])
        STATS[f"{col}_cut"] = [float(lo), float(hi)]
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    STATS["n_clean"] = int(len(df))
    print(f"Sau làm sạch: {len(df)}/{before} dòng")
    df["log_area"] = np.log(df["area"])     #Thêm log(area) để đường thẳng khớp tốt hơn
    return df.reset_index(drop=True)

def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_LABEL)),
                            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)),])
    pre = ColumnTransformer([("num", numeric, numeric_cols),
                             ("cat", categorical, categorical_cols),])
    return Pipeline([("pre", pre), ("model", LinearRegression())])

def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y_true, y_pred)),}

def save_plots(y_test, y_pred, pipe, feature_names_source):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, s=12, alpha=0.5)
    lim = [0, max(y_test.max(), y_pred.max())]
    plt.plot(lim, lim, "r--", label="Dự đoán hoàn hảo")
    plt.xlabel("Giá thực tế (tỷ VNĐ)")
    plt.ylabel("Giá dự đoán (tỷ VNĐ)")
    plt.title("Giá thực tế vs giá dự đoán (tập test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "actual_vs_pred.png", dpi=150)
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.hist(y_test - y_pred, bins=40)
    plt.xlabel("Sai số = thực tế - dự đoán (tỷ VNĐ)")
    plt.ylabel("Số căn nhà")
    plt.title("Phân phối sai số")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "residuals.png", dpi=150)
    plt.close()

    #Hệ số lớn nhất
    names = [n.replace("cat__district_key_", "Quận: ").replace("cat__ward_key_", "Phường: ").replace("cat__", "").replace("num__", "")
        for n in pipe.named_steps["pre"].get_feature_names_out()]
    coefs = pipe.named_steps["model"].coef_
    top = np.argsort(np.abs(coefs))[-15:]
    plt.figure(figsize=(8, 5))
    plt.barh([names[i] for i in top], [coefs[i] for i in top])
    plt.title("15 hệ số có ảnh hưởng lớn nhất (w)")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "coefficients.png", dpi=150)
    plt.close()

def run_experiments(df, train_idx, test_idx, numeric_cols, categorical_cols, selected_is_log):
    if "city" not in df.columns:
        return []
    num_base = [c for c in numeric_cols if c != "log_area"]
    cat_base = ["legal_status", "furniture_state"]
    plans = [("Chỉ có tỉnh/thành phố (chưa tách quận, phường)", num_base, ["city"] + cat_base, False),
             ("Tách thêm quận và phường", num_base, ["district_key", "ward_key"] + cat_base, False),
             ("Thêm log(diện tích)", num_base + ["log_area"], ["district_key", "ward_key"] + cat_base, False),
             ("Thêm hướng nhà và hướng ban công", numeric_cols, categorical_cols, False),
             ("Như trên nhưng dự đoán log(giá) thay vì giá gốc", numeric_cols, categorical_cols, True),]
    results = []
    for name, num, cat, use_log in plans:
        cols = [c for c in num + cat if c in df.columns]
        num_ok = [c for c in num if c in df.columns]
        cat_ok = [c for c in cat if c in df.columns]
        pipe = build_pipeline(num_ok, cat_ok)
        model = (TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
                 if use_log else pipe)
        model.fit(df.loc[train_idx, cols], df.loc[train_idx, "price"])
        m = evaluate(df.loc[test_idx, "price"], model.predict(df.loc[test_idx, cols]))
        full = (num == numeric_cols and cat == categorical_cols)
        results.append({"name": name, "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                        "selected": bool(full and use_log == selected_is_log)})
        print(f"  {name:55s} R2={m['R2']:.3f}  MAE={m['MAE']:.3f}")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE_DIR / "data" / "gia_nha.csv"))
    args = ap.parse_args()
    df = load_and_clean(Path(args.data))
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    feature_cols = numeric_cols + categorical_cols
    print("Đặc trưng số:", numeric_cols)
    print("Đặc trưng phân loại:", categorical_cols)
    X, y = df[feature_cols], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    #Linear Regression thuần
    base = build_pipeline(numeric_cols, categorical_cols).fit(X_train, y_train)
    base_metrics = evaluate(y_test, base.predict(X_test))

    #Linear Regression trên log(giá)
    log_model = TransformedTargetRegressor(regressor=build_pipeline(numeric_cols, categorical_cols),func=np.log1p, inverse_func=np.expm1,).fit(X_train, y_train)
    log_pred = log_model.predict(X_test)
    log_metrics = evaluate(y_test, log_pred)
    print("\nLinear Regression (giá gốc):", base_metrics)
    print("Linear Regression (log giá): ", log_metrics)
    if log_metrics["R2"] > base_metrics["R2"]:
        best, best_name, best_metrics, best_pred = log_model, "Linear Regression (log giá)", log_metrics, log_pred
        coef_pipe = log_model.regressor_
    else:
        best, best_name, best_metrics, best_pred = base, "Linear Regression", base_metrics, base.predict(X_test)
        coef_pipe = base
    print(f"\n=> Chọn: {best_name}")

    #Số liệu bổ sung
    train_metrics = evaluate(y_train, best.predict(X_train))
    print("Trên tập train:", train_metrics)

    #Cross-validation 
    cv = cross_validate(best, X, y, cv=5,scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},)
    cv_metrics = {"R2": [float(cv["test_r2"].mean()), float(cv["test_r2"].std())],
        "RMSE": [float(-cv["test_rmse"].mean()), float(cv["test_rmse"].std())],"MAE": [float(-cv["test_mae"].mean()), float(cv["test_mae"].std())],}
    print("Cross-validation 5 lần:", cv_metrics)

    #Phân bố sai số trên test
    resid = y_test.to_numpy() - np.asarray(best_pred)
    high_mask, low_mask = y_test.to_numpy() > 8, y_test.to_numpy() < 3
    error_info = {"within1": float((np.abs(resid) <= 1).mean() * 100),"within2": float((np.abs(resid) <= 2).mean() * 100),
        "mean_error": float(resid.mean()),"high_cut": 8, "low_cut": 3,"high": {"n": int(high_mask.sum()), "bias": float(resid[high_mask].mean()) if high_mask.any() else 0.0},
        "low": {"n": int(low_mask.sum()), "bias": float(resid[low_mask].mean()) if low_mask.any() else 0.0},}
    print("\nCác phương án đã thử (cùng tập test):")
    train_idx, test_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    experiments = run_experiments(df, train_idx, test_idx, numeric_cols, categorical_cols,selected_is_log=best is log_model)

    #Thông số chuẩn hóa
    num_pipe = coef_pipe.named_steps["pre"].named_transformers_["num"]
    scaler = {c: {"median": float(med), "mean": float(mu), "std": float(sd)}
        for c, med, mu, sd in zip(
            numeric_cols, num_pipe.named_steps["imputer"].statistics_,
            num_pipe.named_steps["scaler"].mean_, num_pipe.named_steps["scaler"].scale_)}
    n_coef = int(len(coef_pipe.named_steps["model"].coef_))

    #Dữ liệu minh họa cho phần tương tác
    demo = None
    if "district_key" in df.columns:
        top_key = df["district_key"].value_counts().index[0]
        sub = df[df["district_key"] == top_key]
        sub = sub.sample(min(100, len(sub)), random_state=RANDOM_STATE)
        demo = {"place": top_key.replace("|", ", ")[::1],"points": [[round(float(a), 1), round(float(p), 2)] for a, p in zip(sub["area"], sub["price"])]}
        city_, district_ = top_key.split("|")
        demo["place"] = f"{district_}, {city_}"
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    save_plots(y_test, best_pred, coef_pipe, feature_cols)

    #Thông tin cho giao diện web
    options = {}
    for c in ("legal_status", "furniture_state", "house_direction", "balcony_direction"):
        if c in df.columns:
            options[c] = sorted(df[c].dropna().astype(str).unique().tolist())
    locations = {}
    if "ward" in df.columns:
        dcount = df.groupby(["city", "district"]).size()
        wcount = df.groupby(["city", "district", "ward"]).size()
        wards_of = {}
        for (c, d, w), k in wcount.items():
            if k >= 3:
                wards_of.setdefault((c, d), []).append(w)
        for (c, d), k in dcount.items():
            if k >= 5:
                locations.setdefault(c, {})[d] = sorted(wards_of.get((c, d), []))
        locations = {c: dict(sorted(ds.items())) for c, ds in sorted(locations.items())}
    meta = {"model_name": best_name, "numeric_cols": numeric_cols, "form_numeric": [c for c in numeric_cols if c != "log_area"],
            "categorical_cols": categorical_cols, "options": options, "locations": locations, "missing_label": MISSING_LABEL,
            "defaults": {c: float(df[c].median()) for c in numeric_cols if df[c].notna().any()}, 
            "price_unit": "tỷ VNĐ", "price_min": float(df["price"].min()), "price_max": float(df["price"].max()), 
            "n_train": len(X_train), "n_test": len(X_test),
            "metrics": best_metrics, "metrics_base": base_metrics, "metrics_log": log_metrics, "metrics_train": train_metrics, "metrics_cv": cv_metrics,
            "experiments": experiments,"error_info": error_info, "scaler": scaler,"n_coef": n_coef,"n_num_coef": len(numeric_cols),"data": dict(STATS),"demo": demo,}
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu: {MODEL_PATH.name}, {META_PATH.name}, biểu đồ trong static/images/")

if __name__ == "__main__":
    main()