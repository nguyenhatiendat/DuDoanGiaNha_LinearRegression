"""
app.py - Ứng dụng web Flask dự đoán giá nhà.

Chạy:  python app.py   rồi mở http://127.0.0.1:5000
(cần chạy train_model.py trước để có models/linear_regression.pkl)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "linear_regression.pkl"
META_PATH = BASE_DIR / "models" / "meta.json"

app = Flask(__name__)

LABELS = {
    "area": "Diện tích (m²)",
    "frontage": "Mặt tiền (m)",
    "access_road": "Đường vào (m)",
    "floors": "Số tầng",
    "bedrooms": "Số phòng ngủ",
    "bathrooms": "Số phòng tắm",
    "city": "Tỉnh / Thành phố",
    "district": "Quận / Huyện",
    "ward": "Phường / Xã",
    "legal_status": "Pháp lý",
    "furniture_state": "Nội thất",
    "house_direction": "Hướng nhà",
    "balcony_direction": "Hướng ban công",
}

# Dataset ghi một số giá trị bằng tiếng Anh. Mô hình vẫn học bằng giá trị gốc,
# giao diện chỉ hiển thị bản tiếng Việt.
VALUE_LABELS = {
    "legal_status": {"Have certificate": "Đã có sổ", "Sale contract": "Hợp đồng mua bán"},
    "furniture_state": {"Basic": "Cơ bản", "Full": "Đầy đủ"},
}

model = None
meta = None


def load_model():
    global model, meta
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise SystemExit("Chưa có mô hình. Hãy chạy: python train_model.py")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))


def parse_float(value):
    value = (value or "").strip().replace(",", ".")
    return float(value) if value else np.nan


@app.template_filter("vn_price")
def vn_price(value):
    """1.7 -> '1,70 tỷ đồng'; 0.85 -> '850 triệu đồng'."""
    if value < 1:
        return f"{round(value * 1000):,} triệu đồng".replace(",", ".")
    return f"{value:.2f}".replace(".", ",") + " tỷ đồng"


@app.template_filter("vn_num")
def vn_num(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def render_form(**kwargs):
    return render_template("index.html", meta=meta, labels=LABELS,
                           value_labels=VALUE_LABELS, **kwargs)


@app.route("/")
def index():
    return render_form()


@app.route("/predict", methods=["POST"])
def predict():
    row, shown = {}, []
    try:
        for col in meta["form_numeric"]:
            row[col] = parse_float(request.form.get(col))
            if not np.isnan(row[col]):
                shown.append((LABELS.get(col, col), f"{row[col]:g}"))
        if np.isnan(row["area"]) or row["area"] <= 0:
            raise ValueError("Diện tích phải là số lớn hơn 0.")
        row["log_area"] = float(np.log(row["area"]))

        city = (request.form.get("city") or "").strip()
        district = (request.form.get("district") or "").strip()
        ward = (request.form.get("ward") or "").strip()
        if not city or not district:
            raise ValueError("Hãy chọn tỉnh/thành phố và quận/huyện.")
        row["district_key"] = f"{city}|{district}"
        row["ward_key"] = f"{city}|{district}|{ward}" if ward else np.nan
        place = [p for p in (ward, district, city) if p]

        for col in meta["options"]:
            val = (request.form.get(col) or "").strip()
            row[col] = val if val else np.nan
            if val:
                shown.append((LABELS.get(col, col), VALUE_LABELS.get(col, {}).get(val, val)))
    except ValueError as exc:
        return render_form(error=str(exc) or "Dữ liệu nhập không hợp lệ.",
                           form=request.form), 400

    X = pd.DataFrame([row], columns=meta["numeric_cols"] + meta["categorical_cols"])
    price = max(float(model.predict(X)[0]), 0.0)

    mae = meta["metrics"]["MAE"]
    lo_price, hi_price = max(price - mae, 0.0), price + mae
    # vị trí trên thang giá của dữ liệu (để vẽ thanh minh họa)
    span = meta["price_max"] - meta["price_min"]

    def pct(v):
        return float(min(max((v - meta["price_min"]) / span * 100, 0), 100))

    return render_template(
        "result.html", meta=meta, price=price, lo=lo_price, hi=hi_price, mae=mae,
        pos=pct(price), pos_lo=pct(lo_price), pos_hi=pct(hi_price),
        place=", ".join(place), inputs=shown,
        floors=None if np.isnan(row["floors"]) else float(row["floors"]),
        frontage=None if np.isnan(row["frontage"]) else float(row["frontage"]),
        out_of_range=price > meta["price_max"] or price < meta["price_min"],
    )


if __name__ == "__main__":
    load_model()
    app.run(debug=True)
