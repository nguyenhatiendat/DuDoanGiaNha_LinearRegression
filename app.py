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

# Ý nghĩa từng cột trong dataset (tên cột lấy theo file CSV)
COLUMN_INFO = {
    "address": ("Address", "Địa chỉ đầy đủ: đường, phường/xã, quận/huyện, tỉnh/thành"),
    "area": ("Area", "Diện tích (m²)"),
    "frontage": ("Frontage", "Chiều rộng mặt tiền (m)"),
    "access_road": ("Access Road", "Chiều rộng đường vào nhà (m)"),
    "house_direction": ("House direction", "Hướng nhà"),
    "balcony_direction": ("Balcony direction", "Hướng ban công"),
    "floors": ("Floors", "Số tầng"),
    "bedrooms": ("Bedrooms", "Số phòng ngủ"),
    "bathrooms": ("Bathrooms", "Số phòng tắm"),
    "legal_status": ("Legal status", "Pháp lý: Have certificate (đã có sổ) hoặc Sale contract (hợp đồng mua bán)"),
    "furniture_state": ("Furniture state", "Nội thất: Basic (cơ bản) hoặc Full (đầy đủ)"),
    "price": ("Price", "Giá bán (tỷ đồng). Đây là cột cần dự đoán"),
}

# Căn nhà mẫu để minh họa cách mô hình tính trong trang giải thích
EXAMPLE_VALUES = {"area": 56.0, "frontage": 4.5, "floors": 5.0, "bedrooms": 3.0, "bathrooms": 3.0}
EXAMPLE_PLACE = ("Hà Nội", "Cầu Giấy")

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


@app.template_filter("vn_int")
def vn_int(value):
    """29319 -> '29.319'."""
    return f"{int(round(value)):,}".replace(",", ".")


@app.template_filter("vn_signed")
def vn_signed(value):
    """0.66 -> '+0,66', -0.05 -> '−0,05' (đơn vị tỷ đồng)."""
    shown = round(value, 2)
    if shown == 0:  # tránh hiện "−0,00"
        return "0,00"
    sign = "+" if shown > 0 else "−"
    return f"{sign}{abs(shown):.2f}".replace(".", ",")


# Mỗi hệ số w của mô hình thuộc về một nhóm thông tin, dùng để gộp lại cho dễ đọc
GROUPS = [
    ("area", "Diện tích", ("area", "log_area")),
    ("frontage", "Mặt tiền", ("frontage",)),
    ("access_road", "Đường vào", ("access_road",)),
    ("floors", "Số tầng", ("floors",)),
    ("bedrooms", "Số phòng ngủ", ("bedrooms",)),
    ("bathrooms", "Số phòng tắm", ("bathrooms",)),
    ("other", "Pháp lý, nội thất, hướng",
     ("legal_status", "furniture_state", "house_direction", "balcony_direction")),
]


def explain(X):
    """
    Tách giá dự đoán thành các phần cộng dồn: y = w0 + w1*x1 + ... + wn*xn.
    Chỉ làm được khi mô hình là Linear Regression thuần (không biến đổi log giá).
    Trả về None nếu không tách được.
    """
    if not hasattr(model, "named_steps"):
        return None
    pre, lr = model.named_steps["pre"], model.named_steps["model"]
    Xt = pre.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    parts = Xt[0] * lr.coef_  # w_i * x_i của từng cột sau khi mã hóa

    # w0 + các cột quận/phường gộp lại thành "giá nền" của khu vực.
    # (Các cột one-hot cộng lại luôn bằng 1 nên không tách riêng w0 được.)
    base = float(lr.intercept_)
    totals = {key: 0.0 for key, _, _ in GROUPS}
    for name, value in zip(pre.get_feature_names_out(), parts):
        col = name.split("__", 1)[1]
        if col.startswith(("district_key", "ward_key")):
            base += float(value)
            continue
        for key, _, prefixes in GROUPS:
            if any(col == p or col.startswith(p + "_") for p in prefixes):
                totals[key] += float(value)
                break

    rows = [{"label": label, "value": totals[key]} for key, label, _ in GROUPS]
    biggest = max((abs(r["value"]) for r in rows), default=0.0)
    for r in rows:  # độ dài thanh: tối đa 50% bề rộng (một nửa thanh trục giữa)
        r["bar"] = abs(r["value"]) / biggest * 50 if biggest else 0.0
    return {"base": base, "rows": rows, "total": base + sum(totals.values())}


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
        explain=explain(X),
        floors=None if np.isnan(row["floors"]) else float(row["floors"]),
        frontage=None if np.isnan(row["frontage"]) else float(row["frontage"]),
        out_of_range=price > meta["price_max"] or price < meta["price_min"],
    )


def build_example():
    """Tính giá và cách cộng dồn cho căn nhà mẫu."""
    city, district = EXAMPLE_PLACE
    row = {c: EXAMPLE_VALUES.get(c, np.nan) for c in meta["form_numeric"]}
    row["log_area"] = float(np.log(row["area"]))
    row["district_key"] = f"{city}|{district}"
    row["ward_key"] = np.nan
    for col in meta["options"]:
        row[col] = np.nan
    X = pd.DataFrame([row], columns=meta["numeric_cols"] + meta["categorical_cols"])

    floors = None
    if hasattr(model, "named_steps") and "floors" in meta.get("scaler", {}):
        pre, lr = model.named_steps["pre"], model.named_steps["model"]
        names = list(pre.get_feature_names_out())
        if "num__floors" in names:
            s = meta["scaler"]["floors"]
            value = float(row["floors"])
            z = (value - s["mean"]) / s["std"]
            w = float(lr.coef_[names.index("num__floors")])
            floors = {"value": value, "mean": s["mean"], "std": s["std"], "z": z, "w": w, "contrib": z * w}

    return {
        "place": f"{district}, {city}",
        "values": EXAMPLE_VALUES,
        "price": max(float(model.predict(X)[0]), 0.0),
        "breakdown": explain(X),
        "floors": floors,
    }


@app.route("/giai-thich")
def explain_page():
    info = meta.get("data", {})
    columns = [
        {**c, "csv_name": COLUMN_INFO.get(c["col"], (c["col"], ""))[0],
         "meaning": COLUMN_INFO.get(c["col"], (c["col"], ""))[1]}
        for c in info.get("columns", [])
    ]
    return render_template("giai_thich.html", meta=meta, data=info, columns=columns,
                           ex=build_example(), labels=LABELS)


if __name__ == "__main__":
    load_model()
    app.run(debug=True)