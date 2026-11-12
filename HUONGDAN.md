# Hướng dẫn chạy

## Chạy web (mô hình đã có sẵn)

Mở terminal trong thư mục `HousePricePrediction` (thư mục có file `app.py`), gõ:

```
python -m pip install -r requirements.txt
python app.py
```

Mở trình duyệt vào http://127.0.0.1:5000. Tắt web bằng `Ctrl + C`.

## Huấn luyện lại mô hình (nếu muốn)

```
python train_model.py
```

Lệnh này đọc `data/gia_nha.csv`, in ra MAE, MSE, RMSE, R² và tạo lại:
- `models/linear_regression.pkl` và `models/meta.json`
- 3 biểu đồ trong `static/images/`

Chạy xong thì chạy lại `python app.py`.

## Xem phân tích dữ liệu

Mở `notebooks/eda.ipynb` trong VS Code, chọn Python ở "Select Kernel", bấm Run All.
Nếu thiếu thư viện: `python -m pip install seaborn ipykernel`

## Lỗi thường gặp

| Lỗi | Cách sửa |
|---|---|
| `No module named 'flask'` (hoặc thư viện khác) | Chạy `python -m pip install -r requirements.txt` |
| `Could not open requirements file` | Terminal đang sai thư mục, dùng `cd` vào thư mục có `app.py` |
| `Chưa có mô hình` | Chạy `python train_model.py` trước |
| Web hiện bản cũ | Bấm `Ctrl + F5` |
| Cổng 5000 đang bị dùng | Tắt server cũ bằng `Ctrl + C` rồi chạy lại |
