# HousePricePrediction

> **Dự đoán giá nhà bằng Linear Regression.** Web nhỏ cho phép nhập thông tin một căn nhà và xem giá dự đoán, mô hình học từ Vietnam Housing Dataset 2024.

## Giới thiệu

Đây là bài tập lớn môn Học máy và ứng dụng. Mình dùng thuật toán **Linear Regression** (hồi quy tuyến tính) để tìm mối quan hệ giữa các thông tin của căn nhà (diện tích, số tầng, số phòng, vị trí, pháp lý...) và giá bán. Sau khi học xong, người dùng nhập thông tin một căn nhà mới và mô hình trả về giá ước tính. Phần giao diện làm bằng **Flask**.

Dữ liệu lấy từ Kaggle: [Vietnam Housing Dataset 2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024). File gốc có 30.229 tin rao bán, sau khi làm sạch còn 29.319 tin. Giá tính theo đơn vị tỷ đồng.

## Chức năng chính

- **Nhập thông tin nhà:** chọn tỉnh, quận, phường rồi nhập diện tích, mặt tiền, đường vào, số tầng, số phòng ngủ, số phòng tắm, pháp lý, nội thất, hướng nhà, hướng ban công. Chỉ có tỉnh, quận và diện tích là bắt buộc, ô nào chưa biết thì để trống.
- **Dự đoán giá:** trả về giá ước tính kèm khoảng sai số theo MAE của mô hình.
- **Cảnh báo:** nếu giá dự đoán nằm ngoài khoảng giá mô hình đã học (1,45 đến 10 tỷ) thì web báo kết quả kém tin cậy.
- **Xem độ chính xác:** trang chủ có các chỉ số MAE, RMSE, R² và biểu đồ giá thực tế so với giá dự đoán.
- **Phân tích dữ liệu:** notebook `eda.ipynb` có các biểu đồ khám phá dữ liệu.

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python |
| Thư viện học máy | Scikit-learn |
| Xử lý dữ liệu | Pandas, NumPy |
| Lưu mô hình | Joblib |
| Vẽ biểu đồ | Matplotlib, Seaborn |
| Backend | Flask |
| Giao diện | HTML, CSS, JavaScript, Bootstrap |

## Cấu trúc dự án

```text
HousePricePrediction/
├── app.py                  # Web Flask
├── train_model.py          # Làm sạch dữ liệu, huấn luyện, đánh giá, lưu mô hình
├── requirements.txt        # Các thư viện cần cài
├── README.md
├── HUONGDAN.md             # Hướng dẫn chạy nhanh
├── data/
│   └── gia_nha.csv         # Dữ liệu
├── models/
│   ├── linear_regression.pkl   # Mô hình đã huấn luyện
│   └── meta.json               # Danh sách tỉnh/quận/phường và các chỉ số
├── notebooks/
│   └── eda.ipynb           # Phân tích dữ liệu
├── templates/
│   ├── index.html          # Trang nhập thông tin
│   └── result.html         # Trang kết quả
└── static/
    ├── css/style.css
    ├── js/house.js         # Vẽ ngôi nhà ở đầu trang
    └── images/             # Biểu đồ và họa tiết
```

## Yêu cầu

- Python 3.10 trở lên.
- Trình duyệt như Chrome hoặc Edge.
- Cần mạng để tải font chữ và Bootstrap, nếu mất mạng thì web vẫn chạy nhưng dùng font mặc định.

## Cài đặt và chạy trên Windows

Mở Terminal tại thư mục dự án:

```bat
cd HousePricePrediction
```

### 1. (Khuyến nghị) Tạo môi trường ảo

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 2. Cài thư viện

```bat
python -m pip install -r requirements.txt
```

### 3. Chạy web

```bat
python app.py
```

Mở trình duyệt vào **http://127.0.0.1:5000**. Muốn tắt thì quay lại Terminal nhấn `Ctrl + C`.

Mô hình đã được huấn luyện sẵn trong `models/` nên không cần chạy thêm bước nào khác.

## Huấn luyện lại mô hình

```bat
python train_model.py
```

Lệnh này đọc `data/gia_nha.csv`, huấn luyện, in ra các chỉ số đánh giá rồi lưu lại mô hình và biểu đồ. Nếu muốn dùng file dữ liệu khác:

```bat
python train_model.py --data data/ten_file.csv
```

## Xử lý dữ liệu và mô hình

**Làm sạch:**
- Tách cột `Address` thành tỉnh, quận, phường. Gộp các cách viết khác nhau của cùng một tỉnh (ví dụ "HN" thành "Hà Nội"), bỏ chữ "Quận/Huyện" thừa.
- Bỏ các tỉnh chỉ xuất hiện dưới 5 lần (đa phần là tên nhập sai).
- Bỏ giá trị ngoại lai của `Price` và `Area` (ngoài khoảng phân vị 1% đến 99%).
- Thêm đặc trưng `log(Area)` vì giá không tăng đều theo diện tích.

**Tiền xử lý:** cột số điền giá trị thiếu bằng `SimpleImputer` (median) rồi chuẩn hóa bằng `StandardScaler`. Cột phân loại điền "Không rõ" rồi mã hóa bằng `OneHotEncoder`.

**Huấn luyện:** chia 80% train (23.455 căn) và 20% test (5.864 căn), `random_state = 42`.

**Kết quả trên tập test:**

| Chỉ số | Giá trị |
|---|---|
| MAE | 1,00 tỷ |
| MSE | 1,78 |
| RMSE | 1,33 tỷ |
| R² | 0,62 |

Mình có thử so sánh các cách làm với cùng dữ liệu:

| Cách làm | R² |
|---|---|
| Chỉ tách tỉnh | 0,34 |
| Tách thêm quận và phường | 0,57 |
| Thêm `log(Area)` | 0,62 |
| Dự đoán `log(Price)` thay vì giá gốc | 0,60 |

Tách đến quận và phường là thay đổi giúp mô hình tốt lên nhiều nhất. Dự đoán log giá không tốt hơn nên giữ cách dự đoán trực tiếp giá.

## Các đường dẫn của web

| Phương thức | Đường dẫn | Chức năng |
|---|---|---|
| `GET` | `/` | Trang nhập thông tin căn nhà |
| `POST` | `/predict` | Nhận thông tin, trả về trang kết quả |

## Một số lỗi thường gặp

**1. `ModuleNotFoundError: No module named 'flask'`**

Chưa cài thư viện. Chạy lại:

```bat
python -m pip install -r requirements.txt
```

**2. `Could not open requirements file`**

Terminal đang đứng sai thư mục. Dùng `cd` vào thư mục có file `app.py` rồi chạy lại.

**3. `Chưa có mô hình`**

Thiếu file trong `models/`. Chạy `python train_model.py` để tạo lại.

**4. Web vẫn hiện bản cũ**

Nhấn `Ctrl + F5`. Nếu vẫn vậy thì tắt server cũ bằng `Ctrl + C` rồi chạy lại.

## Hạn chế

- R² khoảng 0,62 nên mô hình chỉ ước tính ở mức tham khảo. Giá nhà còn phụ thuộc nhiều thứ dataset không có, như vị trí chính xác trên đường, chất lượng xây dựng, tình trạng nhà.
- Dữ liệu chỉ có nhà từ khoảng 1,45 đến 10 tỷ, nên không dùng để định giá nhà đắt hơn.
- Hai cột hướng nhà và hướng ban công thiếu rất nhiều (70% và 83%), mô hình coi "không ghi" là một nhóm riêng.
- Flask đang chạy bằng server phát triển, chỉ dùng để chạy trên máy cá nhân.

---

**HousePricePrediction**: bài tập lớn môn Học máy và ứng dụng.
