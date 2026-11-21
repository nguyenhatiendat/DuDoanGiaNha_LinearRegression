# HousePricePrediction

> **Dự đoán giá nhà bằng Linear Regression.** Web nhỏ cho phép nhập thông tin một căn nhà và xem giá dự đoán, mô hình học từ Vietnam Housing Dataset 2024.

## Giới thiệu

Đây là bài tập lớn môn Học máy và ứng dụng. Mình dùng thuật toán **Linear Regression** (hồi quy tuyến tính) để tìm mối quan hệ giữa các thông tin của căn nhà (diện tích, số tầng, số phòng, vị trí, pháp lý...) và giá bán. Sau khi học xong, người dùng nhập thông tin một căn nhà mới và mô hình trả về giá ước tính. Phần giao diện làm bằng **Flask**.

Dữ liệu lấy từ Kaggle: [Vietnam Housing Dataset 2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024). File gốc có 30.229 tin rao bán, sau khi làm sạch còn 29.319 tin. Giá tính theo đơn vị tỷ đồng.

## Chức năng chính

- **Nhập thông tin nhà:** chọn tỉnh, quận, phường rồi nhập diện tích, mặt tiền, đường vào, số tầng, số phòng ngủ, số phòng tắm, pháp lý, nội thất, hướng nhà, hướng ban công. Chỉ có tỉnh, quận và diện tích là bắt buộc, ô nào chưa biết thì để trống.
- **Dự đoán giá:** trả về giá ước tính kèm khoảng sai số theo MAE của mô hình.
- **Cảnh báo:** nếu giá dự đoán nằm ngoài khoảng giá mô hình đã học (1,45 đến 10 tỷ) thì web báo kết quả kém tin cậy.
- **Giải thích giá:** trang kết quả có bảng cho biết giá được cộng dồn từ đâu (giá nền của khu vực, rồi mỗi thông tin cộng thêm hoặc trừ bớt bao nhiêu).
- **Xem độ chính xác:** trang chủ có các chỉ số MAE, RMSE, R² và biểu đồ giá thực tế so với giá dự đoán.
- **Trang giải thích mô hình (`/giai-thich`):** trình bày từ dữ liệu, tiền xử lý, nguyên lý Linear Regression (kèm công thức), chỉ số MAE/MSE/RMSE/R², kết quả train/test/cross-validation đến các phương án đã thử. Có phần kéo thử đường thẳng để thấy mô hình "học" là gì. Số liệu trên trang lấy từ lần huấn luyện gần nhất nên luôn khớp với mô hình đang chạy.
- **Phân tích dữ liệu:** notebook `eda.ipynb` có các biểu đồ khám phá dữ liệu.

## Thuật toán Linear Regression

**Ý tưởng.** Linear Regression (hồi quy tuyến tính) coi giá nhà là tổng có trọng số của các thông tin đầu vào. Mỗi thông tin `x` (diện tích, số tầng, vị trí...) có một hệ số `w` cho biết nó ảnh hưởng đến giá nhiều hay ít, cộng thêm một số tự do `w0`:

```text
ŷ = w0 + w1·x1 + w2·x2 + ... + wn·xn
```

Trong đó `ŷ` là giá dự đoán, `x` là thông tin của căn nhà, `w` là các hệ số mà mô hình tự học từ dữ liệu.

**Hàm mất mát.** Để biết bộ `w` nào tốt, mô hình lấy giá dự đoán so với giá thật của từng căn trong tập train, rồi tính trung bình bình phương sai số (MSE):

```text
J(w) = (1/n) · Σ (yi − ŷi)²        với i = 1 ... n
```

Sai số bị bình phương nên căn nào dự đoán lệch nhiều sẽ bị phạt nặng hơn. Huấn luyện chính là đi tìm `w` làm cho `J(w)` nhỏ nhất.

**Cách tìm w.** Scikit-learn giải thẳng bài toán bình phương tối thiểu, về bản chất là `w = (XᵀX)⁻¹ Xᵀ y`, nên không phải chọn tốc độ học hay số vòng lặp. Cách còn lại là Gradient Descent, cập nhật dần `w ← w − α·∇J(w)`, cho cùng đáp số nhưng phải chọn `α`.

**Tham số và siêu tham số.**

| | Là gì | Trong bài này |
|---|---|---|
| Tham số | Giá trị mô hình tự học khi huấn luyện | `w0` và 856 hệ số `w` (7 hệ số cho thông tin dạng số, 849 hệ số cho các cột quận, phường, pháp lý, nội thất, hướng sau khi mã hóa) |
| Siêu tham số | Giá trị mình đặt trước khi huấn luyện | Tỉ lệ test 20%, `random_state = 42`, điền thiếu bằng median, `min_frequency = 5` của OneHotEncoder, ngưỡng cắt ngoại lai 1% đến 99% |

Linear Regression thuần gần như không có siêu tham số để chỉnh, nên phần lớn công sức nằm ở việc xử lý dữ liệu và chọn đặc trưng.

**Ví dụ tính một căn nhà.** Căn 56 m², mặt tiền 4,5 m, 5 tầng, 3 phòng ngủ, 3 phòng tắm ở Cầu Giấy, Hà Nội (các ô còn lại để trống). Mô hình cộng dồn như sau:

| Phần | Đóng góp (tỷ đồng) |
|---|---|
| Giá nền của khu vực (`w0` + hệ số quận + hệ số phường) | 8,65 |
| Diện tích | −0,04 |
| Mặt tiền | 0,00 |
| Đường vào | −0,05 |
| Số tầng | +0,66 |
| Số phòng ngủ | 0,00 |
| Số phòng tắm | −0,05 |
| Pháp lý, nội thất, hướng | +0,01 |
| **Giá dự đoán** | **9,18** |

Lấy riêng số tầng: trong dữ liệu trung bình có 3,39 tầng, độ lệch chuẩn 1,25. Căn này 5 tầng nên giá trị sau chuẩn hóa là (5 − 3,39) / 1,25 = 1,29. Hệ số của số tầng là `w = 0,515`, vậy đóng góp là 1,29 × 0,515 ≈ 0,66 tỷ. Các cột số đều được chuẩn hóa nên `w` hiểu là "tăng 1 độ lệch chuẩn thì giá thay đổi bao nhiêu tỷ", việc này không làm đổi giá dự đoán mà chỉ giúp so sánh các hệ số với nhau. Bảng trên cũng chính là bảng trong trang kết quả của web.

Phần "giá nền" gộp `w0` với hệ số quận và phường vì các cột quận cộng lại luôn bằng 1, nên không tách riêng `w0` ra để đọc được.

**Đọc các hệ số.**
- Vị trí quan trọng nhất. Chỉ cần thêm quận và phường là R² tăng từ 0,34 lên 0,57 (xem bảng so sánh ở dưới).
- Số tầng có hệ số dương rõ: nhiều hơn khoảng 1,25 tầng thì giá tăng khoảng 0,5 tỷ, khi các thông tin khác như nhau.
- Hệ số số phòng ngủ gần bằng 0. Lý do là số phòng ngủ và số phòng tắm tương quan mạnh (0,78), phần ảnh hưởng đã được tính vào số phòng tắm.
- Diện tích và log(diện tích) tương quan 0,95, gần như là một thông tin. Hệ số của hai cột này (−0,52 và +1,94) không đọc riêng được, phải đọc gộp lại.

**Hạn chế của Linear Regression.**
- Giả sử quan hệ giữa thông tin và giá là đường thẳng. Giá không tăng đều theo diện tích nên mình thêm cột `log(diện tích)`.
- Nhạy với giá trị ngoại lai, vì vậy phải cắt ngoại lai trước khi huấn luyện.
- Nhà càng đắt thì mô hình càng có xu hướng dự đoán thấp (thấy rõ ở biểu đồ giá thực tế và giá dự đoán).
- Khi hai cột quá giống nhau, hệ số từng cột khó đọc.

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
│   ├── result.html         # Trang kết quả
│   ├── giai_thich.html     # Trang giải thích mô hình
│   └── _nav.html           # Thanh điều hướng dùng chung
└── static/
    ├── css/style.css
    ├── css/giai-thich.css  # Giao diện trang giải thích
    ├── js/house.js         # Vẽ ngôi nhà ở đầu trang
    ├── js/demo.js          # Phần kéo đường thẳng ở trang giải thích
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

Mỗi khối lệnh dưới đây là một lệnh, gõ (hoặc dán) từng lệnh rồi nhấn Enter, chờ chạy xong mới sang lệnh tiếp theo.

### 1. (Khuyến nghị) Tạo môi trường ảo

Tạo môi trường:

```bat
python -m venv .venv
```

Kích hoạt môi trường (đầu dòng sẽ hiện thêm `(.venv)`):

```bat
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
| `GET` | `/giai-thich` | Trang giải thích mô hình và kết quả |

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
