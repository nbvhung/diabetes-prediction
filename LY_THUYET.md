# PHẦN 1 — LÝ THUYẾT: HỆ CHẨN ĐOÁN BỆNH TIỂU ĐƯỜNG

> Bài này là **Phần 1 — Lý thuyết** của Assignment 01, theo khung Slide 01: *Understand → Represent → Learn → Experiment → Apply*. Hệ thống được mô tả như một **hệ thông minh thu nhỏ**, không phải tập hợp rời rạc các ví dụ Scikit-learn.

## 1. Định nghĩa hệ thông minh

**Bài toán thực tế:** Hỗ trợ bác sĩ sàng lọc nhanh nguy cơ tiểu đường từ 8 chỉ số lâm sàng đo được trong một lần khám.

```
Môi trường (bệnh nhân) → Đầu vào (8 chỉ số) → Biểu diễn (vector x) → Mô hình học (fθ) → Quyết định (ŷ∈{0,1}) → Ứng dụng (web/mobile)
```

**Đoạn mô tả bắt buộc (dán vào báo cáo):**
> Hệ thống chẩn đoán tiểu đường là một hệ thống học có giám sát dạng **phân loại nhị phân**. Hệ thống tiếp nhận 8 chỉ số y tế của một bệnh nhân, biểu diễn chúng thành vector đặc trưng `x ∈ R⁸`, học quan hệ `ŷ = fθ(x)` từ dữ liệu Pima Indians Diabetes, và đưa ra dự đoán `ŷ = 1` (dương tính, có nguy cơ) hoặc `ŷ = 0` (âm tính, bình thường) kèm xác suất, phục vụ quyết định sàng lọc ban đầu.

**Sơ đồ hệ thống (vẽ trong Word):**
```
[BN: Glucose 148, BMI 33.6, ...] → [ x=[6,148,72,35,0,33.6,0.627,50] ] → [Scaler → LogisticRegression/KNN/SVM/RandomForest] → [ŷ=1, P=78.7%]
```

## 2. Nguồn và mô tả dữ liệu

- **Nguồn:** `diabetes.csv` — Pima Indians Diabetes Database (Kaggle / UCI), bản chuẩn 768 quan sát.
- **Một quan sát:** một bệnh nhân nữ.
- **Đặc trưng (d=8, đều là Numerical, continuous):**

| # | Tên | Ý nghĩa y học | Kiểu |
|---|-----|---------------|------|
| 1 | Pregnancies | Số lần mang thai | số đếm |
| 2 | Glucose | Nồng độ glucose huyết tương (mg/dL) | thực |
| 3 | BloodPressure | Huyết áp tâm trương (mm Hg) | thực |
| 4 | SkinThickness | Độ dày nếp gấp da cơ tam đầu (mm) | thực |
| 5 | Insulin | Insulin huyết thanh 2h (mu U/ml) | thực |
| 6 | BMI | Chỉ số khối cơ thể | thực |
| 7 | DiabetesPedigreeFunction | Hàm phả hệ tiểu đường (yếu tố di truyền) | thực |
| 8 | Age | Tuổi | số nguyên |

- **Target:** `Outcome ∈ {0,1}` — 0 âm tính, 1 dương tính. **Categorical, nhị phân → bài toán phân loại.**
- **Quy mô:** N=768, d=8, không có đặc trưng categorical gốc (không cần one-hot; chỉ cần chuẩn hóa).
- **Tiền xử lý:** Thay 0 bất hợp lý (Glucose, BloodPressure, SkinThickness, Insulin, BMI = 0) bằng NaN → điền median; chuẩn hóa `StandardScaler` cho KNN/SVM/Logistic.

**8 câu hỏi dataset (trả lời trong notebook):**
1. Hiện tượng: nguy cơ tiểu đường. 2. Một quan sát: một bệnh nhân. 3. Features: 8 chỉ số trên. 4. Target: Outcome. 5. Categorical. 6. Classification. 7. N=768. 8. d=8. 9. Tất cả numerical. 10. Không có categorical.

## 3. Biểu diễn dữ liệu

Một bệnh nhân được trừu tượng hóa thành:

```
xᵢ = [x₁, x₂, ..., x₈] = [Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age] ∈ R⁸
```

Tập huấn luyện:

```
D = {(xᵢ, yᵢ)}ᴺᵢ₌₁ ,  X ∈ Rᴺˣ⁸
```

Bảng biểu diễn (dán vào báo cáo):

| Feature | Type | Representation | Meaning |
|---------|------|----------------|---------|
| Age | Numerical | real value | Tuổi bệnh nhân |
| BMI | Numerical | real value | Chỉ số khối cơ thể |
| ... | ... | ... | ... |
| Outcome | Categorical | binary {0,1} | Nhãn |

**Raw feature ≠ encoded feature ≠ model input:** Giá trị thô sau khi điền khuyết và chuẩn hóa mới là model input `x̃ = (x - μ)/σ`.

## 4. Phát biểu bài toán học

```
D = {(xᵢ, yᵢ)}ᴺᵢ₌₁ ,  ŷ = fθ(x) ,  θ* = argmin_θ (1/N) Σ ℓ(fθ(xᵢ), yᵢ)
```

- Input: vector 8 chiều đã chuẩn hóa.
- Target: nhãn nhị phân.
- Model: một trong 4 mô hình dưới đây.
- Tham số: vector trọng số w, bias b (Logistic), ngưỡng k, siêu phẳng, tập cây...
- Prediction: xác suất P(y=1|x) → ngưỡng 0.5 → nhãn.
- Loss: Binary Cross-Entropy (Logistic), Hinge (SVM), Gini (Tree).
- Training/Testing: chia 80/20, stratify để giữ tỉ lệ dương tính.
- Generalization: đánh giá trên tập test chưa từng thấy.

**Câu một-sentence (bắt buộc):**
> *Cho vector 8 chỉ số đã chuẩn hóa của một bệnh nhân chưa từng thấy, hãy dự đoán nhãn ŷ ∈ {0,1} về nguy cơ tiểu đường.*

**Phân loại vs hồi quy:** Đây là **phân loại** vì target rời rạc; nếu dự đoán giá trị liên tục (giá nhà) thì là hồi quy.

## 5. Chia Train/Test và Baseline

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

Test set tuyệt đối không dùng để train — nó ước lượng khả năng tổng quát hóa.

**Baseline (R6 bắt buộc):**

```python
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
```

Baseline luôn đoán lớp đa số (≈65% âm tính) → Accuracy ≈65%. Mọi mô hình học được phải vượt qua mốc này mới có ý nghĩa.

## 6. Bốn mô hình truyền thống

| Mô hình | Công thức cốt lõi | Tham số học được | Tiêu chí học | Giả định | Mạnh | Yếu |
|---------|-------------------|------------------|--------------|----------|------|-----|
| **Logistic Regression** | z=wᵀx+b, P=σ(z)=1/(1+e⁻ᶻ) | w∈R⁸, b | Cực đại likelihood / BCE | Quan hệ tuyến tính giữa x và log-odds | Xác suất có thể giải thích, nhẹ, ít overfit | Chỉ biên tuyến tính |
| **K-Nearest Neighbors** | d(x,xᵢ)=√Σ(xⱼ−xᵢⱼ)², vote K láng giềng | K, metric | Không có giai đoạn học tham số | Bệnh nhân gần nhau trong không gian đặc trưng thì cùng nhãn | Không giả định phân phối, dễ hiểu | Nhạy thang đo, chậm khi N lớn, cần chuẩn hóa |
| **Support Vector Machine** | max margin: min ½‖w‖² s.t. yᵢ(wᵀxᵢ+b)≥1 | w,b, support vectors | Hinge loss + margin | Tồn tại siêu phẳng phân tách với lề lớn | Tổng quát tốt, hiệu quả với d nhỏ | Nhạy C, kernel, cần scale |
| **Random Forest** | ŷ=Vote(T₁,...,Tʙ) | Tập B cây, ngưỡng split | Gini/Entropy, Bagging + random feature | Tổ hợp nhiều cây giảm phương sai | Chống overfit tốt hơn cây đơn, xử lý phi tuyến | Kém giải thích, nặng hơn |

*Notebook thực tế chạy đủ 4 mô hình trên, lưu cross-val và test accuracy để so sánh.*

## 7. Ba thí nghiệm có kiểm soát (R8)

**TN1 — So sánh mô hình (cùng protocol, cùng split, cùng scaler):**
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic | — | — | — | — |
| KNN | — | — | — | — |
| SVM | — | — | — | — |
| Random Forest | — | — | — | — |
*Điền số từ notebook; kèm Confusion Matrix. Trả lời: mô hình nào tổng quát nhất? Vì sao?*

**TN2 — Điều tra siêu tham số (đặt câu hỏi trước):**
*“Tăng K trong KNN từ 3 → 15 có cải thiện F1 không? Hay K lớn làm mịn quá mức?”* — Vẽ đường cong Accuracy theo K.

**TN3 — Điều tra biểu diễn (Slide 01):**
*“Chuẩn hóa có ảnh hưởng gì?”* — So sánh `X_raw vs X_standardized` trên KNN và SVM (Logistic ít nhạy hơn). Giải thích: KNN/SVM dùng khoảng cách → không chuẩn hóa thì Glucose (~100) át Age (~30).

## 8. Đánh giá

Với phân loại nhị phân phải báo cáo đủ:

```
Accuracy  = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)   — trong số dự đoán dương, bao nhiêu đúng
Recall    = TP/(TP+FN)   — trong số dương thật, bắt được bao nhiêu
F1 = 2PR/(P+R)
Confusion Matrix 2×2
```

*Giải thích chọn metric:* Với y tế, **Recall quan trọng** (bỏ sót ca dương tính nguy hiểm hơn báo nhầm), nên cân bằng với Precision qua F1.

## 9. Từ mô hình sang ứng dụng thông minh

```
Input (form web/mobile) → Vector x → Scaler transform → model.predict_proba → ŷ + confidence → Hiển thị
```

```python
def predict(model, scaler, sample):
    x = scaler.transform([sample])
    proba = model.predict_proba(x)[0]
    return int(proba.argmax()), float(proba.max())
```

Ứng dụng (FastAPI `api.py` + `web.html` + `mobile/App.js`) thể hiện đúng pipeline trên, demo ít nhất 3 ca: một ca âm tính rõ, một ca dương tính rõ, một ca biên (xác suất ~50%).

## 10. Suy ngẫm bắt buộc (R13)

1. Hệ thống nhận gì? 8 chỉ số đo được.
2. Biểu diễn nội bộ? Vector R⁸ đã chuẩn hóa.
3. Mô hình học gì? Quan hệ thống kê giữa chỉ số và nhãn từ ví dụ.
4. Dự đoán gì? Nhãn + xác suất.
5. Vì sao xử lý được ca chưa thấy? Vì học được biên quyết định tổng quát, không ghi nhớ từng ca.
6. Phần nào gọi là “thông minh”? Khả năng khái quát hóa và dự đoán có cơ sở xác suất, vượt baseline.
7. Hạn chế? Chỉ 8 số, mất thông tin bệnh sử chi tiết, hình ảnh, gen; không giải thích nhân quả; phụ thuộc chất lượng đo.
8. Mô hình ≠ hệ thống hoàn chỉnh: hệ thống còn cần nhập liệu, biểu diễn, tiền xử lý, API, giao diện, xử lý lỗi.

**Suy ngẫm về biểu diễn:**
- Phù hợp vì dữ liệu có cấu trúc, vector đủ cho ML truyền thống.
- Giữ lại giá trị số học, mất thông tin phi cấu trúc (triệu chứng mô tả, tiền sử văn bản).
- Có thể biểu diễn dạng ảnh (scan), chuỗi thời gian (theo dõi glucose), đồ thị (quan hệ bệnh nhân-bác sĩ), embedding học được — mỗi dạng sẽ đòi hỏi CNN/RNN/GNN khác nhau.

## 11. Vị trí trong lịch sử AI

```
Ký hiệu/Rules → Feature vector + Statistical ML (hệ này) → Tensor + Deep Learning → Embedding + Foundation → Interactive/Agentic
```

Hệ này thuộc giai đoạn **Feature Engineering + Traditional ML**, là nền tảng trước khi học representation tự động.

---
**Ghi chú để dán vào Word:** Mỗi biểu đồ trong notebook (heatmap tương quan, phân phối Glucose/BMI theo Outcome) phải có một đoạn văn giải thích ngay dưới ảnh. Link deployed dán ở trang bìa và mục Ứng dụng.
