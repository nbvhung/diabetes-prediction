# 🩺 Hệ Chẩn đoán Bệnh Tiểu đường

Hệ thống Machine Learning dự đoán nguy cơ tiểu đường từ 8 chỉ số xét nghiệm (dataset Pima Indians Diabetes). Mô hình **Logistic Regression** kết hợp **StandardScaler**, triển khai đầy đủ dạng **Web**, **API** và **App Mobile (Android)**.

**Link deployed (Render):** https://diabetes-prediction-z0gh.onrender.com

## Kiến trúc

```
[Web HTML]  [App React Native]
      \        /
       fetch POST /predict (JSON)
            |
        [FastAPI api.py]
            |
   [LogisticRegression + StandardScaler]
   (diabetes_model.sav + scaler.sav)
```

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Machine Learning | scikit-learn 1.7.2 (Logistic Regression) |
| Backend API | Python 3.11, FastAPI, Uvicorn |
| Web | HTML/CSS/JS (do FastAPI serve trực tiếp) |
| Mobile | React Native — Expo SDK 54 |
| Deploy | Render (Docker-free, auto deploy từ GitHub) |

## Cấu trúc thư mục

```
.
├── api.py              # FastAPI: serve web + API POST /predict
├── web.html            # Giao diện web (responsive, dùng được trên điện thoại)
├── diabetes_model.sav  # Model LogisticRegression đã train
├── scaler.sav          # StandardScaler đã fit
├── diabetes.csv        # Dataset Pima Indians Diabetes (768 mẫu, 8 đặc trưng)
├── diabetes.ipynb      # Notebook: EDA, biểu đồ, train, đánh giá mô hình
├── requirements.txt    # Dependencies (pin đúng phiên bản lúc train)
└── mobile/             # App Android (React Native + Expo)
    ├── App.js          # Form nhập 8 chỉ số + gọi API + hiện kết quả
    ├── api.js          # ⚙️ URL của server (đổi tại đây khi cần)
    ├── components/Field.js
    ├── START_EXPO.bat  # Nhấp đúp để chạy server Expo (Windows)
    └── app.json
```

## Chạy local — Backend + Web (Anaconda Prompt)

```bat
conda create -n diabetes-ml python=3.11 -y
conda activate diabetes-ml
cd /d "duong-dan-den-thu-muc-diabetes-prediction"
pip install -r requirements.txt
python -m uvicorn api:app --reload --port 8000
```

- Web UI: http://127.0.0.1:8000
- Tài liệu API tự động (Swagger): http://127.0.0.1:8000/docs

## Chạy local — Notebook

```bat
conda activate diabetes-ml
jupyter notebook diabetes.ipynb
```

## Chạy local — App Mobile

Yêu cầu: Node.js ≥ 20, app **Expo Go** (bản hỗ trợ **SDK 54**) trên Android.

```bat
cd mobile
npm install
npx expo start --tunnel
```

- Quét mã QR bằng app Expo Go (chế độ tunnel chạy được kể cả khác Wi-Fi)
- Điện thoại cùng Wi-Fi với máy tính có thể bỏ `--tunnel`
- Windows: nhấp đúp `START_EXPO.bat` là chạy luôn
- Đổi địa chỉ server: sửa `mobile/api.js`

## API

`POST /predict`

```json
// Request
{
  "pregnancies": 6, "glucose": 148, "blood_pressure": 72,
  "skin_thickness": 35, "insulin": 0, "bmi": 33.6,
  "diabetes_pedigree": 0.627, "age": 50
}

// Response
{
  "prediction": 1,
  "label": "Dương tính (Có nguy cơ Tiểu đường)",
  "confidence_score": 78.67,
  "message": "Chẩn đoán: Dương tính (Xác suất: 78.67%)"
}
```

## Deploy lên Render

| Cấu hình | Giá trị |
|---|---|
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Environment | `PYTHON_VERSION=3.11.9` |
| Instance | Free |

> ⚠️ Gói Free tự ngủ sau ~15 phút không dùng — request đầu tiên mất ~1 phút để server thức dậy.

## Build APK (EAS Build)

```bat
cd mobile
npm install -g eas-cli
eas login
eas build -p android --profile preview
```

File `.apk` tải về cài trực tiếp trên Android, không cần Expo Go.
