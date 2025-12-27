# Hệ thống Gợi ý Game Steam (Steam Game Recommendation System)

Dự án này là một hệ thống gợi ý game toàn diện dựa trên dữ liệu từ Steam, sử dụng kết hợp nhiều thuật toán gợi ý (Content-based, Hybrid, Context-aware) để đưa ra các đề xuất phù hợp nhất cho người dùng. Hệ thống cũng tích hợp khả năng kiểm tra tương thích cấu hình thiết bị để đảm bảo người chơi có trải nghiệm tốt nhất.

## 📋 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Tính năng nổi bật](#tính-năng-nổi-bật)
- [Quy trình Dữ liệu (Data Pipeline)](#quy-trình-dữ-liệu-data-pipeline)
  - [1. Thu thập dữ liệu (Crawling)](#1-thu-thập-dữ-liệu-crawling)
  - [2. Xử lý dữ liệu (Processing)](#2-xử-lý-dữ-liệu-processing)
- [Thuật toán Gợi ý](#thuật-toán-gợi-ý)
- [Cài đặt và Sử dụng](#cài-đặt-và-sử-dụng)
- [Đánh giá Mô hình](#đánh-giá-mô-hình)
- [Cấu trúc Dự án](#cấu-trúc-dự-án)

---

## 🌟 Giới thiệu

Hệ thống được xây dựng để giải quyết vấn đề "information overload" trên Steam, giúp người dùng tìm kiếm game mới dựa trên sở thích cá nhân, lịch sử xem, và đặc biệt là cấu hình máy tính của họ.

## 🚀 Tính năng nổi bật

- **Đa dạng thuật toán**: Hỗ trợ Content-based (dựa trên nội dung), Hybrid (kết hợp chất lượng), và Popularity-based.
- **Context-Aware**: Gợi ý dựa trên ngữ cảnh người dùng như thời gian trong ngày (sáng/tối), ngày nghỉ (cuối tuần), và cấu hình thiết bị.
- **Kiểm tra tương thích**: Tự động phân tích cấu hình máy (CPU, GPU, RAM) để cảnh báo khả năng chơi mượt game.
- **Giao diện trực quan**: Ứng dụng web tương tác xây dựng bằng Streamlit.
- **Quản lý người dùng**: Đăng ký/Đăng nhập và lưu lịch sử xem/yêu thích.

---

## 🔄 Quy trình Dữ liệu (Data Pipeline)

### 1. Thu thập dữ liệu (Crawling)
File: `steam_crawler.py`

Hệ thống thu thập dữ liệu từ hai nguồn chính:
*   **Steam Store API**: Lấy thông tin chi tiết (giá, mô tả, yêu cầu hệ thống, hình ảnh).
*   **SteamSpy API**: Lấy thông tin thống kê (số lượng người sở hữu, tags, ratings).

```bash
# Cách chạy crawler (thu thập 100 game)
python steam_crawler.py --limit 100

# Crawl danh sách AppList mới từ Steam và thu thập 1000 game
python steam_crawler.py --crawl-applist --limit 1000
```
Dữ liệu thô được lưu vào thư mục `data/` dưới dạng các file CSV (`steam.csv`, `steam_description_data.csv`, ...).

### 2. Xử lý dữ liệu (Processing)
File: `data_processor.py`

Dữ liệu thô được làm sạch và trích xuất đặc trưng:
*   **Cleaning**: Xử lý dữ liệu thiếu, loại bỏ các game không đủ thông tin quan trọng.
*   **Feature Engineering**:
    *   `popularity_score`: Điểm phổ biến (dựa trên số lượng ratings).
    *   `quality_score`: Điểm chất lượng tổng hợp (kết hợp giữa đánh giá tích cực và mức độ phổ biến).
    *   `game_age_days`: Tuổi đời của game.
    *   `price_category`: Phân loại giá (Free, Budget, Premium...).
*   **Vectorization**: Sử dụng TF-IDF để chuyển đổi văn bản (mô tả, thể loại) thành vector phục vụ cho việc tính toán độ tương đồng.

---

## 🧠 Thuật toán Gợi ý
File: `recommender.py` & `context_aware.py`

1.  **Content-Based Filtering**:
    *   Sử dụng **TF-IDF** (hoặc Sentence Transformers nếu có) để phân tích mô tả game, thể loại, tags.
    *   Tính toán **Cosine Similarity** để tìm các game có nội dung tương tự game người dùng đang xem.

2.  **Hybrid Recommendation**:
    *   Kết hợp điểm tương đồng nội dung (`content_score`) và điểm chất lượng game (`quality_score`).
    *   Giúp gợi ý không chỉ các game giống nhau mà còn là các game hay, được cộng đồng đánh giá cao.

3.  **Context-Aware Recommendation**:
    *   **Thiết bị**: Lọc hoặc cảnh báo các game không tương thích với cấu hình phần cứng người dùng.
    *   **Thời gian**: Ưu tiên các thể loại phù hợp với thời điểm (ví dụ: game nhẹ nhàng, giải đố vào buổi sáng; game nhập vai, hành động vào buổi tối/cuối tuần).

---

## 💻 Cài đặt và Sử dụng

### Yêu cầu
*   Python 3.8+
*   Các thư viện trong `requirements.txt`

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng
```bash
streamlit run app.py
```
Truy cập vào địa chỉ hiển thị trên terminal (thường là `http://localhost:8501`).

---

## 📊 Đánh giá Mô hình
File: `run_evaluation.py`

Hệ thống đi kèm công cụ đánh giá hiệu suất các thuật toán gợi ý sử dụng các chỉ số như Precision@K, Recall@K, RMSE, MAE.

Cách chạy đánh giá:
```bash
# Đánh giá với 50 mẫu thử, K=10
python run_evaluation.py --samples 50 --k 10
```
Kết quả sẽ được in ra màn hình và lưu vào file JSON trong thư mục `results/`.

---

## 📂 Cấu trúc Dự án

```
├── app.py                 # Main Streamlit application
├── auth.py                # Quản lý xác thực người dùng
├── context_aware.py       # Logic gợi ý theo ngữ cảnh
├── data_processor.py      # Xử lý và làm sạch dữ liệu
├── device_config.py       # Cấu hình và kiểm tra thiết bị
├── evaluator.py           # Class đánh giá mô hình
├── recommender.py         # Core logic gợi ý (Content-based, Hybrid)
├── run_evaluation.py      # Script chạy đánh giá
├── steam_crawler.py       # Tool thu thập dữ liệu
├── user_history.py        # Quản lý lịch sử người dùng
├── requirements.txt       # Danh sách thư viện
├── README.md              # Tài liệu dự án
└── data/                  # Thư mục chứa dữ liệu CSV (sau khi crawl)
```
### 3. Kết quả demo
