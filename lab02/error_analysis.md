## 1. Prediction đúng — `couldn → t`

* Context: `couldn`
* Model prediction: `t`
* Expected: `t`
* Probability: `1.0`

Mô hình dự đoán chính xác từ `t`, trùng với từ thực tế.

Xác suất bằng `1.0` cho thấy trong tập huấn luyện, sau context `couldn`, mô hình chỉ quan sát thấy continuation `t`. Do đó:

$$
P(t \mid couldn't) = 1.0
$$

Trường hợp này cho thấy mô hình có thể dự đoán rất chính xác đối với các context có continuation ổn định trong dữ liệu huấn luyện.

Nguyên nhân: Mô hình đã quan sát đầy đủ mẫu n-gram tương ứng trong tập huấn luyện.

---

## 2. Prediction đúng — `burkina → faso`

* Context: `burkina`
* Model prediction: `faso`
* Expected: `faso`
* Probability: `1.0`

Mô hình dự đoán đúng `faso` sau `burkina`.

Xác suất `1.0` cho thấy trong dữ liệu huấn luyện, `faso` là continuation được quan sát cho context này.

Nguyên nhân: N-gram xuất hiện rõ ràng và có tính đặc trưng cao trong dữ liệu huấn luyện.

---

## 3. Prediction sai — `<s> → the` thay vì `dehydrate`

* Context: `<s>`
* Model prediction: `the`
* Expected: `dehydrate`
* Probability: `0.0899`

Mô hình dự đoán `the`, trong khi từ thực tế ở đầu câu là `dehydrate`.
Với bigram, context chỉ là `<s>`, tức mô hình chỉ biết rằng đây là vị trí bắt đầu câu. Context này quá ngắn để phân biệt giữa rất nhiều từ có thể xuất hiện ở đầu câu.
Mô hình chọn `the` vì đây là một continuation có xác suất tương đối cao trong dữ liệu huấn luyện. Trong khi đó, `<s> dehydrate` có thể là một unseen n-gram hoặc xuất hiện quá ít để mô hình học được.

---

## 4. Prediction sai — `rosebery → kids` thay vì `mascot`

* Context: `rosebery`
* Model prediction: `kids`
* Expected: `mascot`
* Probability: `1.0`

Mô hình dự đoán `kids`, nhưng từ thực tế là `mascot`.
Đáng chú ý, mô hình gán xác suất `1.0` cho `kids`. Điều này xảy ra khi trong dữ liệu huấn luyện, đối với context `rosebery`, mô hình chỉ quan sát được `kids` là continuation.
Do đó:
$$
P(kids \mid rosebery) = 1.0
$$
Tuy nhiên, trong tập đánh giá, continuation thực tế lại là `mascot`. Điều này cho thấy mô hình rất nhạy với các context hiếm: nếu một context chỉ xuất hiện với một continuation trong training set, mô hình có thể gán xác suất tuyệt đối cho continuation đó, ngay cả khi nó không đúng trong trường hợp mới.

---
