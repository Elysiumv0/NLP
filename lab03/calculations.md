## 1. Word–Context Matrix

| **Word**  | **cat** | **dog** | **eats** | **likes** | **fish** | **milk** | **meat** |
| --------- | ------: | ------: | -------: | --------: | -------: | -------: | -------: |
| **cat**   |       0 |       0 |        1 |         1 |        0 |        0 |        0 |
| **dog**   |       0 |       0 |        1 |         1 |        0 |        0 |        0 |
| **eats**  |       1 |       1 |        0 |         0 |        2 |        0 |        0 |
| **likes** |       1 |       1 |        0 |         0 |        0 |        1 |        1 |

Các vector:

```text
cat   = [0, 0, 1, 1, 0, 0, 0]
dog   = [0, 0, 1, 1, 0, 0, 0]
eats  = [1, 1, 0, 0, 2, 0, 0]
likes = [1, 1, 0, 0, 0, 1, 1]
```

---

## 2. Cosine Similarity

Ta có:

```text
x · y = 1·2 + 2·4 + 1·2
      = 2 + 8 + 2
      = 12
```

Độ dài của hai vector:

```text
|x| = √(1² + 2² + 1²)
   = √6

|y| = √(2² + 4² + 2²)
   = √24
   = 2√6
```

Do đó:

```text
cos(x, y) = (x · y) / (|x| · |y|)
           = 12 / (√6 · 2√6)
           = 12 / (2 · 6)
           = 12 / 12
           = 1
```

**Ý nghĩa:** Hai vector cùng hướng, chỉ khác độ lớn. Cosine similarity chỉ đo góc giữa hai vector và bỏ qua độ lớn. Điều này có nghĩa là hai đối tượng có cùng phân bố tương đối của các đặc trưng thì được coi là giống nhau bất kể tổng số lần xuất hiện.

---

## 3. Cosine Similarity giữa các từ

**Dự đoán:** `physician` gần `doctor` hơn `banana`.

### `cos(doctor, physician)`

```text
doctor · physician
= 0.8·0.7 + 0.1·0.2 + 0.7·0.8
= 0.56 + 0.02 + 0.56
= 1.14
```

Độ dài:

```text
|doctor|² = 0.64 + 0.01 + 0.49
         = 1.14

|doctor| = √1.14
         ≈ 1.0677
```

```text
|physician|² = 0.49 + 0.04 + 0.64
             = 1.17

|physician| = √1.17
             ≈ 1.0817
```

Cosine similarity:

```text
cos(doctor, physician)
= 1.14 / (1.0677 · 1.0817)
≈ 1.14 / 1.1549
≈ 0.987
```

### `cos(doctor, banana)`

```text
doctor · banana
= 0.8·(-0.2) + 0.1·0.9 + 0.7·(-0.1)
= -0.16 + 0.09 - 0.07
= -0.14
```

```text
|banana|² = 0.04 + 0.81 + 0.01
          = 0.86

|banana| ≈ 0.9274
```

Do đó:

```text
cos(doctor, banana)
= -0.14 / (1.0677 · 0.9274)
≈ -0.14 / 0.9902
≈ -0.141
```

Vì:

```text
cos(doctor, physician) ≈ 0.987
cos(doctor, banana)    ≈ -0.141
```

nên **`physician` gần `doctor` hơn `banana`**.

---

## 4. Sparse vs Dense Vector

### 1. Sparse

Vector biểu diễn dưới dạng **sparse vector**.

### 2. Dense

Vector biểu diễn dưới dạng **dense vector**.

### 3. Tại sao dense vector có thể biểu diễn similarity tốt hơn sparse vector?

* Hai vector sparse chỉ giống nhau khi chúng trùng đúng các cột. Hai từ gần nghĩa nhưng không chia sẻ cột nào thì có thể có `cos = 0`, dù thực tế chúng có context liên quan.
* Dense vector nén một không gian context rất lớn, ví dụ 10.000 chiều, xuống khoảng 300 chiều.
* Các context có quan hệ ngữ nghĩa có thể được biểu diễn ở những chiều gần nhau, từ đó similarity có thể khái quát hóa qua các context tương tự.

### 4. Dense có luôn tốt hơn sparse không?

**Không.**

* Sparse dễ diễn giải và khớp từ khóa chính xác rất tốt.
* Dense cần huấn luyện, nhạy với corpus và siêu tham số, kém minh bạch hơn, đồng thời có thể làm mờ từ hiếm và các nghĩa khác nhau của từ đa nghĩa.
* Về bộ nhớ **trên mỗi vector**, sparse lưu 30 cặp `(chỉ số, giá trị)` có thể còn nhỏ hơn 300 số thực của dense.
* Vấn đề chính của sparse là **chiều không gian rất lớn**, có thể lên tới `|V| × |V|`.

---

## 5. CBOW vs Skip-gram

### CBOW

CBOW (**Continuous Bag-of-Words**) tổng hợp hoặc lấy trung bình các embedding của các từ trong context rồi dùng chúng để dự đoán từ ở giữa.

### Skip-gram

Skip-gram sử dụng từ trung tâm để dự đoán các từ trong context.

Với cùng một cửa sổ context, Skip-gram sinh ra nhiều cặp `(center word, context word)` hơn, nên thường học tốt hơn đối với các từ hiếm nhưng có tốc độ huấn luyện chậm hơn.

---

## 6. Word Analogy: `king - man + woman`

Ta có:

```text
king − man + woman
= [8−5+5, 2−1+3, 7−5+5]
= [8, 4, 7]
```

Ta phân tích:

```text
king − man = [3, 1, 2]
```

Phần này biểu diễn những đặc trưng còn lại của `king` sau khi loại bỏ thành phần `man`.

Tiếp theo:

```text
woman − man = [0, 2, 0]
```

Chỉ có chiều thứ 2 thay đổi, do đó chiều 2 có thể đóng vai trò như một trục biểu diễn giới tính.

Vì vậy:

```text
king + (woman − man)
= [8, 2, 7] + [0, 2, 0]
= [8, 4, 7]
```

Kết quả giữ nguyên các thuộc tính của **`king`**, đồng thời thay đổi thành phần liên quan đến giới tính từ `man` sang `woman`.

Do đó:

```text
[8, 4, 7] ≈ queen
```

**Kết luận:**

```text
king − man + woman ≈ queen
```
