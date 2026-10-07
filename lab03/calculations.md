## 1 

| Word  | cat | dog | eats | likes | fish | milk | meat |
|-------|:---:|:---:|:----:|:-----:|:----:|:----:|:----:|
| cat   | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| dog   | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| eats  | 1 | 1 | 0 | 0 | 2 | 0 | 0 |
| likes | 1 | 1 | 0 | 0 | 0 | 1 | 1 |

Vector: 
`cat = [0,0,1,1,0,0,0]`
`dog = [0,0,1,1,0,0,0]`
`eats = [1,1,0,0,2,0,0]`
`likes = [1,1,0,0,0,1,1]`.

## 2

x · y  = 1·2 + 2·4 + 1·2 = 2 + 8 + 2 = 12
|x|    = √(1² + 2² + 1²) = √6
|y|    = √(2² + 4² + 2²) = √24 = 2√6
cos(x, y) = 12 / (√6 · 2√6) = 12 / (2·6) = 12/12 = 1
```

Ý nghĩa: hai vector cùng hướng, chỉ khác độ lớn. Cosine similarity chỉ đo góc giữa hai vector và bỏ qua độ lớn. Điều này có nghĩa là hai đối tượng có cùng phân bố tương đối của các đặc trưng thì được coi là giống nhau bất kể tổng số lần xuất hiện.

## 3 

Dự đoán: `physician` gần `doctor` hơn `banana`.

cos(doctor, physician)
```
doctor · physician = 0.8·0.7 + 0.1·0.2 + 0.7·0.8 = 0.56 + 0.02 + 0.56 = 1.14
|doctor|² = 0.64 + 0.01 + 0.49 = 1.14      → |doctor| = √1.14 ≈ 1.0677
|physician|² = 0.49 + 0.04 + 0.64 = 1.17   → |physician| = √1.17 ≈ 1.0817
cos = 1.14 / (1.0677 · 1.0817) = 1.14 / 1.1549 ≈ 0.987
```

cos(doctor, banana)
```
doctor · banana = 0.8·(−0.2) + 0.1·0.9 + 0.7·(−0.1) = −0.16 + 0.09 − 0.07 = −0.14
|banana|² = 0.04 + 0.81 + 0.01 = 0.86      → |banana| ≈ 0.9274
cos = −0.14 / (1.0677 · 0.9274) = −0.14 / 0.9902 ≈ −0.141
```

## 4

1. 1 sparse
2. 2 dense
3.
- Hai vector sparse chỉ giống nhau khi trùng đúng cột. Hai từ gần nghĩa nhưng không chia sẻ cột nào thì `cos = 0`
dù thực ra có context liên quan. Dense nén 10.000 context thành 300 chiều nên các context gần nhau được gộp vào cùng những chiều → similarity
khái quát hoá được qua các context tương tự.
4.
Không.
- Sparse dễ diễn giải và khớp từ khoá chính xác rất tốt.
- Dense cần huấn luyện, nhạy với corpus/siêu tham số, kém minh bạch, và làm mờ từ hiếm và các nghĩa khác nhau của từ đa nghĩa.
- Về bộ nhớ *trên mỗi vector*, sparse lưu 30 cặp (chỉ số, giá trị) có thể còn nhỏ hơn 300 số thực của dense; vấn đề của sparse là chiều không gian
     rất lớn (|V|×|V|).
---

## 5

Khác nhau
+, CBOW: tổng hợp/trung bình các embedding của context rồi dự đoán từ giữa;
+, Skip-gram: Cùng một cửa sổ, Skip-gram sinh nhiều mẫu hơn nên thường học tốt hơn cho từ hiếm nhưng chậm hơn
---

## 6 
```
king − man + woman = [8−5+5,  2−1+3,  7−5+5] = [8, 4, 7]
```

- `king − man = [3, 1, 2]` — phần còn lại khi bỏ "man".
- `woman − man = [0, 2, 0]` — chỉ chiều thứ 2 thay đổi → chiều 2 đóng vai trò trục giới tính.
- Vậy kết quả = `king + [0, 2, 0]`: giữ nguyên các thuộc tính "vua", chỉ chuyển trục giới tính sang "nữ" → vector `[8, 4, 7]` là "vua nữ" ≈ **queen**.
