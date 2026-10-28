# Calculations

## Bài 1 — Attention scores

Với `q = [1, 2]` và các key `k1 = [1, 0]`, `k2 = [0, 1]`, `k3 = [1, 1]`:

- `s1 = qᵀk1 = 1 × 1 + 2 × 0 = 1`
- `s2 = qᵀk2 = 1 × 0 + 2 × 1 = 2`
- `s3 = qᵀk3 = 1 × 1 + 2 × 1 = 3`

Score vector: `s = [1, 2, 3]`.

## Bài 2 — Softmax

Với `s = [1, 2, 3]`, trừ giá trị lớn nhất trước khi tính exponential:

`softmax(s) = [0.090031, 0.244728, 0.665241]`.

Tổng các weights bằng `1`. Score lớn nhất cho weight lớn nhất vì hàm exponential tăng đơn điệu và mẫu số giống nhau cho mọi phần tử trong cùng phân phối.

## Bài 3 — Context vector

Với `α = [0.1, 0.2, 0.7]`, `v1 = [1, 0]`, `v2 = [0, 1]`, `v3 = [2, 2]`:

`c = 0.1[1, 0] + 0.2[0, 1] + 0.7[2, 2]`

`c = [0.1, 0] + [0, 0.2] + [1.4, 1.4] = [1.5, 1.6]`.

`v3` đóng góp lớn nhất: phần đóng góp có trọng số là `[1.4, 1.4]`.

## Bài 4 — Toàn bộ attention computation

Với `q = [1, 0]`, `k1 = [1, 0]`, `k2 = [0, 1]`, `k3 = [1, 1]`:

` s = [1, 0, 1] `

`α = softmax(s) = [0.422319, 0.155362, 0.422319]`.

Với `v1 = [1, 0]`, `v2 = [0, 2]`, `v3 = [2, 1]`:

`c = 0.422319[1, 0] + 0.155362[0, 2] + 0.422319[2, 1]`

`c ≈ [1.266958, 0.733042]`.

## Bài scaling

`dk = 4`, nên `√dk = 2`. Với score chưa scaling `[4, 2, 0]`, score sau scaling là `[2, 1, 0]`.

| Trường hợp | Scores | Softmax |
|---|---|---|
| Không scaling | `[4, 2, 0]` | `[0.866813, 0.117310, 0.015876]` |
| Chia cho `√dk` | `[2, 1, 0]` | `[0.665241, 0.244728, 0.090031]` |

Scaling làm distribution bớt tập trung vào score cao nhất trong ví dụ này. Khi vector dimension tăng, dot product thường có độ lớn tăng; chia cho `√dk` giúp kiểm soát độ lớn score trước softmax.

## Temperature

Dùng score vector `[1, 2, 3]` và `softmax(s / T)`.

| Temperature | Weights | Max weight | Entropy, nats |
|---:|---|---:|---:|
| 0.5 | `[0.015876, 0.117310, 0.866813]` | 0.866813 | 0.4411 |
| 1.0 | `[0.090031, 0.244728, 0.665241]` | 0.665241 | 0.8324 |
| 2.0 | `[0.186324, 0.307196, 0.506480]` | 0.506480 | 1.0202 |

Temperature thấp làm phân phối tập trung hơn. Temperature cao làm phân phối phẳng hơn.

## Numerical stability

Với `x = [1000, 1001, 1002]`, softmax ổn định số là `[0.090031, 0.244728, 0.665241]`.

Phép trừ `max(x)` không đổi kết quả vì:

`exp(xᵢ − m) / Σⱼ exp(xⱼ − m) = exp(−m)exp(xᵢ) / (exp(−m)Σⱼexp(xⱼ))`.

Hệ số `exp(−m)` triệt tiêu ở tử và mẫu. Cách này giảm nguy cơ overflow khi lấy exponential.

## Chi phí theo sequence length

| Số token `T` | Attention matrix | Số phần tử |
|---:|---|---:|
| 10 | `10 × 10` | 100 |
| 100 | `100 × 100` | 10,000 |
| 500 | `500 × 500` | 250,000 |
| 1,000 | `1,000 × 1,000` | 1,000,000 |
| 10,000 | `10,000 × 10,000` | 100,000,000 |

Full self-attention có số phần tử trong attention matrix tăng theo `O(T²)`. Khi sequence dài gấp đôi, số phần tử tăng gấp bốn.
