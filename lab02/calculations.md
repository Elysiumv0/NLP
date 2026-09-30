1.

vocab = 7

token = 12

$$
P(\text{the}) = \frac{3}{12}
$$

$$
P(\text{cat}) = \frac{2}{12}
$$

$$
P(\text{fish}) = \frac{2}{12}
$$

$$
P(\text{dog}) = \frac{1}{12}
$$

Với câu $W$:

$$
P(W) = \frac{3+2+2+2+1+1+1}{12} = 1
$$

2. 

Có 9 bigram.

Xác suất của các bigram

$$
P(\text{cat}\mid\text{the})
= \frac{P(\text{the},\text{cat})}{P(\text{the})}
= \frac{2/9}{3/9}
= \frac{2}{3}
$$

$$
P(\text{dog}\mid\text{the})
= \frac{P(\text{the},\text{dog})}{P(\text{the})}
= \frac{1/9}{3/9}
= \frac{1}{3}
$$

$$
P(\text{eats}\mid\text{cat})
= \frac{P(\text{cat},\text{eats})}{P(\text{cat})}
= \frac{1/9}{2/9}
= \frac{1}{2}
$$

$$
P(\text{likes}\mid\text{cat})
= \frac{P(\text{cat},\text{likes})}{P(\text{cat})}
= \frac{1/9}{2/9}
= \frac{1}{2}
$$

Tổng xác suất các từ đứng sau the

Tổng xác suất của các từ đứng sau the phải bằng 1 vì sau the luôn có một từ trong vocabulary theo sau, nên:

$$
\sum_w C(\text{the},w) = C(\text{the})
$$

Chia cả hai vế cho $C(\text{the})$, ta được đpcm:

$$
\sum_w P(w\mid\text{the}) = 1
$$

3. 

$$
P(W) = \frac{1}{4}\times\frac{2}{3}\times\frac{1}{2}\times\frac{1}{2}
= \frac{1}{24}
$$

Thêm một từ vào câu thì xác suất không thể tăng, vì phải nhân thêm một xác suất có điều kiện trong khoảng $[0,1]$.

9

9.1. 

$$
\operatorname{Count}(\text{study AI}) = 0
$$

9.2. 

$$
P(\text{AI}\mid\text{Study})
= \frac{P(\text{study},\text{AI})}{P(\text{Study})}
= 0
$$

9.3. 

$$
P = 0
$$

11. 

$$
P = \frac{0+1}{10+5} = \frac{1}{15}
$$

Khi $C=3$:

$$
P = \frac{3+1}{10+5} = \frac{4}{15}
$$

Smoothing làm giảm xác suất của các bigram đã thấy. Ở đây, xác suất giảm từ:

$$
0.3 \rightarrow \frac{4}{15}
$$

để tăng xác suất cho các bigram chưa thấy.

18. 

Trường hợp 1

$$
P(W) = 0.5 \times 0.25 \times 0.5 = \frac{1}{16}
$$

$$
PP(W) = \left(\frac{1}{16}\right)^{-1/3} \approx 2.52
$$


Nếu:

$$
P(w_2\mid w_1)=0.1
$$

thì:

$$
P(W) = 0.5 \times 0.1 \times 0.5 = \frac{1}{40}
$$

$$
PP(W) = \left(\frac{1}{40}\right)^{-1/3} \approx 3.42
$$

Vì Perplexity là phép lấy log, xác suất càng nhỏ thì Perplexity càng lớn.
