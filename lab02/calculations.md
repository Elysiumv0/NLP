1. Unigram

vocab = 7

token = 12

P(the) = 3/12

P(cat) = 2/12

P(fish) = 2/12

P(dog) = 1/12

[
P(W) = \frac{3+2+2+2+1+1+1}{12} = 1
]

2. Bigram

Có 9 bigram.

[
P(cat \mid the) = \frac{P(the, cat)}{P(the)} = \frac{2/9}{3/9} = \frac{2}{3}
]

[
P(dog \mid the) = \frac{P(the, dog)}{P(the)} = \frac{1/9}{3/9} = \frac{1}{3}
]

[
P(eats \mid cat) = \frac{P(cat, eats)}{P(cat)} = \frac{1/9}{2/9} = \frac{1}{2}
]

[
P(likes \mid cat) = \frac{P(cat, likes)}{P(cat)} = \frac{1/9}{2/9} = \frac{1}{2}
]

Tổng xác suất của các từ đứng sau the phải bằng 1 vì sau the luôn có 1 từ trong vocab theo sau nên

[
\sum_w C(the,w) = C(the)
]

Chia cả 2 vế cho (C(the)) ta được đpcm.

3. Xác suất câu

[
P = \frac{1}{4} \times \frac{2}{3} \times \frac{1}{2} \times \frac{1}{2} = \frac{1}{24}
]

Thêm 1 từ vào câu thì xác suất không thể tăng khi phải nhân thêm 1 xác suất điều kiện trong khoảng ([0,1]).

9

1. Count

Count(study AI) = 0

2. Xác suất điều kiện

[
P(AI \mid Study) = \frac{P(study, AI)}{P(Study)} = 0
]

3. Xác suất

[
P = 0
]

11. Smoothing

[
P = \frac{0+1}{10+5} = \frac{1}{15}
]

Khi (C = 3):

[
P = \frac{3+1}{10+5} = \frac{4}{15}
]

Smoothing làm giảm xác suất của các bigram đã thấy, ở đây là từ 0.3 → (4/15), để tăng xác suất cho các bigram chưa thấy.

18. Perplexity

[
P(W) = 0.5 \times 0.25 \times 0.5 = \frac{1}{16}
]

[
PP(W) = \left(\frac{1}{16}\right)^{-1/3} \approx 2.52
]

Nếu (P(w_2 \mid w_1)=0.1) thì:

[
P(W) = 0.5 \times 0.1 \times 0.5 = \frac{1}{40}
]

[
PP(W) = \left(\frac{1}{40}\right)^{-1/3} \approx 3.42
]

Vì Perplexity là phép lấy log, số càng nhỏ thì log càng lớn.
