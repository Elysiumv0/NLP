## E1
```text
D1 = [1, 0, 1, 1, 0]
D2 = [0, 1, 1, 1, 0]
D3 = [1, 0, 0, 1, 1]
```

## E2
```text
|D1| = 3
```
```text
tf(cat, D1)  = 1/3
tf(eats, D1) = 1/3
tf(fish, D1) = 1/3
```
Kiểm tra:
```text
1/3 + 1/3 + 1/3 = 1
```

## E3
```text
idf(cat)   = log(3/2) ≈ 0.4054651081
idf(dog)   = log(3)   ≈ 1.0986122887
idf(eats)  = log(3/2) ≈ 0.4054651081
idf(fish)  = log(3/3) = 0
idf(likes) = log(3)   ≈ 1.0986122887
```
Term có IDF thấp nhất là `fish` vì nó xuất hiện trong cả 3 documents.

## E4
```text
tfidf(cat, D1)
= (1/3) * log(3/2)
≈ 0.1351550360
tfidf(eats, D1)
= (1/3) * log(3/2)
≈ 0.1351550360
tfidf(fish, D1)
= (1/3) * 0
= 0
```
`fish` có TF-IDF bằng 0 vì `df(fish)=N`, nên `idf(fish)=0`.

## E5
```text
x · y = 2
||x|| = sqrt(3)
||y|| = sqrt(2)
```
=>
```text
cos(x,y) = 2 / sqrt(6)
         ≈ 0.8164965809
```
Không phải `2/3` vì cosine chuẩn hóa dot product bởi độ dài Euclid của hai vector, không phải tỷ lệ số term trùng nhau.

## E6
1. Similarity cao nhất: **D1** vì trùng toàn bộ query.
2. Similarity thấp nhất: **D3** vì không chia sẻ các term chính với query.
3. Term có thể có IDF thấp: **medical** và **image** vì xuất hiện ở D1 và D2.
4. Nếu dùng count vector thay TF-IDF, ranking trong ví dụ này dự kiến vẫn là D1 > D2 > D3, nhưng các term phổ biến như `medical` và `image` không bị giảm trọng số.
