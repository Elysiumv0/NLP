1, 
vocab = 7
token = 12
P(the) = 3/12
P(cat) = 2/12
P(fish) = 2/12
P(dog) = 1/12
P(W) = (3+2+2+2+1+1+1)/12 = 1
2, 
Có 9 biagram
P(cat|the) = P(the, cat)|P(the) = 2/9 / 3/9 = 2/3
P(dog|the) = P(the, dog)|P(the) = 1/9 / 3/9 = 1/3
P(eats|cat) = P(cat, eats)|P(cat) = 1/9 / 2/9 = 1/2
P(likes|cat) = P(cat, likes)|P(cat) = 1/9 / 2/9 = 1/2
Tổng xác suất của các từ đứng sau the phải bằng 1 vì sau the luôn có 1 từ trong vocab theo sau nên /sigma(w) C(the,w) = C(the). Chia cả 2 vế cho C(the) ta được đpcm.
3, 
P = 1/4*2/3*1/2*1/2 = 1/24
Thêm 1 từ vào câu thì xác suất không thể tăng khi phải nhân thêm 1 xác suất điều kiện trong khoảng [0,1]
#9
##1, Count(study AI) = 0
##2, P(AI|Study) = P(study, AI)|P(Study) = 0
##3, P = 0
#11, 
P = 0+1/10+5 = 1/15
Khi C = 3:
P = 3+1/10+5 = 4/15
Smoothing làm giảm xác suất của các bigram đã thấy, ở đây là từ 0.3 -> 4/15, để tăng xác suất cho các bigram chưa thấy.
18,
P(W) = 0.5*0.25*0.5 = 1/16
PP(W) = (1/16)^(-1/3) ~ 2.52
Nếu P(w2|w1)=0.1 thì:
P(W) = 0.5*0.1*0.5 = 1/40
PP(W) = (1/40)^(-1/3) ~ 3.42
Vì Perplexity là phép lấy log, số càng nhỏ thì log càng lớn.
