## 1
Vocabulary của corpus khoảng 30K documents sẽ ở mức 100000 vì nhiều terms được lặp lại giữa documents nên vocabulary nhỏ hơn tổng số token rất nhiều, nhưng từ hiếm, tên riêng và thuật ngữ vẫn làm vocabulary lớn.

## 2 
TF-IDF matrix sẽ sparse, với tỷ lệ zero entries > 95%, có thể cao hơn tùy corpus vì mỗi document chỉ dùng một phần nhỏ vocabulary.

## 3
Top documents của TF-IDF search không nhất thiết là những documents gần nghĩa nhất vì TF-IDF dựa chủ yếu trên lexical overlap và term statistics. Synonym/paraphrase không chia sẻ token có thể bị xếp hạng thấp.
