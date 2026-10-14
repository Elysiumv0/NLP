# Reflection

## Experimental setup

- Dataset: IMDb Movie Reviews.
- Training samples: 8000.
- Validation samples: 2000.
- Test samples: 2485.
- Random seed: 42.
- Baseline test accuracy: 0.8716.
- Baseline test precision: 0.8675.
- Baseline test recall: 0.8779.
- Baseline test F1: 0.8727.

## Representation comparison

Trong lần chạy này, TF-IDF + Logistic Regression có F1 không thấp hơn Average Word2Vec + Logistic Regression. Điều này phù hợp với giả thuyết rằng lexical feature có tín hiệu hữu ích cho sentiment classification.

Average Word2Vec creates a dense fixed-size vector but averaging removes word order and can dilute the effect of negation. TF-IDF preserves separate lexical features and can represent selected phrases when n-grams are enabled.

## N-gram comparison

Unigram + bigram đạt F1 không thấp hơn unigram trong lần chạy này. Cụm từ có thể bổ sung tín hiệu, nhưng cần cân nhắc vocabulary size và training time.

Unigram vocabulary size: 27352.

Unigram plus bigram vocabulary size: 80000.

## Classifier comparison

F1 for TF-IDF + Logistic Regression: 0.8727.

F1 for TF-IDF + Linear SVM: 0.8669.

Best F1 in this experiment: TF-IDF unigram + bigram + Logistic Regression at 0.8779.

## Data leakage

TF-IDF was fit inside a training-only pipeline. Validation and test documents were transformed using the vocabulary and inverse document frequencies learned from training data. Word2Vec was also trained only from training text. The held-out test split was not used for fitting.

## Evaluation beyond accuracy

Precision and recall describe different errors. False positives are negative reviews classified as positive; false negatives are positive reviews classified as negative. For the baseline, FP = 167 and FN = 152. The larger of these two counts identifies the more frequent error type in this test sample. Precision is especially sensitive to false positives, while recall is especially sensitive to false negatives. Confusion matrix and selected examples are required to understand the error pattern. A higher accuracy alone is not sufficient evidence of a better classifier.

## Limitations

The experiment uses a stratified sample for practical runtime. Results can change with random seed, sample size, preprocessing, hyperparameters and the dataset version. Error categories are heuristic labels that require manual confirmation. The scores are not claims about every sentiment dataset.

## Prediction review

- P1: check the baseline metrics above and compare against the other representations.
- P2: compare the F1 values of TF-IDF and Average Word2Vec.
- P3: compare unigram and unigram plus bigram vocabulary sizes and F1.
- P4: stopwords were preserved so that terms such as `not` remain available to the model.
- P5: compare accuracy, precision, recall and F1 in `results.csv` to determine whether the metrics rank models differently.
