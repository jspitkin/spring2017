python3 ner.py train.txt test.txt word
./liblinear-1.93/train -s 0 train.txt.word classifier.word
./liblinear-1.93/predict test.txt.word classifier.word word_predictions.txt > word_accuracy.txt
more word_accuracy.txt
