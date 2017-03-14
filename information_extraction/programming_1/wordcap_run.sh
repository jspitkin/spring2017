python3 ner.py train.txt test.txt wordcap
./liblinear-1.93/train -s 0 train.txt.wordcap classifier.wordcap
./liblinear-1.93/predict test.txt.wordcap classifier.wordcap wordcap_predictions.txt > wordcap_accuracy.txt
more wordcap_accuracy.txt
