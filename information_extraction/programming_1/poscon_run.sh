python3 ner.py train.txt test.txt poscon
./liblinear-1.93/train -s 0 train.txt.poscon classifier.poscon
./liblinear-1.93/predict test.txt.poscon classifier.poscon poscon_predictions.txt > poscon_accuracy.txt
more poscon_accuracy.txt
