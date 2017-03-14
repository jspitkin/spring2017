python3 ner.py train.txt test.txt bothcon
./liblinear-1.93/train -s 0 train.txt.bothcon classifier.bothcon
./liblinear-1.93/predict test.txt.bothcon classifier.bothcon bothcon_predictions.txt > bothcon_accuracy.txt
more bothcon_accuracy.txt
