python3 ner.py train.txt test.txt lexcon
./liblinear-1.93/train -s 0 train.txt.lexcon classifier.lexcon
./liblinear-1.93/predict test.txt.lexcon classifier.lexcon lexcon_predictions.txt > lexcon_accuracy.txt
more lexcon_accuracy.txt
