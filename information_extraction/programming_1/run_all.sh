python3 ner.py train.txt test.txt word
./liblinear-1.93/train -s 0 train.txt.word classifier.word
./liblinear-1.93/predict test.txt.word classifier.word word_predictions.txt > word_accuracy.txt
python3 ner.py train.txt test.txt wordcap
./liblinear-1.93/train -s 0 train.txt.wordcap classifier.wordcap
./liblinear-1.93/predict test.txt.wordcap classifier.wordcap wordcap_predictions.txt > wordcap_accuracy.txt
python3 ner.py train.txt test.txt poscon
./liblinear-1.93/train -s 0 train.txt.poscon classifier.poscon
./liblinear-1.93/predict test.txt.poscon classifier.poscon poscon_predictions.txt > poscon_accuracy.txt
python3 ner.py train.txt test.txt lexcon
./liblinear-1.93/train -s 0 train.txt.lexcon classifier.lexcon
./liblinear-1.93/predict test.txt.lexcon classifier.lexcon lexcon_predictions.txt > lexcon_accuracy.txt
python3 ner.py train.txt test.txt bothcon
./liblinear-1.93/train -s 0 train.txt.bothcon classifier.bothcon
./liblinear-1.93/predict test.txt.bothcon classifier.bothcon bothcon_predictions.txt > bothcon_accuracy.txt
printf "word:   "
more word_accuracy.txt
printf "wordcap:   "
more wordcap_accuracy.txt
printf "poscon:   "
more poscon_accuracy.txt
printf "lexcon:   "
more lexcon_accuracy.txt
printf "bothcon:   "
more bothcon_accuracy.txt
