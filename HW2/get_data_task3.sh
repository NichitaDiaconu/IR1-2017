
#!/bin/bash

#-m 1 : lsi on bow corpus
#-m 2 : lsi on shifted bow corpus
#-m 3 : lsi on tfidf corpus
#-m 4 : lda on bow corpus

python3 lsi.py -m 1 -k 200
python3 lsi.py -m 2 -k 200

python3 lsi.py -m 1 -k 300
python3 lsi.py -m 1 -k 400
python3 lsi.py -m 1 -k 500
python3 lsi.py -m 1 -k 50
python3 lsi.py -m 1 -k 1000
#python3 lsi.py -m 1 -k 400
#python3 lsi.py -m 1 -k 500
#python3 lsi.py -m 1 -k 800
#python3 lsi.py -m 1 -k 50

#python3 lsi.py -m 2 -k 200
#python3 lsi.py -m 2 -k 300
#python3 lsi.py -m 2 -k 400
#python3 lsi.py -m 2 -k 500
#python3 lsi.py -m 2 -k 1000
#python3 lsi.py -m 2 -k 50

#python3 lsi.py -m 3 -k 200
#python3 lsi.py -m 3 -k 300
#python3 lsi.py -m 3 -k 400
#python3 lsi.py -m 3 -k 500

#python3 lsi.py -m 4 -k 800
#python3 lsi.py -m 3 -k 50

#python3 lsi.py -m 4 -k 200
#python3 lsi.py -m 4 -k 300
#python3 lsi.py -m 4 -k 400
#python3 lsi.py -m 4 -k 500
#python3 lsi.py -m 4 -k 1000

python3 lsi.py -m 4 -k 50 -a 10 -b 10
python3 lsi.py -m 4 -k 200 -a 10 -b 10
python3 lsi.py -m 4 -k 350 -a 10 -b 10
python3 lsi.py -m 4 -k 500 -a 10 -b 10
python3 lsi.py -m 4 -k 1000 -a 10 -b 10

python3 lsi.py -m 4 -k 50 -a 0.1 -b 10
python3 lsi.py -m 4 -k 200 -a 0.1 -b 10
python3 lsi.py -m 4 -k 350 -a 0.1 -b 10
python3 lsi.py -m 4 -k 500 -a 0.1 -b 10
python3 lsi.py -m 4 -k 1000 -a 0.1 -b 10

python3 lsi.py -m 4 -k 50 -a 1 -b 10
python3 lsi.py -m 4 -k 200 -a 1 -b 10
python3 lsi.py -m 4 -k 350 -a 1 -b 10
python3 lsi.py -m 4 -k 500 -a 1 -b 10
python3 lsi.py -m 4 -k 1000 -a 1 -b 10

python3 lsi.py -m 4 -k 50 -a 10 -b 1
python3 lsi.py -m 4 -k 200 -a 10 -b 1
python3 lsi.py -m 4 -k 350 -a 10 -b 1
python3 lsi.py -m 4 -k 500 -a 10 -b 1
python3 lsi.py -m 4 -k 1000 -a 10 -b 1

python3 lsi.py -m 4 -k 50 -a 0.1 -b 1
python3 lsi.py -m 4 -k 200 -a 0.1 -b 1
python3 lsi.py -m 4 -k 350 -a 0.1 -b 1
python3 lsi.py -m 4 -k 500 -a 0.1 -b 1
python3 lsi.py -m 4 -k 1000 -a 0.1 -b 1

python3 lsi.py -m 4 -k 50 -a 1 -b 1
python3 lsi.py -m 4 -k 200 -a 1 -b 1
python3 lsi.py -m 4 -k 350 -a 1 -b 1
python3 lsi.py -m 4 -k 500 -a 1 -b 1
python3 lsi.py -m 4 -k 1000 -a 1 -b 1

python3 lsi.py -m 4 -k 50 -a 10 -b 0.1
python3 lsi.py -m 4 -k 200 -a 10 -b 0.1
python3 lsi.py -m 4 -k 350 -a 10 -b 0.1
python3 lsi.py -m 4 -k 500 -a 10 -b 0.1
python3 lsi.py -m 4 -k 1000 -a 10 -b 0.1

python3 lsi.py -m 4 -k 50 -a 0.1 -b 0.1
python3 lsi.py -m 4 -k 200 -a 0.1 -b 0.1
python3 lsi.py -m 4 -k 350 -a 0.1 -b 0.1
python3 lsi.py -m 4 -k 500 -a 0.1 -b 0.1
python3 lsi.py -m 4 -k 1000 -a 0.1 -b 0.1

python3 lsi.py -m 4 -k 50 -a 1 -b 0.1
python3 lsi.py -m 4 -k 200 -a 1 -b 0.1
python3 lsi.py -m 4 -k 350 -a 1 -b 0.1
python3 lsi.py -m 4 -k 500 -a 1 -b 0.1
python3 lsi.py -m 4 -k 1000 -a 1 -b 0.1