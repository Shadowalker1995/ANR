#!/usr/bin/env bash

python preprocessing/preprocessing_simple.py -d Automotive -dev_test_in_train 1 | tee preprocessing/Automotive_simple.log
python preprocessing/pretrained_vectors_simple.py -d Automotive | tee preprocessing/Automotive_vectors.log

python preprocessing/reprocessing_simple.py -d Baby -dev_test_in_train 1 | tee preprocessing/Baby_simple.log
python preprocessing/retrained_vectors_simple.py -d Baby | tee preprocessing/Baby_vectors.log

python preprocessing/preprocessing_simple.py -d Beauty -dev_test_in_train 1 | tee preprocessing/Beauty_simple.log
python preprocessing/pretrained_vectors_simple.py -d Beauty | tee preprocessing/Beauty_vectors.log

python preprocessing/preprocessing_simple.py -d Books -dev_test_in_train 1 | tee preprocessing/Books_simple.log
python preprocessing/pretrained_vectors_simple.py -d Books | tee preprocessing/Books_vectors.log

python preprocessing/preprocessing_simple.py -d CDs_and_Vinyl -dev_test_in_train 1 | tee preprocessing/CDs_and_Vinyl_simple.log
python preprocessing/pretrained_vectors_simple.py -d CDs_and_Vinyl | tee preprocessing/CDs_and_Vinyl_vectors.log

python preprocessing/preprocessing_simple.py -d Cell_Phones_and_Accessories -dev_test_in_train 1 | tee preprocessing/Cell_Phones_and_Accessories_simple.log
python preprocessing/pretrained_vectors_simple.py -d Cell_Phones_and_Accessories | tee preprocessing/Cell_Phones_and_Accessories_vectors.log

python preprocessing/preprocessing_simple.py -d Clothing_Shoes_and_Jewelry -dev_test_in_train 1 | tee preprocessing/Clothing_Shoes_and_Jewelry_simple.log
python preprocessing/pretrained_vectors_simple.py -d Clothing_Shoes_and_Jewelry | tee preprocessing/Clothing_Shoes_and_Jewelry_vectors.log

python preprocessing/preprocessing_simple.py -d Digital_Music -dev_test_in_train 1 | tee preprocessing/Digital_Music_simple.log
python preprocessing/pretrained_vectors_simple.py -d Digital_Music | tee preprocessing/Digital_Music_vectors.log

python preprocessing/preprocessing_simple.py -d Electronics -dev_test_in_train 1 | tee preprocessing/Electronics_simple.log
python preprocessing/pretrained_vectors_simple.py -d Electronics | tee preprocessing/Electronics_vectors.log

python preprocessing/preprocessing_simple.py -d Grocery_and_Gourmet_Food -dev_test_in_train 1 | tee preprocessing/Grocery_and_Gourmet_Food_simple.log
python preprocessing/pretrained_vectors_simple.py -d Grocery_and_Gourmet_Food | tee preprocessing/Grocery_and_Gourmet_Food_vectors.log

python preprocessing/preprocessing_simple.py -d Health_and_Personal_Care -dev_test_in_train 1 | tee preprocessing/Health_and_Personal_Care_simple.log
python preprocessing/pretrained_vectors_simple.py -d Health_and_Personal_Care | tee preprocessing/Health_and_Personal_Care_vectors.log

python preprocessing/preprocessing_simple.py -d Home_and_Kitchen -dev_test_in_train 1 | tee preprocessing/Home_and_Kitchen_simple.log
python preprocessing/pretrained_vectors_simple.py -d Home_and_Kitchen | tee preprocessing/Home_and_Kitchen_vectors.log

python preprocessing/preprocessing_simple.py -d Kindle_Store -dev_test_in_train 1 | tee preprocessing/Kindle_Store_simple.log
python preprocessing/pretrained_vectors_simple.py -d Kindle_Store | tee preprocessing/Kindle_Store_vectors.log

python preprocessing/preprocessing_simple.py -d Movies_and_TV -dev_test_in_train 1 | tee preprocessing/Movies_and_TV_simple.log
python preprocessing/pretrained_vectors_simple.py -d Movies_and_TV | tee preprocessing/Movies_and_TV_vectors.log

python preprocessing/preprocessing_simple.py -d Musical_Instruments -dev_test_in_train 1 | tee preprocessing/Musical_Instruments_simple.log
python preprocessing/pretrained_vectors_simple.py -d Musical_Instruments | tee preprocessing/Musical_Instruments_vectors.log

python preprocessing/preprocessing_simple.py -d Office_Products -dev_test_in_train 1 | tee preprocessing/Office_Products_simple.log
python preprocessing/pretrained_vectors_simple.py -d Office_Products | tee preprocessing/Office_Products_vectors.log

python preprocessing/preprocessing_simple.py -d Patio_Lawn_and_Garden -dev_test_in_train 1 | tee preprocessing/Patio_Lawn_and_Garden_simple.log
python preprocessing/pretrained_vectors_simple.py -d Patio_Lawn_and_Garden | tee preprocessing/Patio_Lawn_and_Garden_vectors.log

python preprocessing/preprocessing_simple.py -d Pet_Supplies -dev_test_in_train 1 | tee preprocessing/Pet_Supplies_simple.log
python preprocessing/pretrained_vectors_simple.py -d Pet_Supplies | tee preprocessing/Pet_Supplies_vectors.log

python preprocessing/preprocessing_simple.py -d Sports_and_Outdoors -dev_test_in_train 1 | tee preprocessing/Sports_and_Outdoors_simple.log
python preprocessing/pretrained_vectors_simple.py -d Sports_and_Outdoors | tee preprocessing/Sports_and_Outdoors_vectors.log

python preprocessing/preprocessing_simple.py -d Tools_and_Home_Improvement -dev_test_in_train 1 | tee preprocessing/Tools_and_Home_Improvement_simple.log
python preprocessing/pretrained_vectors_simple.py -d Tools_and_Home_Improvement | tee preprocessing/Tools_and_Home_Improvement_vectors.log

python preprocessing/preprocessing_simple.py -d Toys_and_Games -dev_test_in_train 1 | tee preprocessing/Toys_and_Games_simple.log
python preprocessing/pretrained_vectors_simple.py -d Toys_and_Games | tee preprocessing/Toys_and_Games_vectors.log

python preprocessing/preprocessing_simple.py -d Video_Games -dev_test_in_train 1 | tee preprocessing/Video_Games_simple.log
python preprocessing/pretrained_vectors_simple.py -d Video_Games | tee preprocessing/Video_Games_vectors.log
