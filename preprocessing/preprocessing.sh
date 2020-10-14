#!/usr/bin/env bash

python preprocessing_simple.py -d Automotive -dev_test_in_train 1 | tee Automotive_simple.log
python pretrained_vectors_simple.py -d Automotive | tee Automotive_vectors.log

python preprocessing_simple.py -d Baby -dev_test_in_train 1 | tee Baby_simple.log
python pretrained_vectors_simple.py -d Baby | tee Baby_vectors.log

python preprocessing_simple.py -d Beauty -dev_test_in_train 1 | tee Beauty_simple.log
python pretrained_vectors_simple.py -d Beauty | tee Beauty_vectors.log

python preprocessing_simple.py -d Books -dev_test_in_train 1 | tee Books_simple.log
python pretrained_vectors_simple.py -d Books | tee Books_vectors.log

python preprocessing_simple.py -d CDs_and_Vinyl -dev_test_in_train 1 | tee CDs_and_Vinyl_simple.log
python pretrained_vectors_simple.py -d CDs_and_Vinyl | tee CDs_and_Vinyl_vectors.log

python preprocessing_simple.py -d Cell_Phones_and_Accessories -dev_test_in_train 1 | tee Cell_Phones_and_Accessories_simple.log
python pretrained_vectors_simple.py -d Cell_Phones_and_Accessories | tee Cell_Phones_and_Accessories_vectors.log

python preprocessing_simple.py -d Clothing_Shoes_and_Jewelry -dev_test_in_train 1 | tee Clothing_Shoes_and_Jewelry_simple.log
python pretrained_vectors_simple.py -d Clothing_Shoes_and_Jewelry | tee Clothing_Shoes_and_Jewelry_vectors.log

python preprocessing_simple.py -d Digital_Music -dev_test_in_train 1 | tee Digital_Music_simple.log
python pretrained_vectors_simple.py -d Digital_Music | tee Digital_Music_vectors.log

python preprocessing_simple.py -d Electronics -dev_test_in_train 1 | tee Electronics_simple.log
python pretrained_vectors_simple.py -d Electronics | tee Electronics_vectors.log

python preprocessing_simple.py -d Grocery_and_Gourmet_Food -dev_test_in_train 1 | tee Grocery_and_Gourmet_Food_simple.log
python pretrained_vectors_simple.py -d Grocery_and_Gourmet_Food | tee Grocery_and_Gourmet_Food_vectors.log

python preprocessing_simple.py -d Health_and_Personal_Care -dev_test_in_train 1 | tee Health_and_Personal_Care_simple.log
python pretrained_vectors_simple.py -d Health_and_Personal_Care | tee Health_and_Personal_Care_vectors.log

python preprocessing_simple.py -d Home_and_Kitchen -dev_test_in_train 1 | tee Home_and_Kitchen_simple.log
python pretrained_vectors_simple.py -d Home_and_Kitchen | tee Home_and_Kitchen_vectors.log

python preprocessing_simple.py -d Kindle_Store -dev_test_in_train 1 | tee Kindle_Store_simple.log
python pretrained_vectors_simple.py -d Kindle_Store | tee Kindle_Store_vectors.log

python preprocessing_simple.py -d Movies_and_TV -dev_test_in_train 1 | tee Movies_and_TV_simple.log
python pretrained_vectors_simple.py -d Movies_and_TV | tee Movies_and_TV_vectors.log

python preprocessing_simple.py -d Musical_Instruments -dev_test_in_train 1 | tee Musical_Instruments_simple.log
python pretrained_vectors_simple.py -d Musical_Instruments | tee Musical_Instruments_vectors.log

python preprocessing_simple.py -d Office_Products -dev_test_in_train 1 | tee Office_Products_simple.log
python pretrained_vectors_simple.py -d Office_Products | tee Office_Products_vectors.log

python preprocessing_simple.py -d Patio_Lawn_and_Garden -dev_test_in_train 1 | tee Patio_Lawn_and_Garden_simple.log
python pretrained_vectors_simple.py -d Patio_Lawn_and_Garden | tee Patio_Lawn_and_Garden_vectors.log

python preprocessing_simple.py -d Pet_Supplies -dev_test_in_train 1 | tee Pet_Supplies_simple.log
python pretrained_vectors_simple.py -d Pet_Supplies | tee Pet_Supplies_vectors.log

python preprocessing_simple.py -d Sports_and_Outdoors -dev_test_in_train 1 | tee Sports_and_Outdoors_simple.log
python pretrained_vectors_simple.py -d Sports_and_Outdoors | tee Sports_and_Outdoors_vectors.log

python preprocessing_simple.py -d Tools_and_Home_Improvement -dev_test_in_train 1 | tee Tools_and_Home_Improvement_simple.log
python pretrained_vectors_simple.py -d Tools_and_Home_Improvement | tee Tools_and_Home_Improvement_vectors.log

python preprocessing_simple.py -d Toys_and_Games -dev_test_in_train 1 | tee Toys_and_Games_simple.log
python pretrained_vectors_simple.py -d Toys_and_Games | tee Toys_and_Games_vectors.log

python preprocessing_simple.py -d Video_Games -dev_test_in_train 1 | tee Video_Games_simple.log
python pretrained_vectors_simple.py -d Video_Games | tee Video_Games_vectors.log
