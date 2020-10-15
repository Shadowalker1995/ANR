#!/usr/bin/env bash
# Example script for ANR
# We repeat the process 5 times using different random seeds

# If pretrained ARL weights are available, specify it using -ARL_path ...
# The saved ARNS model weights should be in ./__saved_models__/[dataset] - ARNS/[dataset]_ANRS_[random_seed].pth
# E.g. ./__saved_models__/amazon_instant_video - ARNS/amazon_instant_video_ANRS_1337.pth
#python PyTorchTEST.py -d "Automotive" -m "ANR" -e 15 -p 1 -v 50000 -rs 1337 -gpu 0 -vb 1 -sm "Automotive_ANR" -ARL_path "Automotive_ANRS_1337"
python PyTorchTEST.py -d "Automotive" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Automotive_ANR" -ARL_path "Automotive_ANRS_1337"
#python PyTorchTEST.py -d "Automotive" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Automotive_ANR" -ARL_path "Automotive_ANRS_1337"
python PyTorchTEST.py -d "Automotive" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Automotive_ANR" -ARL_path "Automotive_ANRS_1337"
#python PyTorchTEST.py -d "Automotive" -m "ANR" -e 15 -p 1 -v 50000 -rs 2468 -gpu 0 -vb 1 -sm "Automotive_ANR" -ARL_path "Automotive_ANRS_1337"

python PyTorchTEST.py -d "Baby" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Baby_ANR" -ARL_path "Baby_ANRS_1337"
python PyTorchTEST.py -d "Baby" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Baby_ANR" -ARL_path "Baby_ANRS_1337"
python PyTorchTEST.py -d "Baby" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Baby_ANR" -ARL_path "Baby_ANRS_1337"

python PyTorchTEST.py -d "Beauty" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Beauty_ANR" -ARL_path "Beauty_ANRS_1337"
python PyTorchTEST.py -d "Beauty" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Beauty_ANR" -ARL_path "Beauty_ANRS_1337"
python PyTorchTEST.py -d "Beauty" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Beauty_ANR" -ARL_path "Beauty_ANRS_1337"

python PyTorchTEST.py -d "Books" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Books_ANR" -ARL_path "Books_ANRS_1337"
python PyTorchTEST.py -d "Books" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Books_ANR" -ARL_path "Books_ANRS_1337"
python PyTorchTEST.py -d "Books" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Books_ANR" -ARL_path "Books_ANRS_1337"

python PyTorchTEST.py -d "CDs_and_Vinyl" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_ANR" -ARL_path "CDs_and_Vinyl_ANRS_1337"
python PyTorchTEST.py -d "CDs_and_Vinyl" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_ANR" -ARL_path "CDs_and_Vinyl_ANRS_1337"
python PyTorchTEST.py -d "CDs_and_Vinyl" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_ANR" -ARL_path "CDs_and_Vinyl_ANRS_1337"

python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_ANR" -ARL_path "Cell_Phones_and_Accessories_ANRS_1337"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_ANR" -ARL_path "Cell_Phones_and_Accessories_ANRS_1337"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_ANR" -ARL_path "Cell_Phones_and_Accessories_ANRS_1337"

python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_ANR" -ARL_path "Clothing_Shoes_and_Jewelry_ANRS_1337"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_ANR" -ARL_path "Clothing_Shoes_and_Jewelry_ANRS_1337"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_ANR" -ARL_path "Clothing_Shoes_and_Jewelry_ANRS_1337"

python PyTorchTEST.py -d "Digital_Music" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Digital_Music_ANR" -ARL_path "Digital_Music_ANRS_1337"
python PyTorchTEST.py -d "Digital_Music" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Digital_Music_ANR" -ARL_path "Digital_Music_ANRS_1337"
python PyTorchTEST.py -d "Digital_Music" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Digital_Music_ANR" -ARL_path "Digital_Music_ANRS_1337"

python PyTorchTEST.py -d "Electronics" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_ANR" -ARL_path "Electronics_ANRS_1337"
python PyTorchTEST.py -d "Electronics" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Electronics_ANR" -ARL_path "Electronics_ANRS_1337"
python PyTorchTEST.py -d "Electronics" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Electronics_ANR" -ARL_path "Electronics_ANRS_1337"

python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_ANR" -ARL_path "Grocery_and_Gourmet_Food_ANRS_1337"
python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_ANR" -ARL_path "Grocery_and_Gourmet_Food_ANRS_1337"
python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_ANR" -ARL_path "Grocery_and_Gourmet_Food_ANRS_1337"

python PyTorchTEST.py -d "Health_and_Personal_Care" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_ANR" -ARL_path "Health_and_Personal_Care_ANRS_1337"
python PyTorchTEST.py -d "Health_and_Personal_Care" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_ANR" -ARL_path "Health_and_Personal_Care_ANRS_1337"
python PyTorchTEST.py -d "Health_and_Personal_Care" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_ANR" -ARL_path "Health_and_Personal_Care_ANRS_1337"

python PyTorchTEST.py -d "Home_and_Kitchen" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Home_and_Kitchen_ANR" -ARL_path "Home_and_Kitchen_ANRS_1337"
python PyTorchTEST.py -d "Home_and_Kitchen" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Home_and_Kitchen_ANR" -ARL_path "Home_and_Kitchen_ANRS_1337"
python PyTorchTEST.py -d "Home_and_Kitchen" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Home_and_Kitchen_ANR" -ARL_path "Home_and_Kitchen_ANRS_1337"

python PyTorchTEST.py -d "Kindle_Store" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Kindle_Store_ANR" -ARL_path "Kindle_Store_ANRS_1337"
python PyTorchTEST.py -d "Kindle_Store" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Kindle_Store_ANR" -ARL_path "Kindle_Store_ANRS_1337"
python PyTorchTEST.py -d "Kindle_Store" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Kindle_Store_ANR" -ARL_path "Kindle_Store_ANRS_1337"

python PyTorchTEST.py -d "Movies_and_TV" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Movies_and_TV_ANR" -ARL_path "Movies_and_TV_ANRS_1337"
python PyTorchTEST.py -d "Movies_and_TV" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Movies_and_TV_ANR" -ARL_path "Movies_and_TV_ANRS_1337"
python PyTorchTEST.py -d "Movies_and_TV" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Movies_and_TV_ANR" -ARL_path "Movies_and_TV_ANRS_1337"

python PyTorchTEST.py -d "Musical_Instruments" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Musical_Instruments_ANR" -ARL_path "Musical_Instruments_ANRS_1337"
python PyTorchTEST.py -d "Musical_Instruments" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Musical_Instruments_ANR" -ARL_path "Musical_Instruments_ANRS_1337"
python PyTorchTEST.py -d "Musical_Instruments" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Musical_Instruments_ANR" -ARL_path "Musical_Instruments_ANRS_1337"

python PyTorchTEST.py -d "Office_Products" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Office_Products_ANR" -ARL_path "Office_Products_ANRS_1337"
python PyTorchTEST.py -d "Office_Products" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Office_Products_ANR" -ARL_path "Office_Products_ANRS_1337"
python PyTorchTEST.py -d "Office_Products" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Office_Products_ANR" -ARL_path "Office_Products_ANRS_1337"

python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_ANR" -ARL_path "Patio_Lawn_and_Garden_ANRS_1337"
python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_ANR" -ARL_path "Patio_Lawn_and_Garden_ANRS_1337"
python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_ANR" -ARL_path "Patio_Lawn_and_Garden_ANRS_1337"

python PyTorchTEST.py -d "Pet_Supplies" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Pet_Supplies_ANR" -ARL_path "Pet_Supplies_ANRS_1337"
python PyTorchTEST.py -d "Pet_Supplies" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Pet_Supplies_ANR" -ARL_path "Pet_Supplies_ANRS_1337"
python PyTorchTEST.py -d "Pet_Supplies" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Pet_Supplies_ANR" -ARL_path "Pet_Supplies_ANRS_1337"

python PyTorchTEST.py -d "Sports_and_Outdoors" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_ANR" -ARL_path "Sports_and_Outdoors_ANRS_1337"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_ANR" -ARL_path "Sports_and_Outdoors_ANRS_1337"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_ANR" -ARL_path "Sports_and_Outdoors_ANRS_1337"

python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_ANR" -ARL_path "Tools_and_Home_Improvement_ANRS_1337"
python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_ANR" -ARL_path "Tools_and_Home_Improvement_ANRS_1337"
python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_ANR" -ARL_path "Tools_and_Home_Improvement_ANRS_1337"

python PyTorchTEST.py -d "Toys_and_Games" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Toys_and_Games_ANR" -ARL_path "Toys_and_Games_ANRS_1337"
python PyTorchTEST.py -d "Toys_and_Games" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Toys_and_Games_ANR" -ARL_path "Toys_and_Games_ANRS_1337"
python PyTorchTEST.py -d "Toys_and_Games" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Toys_and_Games_ANR" -ARL_path "Toys_and_Games_ANRS_1337"

python PyTorchTEST.py -d "Video_Games" -m "ANR" -e 15 -p 1 -v 50000 -rs 1234 -gpu 0 -vb 1 -sm "Video_Games_ANR" -ARL_path "Video_Games_ANRS_1337"
python PyTorchTEST.py -d "Video_Games" -m "ANR" -e 15 -p 1 -v 50000 -rs 5678 -gpu 0 -vb 1 -sm "Video_Games_ANR" -ARL_path "Video_Games_ANRS_1337"
python PyTorchTEST.py -d "Video_Games" -m "ANR" -e 15 -p 1 -v 50000 -rs 1357 -gpu 0 -vb 1 -sm "Video_Games_ANR" -ARL_path "Video_Games_ANRS_1337"

