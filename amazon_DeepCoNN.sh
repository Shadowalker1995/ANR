#!/usr/bin/env bash
# Example script for ANR
# We repeat the process 5 times using different random seeds

# E.g. ./__saved_models__/amazon_instant_video - ARNS/amazon_instant_video_ANRS_1337.pth
#python PyTorchTEST.py -d "Automotive" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Automotive_DeepCoNN"
python PyTorchTEST.py -d "Automotive" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Automotive_DeepCoNN"
#python PyTorchTEST.py -d "Automotive" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Automotive_DeepCoNN"
#python PyTorchTEST.py -d "Automotive" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Automotive_DeepCoNN"
#python PyTorchTEST.py -d "Automotive" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Automotive_DeepCoNN"

#python PyTorchTEST.py -d "Baby" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Baby_DeepCoNN"
python PyTorchTEST.py -d "Baby" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Baby_DeepCoNN"
#python PyTorchTEST.py -d "Baby" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Baby_DeepCoNN"
#python PyTorchTEST.py -d "Baby" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Baby_DeepCoNN"
#python PyTorchTEST.py -d "Baby" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Baby_DeepCoNN"

#python PyTorchTEST.py -d "Beauty" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Beauty_DeepCoNN"
python PyTorchTEST.py -d "Beauty" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Beauty_DeepCoNN"
#python PyTorchTEST.py -d "Beauty" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Beauty_DeepCoNN"
#python PyTorchTEST.py -d "Beauty" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Beauty_DeepCoNN"
#python PyTorchTEST.py -d "Beauty" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Beauty_DeepCoNN"

#python PyTorchTEST.py -d "Books" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Books_DeepCoNN"
python PyTorchTEST.py -d "Books" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Books_DeepCoNN"
#python PyTorchTEST.py -d "Books" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Books_DeepCoNN"
#python PyTorchTEST.py -d "Books" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Books_DeepCoNN"
#python PyTorchTEST.py -d "Books" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Books_DeepCoNN"

#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DeepCoNN"
python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DeepCoNN"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DeepCoNN"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DeepCoNN"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DeepCoNN"

#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DeepCoNN"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DeepCoNN"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DeepCoNN"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DeepCoNN"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DeepCoNN"

#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DeepCoNN"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DeepCoNN"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DeepCoNN"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DeepCoNN"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DeepCoNN"

#python PyTorchTEST.py -d "Digital_Music" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Digital_Music_DeepCoNN"
python PyTorchTEST.py -d "Digital_Music" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Digital_Music_DeepCoNN"
#python PyTorchTEST.py -d "Digital_Music" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Digital_Music_DeepCoNN"
#python PyTorchTEST.py -d "Digital_Music" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Digital_Music_DeepCoNN"
#python PyTorchTEST.py -d "Digital_Music" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Digital_Music_DeepCoNN"

#python PyTorchTEST.py -d "Electronics" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Electronics_DeepCoNN"
python PyTorchTEST.py -d "Electronics" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_DeepCoNN"
#python PyTorchTEST.py -d "Electronics" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Electronics_DeepCoNN"
#python PyTorchTEST.py -d "Electronics" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Electronics_DeepCoNN"
#python PyTorchTEST.py -d "Electronics" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Electronics_DeepCoNN"

#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DeepCoNN"
python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DeepCoNN"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DeepCoNN"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DeepCoNN"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DeepCoNN"

#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DeepCoNN"
python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DeepCoNN"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DeepCoNN"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DeepCoNN"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DeepCoNN"

#python PyTorchTEST.py -d "Home_and_Kitchen" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DeepCoNN"
python PyTorchTEST.py -d "Home_and_Kitchen" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DeepCoNN"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DeepCoNN"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DeepCoNN"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DeepCoNN"

#python PyTorchTEST.py -d "Kindle_Store" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Kindle_Store_DeepCoNN"
python PyTorchTEST.py -d "Kindle_Store" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Kindle_Store_DeepCoNN"
#python PyTorchTEST.py -d "Kindle_Store" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Kindle_Store_DeepCoNN"
#python PyTorchTEST.py -d "Kindle_Store" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Kindle_Store_DeepCoNN"
#python PyTorchTEST.py -d "Kindle_Store" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Kindle_Store_DeepCoNN"

#python PyTorchTEST.py -d "Movies_and_TV" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Movies_and_TV_DeepCoNN"
python PyTorchTEST.py -d "Movies_and_TV" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Movies_and_TV_DeepCoNN"
#python PyTorchTEST.py -d "Movies_and_TV" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Movies_and_TV_DeepCoNN"
#python PyTorchTEST.py -d "Movies_and_TV" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Movies_and_TV_DeepCoNN"
#python PyTorchTEST.py -d "Movies_and_TV" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Movies_and_TV_DeepCoNN"

#python PyTorchTEST.py -d "Musical_Instruments" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Musical_Instruments_DeepCoNN"
python PyTorchTEST.py -d "Musical_Instruments" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Musical_Instruments_DeepCoNN"
#python PyTorchTEST.py -d "Musical_Instruments" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Musical_Instruments_DeepCoNN"
#python PyTorchTEST.py -d "Musical_Instruments" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Musical_Instruments_DeepCoNN"
#python PyTorchTEST.py -d "Musical_Instruments" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Musical_Instruments_DeepCoNN"

#python PyTorchTEST.py -d "Office_Products" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Office_Products_DeepCoNN"
python PyTorchTEST.py -d "Office_Products" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Office_Products_DeepCoNN"
#python PyTorchTEST.py -d "Office_Products" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Office_Products_DeepCoNN"
#python PyTorchTEST.py -d "Office_Products" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Office_Products_DeepCoNN"
#python PyTorchTEST.py -d "Office_Products" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Office_Products_DeepCoNN"

#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DeepCoNN"
python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DeepCoNN"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DeepCoNN"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DeepCoNN"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DeepCoNN"

#python PyTorchTEST.py -d "Pet_Supplies" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Pet_Supplies_DeepCoNN"
python PyTorchTEST.py -d "Pet_Supplies" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Pet_Supplies_DeepCoNN"
#python PyTorchTEST.py -d "Pet_Supplies" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Pet_Supplies_DeepCoNN"
#python PyTorchTEST.py -d "Pet_Supplies" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Pet_Supplies_DeepCoNN"
#python PyTorchTEST.py -d "Pet_Supplies" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Pet_Supplies_DeepCoNN"

#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DeepCoNN"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DeepCoNN"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DeepCoNN"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DeepCoNN"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DeepCoNN"

#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DeepCoNN"
python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DeepCoNN"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DeepCoNN"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DeepCoNN"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DeepCoNN"

#python PyTorchTEST.py -d "Toys_and_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Toys_and_Games_DeepCoNN"
python PyTorchTEST.py -d "Toys_and_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Toys_and_Games_DeepCoNN"
#python PyTorchTEST.py -d "Toys_and_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Toys_and_Games_DeepCoNN"
#python PyTorchTEST.py -d "Toys_and_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Toys_and_Games_DeepCoNN"
#python PyTorchTEST.py -d "Toys_and_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Toys_and_Games_DeepCoNN"

#python PyTorchTEST.py -d "Video_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Video_Games_DeepCoNN"
python PyTorchTEST.py -d "Video_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Video_Games_DeepCoNN"
#python PyTorchTEST.py -d "Video_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Video_Games_DeepCoNN"
#python PyTorchTEST.py -d "Video_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Video_Games_DeepCoNN"
#python PyTorchTEST.py -d "Video_Games" -m "DeepCoNN" -e 10 -dr 0.5 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -filters_num 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Video_Games_DeepCoNN"

