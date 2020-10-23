#!/usr/bin/env bash
# Example script for ANR
# We repeat the process 5 times using different random seeds

# E.g. ./__saved_models__/amazon_instant_video - ARNS/amazon_instant_video_ANRS_1337.pth
#python PyTorchTEST.py -d "Automotive" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Automotive_VANRA"
python PyTorchTEST.py -d "Automotive" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Automotive_VANRA"
#python PyTorchTEST.py -d "Automotive" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Automotive_VANRA"
#python PyTorchTEST.py -d "Automotive" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Automotive_VANRA"
#python PyTorchTEST.py -d "Automotive" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Automotive_VANRA"

#python PyTorchTEST.py -d "Baby" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Baby_VANRA"
python PyTorchTEST.py -d "Baby" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Baby_VANRA"
#python PyTorchTEST.py -d "Baby" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Baby_VANRA"
#python PyTorchTEST.py -d "Baby" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Baby_VANRA"
#python PyTorchTEST.py -d "Baby" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Baby_VANRA"

#python PyTorchTEST.py -d "Beauty" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Beauty_VANRA"
python PyTorchTEST.py -d "Beauty" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Beauty_VANRA"
#python PyTorchTEST.py -d "Beauty" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Beauty_VANRA"
#python PyTorchTEST.py -d "Beauty" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Beauty_VANRA"
#python PyTorchTEST.py -d "Beauty" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Beauty_VANRA"

#python PyTorchTEST.py -d "Books" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Books_VANRA"
python PyTorchTEST.py -d "Books" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Books_VANRA"
#python PyTorchTEST.py -d "Books" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Books_VANRA"
#python PyTorchTEST.py -d "Books" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Books_VANRA"
#python PyTorchTEST.py -d "Books" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Books_VANRA"

#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_VANRA"
python PyTorchTEST.py -d "CDs_and_Vinyl" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_VANRA"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_VANRA"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_VANRA"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_VANRA"

#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"

#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"

#python PyTorchTEST.py -d "Digital_Music" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Digital_Music_VANRA"
python PyTorchTEST.py -d "Digital_Music" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Digital_Music_VANRA"
#python PyTorchTEST.py -d "Digital_Music" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Digital_Music_VANRA"
#python PyTorchTEST.py -d "Digital_Music" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Digital_Music_VANRA"
#python PyTorchTEST.py -d "Digital_Music" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Digital_Music_VANRA"

#python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
#python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Electronics_VANRA"
#python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Electronics_VANRA"
#python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Electronics_VANRA"

#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_VANRA"
python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_VANRA"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_VANRA"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_VANRA"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_VANRA"

#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_VANRA"
python PyTorchTEST.py -d "Health_and_Personal_Care" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_VANRA"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_VANRA"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_VANRA"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_VANRA"

#python PyTorchTEST.py -d "Home_and_Kitchen" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Home_and_Kitchen_VANRA"
python PyTorchTEST.py -d "Home_and_Kitchen" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Home_and_Kitchen_VANRA"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Home_and_Kitchen_VANRA"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Home_and_Kitchen_VANRA"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Home_and_Kitchen_VANRA"

#python PyTorchTEST.py -d "Kindle_Store" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Kindle_Store_VANRA"
python PyTorchTEST.py -d "Kindle_Store" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Kindle_Store_VANRA"
#python PyTorchTEST.py -d "Kindle_Store" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Kindle_Store_VANRA"
#python PyTorchTEST.py -d "Kindle_Store" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Kindle_Store_VANRA"
#python PyTorchTEST.py -d "Kindle_Store" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Kindle_Store_VANRA"

#python PyTorchTEST.py -d "Movies_and_TV" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Movies_and_TV_VANRA"
python PyTorchTEST.py -d "Movies_and_TV" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Movies_and_TV_VANRA"
#python PyTorchTEST.py -d "Movies_and_TV" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Movies_and_TV_VANRA"
#python PyTorchTEST.py -d "Movies_and_TV" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Movies_and_TV_VANRA"
#python PyTorchTEST.py -d "Movies_and_TV" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Movies_and_TV_VANRA"

#python PyTorchTEST.py -d "Musical_Instruments" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Musical_Instruments_VANRA"
python PyTorchTEST.py -d "Musical_Instruments" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Musical_Instruments_VANRA"
#python PyTorchTEST.py -d "Musical_Instruments" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Musical_Instruments_VANRA"
#python PyTorchTEST.py -d "Musical_Instruments" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Musical_Instruments_VANRA"
#python PyTorchTEST.py -d "Musical_Instruments" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Musical_Instruments_VANRA"

#python PyTorchTEST.py -d "Office_Products" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Office_Products_VANRA"
python PyTorchTEST.py -d "Office_Products" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Office_Products_VANRA"
#python PyTorchTEST.py -d "Office_Products" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Office_Products_VANRA"
#python PyTorchTEST.py -d "Office_Products" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Office_Products_VANRA"
#python PyTorchTEST.py -d "Office_Products" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Office_Products_VANRA"

#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_VANRA"
python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_VANRA"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_VANRA"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_VANRA"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_VANRA"

#python PyTorchTEST.py -d "Pet_Supplies" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Pet_Supplies_VANRA"
python PyTorchTEST.py -d "Pet_Supplies" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Pet_Supplies_VANRA"
#python PyTorchTEST.py -d "Pet_Supplies" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Pet_Supplies_VANRA"
#python PyTorchTEST.py -d "Pet_Supplies" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Pet_Supplies_VANRA"
#python PyTorchTEST.py -d "Pet_Supplies" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Pet_Supplies_VANRA"

#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"

#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_VANRA"
python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_VANRA"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_VANRA"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_VANRA"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_VANRA"

#python PyTorchTEST.py -d "Toys_and_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Toys_and_Games_VANRA"
python PyTorchTEST.py -d "Toys_and_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Toys_and_Games_VANRA"
#python PyTorchTEST.py -d "Toys_and_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Toys_and_Games_VANRA"
#python PyTorchTEST.py -d "Toys_and_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Toys_and_Games_VANRA"
#python PyTorchTEST.py -d "Toys_and_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Toys_and_Games_VANRA"

#python PyTorchTEST.py -d "Video_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Video_Games_VANRA"
python PyTorchTEST.py -d "Video_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Video_Games_VANRA"
#python PyTorchTEST.py -d "Video_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Video_Games_VANRA"
#python PyTorchTEST.py -d "Video_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Video_Games_VANRA"
#python PyTorchTEST.py -d "Video_Games" -m "VANRA" -e 15 -dr 0.9 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -c_local 10 -c_global 10 -hiden_size 100 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Video_Games_VANRA"

