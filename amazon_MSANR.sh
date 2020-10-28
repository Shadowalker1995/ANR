#!/usr/bin/env bash
# Example script for ANR
# We repeat the process 5 times using different random seeds

# E.g. ./__saved_models__/amazon_instant_video - ARNS/amazon_instant_video_ANRS_1337.pth
#python PyTorchTEST.py -d "Automotive" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Automotive_MSANR"
python PyTorchTEST.py -d "Automotive" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Automotive_MSANR"
#python PyTorchTEST.py -d "Automotive" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Automotive_MSANR"
#python PyTorchTEST.py -d "Automotive" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Automotive_MSANR"
#python PyTorchTEST.py -d "Automotive" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Automotive_MSANR"

#python PyTorchTEST.py -d "Baby" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Baby_MSANR"
python PyTorchTEST.py -d "Baby" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Baby_MSANR"
#python PyTorchTEST.py -d "Baby" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Baby_MSANR"
#python PyTorchTEST.py -d "Baby" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Baby_MSANR"
#python PyTorchTEST.py -d "Baby" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Baby_MSANR"

#python PyTorchTEST.py -d "Beauty" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Beauty_MSANR"
python PyTorchTEST.py -d "Beauty" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Beauty_MSANR"
#python PyTorchTEST.py -d "Beauty" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Beauty_MSANR"
#python PyTorchTEST.py -d "Beauty" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Beauty_MSANR"
#python PyTorchTEST.py -d "Beauty" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Beauty_MSANR"

#python PyTorchTEST.py -d "Books" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Books_MSANR"
python PyTorchTEST.py -d "Books" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Books_MSANR"
#python PyTorchTEST.py -d "Books" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Books_MSANR"
#python PyTorchTEST.py -d "Books" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Books_MSANR"
#python PyTorchTEST.py -d "Books" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Books_MSANR"

#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_MSANR"
python PyTorchTEST.py -d "CDs_and_Vinyl" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_MSANR"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_MSANR"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_MSANR"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_MSANR"

#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"

#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"

#python PyTorchTEST.py -d "Digital_Music" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Digital_Music_MSANR"
python PyTorchTEST.py -d "Digital_Music" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Digital_Music_MSANR"
#python PyTorchTEST.py -d "Digital_Music" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Digital_Music_MSANR"
#python PyTorchTEST.py -d "Digital_Music" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Digital_Music_MSANR"
#python PyTorchTEST.py -d "Digital_Music" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Digital_Music_MSANR"

#python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Electronics_MSANR"
python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_MSANR"
#python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Electronics_MSANR"
#python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Electronics_MSANR"
#python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Electronics_MSANR"

#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_MSANR"
python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_MSANR"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_MSANR"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_MSANR"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_MSANR"

#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_MSANR"
python PyTorchTEST.py -d "Health_and_Personal_Care" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_MSANR"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_MSANR"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_MSANR"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_MSANR"

#python PyTorchTEST.py -d "Home_and_Kitchen" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Home_and_Kitchen_MSANR"
python PyTorchTEST.py -d "Home_and_Kitchen" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Home_and_Kitchen_MSANR"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Home_and_Kitchen_MSANR"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Home_and_Kitchen_MSANR"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Home_and_Kitchen_MSANR"

#python PyTorchTEST.py -d "Kindle_Store" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Kindle_Store_MSANR"
python PyTorchTEST.py -d "Kindle_Store" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Kindle_Store_MSANR"
#python PyTorchTEST.py -d "Kindle_Store" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Kindle_Store_MSANR"
#python PyTorchTEST.py -d "Kindle_Store" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Kindle_Store_MSANR"
#python PyTorchTEST.py -d "Kindle_Store" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Kindle_Store_MSANR"

#python PyTorchTEST.py -d "Movies_and_TV" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Movies_and_TV_MSANR"
python PyTorchTEST.py -d "Movies_and_TV" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Movies_and_TV_MSANR"
#python PyTorchTEST.py -d "Movies_and_TV" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Movies_and_TV_MSANR"
#python PyTorchTEST.py -d "Movies_and_TV" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Movies_and_TV_MSANR"
#python PyTorchTEST.py -d "Movies_and_TV" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Movies_and_TV_MSANR"

#python PyTorchTEST.py -d "Musical_Instruments" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Musical_Instruments_MSANR"
python PyTorchTEST.py -d "Musical_Instruments" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Musical_Instruments_MSANR"
#python PyTorchTEST.py -d "Musical_Instruments" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Musical_Instruments_MSANR"
#python PyTorchTEST.py -d "Musical_Instruments" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Musical_Instruments_MSANR"
#python PyTorchTEST.py -d "Musical_Instruments" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Musical_Instruments_MSANR"

#python PyTorchTEST.py -d "Office_Products" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Office_Products_MSANR"
python PyTorchTEST.py -d "Office_Products" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Office_Products_MSANR"
#python PyTorchTEST.py -d "Office_Products" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Office_Products_MSANR"
#python PyTorchTEST.py -d "Office_Products" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Office_Products_MSANR"
#python PyTorchTEST.py -d "Office_Products" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Office_Products_MSANR"

#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_MSANR"
python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_MSANR"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_MSANR"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_MSANR"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_MSANR"

#python PyTorchTEST.py -d "Pet_Supplies" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Pet_Supplies_MSANR"
python PyTorchTEST.py -d "Pet_Supplies" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Pet_Supplies_MSANR"
#python PyTorchTEST.py -d "Pet_Supplies" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Pet_Supplies_MSANR"
#python PyTorchTEST.py -d "Pet_Supplies" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Pet_Supplies_MSANR"
#python PyTorchTEST.py -d "Pet_Supplies" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Pet_Supplies_MSANR"

#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"

#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_MSANR"
python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_MSANR"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_MSANR"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_MSANR"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_MSANR"

#python PyTorchTEST.py -d "Toys_and_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Toys_and_Games_MSANR"
python PyTorchTEST.py -d "Toys_and_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Toys_and_Games_MSANR"
#python PyTorchTEST.py -d "Toys_and_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Toys_and_Games_MSANR"
#python PyTorchTEST.py -d "Toys_and_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Toys_and_Games_MSANR"
#python PyTorchTEST.py -d "Toys_and_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Toys_and_Games_MSANR"

#python PyTorchTEST.py -d "Video_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Video_Games_MSANR"
python PyTorchTEST.py -d "Video_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Video_Games_MSANR"
#python PyTorchTEST.py -d "Video_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Video_Games_MSANR"
#python PyTorchTEST.py -d "Video_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Video_Games_MSANR"
#python PyTorchTEST.py -d "Video_Games" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -h2 50 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Video_Games_MSANR"

