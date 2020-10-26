#!/usr/bin/env bash
# Example script for ANR
# We repeat the process 5 times using different random seeds

# E.g. ./__saved_models__/amazon_instant_video - ARNS/amazon_instant_video_ANRS_1337.pth
#python PyTorchTEST.py -d "Automotive" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Automotive_DAttn"
#python PyTorchTEST.py -d "Automotive" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Automotive_DAttn"
#python PyTorchTEST.py -d "Automotive" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Automotive_DAttn"
#python PyTorchTEST.py -d "Automotive" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Automotive_DAttn"
#python PyTorchTEST.py -d "Automotive" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Automotive_DAttn"

#python PyTorchTEST.py -d "Baby" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Baby_DAttn"
#python PyTorchTEST.py -d "Baby" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Baby_DAttn"
#python PyTorchTEST.py -d "Baby" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Baby_DAttn"
#python PyTorchTEST.py -d "Baby" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Baby_DAttn"
#python PyTorchTEST.py -d "Baby" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Baby_DAttn"

#python PyTorchTEST.py -d "Beauty" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Beauty_DAttn"
#python PyTorchTEST.py -d "Beauty" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Beauty_DAttn"
#python PyTorchTEST.py -d "Beauty" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Beauty_DAttn"
#python PyTorchTEST.py -d "Beauty" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Beauty_DAttn"
#python PyTorchTEST.py -d "Beauty" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Beauty_DAttn"

#python PyTorchTEST.py -d "Books" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Books_DAttn"
#python PyTorchTEST.py -d "Books" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Books_DAttn"
#python PyTorchTEST.py -d "Books" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Books_DAttn"
#python PyTorchTEST.py -d "Books" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Books_DAttn"
#python PyTorchTEST.py -d "Books" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Books_DAttn"

#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DAttn"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DAttn"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DAttn"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DAttn"
#python PyTorchTEST.py -d "CDs_and_Vinyl" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "CDs_and_Vinyl_DAttn"

#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DAttn"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DAttn"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DAttn"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DAttn"
#python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_DAttn"

#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DAttn"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DAttn"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DAttn"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DAttn"
#python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_DAttn"

#python PyTorchTEST.py -d "Digital_Music" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Digital_Music_DAttn"
#python PyTorchTEST.py -d "Digital_Music" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Digital_Music_DAttn"
#python PyTorchTEST.py -d "Digital_Music" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Digital_Music_DAttn"
#python PyTorchTEST.py -d "Digital_Music" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Digital_Music_DAttn"
#python PyTorchTEST.py -d "Digital_Music" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Digital_Music_DAttn"

#python PyTorchTEST.py -d "Electronics" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Electronics_DAttn"
#python PyTorchTEST.py -d "Electronics" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_DAttn"
#python PyTorchTEST.py -d "Electronics" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Electronics_DAttn"
#python PyTorchTEST.py -d "Electronics" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Electronics_DAttn"
#python PyTorchTEST.py -d "Electronics" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Electronics_DAttn"

#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DAttn"
python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DAttn"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DAttn"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DAttn"
#python PyTorchTEST.py -d "Grocery_and_Gourmet_Food" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Grocery_and_Gourmet_Food_DAttn"

#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DAttn"
python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DAttn"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DAttn"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DAttn"
#python PyTorchTEST.py -d "Health_and_Personal_Care" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Health_and_Personal_Care_DAttn"

#python PyTorchTEST.py -d "Home_and_Kitchen" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DAttn"
python PyTorchTEST.py -d "Home_and_Kitchen" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DAttn"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DAttn"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DAttn"
#python PyTorchTEST.py -d "Home_and_Kitchen" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Home_and_Kitchen_DAttn"

#python PyTorchTEST.py -d "Kindle_Store" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Kindle_Store_DAttn"
python PyTorchTEST.py -d "Kindle_Store" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Kindle_Store_DAttn"
#python PyTorchTEST.py -d "Kindle_Store" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Kindle_Store_DAttn"
#python PyTorchTEST.py -d "Kindle_Store" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Kindle_Store_DAttn"
#python PyTorchTEST.py -d "Kindle_Store" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Kindle_Store_DAttn"

#python PyTorchTEST.py -d "Movies_and_TV" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Movies_and_TV_DAttn"
python PyTorchTEST.py -d "Movies_and_TV" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Movies_and_TV_DAttn"
#python PyTorchTEST.py -d "Movies_and_TV" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Movies_and_TV_DAttn"
#python PyTorchTEST.py -d "Movies_and_TV" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Movies_and_TV_DAttn"
#python PyTorchTEST.py -d "Movies_and_TV" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Movies_and_TV_DAttn"

#python PyTorchTEST.py -d "Musical_Instruments" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Musical_Instruments_DAttn"
python PyTorchTEST.py -d "Musical_Instruments" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Musical_Instruments_DAttn"
#python PyTorchTEST.py -d "Musical_Instruments" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Musical_Instruments_DAttn"
#python PyTorchTEST.py -d "Musical_Instruments" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Musical_Instruments_DAttn"
#python PyTorchTEST.py -d "Musical_Instruments" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Musical_Instruments_DAttn"

#python PyTorchTEST.py -d "Office_Products" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Office_Products_DAttn"
python PyTorchTEST.py -d "Office_Products" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Office_Products_DAttn"
#python PyTorchTEST.py -d "Office_Products" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Office_Products_DAttn"
#python PyTorchTEST.py -d "Office_Products" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Office_Products_DAttn"
#python PyTorchTEST.py -d "Office_Products" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Office_Products_DAttn"

#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DAttn"
python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DAttn"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DAttn"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DAttn"
#python PyTorchTEST.py -d "Patio_Lawn_and_Garden" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Patio_Lawn_and_Garden_DAttn"

#python PyTorchTEST.py -d "Pet_Supplies" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Pet_Supplies_DAttn"
python PyTorchTEST.py -d "Pet_Supplies" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Pet_Supplies_DAttn"
#python PyTorchTEST.py -d "Pet_Supplies" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Pet_Supplies_DAttn"
#python PyTorchTEST.py -d "Pet_Supplies" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Pet_Supplies_DAttn"
#python PyTorchTEST.py -d "Pet_Supplies" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Pet_Supplies_DAttn"

#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DAttn"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DAttn"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DAttn"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DAttn"
#python PyTorchTEST.py -d "Sports_and_Outdoors" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_DAttn"

#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DAttn"
python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DAttn"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DAttn"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DAttn"
#python PyTorchTEST.py -d "Tools_and_Home_Improvement" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Tools_and_Home_Improvement_DAttn"

#python PyTorchTEST.py -d "Toys_and_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Toys_and_Games_DAttn"
python PyTorchTEST.py -d "Toys_and_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Toys_and_Games_DAttn"
#python PyTorchTEST.py -d "Toys_and_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Toys_and_Games_DAttn"
#python PyTorchTEST.py -d "Toys_and_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Toys_and_Games_DAttn"
#python PyTorchTEST.py -d "Toys_and_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Toys_and_Games_DAttn"

#python PyTorchTEST.py -d "Video_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1337 -gpu 0 -vb 1 -sm "Video_Games_DAttn"
python PyTorchTEST.py -d "Video_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Video_Games_DAttn"
#python PyTorchTEST.py -d "Video_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 5678 -gpu 0 -vb 1 -sm "Video_Games_DAttn"
#python PyTorchTEST.py -d "Video_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 1357 -gpu 0 -vb 1 -sm "Video_Games_DAttn"
#python PyTorchTEST.py -d "Video_Games" -m "DAttn" -e 10 -p 1 -v 50000 -K 5 -h1 10 -h2 50 -WED 100 -c_local 200 -c_global 100 -hiden_size 500 -output_size 50 -rs 2468 -gpu 0 -vb 1 -sm "Video_Games_DAttn"

