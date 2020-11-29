#!/usr/bin/env bash
# Example script for ANR
# We repeat the process 5 times using different random seeds

# E.g. ./__saved_models__/amazon_instant_video - ARNS/amazon_instant_video_ANRS_1337.pth
# python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"
# python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"
# python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"
# python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 5 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"
# python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 5 6 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_MSANR"

# python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"
# python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"
# python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"
# python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 5 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"
# python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 5 6 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_MSANR"

# python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_MSANR"
# python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_MSANR"
# python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_MSANR"
# python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 5 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_MSANR"
# python PyTorchTEST.py -d "Electronics" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 5 6 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_MSANR"

# python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"
# python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"
# python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"
# python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 5 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"
# python PyTorchTEST.py -d "Sports_and_Outdoors" -m "MSANR" -e 10 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 2 3 4 5 6 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_MSANR"



python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"

python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"

python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"

python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"

python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"

python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"
python PyTorchTEST.py -d "Cell_Phones_and_Accessories" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Cell_Phones_and_Accessories_VANRA"


python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"

python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"

python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"

python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"

python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"

python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"
python PyTorchTEST.py -d "Clothing_Shoes_and_Jewelry" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Clothing_Shoes_and_Jewelry_VANRA"


python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"

python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"

python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"

python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"

python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"

python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"
python PyTorchTEST.py -d "Electronics" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Electronics_VANRA"


python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 50 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"

python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 5 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"

python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 10 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"

python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 15 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"

python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 25 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"

python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 5 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 10 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 15 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"
python PyTorchTEST.py -d "Sports_and_Outdoors" -m "VANRA" -e 5 -dr 0.9 -WED 300 -p 1 -v 50000 -K 5 -h1 50 -kernel_list 3 -output_size 25 -rs 1234 -gpu 0 -vb 1 -sm "Sports_and_Outdoors_VANRA"