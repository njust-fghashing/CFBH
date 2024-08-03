##cub_bird aircraft stanford_car nabirds food101 vegfru##
conda activate base
python CFBH.py --dataset cub_bird --num_parts 64 
python CFBH.py --dataset food101  --num_parts 128 --batch_size 256
