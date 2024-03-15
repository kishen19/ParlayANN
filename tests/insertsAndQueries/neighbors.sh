#!/bin/bash
cd ~/ParlayANN/tests/insertsAndQueries
make 

P=/ssd1/data/bigann
./neighbors -R 64 -L 128 -alpha 1.15 -two_pass 0 -data_type uint8 -dist_func Euclidian -graph_type aspen_flat -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-10M -res_path test.csv -base_path $P/base.1B.u8bin.crop_nb_10000000