for ((i=1; i<=20; i++))
do
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --test_FP --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack db_1 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack db_2 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack db_3 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack db_4 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack db_5 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack dl --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack pa_1 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack pa_4 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack pa_5 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack pa_6 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
# python la.py --img_num 5 --start_iter 5 --end_iter 10 --attack pa_7 --cuda cuda:0 --alpha 0.05 --runname LA-v4 --threshold 0.30
done