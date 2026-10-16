export LATTE_PATH=~/LATTE/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/bml/install/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/qmd-progress/install/lib64/

srun --cpu-bind=cores -n 16 -c 8 ./select_gpu_device python main.py --md_iter 1000 --temp 300 --dt 0.5 > out.log &
