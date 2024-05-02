for alpha in 5 15 20 35; do
    for K in 8 16 32 64 ; do
        echo "alpha: $alpha K: $K"
        python validate_2fold_moon.py --num_heads $K --alpha $alpha --num_fold=1 --val_ratio=0.5 --model_name openchat/openchat_3.5
        echo
        echo
    done
done
