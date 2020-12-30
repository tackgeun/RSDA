GPU=0
BATCH=48
LR=1e-3
LR_R=1e-3
IRM=1.0
IRM_R=1.0
for BATCH_R in 48
do
	for SEED in 1 2 3 4
	do
		python main.py --baseline DANN --init_method irm_target --refine_method irm_target --irm_feature last_hidden --irm_weight $IRM --lr $LR --gpu_id $GPU --stages 0 --batch_size $BATCH --irm_weight_refine $IRM_R --lr_refine $LR_R --batch_size_refine $BATCH_R --radius 6 --perf_log random_seed --irm_warmup_step 0 --irm_feature last_hidden --random_seed $SEED
	done
done

