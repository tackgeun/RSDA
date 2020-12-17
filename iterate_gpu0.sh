IRM=1.0
GPU=0
BATCH=36
LR=1e-3
IRM_R=0.01
for BATCH_R in 48
do
	for LR_R in 1e-3
	do
		python main.py --baseline DANN --init_method irm_target --refine_method irm_target --irm_feature last_hidden --irm_weight $IRM --lr $LR --gpu_id $GPU --stages 1 --batch_size $BATCH --irm_weight_refine $IRM_R --lr_refine $LR_R --batch_size_refine $BATCH_R --radius 6 --perf_log stage_radius --irm_warmup_step 0 --max_iter 50
	done
done

