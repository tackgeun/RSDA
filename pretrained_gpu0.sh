GPU=0
LR_R=1e-3
IRM_R=1.0
BATCH_R=48
RADIUS_R=10.0
for SEED in 1
do
		python main.py --baseline DANN --init_method pretrained --refine_method irm_target --irm_feature last_hidden --gpu_id $GPU --stages 1 --irm_weight_refine $IRM_R --lr_refine $LR_R --batch_size_refine $BATCH_R --radius_refine $RADIUS_R --perf_log pretrained_refine --irm_warmup_step 0 --irm_feature last_hidden --random_seed $SEED --pretrained_path data/visda-2017/pseudo_list/list_DANN_R6.0_seed1_irmlast_hidden_irm_target_lr_decay0.001_b48_irm0.1.txt --max_iter 50
done

