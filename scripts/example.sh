python -m torch.distributed.launch --nproc_per_node 4 train.py \
  --name test_run \
  --model roberta-base \
  --task QA \
  --data_dir data/squad-v2.0 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --amp \
  --amp_opt_level O2 \
  --num_epochs 3 \
  --learning_rate 3e-5 \
  --warmup_proportion 0.1
