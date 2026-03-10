python benchmark_console.py \
  --model labram \
  --task full \
  --data-percentages 1.0 \
  --linear-probe \
  --no-wandb \
  --eval-noise-types gaussian one_over_f emg channel_dropout \
  --eval-noise-levels-db 30 20 10 0 -5 \
  --eval-noise-channel-dropout-prob 0.1 \
  --eval-noise-mix-all
