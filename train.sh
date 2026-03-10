cd /media/win/Users/Juampi/Materia/Ciencia\ de\ Datos\ Geográficos/__tesis/TGraphormer/src
python run_pretrain.py \
  --epochs 50 \
  --batch_size 4 \
  --model mae_graph_mini \
  --dataset_name "awto_591_Las Condes_202212_202301" \
  --n_hist 24 \
  --n_pred 12 \
  --output_dir ./output_dir \
  --device cuda \
  --wandb_offline