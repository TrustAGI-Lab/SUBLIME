cd ..

python main.py -dataset cora -downstream_task clustering -ntrials 10 -sparse 0 -epochs 2500 -lr 0.001 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.5 -nlayers 2 -type_learner fgp -k 20 -sim_function cosine -activation_learner relu -gsl_mode structure_refinement -eval_freq 100 -tau 0.9999 -maskfeat_rate_learner 0.1 -maskfeat_rate_anchor 0.8 -contrast_batch_size 0 -c 0
