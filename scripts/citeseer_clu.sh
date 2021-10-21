cd ..

python main.py -dataset citeseer -downstream_task clustering -ntrials 10 -sparse 0 -epochs 1000 -lr 0.001 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.5 -nlayers 2 -type_learner att -k 20 -sim_function cosine -activation_learner tanh -gsl_mode structure_refinement -eval_freq 100 -tau 0.999 -maskfeat_rate_learner 0.4 -maskfeat_rate_anchor 0.9 -contrast_batch_size 0 -c 0
