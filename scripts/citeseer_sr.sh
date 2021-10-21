cd ..

python main.py -dataset citeseer -ntrials 5 -sparse 0 -epochs_cls 200 -lr_cls 0.001 -w_decay_cls 0.05 -hidden_dim_cls 32 -dropout_cls 0.5 -dropedge_cls 0.5 -nlayers_cls 2 -patience_cls 10 -epochs 1000 -lr 0.001 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.25 -nlayers 2 -type_learner att -k 20 -sim_function cosine -activation_learner tanh -gsl_mode structure_refinement -eval_freq 20 -tau 0.9999 -maskfeat_rate_learner 0.6 -maskfeat_rate_anchor 0.8 -contrast_batch_size 0 -c 0
