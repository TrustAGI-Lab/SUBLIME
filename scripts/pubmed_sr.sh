cd ..

python main.py -dataset pubmed -ntrials 5 -sparse 1 -epochs_cls 200 -lr_cls 0.01 -w_decay_cls 0.0005 -hidden_dim_cls 32 -dropout_cls 0.5 -dropedge_cls 0.25 -nlayers_cls 2 -patience_cls 10 -epochs 1500 -lr 0.001 -w_decay 0.0 -hidden_dim 128 -rep_dim 64 -proj_dim 64 -dropout 0.5 -dropedge_rate 0.5 -nlayers 2 -type_learner mlp -k 10 -sim_function cosine -activation_learner relu -gsl_mode structure_refinement -eval_freq 20 -tau 0.999 -maskfeat_rate_learner 0.4 -maskfeat_rate_anchor 0.4 -contrast_batch_size 2000 -c 50
