@echo off
setlocal

set "DATASET=biosnap"
rem set "DATASET=celegans"
rem set "DATASET=yourdata"

rem set "radius=1"
set "radius=2"
rem set "radius=3"

rem set "ngram=2"
set "ngram=3"

set "dim=10"
set "layer_gnn=3"
set "side=5"
set /a window=2*%side%+1
set "layer_cnn=3"
set "layer_output=3"
set "lr=1e-3"
set "lr_decay=0.5"
set "decay_interval=10"
set "weight_decay=1e-6"
set "iteration=60"

set "setting=%DATASET%--radius%radius%--ngram%ngram%--dim%dim%--layer_gnn%layer_gnn%--window%window%--layer_cnn%layer_cnn%--layer_output%layer_output%--lr%lr%--lr_decay%lr_decay%--decay_interval%decay_interval%--weight_decay%weight_decay%--iteration%iteration%"
python "%~dp0run_training.py" %DATASET% %radius% %ngram% %dim% %layer_gnn% %window% %layer_cnn% %layer_output% %lr% %lr_decay% %decay_interval% %weight_decay% %iteration% %setting%
