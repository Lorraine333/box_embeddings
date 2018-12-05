import os
import sys
sys.path.insert(0,os.getcwd())
from models import entail_prob_model
from models import entail_lstm_model


phase = sys.argv[1]
dirname, filename = os.path.split(os.path.abspath(__file__))
DIR = "/".join(dirname.split("/")[:-1])
params = {
    'run_dir': DIR,
    'exp_name_lstm': 'lstm',
    'exp_name_full': 'concat',
    'data_dir': 'snli',

    'train_prob_data': 'snli_predict_train.txt',
    'test_prob_data': 'snli_predict_test.txt',
    'dev_prob_data': 'snli_predict_dev.txt',

    'train_entail_data': 'snli_1.0_train.txt',
    'test_entail_data': 'snli_1.0_test.txt',
    'dev_entail_data': 'snli_1.0_dev.txt',

    'vector_file': 'glove.ALL.txt.gz',

    'method': 'train',  # 'train' or 'test'

    'batch_size': 512,
    'hidden_dim': 100,  # hidden dim of LSTM
    'dropout_lstm': 0.85,  # 1 = no dropout, 0.5 = dropout

    'dropout_cpr': 0.8,
    'output_dim_cpr': 300,

    'num_epochs': 10,

    'phase': phase, # intermed or classifier
}

tmp_dir = DIR+'/tmp/'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if params['phase'] == 'lstm':

    # Train LSTM entailment model from SNLI data
    entail_lstm_model.run(**params)

elif params['phase'] == 'lstm_plus_feat':

    # Train entailment prediction model by appending predicted probability
    # features from file to output of previously trained LSTM model
    entail_prob_model.run(**params)

