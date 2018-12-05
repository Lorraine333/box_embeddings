import sys, os
sys.path.insert(0,os.getcwd())
from models import probability_model

dirname, filename = os.path.split(os.path.abspath(__file__))
DIR = "./".join(dirname.split("/")[:-1])

# python experiments/run_probability.py cpr_train.txt.gz cpr_test.txt.gz \
# train 512 0.5 10 1.0 1.0 0.0001 cube sigmoid norm pre 1e-6 softplus \ 
# 15 kl 11

params = {
    'run_dir': DIR,
    'data_dir': 'den_prob',
    'train_data': str(sys.argv[1]),
    'test_data': str(sys.argv[2]),
    'dev_data': 'cpr_dev.txt.gz',
    'vector_file': 'glove.ALL.txt.gz',

    'method': str(sys.argv[3]), # 'train' or 'test'

    'batch_size': int(sys.argv[4]),
    'hidden_dim': 512,  # hidden dim of LSTM
    'output_dim': 512,
    'dropout': float(sys.argv[5]),  # 1 = no dropout, 0.5 = dropout
    'num_epochs': int(sys.argv[6]),

    'lambda_px': float(sys.argv[7]),
    'lambda_cpr': float(sys.argv[8]),

    'learning_rate': float(sys.argv[9]),

    'mode':str(sys.argv[10]), # cube, cube_fix, poe, bilinear
    'cube': str(sys.argv[11]),

    'layer1_init': str(sys.argv[12]), #norm or pre
    'layer2_init': str(sys.argv[13]), #norm or pre
    'lambda_value': float(sys.argv[14]), #1e-6 or 1e-100
    'delta_acti': str(sys.argv[15]),
    'layer2_init_value': float(sys.argv[16]),
    'loss':str(sys.argv[17]), # kl or corr

    'lstm_seed': int(sys.argv[18])
}

tmp_dir = DIR+'/tmp/'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

probability_model.run(**params)
