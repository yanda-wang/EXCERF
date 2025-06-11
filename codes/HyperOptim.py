import os
import sys
import skorch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch import optim
from HyperOptimUtils import MedRecTopoTrans, MedRecTrainer
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import load as optim_load
from skopt.callbacks import CheckpointSaver

from Parameters import Parameters

parameters = Parameters()
PATIENT_RECORDS_FINAL = parameters.PATIENT_RECORDS_FINAL
VOC_FILE = parameters.VOC_FILE
GRAPH_FILE = parameters.GRAPH_FILE
DEVICE = parameters.DEVICE

DIAGNOSE_INDEX = parameters.DIAGNOSE_INDEX
PROCEDURE_INDEX = parameters.PROCEDURE_INDEX
MEDICATION_INDEX = parameters.MEDICATION_INDEX

OPT_SPLIT_TAG_ADMISSION = parameters.OPT_SPLIT_TAG_ADMISSION
OPT_SPLIT_TAG_CODE = parameters.OPT_SPLIT_TAG_CODE
OPT_MAX_EPOCH = parameters.OPT_MAX_EPOCH

TRAIN_RATIO = parameters.RECORDS_TRAIN_RATIO
TEST_RATIO = parameters.RECORDS_TEST_RATIO

LOG_PATH = 'data/log'
LOG_FILE = 'data/log/MedRec_optimization.log'
CHECKPOINT_PATH = 'data/parameters_model'
CHECKPOINT_FILE = 'data/parameters_model/MedRec_checkpoint.pkl'


def concatenate_single_admission(adm):
    x = adm[DIAGNOSE_INDEX] + [OPT_SPLIT_TAG_CODE] + adm[PROCEDURE_INDEX] + [OPT_SPLIT_TAG_CODE] + adm[MEDICATION_INDEX]
    return x


def concatenate_all_admissions(patient):
    x = concatenate_single_admission(patient[0])
    for adm in patient[1:]:
        current_adm = concatenate_single_admission(adm)
        x = x + [OPT_SPLIT_TAG_ADMISSION] + current_adm
    return x


def get_x_y(patient_records):
    x = []
    y = []
    for patient in patient_records:
        for idx, adm in enumerate(patient):
            current_records = patient[:idx + 1]
            current_x = concatenate_all_admissions(current_records)
            x.append(np.array(current_x))
            target = adm[MEDICATION_INDEX]
            y.append(np.array(target))
    return np.array(x, dtype=object), np.array(y, dtype=object)


def get_data(patient_records_file):
    patient_records = pd.read_pickle(patient_records_file)
    train_count = int(len(patient_records) * TRAIN_RATIO)
    test_count = int(len(patient_records) * TEST_RATIO)
    train_records = patient_records[:train_count]
    test_records = patient_records[train_count:train_count + test_count]

    train_x, train_y = get_x_y(train_records)
    test_x, test_y = get_x_y(test_records)
    return train_x, train_y, test_x, test_y


def get_metric(y_predict, y_target):
    f1 = []
    for yp, yt in zip(y_predict, y_target):
        if yp.shape[0] == 0:
            f1.append(0)
        else:
            intersection = list(set(yp.tolist()) & set(yt.tolist()))
            precision = float(len(intersection)) / len(yp.tolist())
            recall = float(len(intersection)) / len(yt.tolist())
            if precision + recall == 0:
                f1.append(0)
            else:
                f1.append(2.0 * precision * recall / (precision + recall))

    avg_f1 = np.mean(np.array(f1))
    return avg_f1


search_space = [Categorical(categories=['64', '128', '192', '256'], name='hidden_size'),
                Real(low=0, high=0.9, name='code_encoding_dropout_rate'),
                Integer(low=1, high=10, name='trans_layer_num_sub'),
                Real(low=0, high=0.9, name='trans_dropout_rate'),
                Real(low=0, high=0.9, name='trans_attn_dropout_rate'),
                Integer(low=1, high=15, name='trans_head_num'),
                Integer(low=1, high=15, name='global_memory_num'),

                Integer(low=1, high=5, name='gru_n_layers'),
                Real(low=0, high=0.9, name='gru_dropout_rate'),

                Categorical(categories=['dot', 'general', 'concat'], name='attn_type_kv'),
                Categorical(categories=['dot', 'general', 'concat'], name='attn_type_embedding'),
                Real(low=0, high=0.9, name='medications_ehr_dropout_rate'),
                Integer(low=1, high=20, name='multi_hop_count'),

                Real(low=1e-5, high=1e-2, prior='log-uniform', name='optimizer_encoder_lr'),
                Real(low=1e-5, high=1e-2, prior='log-uniform', name='optimizer_decoder_lr')
                ]


@use_named_args(dimensions=search_space)
def fitness(hidden_size, code_encoding_dropout_rate, trans_layer_num_sub, trans_dropout_rate, trans_attn_dropout_rate,
            trans_head_num, global_memory_num, gru_n_layers, gru_dropout_rate, attn_type_kv, attn_type_embedding,
            medications_ehr_dropout_rate, multi_hop_count, optimizer_encoder_lr, optimizer_decoder_lr):
    hidden_size = int(hidden_size)

    print('*' * 30)
    print('hyper-parameters:')
    print('hidden size:', hidden_size)
    print('code encoding dropout rate:', code_encoding_dropout_rate)
    print('trans layer num sub:', trans_layer_num_sub)
    print('trans dropout rate:', trans_dropout_rate)
    print('trans attn dropout rate:', trans_attn_dropout_rate)
    print('trans head num:', trans_head_num)
    print('global memory num:', global_memory_num)

    print('gru n layers:', gru_n_layers)
    print('gru dropout rate:', gru_dropout_rate)
    print('attn type kv:', attn_type_kv)
    print('attn type embedding:', attn_type_embedding)
    print('medication ehr dropout rate:', medications_ehr_dropout_rate)
    print('multi hop count:', multi_hop_count)

    print('encoder optimizer lr:', optimizer_encoder_lr)
    print('decoder optimizer lr:', optimizer_decoder_lr)

    print()

    model = MedRecTrainer(criterion=nn.BCEWithLogitsLoss, optimizer_encoder=optim.Adam,
                          optimizer_decoder=optim.Adam, max_epochs=OPT_MAX_EPOCH, batch_size=1,
                          train_split=None, callbacks=[skorch.callbacks.ProgressBar(batches_per_epoch='auto'), ],
                          device=DEVICE, module=MedRecTopoTrans,

                          module__encoder__hidden_size=hidden_size,
                          module__encoder__code_encoding_dropout_rate=code_encoding_dropout_rate,
                          module__encoder__trans_layer_num_sub=trans_layer_num_sub,
                          module__encoder__trans_dropout_rate=trans_dropout_rate,
                          module__encoder__trans_attn_dropout_rate=trans_attn_dropout_rate,
                          module__encoder__trans_head_num=trans_head_num,
                          module__encoder__global_memory_num=global_memory_num,
                          module__encoder__gru_n_layers=gru_n_layers,
                          module__encoder__gru_dropout_rate=gru_dropout_rate,
                          module__encoder__graph_file=GRAPH_FILE, module__encoder__voc_file=VOC_FILE,
                          module__encoder__device=DEVICE,

                          module__decoder__hidden_size=hidden_size, module__decoder__attn_type_kv=attn_type_kv,
                          module__decoder__attn_type_embedding=attn_type_embedding,
                          module__decoder__medications_ehr_dropout_rate=medications_ehr_dropout_rate,
                          module__decoder__multi_hop_count=multi_hop_count, module__decoder__graph_file=GRAPH_FILE,
                          module__decoder__voc_file=VOC_FILE, module__decoder__device=DEVICE,

                          optimizer_encoder__lr=optimizer_encoder_lr,
                          optimizer_decoder__lr=optimizer_decoder_lr
                          )

    train_x, train_y, test_x, test_y = get_data(PATIENT_RECORDS_FINAL)
    model.fit(train_x, train_y)
    predict_y = model.predict(test_x)
    metric = get_metric(predict_y, test_y)

    print('metric:{0:.4f}'.format(metric))

    return -metric


def optimize(n_calls):
    sys.stdout = open(LOG_FILE, 'a')
    checkpoint_saver = CheckpointSaver(CHECKPOINT_FILE, compress=9)

    # optim_result = optim_load(CHECKPOINT_FILE)
    # examined_values = optim_result.x_iters
    # observed_values = optim_result.func_vals
    # result = gp_minimize(fitness, search_space, n_calls=n_calls, verbose=True, callback=[checkpoint_saver],
    #                      x0=examined_values, y0=observed_values, n_initial_points=-len(examined_values))

    result = gp_minimize(fitness, search_space, n_calls=n_calls, verbose=True, callback=[checkpoint_saver])
    print('**********************************')
    print('best result:')
    print('metric:', -result.fun)
    print('optimal hyper-parameters')

    space_dim_name = [item.name for item in search_space]
    for hyper, value in zip(space_dim_name, result.x):
        print(hyper, value)
    sys.stdout.close()


if __name__ == "__main__":
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)
    optimize(25)
