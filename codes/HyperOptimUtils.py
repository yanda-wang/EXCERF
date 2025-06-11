import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skorch

from torch import optim
from skorch.utils import params_for
from warnings import filterwarnings

from Parameters import Parameters
from Models import Encoder, Decoder

parameters = Parameters()
PATIENT_RECORDS_FINAL = parameters.PATIENT_RECORDS_FINAL
VOC_FILE = parameters.VOC_FILE
GRAPH_FILE = parameters.GRAPH_FILE
DEVICE = parameters.DEVICE

OPT_SPLIT_TAG_ADMISSION = parameters.OPT_SPLIT_TAG_ADMISSION
OPT_SPLIT_TAG_CODE = parameters.OPT_SPLIT_TAG_CODE
OPT_MAX_EPOCH = parameters.OPT_MAX_EPOCH

MEDICATIONS_COUNT = parameters.MEDICATIONS_COUNT

LOSS_PROPORTION_BCE = parameters.LOSS_PROPORTION_BCE
LOSS_PROPORTION_MULTI = parameters.LOSS_PROPORTION_MULTI


class MedRecTopoTrans(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder(**params_for('encoder', kwargs))
        self.decoder = Decoder(**params_for('decoder', kwargs))

    def split_records(self, x):
        records = []
        split_records = np.split(x, np.where(x == OPT_SPLIT_TAG_ADMISSION)[0])
        admission = split_records[0]
        current_records = []
        split_code = np.split(admission, np.where(admission == OPT_SPLIT_TAG_CODE)[0])
        current_records.append(split_code[0].tolist())
        current_records.append(split_code[1][1:].tolist())
        current_records.append(split_code[2][1:].tolist())
        records.append(current_records)

        for admission in split_records[1:]:
            current_records = []
            split_code = np.split(admission[1:], np.where(admission[1:] == OPT_SPLIT_TAG_CODE)[0])
            current_records.append(split_code[0].tolist())
            current_records.append(split_code[1][1:].tolist())
            current_records.append(split_code[2][1:].tolist())
            records.append(current_records)

        return records

    def forward(self, x):
        records = self.split_records(x)
        query, memory_keys, memory_values = self.encoder(records)
        output = self.decoder(query, memory_keys, memory_values)
        return output


class MedRecTrainer(skorch.NeuralNet):
    def __init__(self, *args, optimizer_encoder=optim.Adam, optimizer_decoder=optim.Adam, **kwargs):
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_decoder = optimizer_decoder
        super().__init__(*args, **kwargs)

    def initialize_optimizer(self, triggered_directly=True):
        kwargs = self._get_params_for('optimizer_encoder')
        self.optimizer_encoder_ = self.optimizer_encoder(self.module_.encoder.parameters(), **kwargs)
        kwargs = self._get_params_for('optimizer_decoder')
        self.optimizer_decoder_ = self.optimizer_decoder(self.module_.decoder.parameters(), **kwargs)

    def train_step(self, Xi, yi, **fit_params):
        yi = skorch.utils.to_numpy(yi).tolist()[0]

        self.module_.train()
        self.optimizer_encoder_.zero_grad()
        self.optimizer_decoder_.zero_grad()

        y_pred = self.infer(Xi)
        loss = self.get_loss(y_pred, yi)
        loss.backward()

        self.optimizer_encoder_.step()
        self.optimizer_decoder_.step()

        return {'loss': loss, 'y_pred': y_pred}

    def infer(self, Xi, yi=None):
        Xi = skorch.utils.to_numpy(Xi)[0]
        return self.module_(Xi)

    def get_loss(self, y_pred, y_true, **kwargs):
        loss_bce_target = np.zeros((1, MEDICATIONS_COUNT))
        loss_bce_target[:, y_true] = 1
        loss_multi_target = np.full((1, MEDICATIONS_COUNT), -1)
        for idx, item in enumerate(y_true):
            loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(y_pred, torch.FloatTensor(loss_bce_target).to(self.device))
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(y_pred),
                                              torch.LongTensor(loss_multi_target).to(self.device))
        loss = LOSS_PROPORTION_BCE * loss_bce + LOSS_PROPORTION_MULTI * loss_multi
        return loss

    def _predict(self, X, most_probable=True):
        filterwarnings('error')
        y_probas = []

        for output in self.forward_iter(X, training=False):
            if most_probable:
                predict_prob = skorch.utils.to_numpy(torch.sigmoid(output))[0]
                predict_multi_hot = predict_prob.copy()

                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)

                predict_multi_hot[predict_multi_hot >= 0.5] = 1
                predict_multi_hot[predict_multi_hot < 0.5] = 0
                predict = np.where(predict_multi_hot == 1)[0]
            else:
                predict = skorch.utils.to_numpy(torch.sigmoid(output))[0]
            y_probas.append(predict)

        return np.array(y_probas, dtype=object)

    def predict_proba(self, X):
        return self._predict(X, most_probable=False)

    def predict(self, X):
        return self._predict(X, most_probable=True)
