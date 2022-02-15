import argparse
import datetime
import dill
import os
import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F

from Models import Encoder, Decoder
from Parameters import Parameters
from sklearn.metrics import average_precision_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

parameters = Parameters()


class ModelTraining:
    def __init__(self, patient_records_file, graph_file=parameters.GRAPH_FILE, voc_file=parameters.VOC_FILE,
                 device=parameters.DEVICE):
        self.patient_records_file = patient_records_file
        self.graph_file = graph_file
        graph = dill.load(open(graph_file, 'rb'))
        self.distance_graph_diagnoses = graph['distance_graph_diagnoses']
        self.weight_graph_diagnoses = graph['weight_graph_diagnoses']
        self.distance_graph_procedures = graph['distance_graph_procedures']
        self.weight_graph_procedures = graph['weight_graph_procedures']
        self.ehr_graph = graph['co-occurrence_graph']

        self.voc_file = voc_file
        self.voc = dill.load(open(voc_file, 'rb'))
        self.diagnoses_count = self.voc.get_diagnoses_count()
        self.procedures_count = self.voc.get_procedures_count()
        self.medications_count = self.voc.get_medications_count()
        self.output_size = self.medications_count

        self.device = device
        self.evaluate_utils = EvaluationUtil()

    def loss_function(self, target_medications, predict_medications, proportion_bce, proportion_multi):
        loss_bce_target = np.zeros((1, self.medications_count))
        loss_bce_target[:, target_medications] = 1
        loss_multi_target = np.full((1, self.medications_count), -1)
        for idx, item in enumerate(target_medications):
            loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(predict_medications,
                                                      torch.FloatTensor(loss_bce_target).to(self.device))
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(predict_medications),
                                              torch.LongTensor(loss_multi_target).to(self.device))
        loss = proportion_bce * loss_bce + proportion_multi * loss_multi
        return loss

    def get_performance_on_testset(self, encoder, decoder, patient_records):
        jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg = [], [], [], [], []
        for patient in patient_records:
            for idx, adm in enumerate(patient):
                current_records = patient[:idx + 1]
                query, memory_keys, memory_values = encoder(current_records)
                predict_output = decoder(query, memory_keys, memory_values)

                target_medications = adm[parameters.MEDICATION_INDEX]
                target_multi_hot = np.zeros(self.medications_count)
                target_multi_hot[target_medications] = 1
                predict_prob = torch.sigmoid(predict_output).detach().cpu().numpy()[0]
                predict_multi_hot = predict_prob.copy()

                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)

                predict_multi_hot[predict_multi_hot >= 0.5] = 1
                predict_multi_hot[predict_multi_hot < 0.5] = 0
                predict_medications = list(np.where(predict_multi_hot == 1)[0])

                jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)
                precision = self.evaluate_utils.metric_precision(predict_medications, target_medications)
                recall = self.evaluate_utils.metric_recall(predict_medications, target_medications)
                f1 = self.evaluate_utils.metric_f1(precision, recall)
                prauc = self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

                jaccard_avg.append(jaccard)
                precision_avg.append(precision)
                recall_avg.append(recall)
                f1_avg.append(f1)
                prauc_avg.append(prauc)

        jaccard_avg = np.mean(np.array(jaccard_avg))
        precision_avg = np.mean(np.array(precision_avg))
        recall_avg = np.mean(np.array(recall_avg))
        f1_avg = np.mean(np.array(f1_avg))
        prauc_avg = np.mean(np.array(prauc_avg))

        return jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg

    def trainIters(self, encoder, decoder, encoder_optimizer, decoder_optimizer, proportion_bce, proportion_multi,
                   patient_records_train, patient_records_test, save_model_path, n_epoch, print_every_iteration=100,
                   save_every_epoch=1, trained_epoch=0, trained_iteration=0):
        start_epoch = trained_epoch + 1
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        log_file = open(os.path.join(save_model_path, 'medrec_loss.log'), 'a+')
        encoder_lr_scheduler = ReduceLROnPlateau(encoder_optimizer, mode='max', patience=5, factor=0.1)
        decoder_lr_scheduler = ReduceLROnPlateau(decoder_optimizer, mode='max', patience=5, factor=0.1)

        for epoch in range(start_epoch, start_epoch + n_epoch):
            print_loss = []
            iteration = 0
            for patient in patient_records_train:
                for idx, adm in enumerate(patient):
                    trained_iteration += 1
                    iteration += 1
                    current_records = patient[:idx + 1]
                    target_medications = adm[parameters.MEDICATION_INDEX]
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()

                    query, memory_keys, memory_values = encoder(current_records)
                    predict_output = decoder(query, memory_keys, memory_values)
                    loss = self.loss_function(target_medications, predict_output, proportion_bce, proportion_multi)
                    print_loss.append(loss.item())
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()

                    if iteration % print_every_iteration == 0:
                        print_loss_avg = np.mean(np.array(print_loss))
                        print_loss = []
                        print(
                            'epoch: {}; time: {}; iteration: {}; train loss: {}'.format(epoch, datetime.datetime.now(),
                                                                                        trained_iteration,
                                                                                        print_loss_avg))
                        log_file.write('epoch: {}; time: {}; iteration: {}; train loss: {}\n'.format(epoch,
                                                                                                     datetime.datetime.now(),
                                                                                                     trained_iteration,
                                                                                                     print_loss_avg))

            encoder.eval()
            decoder.eval()
            jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg = self.get_performance_on_testset(encoder,
                                                                                                        decoder,
                                                                                                        patient_records_test)
            encoder.train()
            decoder.train()

            print(
                'epoch: {}; time: {}; iteration: {}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; prauc_test: {:.4f}'.format(
                    epoch, datetime.datetime.now(), trained_iteration, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    prauc_avg))
            log_file.write(
                'epoch: {}; time: {}; iteration: {}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; prauc_test: {:.4f}\n'.format(
                    epoch, datetime.datetime.now(), trained_iteration, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    prauc_avg))

            encoder_lr_scheduler.step(f1_avg)
            decoder_lr_scheduler.step(f1_avg)

            if epoch % save_every_epoch == 0:
                torch.save({'medrec_epoch': epoch,
                            'medrec_iteration': trained_iteration,
                            'encoder': encoder.state_dict(),
                            'decoder': decoder.state_dict(),
                            'encoder_optimizer': encoder_optimizer.state_dict(),
                            'decoder_optimizer': decoder_optimizer.state_dict()
                            }, os.path.join(save_model_path, 'medrec_{}_{}_{:.4f}.checkpoint').format(epoch,
                                                                                                      trained_iteration,
                                                                                                      f1_avg))
        log_file.close()

    def train(self, hidden_size, code_encoding_dropout_rate, trans_layer_num_sub, trans_dropout_rate,
              trans_attn_dropout_rate, trans_head_num, global_memory_num, gru_n_layers, gru_dropout_rate, encoder_lr,
              attn_type_kv, attn_type_embedding, medications_ehr_dropout_rate, multi_hop_count, decoder_lr,
              proportion_bce=parameters.LOSS_PROPORTION_BCE, proportion_multi=parameters.LOSS_PROPORTION_MULTI,
              records_train_ratio=parameters.RECORDS_TRAIN_RATIO, records_test_ratio=parameters.RECORDS_TEST_RATIO,
              n_epoch=40, print_every_iteration=100, save_every_epoch=1, save_model_dir='data/model',
              load_model_name=None):
        print('initializing >>>')
        print('build encoder and decoder >>>')

        encoder = Encoder(hidden_size, code_encoding_dropout_rate, trans_layer_num_sub, trans_dropout_rate,
                          trans_attn_dropout_rate, trans_head_num, global_memory_num, gru_n_layers, gru_dropout_rate,
                          self.graph_file, self.voc_file, self.device)
        decoder = Decoder(hidden_size, attn_type_kv, attn_type_embedding, medications_ehr_dropout_rate, multi_hop_count,
                          self.graph_file, self.voc_file, self.device)

        if load_model_name is not None:
            print('load model from checkpoint file: ', load_model_name)
            if torch.cuda.is_available():
                checkpoint = torch.load(load_model_name)
            else:
                checkpoint = torch.load(load_model_name, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['encoder']
            decoder_sd = checkpoint['decoder']
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.train()
        decoder.train()

        print('build optimizer >>>')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)
        if load_model_name:
            encoder_optimizer_sd = checkpoint['encoder_optimizer']
            decoder_optimizer_sd = checkpoint['decoder_optimizer']
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        print('load patient records >>>')
        patient_records = pd.read_pickle(self.patient_records_file)
        split_point = int(len(patient_records) * records_train_ratio)
        test_count = int(len(patient_records) * records_test_ratio)
        patient_records_train = patient_records[:split_point]
        patient_records_test = patient_records[split_point:split_point + test_count]

        medrec_trained_epoch = 0
        medrec_trained_iteration = 0
        if load_model_name:
            medrec_trained_n_epoch_sd = checkpoint['medrec_epoch']
            medrec_trained_n_iteration_sd = checkpoint['medrec_iteration']
            medrec_trained_epoch = medrec_trained_n_epoch_sd
            medrec_trained_iteration = medrec_trained_n_iteration_sd

        model_structure = str(hidden_size) + '_' + str(trans_layer_num_sub) + '_' + str(trans_head_num) + '_' + str(
            global_memory_num) + '_' + str(gru_n_layers)
        model_parameters = str(trans_dropout_rate) + '_' + str(trans_attn_dropout_rate) + '_' + str(gru_dropout_rate)
        save_model_path = os.path.join(save_model_dir, model_structure, model_parameters)

        print('start training >>>')
        self.trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, proportion_bce, proportion_multi,
                        patient_records_train, patient_records_test, save_model_path, n_epoch, print_every_iteration,
                        save_every_epoch, medrec_trained_epoch, medrec_trained_iteration)


class EvaluationUtil:
    def precision_auc(self, predict_prob, target_prescriptions):
        return average_precision_score(target_prescriptions, predict_prob, average='macro')

    def metric_jaccard_similarity(self, predict_prescriptions, target_prescriptions):
        union = list(set(predict_prescriptions) | set(target_prescriptions))
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        jaccard = float(len(intersection)) / len(union)
        return jaccard

    def metric_precision(self, predict_prescriptions, target_prescriptions):
        if len(set(predict_prescriptions)) == 0:
            return 0
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        # precision = float(len(intersection)) / len(set(predict_prescriptions))
        precision = float(len(intersection)) / len(predict_prescriptions)
        return precision

    def metric_recall(self, predict_prescriptions, target_prescriptions):
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        # recall = float(len(intersection)) / len(set(target_prescriptions))
        recall = float(len(intersection)) / len(target_prescriptions)
        return recall

    def metric_f1(self, precision, recall):
        if precision + recall == 0:
            return 0
        f1 = 2.0 * precision * recall / (precision + recall)
        return f1


class ModelEvaluation:
    def __init__(self, patient_records_file, predict_prob_threshold=0.5, graph_file=parameters.GRAPH_FILE,
                 voc_file=parameters.VOC_FILE, device=parameters.DEVICE):
        self.patient_records_file = patient_records_file
        self.predict_prob_threshold = predict_prob_threshold
        self.graph_file = graph_file
        graph = dill.load(open(graph_file, 'rb'))
        self.distance_graph_diagnoses = graph['distance_graph_diagnoses']
        self.weight_graph_diagnoses = graph['weight_graph_diagnoses']
        self.distance_graph_procedures = graph['distance_graph_procedures']
        self.weight_graph_procedures = graph['weight_graph_procedures']
        self.ehr_graph = graph['co-occurrence_graph']

        self.voc_file = voc_file
        self.voc = dill.load(open(voc_file, 'rb'))
        self.diagnoses_count = self.voc.get_diagnoses_count()
        self.procedures_count = self.voc.get_procedures_count()
        self.medications_count = self.voc.get_medications_count()
        self.output_size = self.medications_count

        self.device = device
        self.evaluate_utils = EvaluationUtil()

    def metric_jaccard_similarity(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)

    def metric_precision(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_precision(predict_medications, target_medications)

    def metric_recall(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_recall(predict_medications, target_medications)

    def metric_f1(self, precision, recall):
        return self.evaluate_utils.metric_f1(precision, recall)

    def metric_prauc(self, predict_prob, target_multi_hot):
        return self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

    def evaluateIters(self, encoder, decoder, patient_records, save_results_file=None):
        jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg = [], [], [], [], []
        predict_result_patient_records = []
        for patient in patient_records:
            current_new_patient_records = []
            for idx, adm in enumerate(patient):
                current_records = patient[:idx + 1]
                query, memory_keys, memory_values = encoder(current_records)
                predict_output = decoder(query, memory_keys, memory_values)

                target_medications = adm[parameters.MEDICATION_INDEX]
                target_multi_hot = np.zeros(self.medications_count)
                target_multi_hot[target_medications] = 1
                predict_prob = torch.sigmoid(predict_output).detach().cpu().numpy()[0]
                predict_multi_hot = predict_prob.copy()

                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)

                predict_multi_hot[predict_multi_hot >= self.predict_prob_threshold] = 1
                predict_multi_hot[predict_multi_hot < self.predict_prob_threshold] = 0
                predict_medications = list(np.where(predict_multi_hot == 1)[0])

                jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)
                precision = self.evaluate_utils.metric_precision(predict_medications, target_medications)
                recall = self.evaluate_utils.metric_recall(predict_medications, target_medications)
                f1 = self.evaluate_utils.metric_f1(precision, recall)
                prauc = self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)
                jaccard_avg.append(jaccard)
                precision_avg.append(precision)
                recall_avg.append(recall)
                f1_avg.append(f1)
                prauc_avg.append(prauc)

                adm.append(predict_medications)
                current_new_patient_records.append(adm)
            predict_result_patient_records.append(current_new_patient_records)

        jaccard_avg = np.mean(np.array(jaccard_avg))
        precision_avg = np.mean(np.array(precision_avg))
        recall_avg = np.mean(np.array(recall_avg))
        f1_avg = np.mean(np.array(f1_avg))
        prauc_avg = np.mean(np.array(prauc_avg))

        dill.dump(obj=predict_result_patient_records,
                  file=open(os.path.join(save_results_file, 'predict_result.pkl'), 'wb'))

        print('evaluation result:')
        print('  jaccard:', jaccard_avg)
        print('precision:', precision_avg)
        print('   recall:', recall_avg)
        print('       f1:', f1_avg)
        print('    prauc:', prauc_avg)

    def evaluate(self, load_model_name, hidden_size, trans_layer_num_sub, trans_head_num, global_memory_num,
                 gru_n_layers, attn_type_kv, attn_type_embedding, multi_hop_count, save_results_path,
                 records_train_ratio=parameters.RECORDS_TRAIN_RATIO, records_test_ratio=parameters.RECORDS_TEST_RATIO):
        print('initializing >>>')
        print('build encoder and decoder >>>')
        encoder = Encoder(hidden_size, 0, trans_layer_num_sub, 0, 0, trans_head_num, global_memory_num, gru_n_layers, 0,
                          self.graph_file, self.voc_file, self.device)
        decoder = Decoder(hidden_size, attn_type_kv, attn_type_embedding, 0, multi_hop_count, self.graph_file,
                          self.voc_file, self.device)

        print('load model from checkpoint file:', load_model_name)
        if torch.cuda.is_available():
            checkpoint = torch.load(load_model_name)
        else:
            checkpoint = torch.load(load_model_name, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['encoder']
        decoder_sd = checkpoint['decoder']
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.eval()
        decoder.eval()

        print('load patient records >>>')
        patient_records = pd.read_pickle(self.patient_records_file)
        split_point = int(len(patient_records) * records_train_ratio)
        test_count = int(len(patient_records) * records_test_ratio)
        # patient_records_train = patient_records[:split_point]
        # patient_records_test = patient_records[split_point:split_point + test_count]
        patient_records_validation = patient_records[split_point + test_count:]

        if not os.path.exists(save_results_path):
            os.makedirs(save_results_path)

        print('start evaluation >>>')
        self.evaluateIters(encoder, decoder, patient_records_validation, save_results_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--patient_records_file', default=parameters.PATIENT_RECORDS_FINAL, type=str, required=False)
    parser.add_argument('--voc_file', default=parameters.VOC_FILE, type=str, required=False)
    parser.add_argument('--graph_file', default=parameters.GRAPH_FILE, type=str, required=False)

    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')

    # parameters of the encoder
    parser.add_argument('--hidden_size', default=192, type=int)
    parser.add_argument('--code_encoding_dropout_rate', default=0, type=float)
    parser.add_argument('--trans_layer_num_sub', default=6, type=int)
    parser.add_argument('--trans_dropout_rate', default=0.19469342, type=float)
    parser.add_argument('--trans_attn_dropout_rate', default=0.17475175, type=float)
    parser.add_argument('--trans_head_num', default=12, type=int)
    parser.add_argument('--global_memory_num', default=10, type=int)
    parser.add_argument('--gru_n_layers', default=1, type=int)
    parser.add_argument('--gru_dropout_rate', default=0.47920831, type=float)
    parser.add_argument('--encoder_lr', default=0.00001146, type=float)

    # parameters of the decoder
    parser.add_argument('--attn_type_kv', default='dot', type=str)
    parser.add_argument('--attn_type_embedding', default='dot', type=str)
    parser.add_argument('--medications_ehr_dropout_rate', default=0.09189291, type=float)
    parser.add_argument('--multi_hop_count', default=18, type=int)
    parser.add_argument('--decoder_lr', default=0.00029095, type=float)
    parser.add_argument('--predict_prob_threshold', default=0.5, type=float)

    parser.add_argument('--loss_proportion_bce', default=parameters.LOSS_PROPORTION_BCE, type=float)
    parser.add_argument('--loss_proportion_multi', default=parameters.LOSS_PROPORTION_MULTI, type=float)
    parser.add_argument('--records_train_ratio', default=parameters.RECORDS_TRAIN_RATIO, type=float)
    parser.add_argument('--records_test_ratio', default=parameters.RECORDS_TEST_RATIO, type=float)
    parser.add_argument('--save_model_dir', default='data/model', type=str, required=False)
    parser.add_argument('--n_epoch', default=40, type=int)
    parser.add_argument('--print_every_iteration', default=100, type=int)
    parser.add_argument('--save_every_epoch', default=1, type=int)
    parser.add_argument('--load_model_name', default=None, type=str)

    parser.add_argument('--save_predict_results_dir', default='data/predict_results', type=str)

    args = parser.parse_args()

    if not args.do_train and not args.do_eval:
        print('choose either --do_train or --do_eval')
        return

    if args.do_eval and args.load_model_name is None:
        print('load_model_name is required if you choose --do_eval')
        return

    if args.do_train:
        model_training = ModelTraining(args.patient_records_file, args.graph_file, args.voc_file, parameters.DEVICE)
        model_training.train(args.hidden_size, args.code_encoding_dropout_rate, args.trans_layer_num_sub,
                             args.trans_dropout_rate, args.trans_attn_dropout_rate, args.trans_head_num,
                             args.global_memory_num, args.gru_n_layers, args.gru_dropout_rate, args.encoder_lr,
                             args.attn_type_kv, args.attn_type_embedding, args.medications_ehr_dropout_rate,
                             args.multi_hop_count, args.decoder_lr, args.loss_proportion_bce,
                             args.loss_proportion_multi, args.records_train_ratio, args.records_test_ratio,
                             args.n_epoch, args.print_every_iteration, args.save_every_epoch, args.save_model_dir,
                             args.load_model_name)

    if args.do_eval:
        model_evaluation = ModelEvaluation(args.patient_records_file, args.predict_prob_threshold, args.graph_file,
                                           args.voc_file, parameters.DEVICE)
        model_evaluation.evaluate(args.load_model_name, args.hidden_size, args.trans_layer_num_sub, args.trans_head_num,
                                  args.global_memory_num, args.gru_n_layers, args.attn_type_kv,
                                  args.attn_type_embedding, args.multi_hop_count, args.save_predict_results_dir,
                                  args.records_train_ratio, args.records_test_ratio)


if __name__ == '__main__':
    main()
