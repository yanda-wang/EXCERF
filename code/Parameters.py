import os
import torch
import dill


class Parameters:
    def __init__(self):
        self.PATIENT_RECORDS_FINAL = 'data/patient_records_final.pkl'
        self.DIAGNOSE_INDEX = 0
        self.PROCEDURE_INDEX = 1
        self.MEDICATION_INDEX = 2

        self.VOC_FILE = 'data/voc.pkl'
        self.GRAPH_FILE = 'data/graph.pkl'

        self.MEDICATIONS_COUNT = 1
        if os.path.exists(self.VOC_FILE):
            voc = dill.load(open(self.VOC_FILE, 'rb'))
            self.MEDICATIONS_COUNT = voc.get_medications_count()

        self.USE_CUDA = torch.cuda.is_available()
        self.DEVICE = torch.device("cuda" if self.USE_CUDA else "cpu")

        self.OPT_SPLIT_TAG_ADMISSION = -1
        self.OPT_SPLIT_TAG_CODE = -2
        self.OPT_MAX_EPOCH = 30

        self.LOSS_PROPORTION_BCE = 0.9
        self.LOSS_PROPORTION_MULTI = 0.1

        self.RECORDS_TRAIN_RATIO = 0.8
        self.RECORDS_TEST_RATIO = 0.1
        self.RECORDS_VALIDATION_RATIO = 1 - self.RECORDS_TRAIN_RATIO - self.RECORDS_TEST_RATIO
