import dill
import itertools
import math

import pandas as pd
import numpy as np

from tqdm import tqdm

med_file = 'data/PRESCRIPTIONS.csv'
diag_file = 'data/DIAGNOSES_ICD.csv'
procedure_file = 'data/PROCEDURES_ICD.csv'

ndc2atc_file = 'data/ndc2atc_level4.csv'
cid_atc = 'data/drug-atc.csv'
ndc2rxnorm_file = 'data/ndc2rxnorm_mapping.txt'

PATIENT_RECORDS_FILE = 'data/patient_records.pkl'  # 以ICD和ATC表示疾病以及药品的病人记录，后续会转换成vocabulary表示的记录
PATIENT_RECORDS_FINAL_FILE = 'data/patient_records_final.pkl'

DIAGNOSES_INDEX = 0
PROCEDURES_INDEX = 1
MEDICATIONS_INDEX = 2

VOC_FILE = 'data/voc.pkl'
GRAPH_FILE = 'data/graph.pkl'


# ===================处理原始EHR数据，选取对应记录================

def process_procedure():
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE': 'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def process_med():
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})
    med_pd.drop(columns=['ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                         'FORMULARY_DRUG_CD', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX',
                         'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'FORM_UNIT_DISP',
                         'ROUTE', 'ENDDATE', 'DRUG'], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    def process_visit_lg2(med_pd):
        a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a

    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')

    return med_pd.reset_index(drop=True)


def process_diag():
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)


def ndc2atc4(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4': 'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]

    return pro_pd.reset_index(drop=True)


def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)


def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'],
                                                                                                         ascending=False).reset_index(
        drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]

    return med_pd.reset_index(drop=True)


def process_ehr():
    med_pd = process_med()
    med_pd = ndc2atc4(med_pd)

    diag_pd = process_diag()
    diag_pd = filter_2000_most_diag(diag_pd)

    pro_pd = process_procedure()

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(
        columns={'ICD9_CODE': 'PRO_CODE'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))

    patient_records = []
    for subject_id in data['SUBJECT_ID'].unique():
        item_df = data[data['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([item for item in row['ICD9_CODE']])  # diagnoses
            admission.append([item for item in row['PRO_CODE']])  # procedures
            admission.append([item for item in row['NDC']])  # medications
            patient.append(admission)
        patient_records.append(patient)

    dill.dump(patient_records, open(PATIENT_RECORDS_FILE, 'wb'))


# ==================构建字典，将原始EHR记录转换为medical code==============
class Voc(object):
    def __init__(self):
        self.idx2code = {}
        self.code2idx = {}

    def add_sentence(self, sentence):
        for code in sentence:
            if code not in self.code2idx:
                self.idx2code[len(self.code2idx)] = code
                self.code2idx[code] = len(self.code2idx)

    def get_token_count(self):
        return len(self.idx2code)


class EHRTokenizer(object):
    def __init__(self):
        self.diagnose_voc = Voc()
        self.procedure_voc = Voc()
        self.medication_voc = Voc()

    def add_diagnoses(self, sentence):
        self.diagnose_voc.add_sentence(sentence)

    def add_procedures(self, sentence):
        self.procedure_voc.add_sentence(sentence)

    def add_medications(self, sentence):
        self.medication_voc.add_sentence(sentence)

    def get_diagnoses_count(self):
        return self.diagnose_voc.get_token_count()

    def get_procedures_count(self):
        return self.procedure_voc.get_token_count()

    def get_medications_count(self):
        return self.medication_voc.get_token_count()

    def code2idx_diagnose(self, code):
        idx = [self.diagnose_voc.code2idx[item] for item in code]
        return idx

    def idx2code_diagnose(self, idx):
        code = [self.diagnose_voc.idx2code[item] for item in idx]
        return code

    def code2idx_procedures(self, code):
        idx = [self.procedure_voc.code2idx[item] for item in code]
        return idx

    def idx2code_procedures(self, idx):
        code = [self.procedure_voc.idx2code[item] for item in idx]
        return code

    def code2idx_medications(self, code):
        idx = [self.medication_voc.code2idx[item] for item in code]
        return idx

    def idx2code_medications(self, idx):
        code = [self.medication_voc.idx2code[item] for item in idx]
        return code


def build_voc():
    voc = EHRTokenizer()
    patient_records = dill.load(open(PATIENT_RECORDS_FILE, 'rb'))

    for patient in patient_records:
        for adm in patient:
            diagnoses, procedures, medications = adm[DIAGNOSES_INDEX], adm[PROCEDURES_INDEX], adm[MEDICATIONS_INDEX]
            voc.add_diagnoses(diagnoses)
            voc.add_procedures(procedures)
            voc.add_medications(medications)

    dill.dump(voc, open(VOC_FILE, 'wb'))


def convert_patient_records():
    voc = dill.load(open(VOC_FILE, 'rb'))
    patient_records = dill.load(open(PATIENT_RECORDS_FILE, 'rb'))
    patient_records_idx = []

    for patient in patient_records:
        current_patient = []
        for adm in patient:
            diagnoses, procedures, medications = adm[DIAGNOSES_INDEX], adm[PROCEDURES_INDEX], adm[MEDICATIONS_INDEX]
            admission = []
            admission.append(voc.code2idx_diagnose(diagnoses))
            admission.append(voc.code2idx_procedures(procedures))
            admission.append(voc.code2idx_medications(medications))
            current_patient.append(admission)
        patient_records_idx.append(current_patient)

    dill.dump(patient_records_idx, open(PATIENT_RECORDS_FINAL_FILE, 'wb'))


def process_patient_records():
    build_voc()
    convert_patient_records()


# ===============================================================
# 依据ATC和ICD，建立疾病与疾病之间，procedure与procedure之间的graph
# 建立药品之间的co-occurrence graph

def build_graph():
    voc = dill.load(open(VOC_FILE, 'rb'))
    diagnoses_count = voc.get_diagnoses_count()
    procedures_count = voc.get_procedures_count()
    medication_count = voc.get_medications_count()
    diagnoses_ICD_structure = {'001-009': '001-139', '010-018': '001-139', '020-027': '001-139', '030-041': '001-139',
                               '042-042': '001-139', '045-049': '001-139', '050-059': '001-139', '060-066': '001-139',
                               '070-079': '001-139', '080-088': '001-139', '090-099': '001-139', '100-104': '001-139',
                               '110-118': '001-139', '120-129': '001-139', '130-136': '001-139', '137-139': '001-139',

                               '140-149': '140-239', '150-159': '140-239', '160-165': '140-239', '170-176': '140-239',
                               '179-189': '140-239', '190-199': '140-239', '200-209': '140-239', '210-229': '140-239',
                               '230-234': '140-239', '235-238': '140-239', '239-239': '140-239',

                               '240-246': '240-279', '249-259': '240-279', '260-269': '240-279', '270-279': '240-279',

                               '280-280': '280-289', '281-281': '280-289', '282-282': '280-289', '283-283': '280-289',
                               '284-284': '280-289', '285-285': '280-289', '286-286': '280-289', '287-287': '280-289',
                               '288-288': '280-289', '289-289': '280-289',

                               '290-294': '290-319', '295-299': '290-319', '300-316': '290-319', '317-319': '290-319',

                               '320-327': '320-389', '330-337': '320-389', '338-338': '320-389', '339-339': '320-389',
                               '340-349': '320-389', '350-359': '320-389', '360-379': '320-389', '380-389': '320-389',

                               '390-392': '390-459', '393-398': '390-459', '401-405': '390-459', '410-414': '390-459',
                               '415-417': '390-459', '420-429': '390-459', '430-438': '390-459', '440-449': '390-459',
                               '451-459': '390-459',

                               '460-466': '460-519', '470-478': '460-519', '480-488': '460-519', '490-496': '460-519',
                               '500-508': '460-519', '510-519': '460-519',

                               '520-529': '520-579', '530-539': '520-579', '540-543': '520-579', '550-553': '520-579',
                               '555-558': '520-579', '560-569': '520-579', '570-579': '520-579',

                               '580-589': '580-629', '590-599': '580-629', '600-608': '580-629', '610-611': '580-629',
                               '614-616': '580-629', '617-629': '580-629',

                               '630-639': '630-679', '640-649': '630-679', '650-659': '630-679', '660-669': '630-679',
                               '670-677': '630-679', '678-679': '630-679',

                               '680-686': '680-709', '690-698': '680-709', '700-709': '680-709',

                               '710-719': '710-739', '720-724': '710-739', '725-729': '710-739', '730-739': '710-739',

                               '740-740': '740-759', '741-741': '740-759', '742-742': '740-759', '743-743': '740-759',
                               '744-744': '740-759', '745-745': '740-759', '746-746': '740-759', '747-747': '740-759',
                               '748-748': '740-759', '749-749': '740-759', '750-750': '740-759', '751-751': '740-759',
                               '752-752': '740-759', '753-753': '740-759', '754-754': '740-759', '755-755': '740-759',
                               '756-756': '740-759', '757-757': '740-759', '758-758': '740-759', '759-759': '740-759',

                               '760-763': '760-779', '764-779': '760-779',

                               '780-789': '780-799', '790-796': '780-799', '797-799': '780-799',

                               '800-804': '800-999', '805-809': '800-999', '810-819': '800-999', '820-829': '800-999',
                               '830-839': '800-999', '840-848': '800-999', '850-854': '800-999', '860-869': '800-999',
                               '870-879': '800-999', '880-887': '800-999', '890-897': '800-999', '900-904': '800-999',
                               '905-909': '800-999', '910-919': '800-999', '920-924': '800-999', '925-929': '800-999',
                               '930-939': '800-999', '940-949': '800-999', '950-957': '800-999', '958-959': '800-999',
                               '960-979': '800-999', '980-989': '800-999', '990-995': '800-999', '996-999': '800-999',

                               'V01-V09': 'V01-V91', 'V10-V19': 'V01-V91', 'V20-V29': 'V01-V91', 'V30-V39': 'V01-V91',
                               'V40-V49': 'V01-V91', 'V50-V59': 'V01-V91', 'V60-V69': 'V01-V91', 'V70-V82': 'V01-V91',
                               'V83-V84': 'V01-V91', 'V85-V85': 'V01-V91', 'V86-V86': 'V01-V91', 'V87-V87': 'V01-V91',
                               'V88-V88': 'V01-V91', 'V89-V89': 'V01-V91', 'V90-V90': 'V01-V91', 'V91-V91': 'V01-V91',

                               'E000-E000': 'E000-E999', 'E001-E030': 'E000-E999', 'E800-E807': 'E000-E999',
                               'E810-E819': 'E000-E999', 'E820-E825': 'E000-E999', 'E826-E829': 'E000-E999',
                               'E830-E838': 'E000-E999', 'E840-E845': 'E000-E999', 'E846-E849': 'E000-E999',
                               'E850-E858': 'E000-E999', 'E860-E869': 'E000-E999', 'E870-E876': 'E000-E999',
                               'E878-E879': 'E000-E999', 'E880-E888': 'E000-E999', 'E890-E899': 'E000-E999',
                               'E900-E909': 'E000-E999', 'E910-E915': 'E000-E999', 'E916-E928': 'E000-E999',
                               'E929-E929': 'E000-E999', 'E930-E949': 'E000-E999', 'E950-E959': 'E000-E999',
                               'E960-E969': 'E000-E999', 'E970-E979': 'E000-E999', 'E980-E989': 'E000-E999',
                               'E990-E999': 'E000-E999'}
    procedures_ICD_structure = ['00-00', '01-05', '06-07', '08-16', '17-17', '18-20', '21-29', '30-34', '35-39',
                                '40-41', '42-54', '55-59', '60-64', '65-71', '72-75', '76-84', '85-86', '87-99']

    distance_graph_diagnoses = np.ones((diagnoses_count, diagnoses_count)) * np.inf
    weight_graph_diagnoses = np.zeros((diagnoses_count, diagnoses_count))
    distance_graph_procedures = np.ones((procedures_count, procedures_count)) * np.inf
    weight_graph_procedures = np.zeros((procedures_count, procedures_count))
    co_occurrence_graph = np.zeros((medication_count, medication_count))

    place_token = ['CLS', 'SEP']
    max_diagnoses_distance = 12
    max_procedures_distance = 10

    # ++++++++++++依据ICD,计算疾病之间的距离===================
    def get_diagnoses_distance(icd1, icd2):
        icd1_path, icd2_path = [], []

        select_length = 3 if icd1[0] != 'E' else 4
        icd1_parent = [key for key in filter(lambda sub: sub.split('-')[0] <= icd1[:select_length] <= sub.split('-')[1],
                                             diagnoses_ICD_structure.keys())][0]
        icd1_path.append(diagnoses_ICD_structure[icd1_parent])
        icd1_path.append(icd1_parent)
        for i in range(select_length, len(icd1) + 1):
            icd1_path.append(icd1[:i])

        select_length = 3 if icd2[0] != 'E' else 4
        icd2_parent = [key for key in filter(lambda sub: sub.split('-')[0] <= icd2[:select_length] <= sub.split('-')[1],
                                             diagnoses_ICD_structure.keys())][0]
        icd2_path.append(diagnoses_ICD_structure[icd2_parent])
        icd2_path.append(icd2_parent)
        for i in range(select_length, len(icd2) + 1):
            icd2_path.append(icd2[:i])

        common_path_length = 0
        for i in range(min(len(icd1_path), len(icd2_path))):
            if icd1_path[i] == icd2_path[i]:
                common_path_length += 1
            else:
                break
        ontology_distance = len(icd1_path) + len(icd2_path) - 2 * common_path_length
        # 两个ICD不在同一个第一级大类分类下，则二者之间距离最大化
        if diagnoses_ICD_structure[icd1_parent] != diagnoses_ICD_structure[icd2_parent]:
            ontology_distance = max_diagnoses_distance
        return ontology_distance

    print('building graphs for diagnoses>>>')
    combination_count = sum(
        1 for _ in itertools.combinations_with_replacement(list(voc.diagnose_voc.code2idx.keys()), 2))
    with tqdm(total=combination_count) as t:
        for (icd1, icd2) in itertools.combinations_with_replacement(list(voc.diagnose_voc.code2idx.keys()), 2):
            if icd1 not in place_token and icd2 not in place_token:
                ontology_distance = get_diagnoses_distance(icd1, icd2)
                [icd1_idx, icd2_idx] = voc.code2idx_diagnose([icd1, icd2])
                distance_graph_diagnoses[icd1_idx, icd2_idx] = ontology_distance
                distance_graph_diagnoses[icd2_idx, icd1_idx] = ontology_distance
                if ontology_distance == 0:
                    ontology_weight = math.log(1) - math.log(1 / max_diagnoses_distance)
                else:
                    ontology_weight = math.log(1 / ontology_distance) - math.log(1 / max_diagnoses_distance)
                weight_graph_diagnoses[icd1_idx, icd2_idx] = ontology_weight
                weight_graph_diagnoses[icd2_idx, icd1_idx] = ontology_weight
            t.update()

    # +++++++++++依据ICD，计算procedure之间的距离+++++++++++++
    def get_procedures_distance(icd1, icd2):
        icd1_path, icd2_path = [], []
        select_length = 2

        icd1_parent = [key for key in filter(lambda sub: sub.split('-')[0] <= icd1[:select_length] <= sub.split('-')[1],
                                             procedures_ICD_structure)][0]
        icd1_path.append(icd1_parent)
        for i in range(select_length, len(icd1) + 1):
            icd1_path.append(icd1[:i])

        icd2_parent = [key for key in filter(lambda sub: sub.split('-')[0] <= icd2[:select_length] <= sub.split('-')[1],
                                             procedures_ICD_structure)][0]
        icd2_path.append(icd2_parent)
        for i in range(select_length, len(icd2) + 1):
            icd2_path.append(icd2[:i])

        common_path_length = 0
        for i in range(min(len(icd1_path), len(icd2_path))):
            if icd1_path[i] == icd2_path[i]:
                common_path_length += 1
            else:
                break
        ontology_distance = len(icd1_path) + len(icd2_path) - 2 * common_path_length
        if icd1_parent != icd2_parent:
            ontology_distance = max_procedures_distance
        return ontology_distance

    print('building graphs for procedures>>>')
    combination_count = sum(
        1 for _ in itertools.combinations_with_replacement(list(voc.procedure_voc.code2idx.keys()), 2))
    with tqdm(total=combination_count) as t:
        for (icd1, icd2) in itertools.combinations_with_replacement(list(voc.procedure_voc.code2idx.keys()), 2):
            if icd1 not in place_token and icd2 not in place_token:
                ontology_distance = get_procedures_distance(icd1, icd2)
                [icd1_idx, icd2_idx] = voc.code2idx_procedures([icd1, icd2])
                distance_graph_procedures[icd1_idx, icd2_idx] = ontology_distance
                distance_graph_procedures[icd2_idx, icd1_idx] = ontology_distance
                if ontology_distance == 0:
                    ontology_weight = math.log(1) - math.log(1 / max_procedures_distance)
                else:
                    ontology_weight = math.log(1 / ontology_distance) - math.log(1 / max_procedures_distance)
                weight_graph_procedures[icd1_idx, icd2_idx] = ontology_weight
                weight_graph_procedures[icd2_idx, icd1_idx] = ontology_weight
            t.update()

    # +++++++++++构建药品的co-occurrence graph++++++++++++++++++++++
    print('building graphs for medications>>>')
    patient_records = dill.load(open(PATIENT_RECORDS_FINAL_FILE, 'rb'))
    for patient in tqdm(patient_records):
        for adm in patient:
            medications = adm[MEDICATIONS_INDEX]
            for (med_1, med_2) in itertools.combinations(medications, 2):
                co_occurrence_graph[med_1, med_2] = 1
                co_occurrence_graph[med_2, med_1] = 1

    dill.dump({'distance_graph_diagnoses': distance_graph_diagnoses,
               'distance_graph_procedures': distance_graph_procedures,
               'weight_graph_diagnoses': weight_graph_diagnoses,
               'weight_graph_procedures': weight_graph_procedures,
               'co-occurrence_graph': co_occurrence_graph}, open(GRAPH_FILE, 'wb'))


if __name__ == '__main__':
    process_ehr()
    process_patient_records()
    build_graph()
