# EXCELF

This is the implement for the model EXCELF in the paper "Beyond EHRs: External Clinical Knowledge and Global Features for Medication Recommendation".

# Data

Experiments are carried out based on [MIMIC-III](https://mimic.physionet.org), a real-world Electoric Healthcare Records (EHRs) dataset comprising deidentified health-related data associated with over forty thousand patients. EXCLEF takes diagnoses and procedures of patients as input, and the medications prescribed in the first 24 hours of each admission are selected out as ground truths.

To prepare data for EXCLEF, put the following files in ./data under your project:

* PRESCRIPTIONS.csv
* DIAGNOSES_ICD.csv
* PROCEDURES_ICD.csv (you can download these three tables from [MIMIC-III](https://mimic.physionet.org))
* ndc2rxnorm_mapping.txt
* ndc2atc_level4.csv
* drug-atc.csv (you can find these three files in EXCLEF/data)

then run

```
python DataProcessing.py
```
 
which would generate the following files in ./data under your project, you can also find them in this repository:

* voc.pkl: vocabularies of diagnoses, procedures and medications.
* graph.pkl: graphs that describe relations of diagnoses, procedures and medications
* patient_records_final.pkl: patient records

# Codes

codes of EXCLEF could be found in EXCLEF/code

* DataProcessing.py: prepare all data files required by EXCLEF from raw medical records.
* HyperOptim.py: hyper-parameters tuning.
* HyperOptimUtils.py: basic modules for HyperOptim.
* Models.py: networks of EXCLEF.
* Parameters.py: basic parameters for EXCLEF.
* run_model.py: code to run EXCLEF

# To run the model

To train EXCLEF(suppose you want to put the model in data/model)
```
python run_model.py --do_train --save_model_dir data/model
```
To evaluate the model(suppose your model is data/model/model_1.checkpoint, and you want to save the predict results in the file data/predict_results)
```
python run_model.py --do_eval --load_model_name data/model/model_1.checkpoint --save_predict_results_dir data/predict_results
```






