# OSHMem

This is the implement for the model OSHMem in the paper Ontology-Enhanced Self-Attention and Hierarchical Memory Neural Network for Medication Recommendation.

# Data

Experiments are carried out based on [MIMIC-III](https://mimic.physionet.org), a real-world Electoric Healthcare Records (EHRs) dataset comprising deidentified health-related data associated with over forty thousand patients. OSHMem takes diagnoses and procedures of patients as input, and the medications prescribed in the first 24 hours of each admission are selected out as ground truths.

To prepare data for OSHMem, put the following file in a file named as 'data' under your project:

+PRESCRIPTIONS.csv

+DIAGNOSES_ICD.csv

+PROCEDURES_ICD.csv (you can download these three tables from [MIMIC-III](https://mimic.physionet.org))

+ndc2rxnorm_mapping.txt

+ndc2atc_level4.csv
  
+drug-atc.csv (you can find these three files in OSHMem/data)

then run python DataProcessing.py, which would generate:

+voc.pkl: vocabularies of diagnoses, procedures and medications.

+graph.pkl: graphs that describe relations of diagnoses, procedures and medications

+patient_records_final.pkl: patient records 


