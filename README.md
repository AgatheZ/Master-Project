# Machine Learning on temporal data from ICU and chronic disease to predict future values and outcomes


## **Additional_analysis**

This folder contains the files for the annex CPA analysis.  model.ipynb is a notebook containing (1) ML models for 5-year mortality prediction, (2) univariate and multivariate survival analysis for 5-year mortality.

## **Archives**

Contains former files such as the preprocessing using for the demo version of MIMIC-III used at the beginning of the project.

## **MIMIC-IV**
Contains the TBI management study files, using MIMIC-IV database.

### **Database analysis**
Contains files for exploratory data analysis of the TBI cohort in MIMIC-IV.

### **SQL Preprocessing Pipeline**

- height_weight_table.sql: computes the height/weight table by performing unit transformation.
- preprocessing_pipeline.sql: SQL preprocessing for TBI cohort extraction, data aggregation, feature extraction, feature engineering. It outputs tables provided in google drive which will be used to train the different models.
- preprocessing_pipeline_augmented.sql: SQL preprocessing for augmented cohort, used for Transfer Learning source task.

### **Tasks code**
Contains code to run for the LOS and BP regression tasks.

- data_augmentation.py: class for data augmentation using VAE.
- evaluation.py: class for models evaluation
- GRU.py: class with the GRU architecture
- hyp_tuning.py: class for hyperparameter tuning using Hyperopt
- preprocessing.py: class for extra-preprocessing (data imputation, outlier removal...)
- main_LOS.py: running this file will train the best model obtained for LOS binary classification
- main_transfer_learning.py: running this file will train the best GRU model for Blood Pressures regression 
- main_task_2.py: running this file will train the best ML model for Blood Pressures regression
- VAE.py: VAE architecture for data augmentation 
- VAE_model.pkl: already trained VAE model 

