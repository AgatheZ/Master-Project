from cProfile import label
from multiprocessing.sharedctypes import Value
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
from sklearn.preprocessing import normalize
import sys 
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    def __init__(self, df_hourly=None, df_24h=None, df_48h=None, df_med=None, df_demographic=None, nb_hours=24, TBI_split=False, random_state=1, imputation='No', diag = None):
        self.df_hourly = df_hourly
        self.df_24h = df_24h
        self.df_48h = df_48h
        self.df_med = df_med
        self.diag = diag
        self.df_demographic = df_demographic
        self.nb_hours = nb_hours
        self.TBI_split = TBI_split
        self.random_state = random_state
        self.imputation = imputation 
        
    def create_batchs(self, ds):
        '''This function creates batches of patients using the complete dataset'''
        batchs = []
        ids = ds.stay_id.unique()
        for i in ids:
            batchs.append(ds.loc[ds['stay_id'] == i])
        return batchs

    def label_severity (self, row):
        '''Compute the label severity based on GCS'''
        if row['gcs'] <= 8 :
            return "severe"
        return "mild"
        
    def remove_missing(df, var, threshold):
    #remove from batch the entries where a too large proportion of the variables var are missing 
        res = df
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame({'column_name': df.columns,
                                        'percent_missing': percent_missing})
        for vital in var: 
            criterion = missing_value_df.loc[missing_value_df.column_name == vital].percent_missing >= threshold 
            if criterion:
                print('entry removed')
                print(missing_value_df.loc[missing_value_df.column_name == vital].percent_missing)
                df.drop([vital], axis = 1)
            else:
                res.append(batch[i])
        return res

    def data_imputation(self, batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic, type, random_state):
        '''Function that takes the non imputed batches and returns batches with data imputation.
        The type of data imputation is defined by the variable type.'''
        
        df_hourly = pd.concat(batch_hourly)
        df_24h = pd.concat(batch_24h)
        df_48h = pd.concat(batch_48h)
        df_med = pd.concat(batch_med)
        df_demographic = pd.concat(batch_demographic)
        gcs_mean = df_demographic.gcs.mean()
        bmi_mean = df_demographic.bmi.mean()
        hourly_mean = df_hourly.mean()
        mean_24h = df_24h.mean()
        mean_48h = df_48h.mean()

        if type == 'linear':
            for i in range(len(batch_hourly)):
                batch_hourly[i] = batch_hourly[i].interpolate(limit = 15)
                batch_24h[i] = batch_24h[i].interpolate(limit = 15)
                batch_48h[i] = batch_48h[i].fillna(mean_48h)
                batch_24h[i] = batch_24h[i].fillna(mean_24h)
                batch_hourly[i] = batch_hourly[i].fillna(hourly_mean)
                batch_demographic[i].bmi = batch_demographic[i].bmi.fillna(bmi_mean)
                batch_demographic[i].gcs = batch_demographic[i].gcs.fillna(gcs_mean)
                batch_demographic[i] = batch_demographic[i].fillna(0)
                batch_hourly[i] = batch_hourly[i].fillna(0)

            return batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic

        if type == 'multivariate': 
            imp_hourly = IterativeImputer(max_iter=10, random_state=random_state)
            imp_24h = IterativeImputer(max_iter=10, random_state=random_state)
            imp_48h = IterativeImputer(max_iter=10, random_state=random_state)
            imp_med = IterativeImputer(max_iter=10, random_state=random_state)
            for i in range(len(batch_hourly)):
                imp_hourly.fit(batch_hourly[i])
                imp_24h.fit(batch_24h[i])
                imp_48h.fit(batch_48h[i])
                imp_med.fit(batch_med[i])


                imp_hourly.transform(batch_hourly[i])
                imp_24h.transform(batch_24h[i])
                imp_48h.transform(batch_48h[i])
                imp_med.transform(batch_med[i])

                batch_48h[i] = batch_48h[i].fillna(mean_48h)
                batch_24h[i] = batch_24h[i].fillna(mean_24h)
                batch_hourly[i] = batch_hourly[i].fillna(hourly_mean)
                batch_demographic[i].bmi = batch_demographic[i].bmi.fillna(bmi_mean)
                batch_demographic[i].gcs = batch_demographic[i].gcs.fillna(gcs_mean)

            return batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic
        
        if type == 'carry_forward': 
            for i in range(len(batch_hourly)):
                batch_hourly[i] = batch_hourly[i].fillna(method = "ffill")
                batch_24h[i] = batch_24h[i].fillna(method = "ffill")
                batch_48h[i] = batch_48h[i].fillna(mean_48h)
                batch_24h[i] = batch_24h[i].fillna(mean_24h)
                batch_hourly[i] = batch_hourly[i].fillna(hourly_mean)
                batch_demographic[i].bmi = batch_demographic[i].bmi.fillna(bmi_mean)
                batch_demographic[i].gcs = batch_demographic[i].gcs.fillna(gcs_mean)
                batch_demographic[i] = batch_demographic[i].fillna(0)
                batch_hourly[i] = batch_hourly[i].fillna(0)
            return batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic
        
        if type == 'No':
            return batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic

    def preprocess_data(self, threshold):
        '''Data preprocessing for the LOS task, for a cut-off value threshold.
        Inputs: - threshold: length-of-stay cut-off 
        Outputs:- final data: preprocessed data for non differentiated cohort 
                - labels: binary classification labels for non differentiated cohort
                - final_data_severe/mild: preprocessed data for severe and mild cohort
                - labels_severe/mild: labels for severe and mild cohort'''

        self.df_hourly = self.df_hourly.drop(columns = ['icu_intime'])
        self.df_24h = self.df_24h.drop(columns = ['icu_intime'])
        self.df_48h = self.df_48h.drop(columns = ['icu_intime'])
        
        ##label extraction 
        labels = self.df_demographic.pop('los')
        labels[labels <= threshold] = 0
        labels[labels > threshold] = 1
        labels = labels.values

        ##pivot the tables 
        if 'std' in self.df_hourly.columns:
            df_std_h = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'std')
            df_std_h = df_std_h.reset_index(level=['hour_from_intime'])
            df_std_24h = self.df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'std')
            df_std_24h = df_std_24h.reset_index(level=['hour_from_intime'])
            self.df_hourly = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
            self.df_24h = self.df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')

        else:
            self.df_hourly = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
            self.df_24h = self.df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        
        self.df_48h = self.df_48h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_med = self.df_med.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'med_name', values = 'amount', aggfunc= np.sum)

        df_std_24h.fillna(0)
        df_std_h.fillna(0)

        ##one-hot encoding for the sex
        self.df_demographic.gender[self.df_demographic.gender == 'F'] = 1
        self.df_demographic.gender[self.df_demographic.gender == 'M'] = 0

        ##create batches 
        self.df_hourly = self.df_hourly.reset_index(level=['stay_id'])
        self.df_24h = self.df_24h.reset_index(level=['stay_id'])
        self.df_48h = self.df_48h.reset_index(level=['stay_id'])
        self.df_med = self.df_med.reset_index(level=['stay_id'])

        batch_hourly = self.create_batchs(self.df_hourly)
        batch_24h = self.create_batchs(self.df_24h)
        batch_48h = self.create_batchs(self.df_48h)
        batch_med = self.create_batchs(self.df_med)
        batch_demographic = self.create_batchs(self.df_demographic)
        self.diag.long_title = self.diag.long_title.astype('category').cat.codes

        ##reindex for patients that don't have entries at the begginning of their stays + cut to 48h
        ##aggregation as well
        for i in range(len(batch_24h)):
            batch_hourly[i] = batch_hourly[i].reindex(range(1, self.nb_hours + 1), fill_value = None) 
            batch_24h[i] = batch_24h[i].reindex(range(1, self.nb_hours//24 + 1), fill_value = None)
            batch_48h[i] = batch_48h[i].reindex(range(1, self.nb_hours//24 + 1), fill_value = None)
            
            #onehotencoding
            batch_med[i] = batch_med[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            batch_med[i] = batch_med[i].fillna(value = 0)
            batch_hourly[i] = batch_hourly[i].drop(columns = 'stay_id')
            batch_24h[i] = batch_24h[i].drop(columns = 'stay_id')
            batch_48h[i] = batch_48h[i].drop(columns = 'stay_id')
            batch_med[i] = batch_med[i].drop(columns = 'stay_id')
            batch_med[i] = batch_med[i].agg([np.sum])
            batch_med[i][batch_med[i] > 0] = 1

            batch_demographic[i] = batch_demographic[i].drop(columns = 'stay_id')
            batch_48h[i] = batch_48h[i].agg([np.mean])


        #separation of the two sub-cohorts
        if self.TBI_split:
            dem = self.df_demographic.drop(columns = 'stay_id').reset_index(drop=True)
            dem['severity'] = dem.apply(lambda row: self.label_severity(row), axis=1)
            mild_idx = np.array(dem[dem['severity'] == 'mild'].index)
            severe_idx = np.array(dem[dem['severity'] == 'severe'].index)
            dem.pop('severity')
        
        self.df_hourly = pd.concat(batch_hourly)
        self.df_24h = pd.concat(batch_24h)
        self.df_48h = pd.concat(batch_48h)
        self.df_med = pd.concat(batch_med)
        self.df_demographic = self.df_demographic.drop(columns = 'stay_id')

        #Data imputation 
        batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic = self.data_imputation(batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic, self.imputation, self.random_state)
        
        #Feature concatenation 
        if self.nb_hours == 24:
            final_data = np.array([[np.concatenate([np.concatenate(batch_demographic[i].values), np.concatenate(batch_med[i].values)])] for i in range(len(batch_hourly))])
        else: 
            final_data = np.array([[np.concatenate([np.concatenate(batch_demographic[i].values), np.concatenate(batch_med[i].values)])] for i in range(len(batch_hourly))])

        final_data = np.squeeze(final_data)

        if self.TBI_split:
            data_mild = np.array([final_data[i] for i in mild_idx])
            data_severe = np.array([final_data[i] for i in severe_idx])
            labels_mild = np.array([labels[i] for i in mild_idx])
            labels_severe = np.array([labels[i] for i in severe_idx])

            return data_mild, data_severe, labels_mild, labels_severe, np.concatenate((batch_demographic[0].columns, batch_med[0].columns))
        else:    
            return final_data, labels, np.concatenate((batch_demographic[0].columns, batch_med[0].columns))
        
    def time_series_pr(self, label, transfer):
        '''Preprocessing for first GRU approach'''

        self.df_hourly = self.df_hourly.drop(columns = ['icu_intime'])
        self.df_24h = self.df_24h.drop(columns = ['icu_intime'])

        ##pivot the tables 
        self.df_hourly = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_24h = self.df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')

        ##label extraction 
        self.df_hourly_copy = self.df_hourly.reset_index(level=['hour_from_intime'])
        labels = self.df_hourly_copy[self.df_hourly_copy.hour_from_intime == 25][label]
        labels = labels.dropna()
        task2_cohort = labels.index.values
        labels = labels.values
        
        ##Restriction of the cohort 
        self.df_hourly = self.df_hourly.reset_index(level=['stay_id'])
        self.df_hourly = self.df_hourly[self.df_hourly['stay_id'].isin(task2_cohort)]
        self.df_demographic = self.df_demographic[self.df_demographic['stay_id'].isin(task2_cohort)]

        self.df_demographic.gender[self.df_demographic.gender == 'F'] = 1
        self.df_demographic.gender[self.df_demographic.gender == 'M'] = 0
        
        ##create batches 
        batch_hourly = self.create_batchs(self.df_hourly)
        batch_demographic = self.create_batchs(self.df_demographic)

        ##reindex for patients that don't have entries at the begginning of their stays + cut to 48h
        ##aggregation as well
        data_pr = []
        for i in range(len(batch_demographic)):
            batch_hourly[i] = batch_hourly[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            
            batch_demographic[i] = batch_demographic[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            batch_demographic[i] = batch_demographic[i].fillna(batch_demographic[i].mean())
            batch_hourly[i] = batch_hourly[i].drop(columns = 'stay_id')
            batch_demographic[i] = batch_demographic[i].drop(columns = 'stay_id')
            data_pr.append(pd.concat([batch_hourly[i], batch_demographic[i]], axis=1))
        
        #divide between severe and mild
        if transfer:
            dem = self.df_demographic.drop(columns = 'stay_id').reset_index(drop=True)
            dem['severity'] = dem.apply(lambda row: self.label_severity(row), axis=1)
    
            mild_idx = np.array(dem[dem['severity'] == 'mild'].index)
            severe_idx = np.array(dem[dem['severity'] == 'severe'].index)
            dem.pop('severity')

        mean = pd.concat(data_pr).mean()
        for i in range(len(batch_hourly)):
                data_pr[i] = data_pr[i].fillna(method = "ffill")
                data_pr[i] = data_pr[i].fillna(mean)
                data_pr[i] = data_pr[i].fillna(0)


        if transfer:
            data_mild = np.array([data_pr[i] for i in mild_idx])
            data_severe = np.array([data_pr[i] for i in severe_idx])
            labels_mild = np.array([labels[i] for i in mild_idx])
            labels_severe = np.array([labels[i] for i in severe_idx])
            return data_pr, labels, data_mild, labels_mild, data_severe, labels_severe

        return data_pr, labels

    

    def std_pr(self, label, transfer, window = 1):
        '''Preprocessing for GRU approach with std prediction'''
        print('std_pr')
        self.df_hourly = self.df_hourly.drop(columns = ['icu_intime'])
        self.df_24h = self.df_24h.drop(columns = ['icu_intime'])

        df_hourly_copy = self.df_hourly
        self.df_hourly = df_hourly_copy.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_24h = self.df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')

        ##pivot the tables 
        df_std = df_hourly_copy.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'std')
        df_std = df_std.reset_index(level=['hour_from_intime'])

        ##outlier transformation
        new_d = self.df_hourly.loc[self.df_hourly.ABPm < 50, 'ABPd'] + (1/3)*(self.df_hourly.loc[self.df_hourly.ABPm < 50, 'ABPs'] - self.df_hourly.loc[self.df_hourly.ABPm < 50, 'ABPd'])
        new_u = self.df_hourly.loc[self.df_hourly.ABPm > 150, 'ABPd'] + (1/3)*(self.df_hourly.loc[self.df_hourly.ABPm > 150, 'ABPs'] - self.df_hourly.loc[self.df_hourly.ABPm > 150, 'ABPd'])

        self.df_hourly.loc[self.df_hourly.ABPm < 50, 'ABPm'] = new_d
        self.df_hourly.loc[self.df_hourly.ABPm > 150, 'ABPm'] = new_u

        labels_std = df_std[df_std.hour_from_intime.isin(range(25,25+window))][label].dropna()
        labels_std = labels_std.groupby('stay_id').mean()

        ##label extraction 
        self.df_hourly_copy = self.df_hourly.reset_index(level=['hour_from_intime'])
        labels = self.df_hourly_copy[self.df_hourly_copy.hour_from_intime.isin(range(25,25+window))][label]
 
        labels = labels.groupby('stay_id').mean()
        labels = labels.dropna()
        task2_cohort = labels.index.values
        labels = labels.to_frame()
        labels_std = labels_std.to_frame()
        test = labels.join(labels_std, on = 'stay_id', how='inner', lsuffix = '', rsuffix = ' std')

        ##Restriction of the cohort 
        self.df_hourly = self.df_hourly.reset_index(level=['stay_id'])
        df_std = df_std.reset_index(level=['stay_id'])

        self.df_hourly = self.df_hourly[self.df_hourly['stay_id'].isin(task2_cohort)]
        self.df_demographic = self.df_demographic[self.df_demographic['stay_id'].isin(task2_cohort)]

        df_std = df_std[df_std['stay_id'].isin(task2_cohort)]

        self.df_demographic.gender[self.df_demographic.gender == 'F'] = 1
        self.df_demographic.gender[self.df_demographic.gender == 'M'] = 0
        df_std = df_std.fillna(0)
  
        ##create batches 
        batch_hourly = self.create_batchs(self.df_hourly)
        batch_demographic = self.create_batchs(self.df_demographic)
        batch_std = self.create_batchs(df_std)

        ##reindex for patients that don't have entries at the begginning of their stays + cut to 48h
        ##aggregation as well
        data_pr = []
        for i in range(len(batch_demographic)):
            batch_hourly[i] = batch_hourly[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            
            batch_demographic[i] = batch_demographic[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            batch_demographic[i] = batch_demographic[i].fillna(batch_demographic[i].mean())
            batch_hourly[i] = batch_hourly[i].drop(columns = 'stay_id')
            batch_demographic[i] = batch_demographic[i].drop(columns = 'stay_id')
            if label == 'ABPs':
                data_pr.append(pd.concat([batch_hourly[i], batch_demographic[i], batch_std[i]], axis=1))
            else:
                data_pr.append(pd.concat([batch_hourly[i], batch_demographic[i]], axis=1))

        #divide between severe and mild
        if transfer:
            dem = self.df_demographic.drop(columns = 'stay_id').reset_index(drop=True)
            dem['severity'] = dem.apply(lambda row: self.label_severity(row), axis=1)
    
            mild_idx = np.array(dem[dem['severity'] == 'mild'].index)
            severe_idx = np.array(dem[dem['severity'] == 'severe'].index)
            dem.pop('severity')

        mean = pd.concat(data_pr).mean()
        for i in range(len(batch_hourly)):
                # data_pr[i] = data_pr[i].fillna(method = "ffill")
                data_pr[i] = data_pr[i].fillna(mean)
                data_pr[i] = data_pr[i].fillna(0)
                data_pr[i] =  normalize(data_pr[i])
        

        if transfer:
            data_mild = np.array([data_pr[i] for i in mild_idx])
            data_severe = np.array([data_pr[i] for i in severe_idx])
            labels_mild = np.array([labels[i] for i in mild_idx])
            labels_severe = np.array([labels[i] for i in severe_idx])
            return data_pr, labels, data_mild, labels_mild, data_severe, labels_severe

        return data_pr, test

    def task_2_pr(self, label):
        '''Data preprocessing for BP regression with ML approach '''
        self.df_hourly = self.df_hourly.drop(columns = ['icu_intime'])
        self.df_24h = self.df_24h.drop(columns = ['icu_intime'])
        self.df_48h = self.df_48h.drop(columns = ['icu_intime'])

        ##pivot the tables 
        if 'std' in self.df_hourly.columns:
            self.df_hourly = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = ['feature_name'], values = 'feature_mean_value')
        else:
            self.df_hourly = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_24h = self.df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_48h = self.df_48h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_med = self.df_med.pivot_table(index = ['stay_id'], columns = 'med_name', values = 'amount')

        ##label extraction 
        self.df_hourly_copy = self.df_hourly.reset_index(level=['hour_from_intime'])
        labels = self.df_hourly_copy[self.df_hourly_copy.hour_from_intime == 25][label]
        labels = labels.dropna()
        task2_cohort = labels.index.values
        labels = labels.values

        ##Restriction of the cohort 
        self.df_hourly = self.df_hourly.reset_index(level=['stay_id'])
        self.df_24h = self.df_24h.reset_index(level=['stay_id'])
        self.df_48h = self.df_48h.reset_index(level=['stay_id'])
        self.df_med = self.df_med.reset_index(level=['stay_id'])
       
        self.df_48h = self.df_48h[self.df_48h['stay_id'].isin(task2_cohort)]
        self.df_24h = self.df_24h[self.df_24h['stay_id'].isin(task2_cohort)]
        self.df_med = self.df_med[self.df_med['stay_id'].isin(task2_cohort)]
        self.df_hourly = self.df_hourly[self.df_hourly['stay_id'].isin(task2_cohort)]
        self.df_demographic = self.df_demographic[self.df_demographic['stay_id'].isin(task2_cohort)]

        ##one-hot encoding for the medication and the sex
        self.df_med = self.df_med.fillna(value = 0)
        self.df_med[self.df_med.iloc[:,1:] > 0] = 1
        self.df_demographic.gender[self.df_demographic.gender == 'F'] = 1
        self.df_demographic.gender[self.df_demographic.gender == 'M'] = 0

        ##outlier transformation
        new_d = self.df_hourly.loc[self.df_hourly.ABPm < 50, 'ABPd'] + (1/3)*(self.df_hourly.loc[self.df_hourly.ABPm < 50, 'ABPs'] - self.df_hourly.loc[self.df_hourly.ABPm < 50, 'ABPd'])
        new_u = self.df_hourly.loc[self.df_hourly.ABPm > 150, 'ABPd'] + (1/3)*(self.df_hourly.loc[self.df_hourly.ABPm > 150, 'ABPs'] - self.df_hourly.loc[self.df_hourly.ABPm > 150, 'ABPd'])

        self.df_hourly.loc[self.df_hourly.ABPm < 50, 'ABPm'] = new_d
        self.df_hourly.loc[self.df_hourly.ABPm > 150, 'ABPm'] = new_u

        ##create batches 
        batch_hourly = self.create_batchs(self.df_hourly)
        batch_24h = self.create_batchs(self.df_24h)
        batch_48h = self.create_batchs(self.df_48h)
        batch_med = self.create_batchs(self.df_med)
        batch_demographic = self.create_batchs(self.df_demographic)

        ##reindex for patients that don't have entries at the begginning of their stays + cut to 48h
        ##aggregation as well
        for i in range(len(batch_24h)):
            batch_hourly[i] = batch_hourly[i].reindex(range(1, self.nb_hours + 1), fill_value = None) 
            batch_24h[i] = batch_24h[i].reindex(range(1, self.nb_hours//24 + 1), fill_value = None)
            batch_48h[i] = batch_48h[i].reindex(range(1, self.nb_hours//24 + 1), fill_value = None)
            
            batch_hourly[i] = batch_hourly[i].drop(columns = 'stay_id')
            batch_24h[i] = batch_24h[i].drop(columns = 'stay_id')
            batch_48h[i] = batch_48h[i].drop(columns = 'stay_id')
            batch_med[i] = batch_med[i].drop(columns = 'stay_id')
            batch_demographic[i] = batch_demographic[i].drop(columns = 'stay_id')
            batch_48h[i] = batch_48h[i].agg([np.mean])

        self.df_hourly = pd.concat(batch_hourly)
        self.df_24h = pd.concat(batch_24h)
        self.df_48h = pd.concat(batch_48h)
        self.df_med = pd.concat(batch_med)

        ##the stay ids column are dropped since we alreasy took care of them being in the same order for all datasets
        self.df_demographic = self.df_demographic.drop(columns = 'stay_id') 
        batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic = self.data_imputation(batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic, self.imputation, self.random_state)
        
        #feature concatenation 
        final_data = np.array([[np.concatenate([np.concatenate(batch_demographic[i].values),  np.concatenate(batch_hourly[i].values)])] for i in range(len(batch_hourly))])

        final_data = np.squeeze(final_data)
     
        return final_data, labels



    def exp_pr(self, label, transfer, window = 1):
        '''Preprocessing for blood pressure experiement (removal of correlated blood pressures)'''
        
        list = ['ABPm', 'ABPs', 'ABPd']
        list.remove(label)
        self.df_hourly = self.df_hourly.drop(columns = ['icu_intime'])
        self.df_24h = self.df_24h.drop(columns = ['icu_intime'])
        
        ##pivot the tables 
        df_std = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'std')
        df_std = df_std.reset_index(level=['hour_from_intime'])
        labels_std = df_std[df_std.hour_from_intime.isin(range(25,25+window))][label].dropna()
        labels_std = labels_std.groupby('stay_id').mean()
        self.df_hourly = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_24h = self.df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        
        #we remove the other two BP as proposed by dr Libert
        self.df_hourly = self.df_hourly.drop(columns = list)
        # self.df_hourly = self.df_hourly.drop(columns = list[1])

        ##label extraction 
        self.df_hourly_copy = self.df_hourly.reset_index(level=['hour_from_intime'])
        labels = self.df_hourly_copy[self.df_hourly_copy.hour_from_intime.isin(range(25,25+window))][label]
        labels = labels.groupby('stay_id').mean()
        labels = labels.dropna()
        task2_cohort = labels.index.values
        labels = labels.to_frame()
        labels_std = labels_std.to_frame()
        test = labels.join(labels_std, on = 'stay_id', how='inner', lsuffix = '', rsuffix = ' std')

        ##Restriction of the cohort 
        self.df_hourly = self.df_hourly.reset_index(level=['stay_id'])
        df_std = df_std.reset_index(level=['stay_id'])

        self.df_hourly = self.df_hourly[self.df_hourly['stay_id'].isin(task2_cohort)]
        self.df_demographic = self.df_demographic[self.df_demographic['stay_id'].isin(task2_cohort)]

        df_std = df_std[df_std['stay_id'].isin(task2_cohort)]

        self.df_demographic.gender[self.df_demographic.gender == 'F'] = 1
        self.df_demographic.gender[self.df_demographic.gender == 'M'] = 0
        df_std = df_std.fillna(0)

        ##create batches 
        batch_hourly = self.create_batchs(self.df_hourly)
        batch_demographic = self.create_batchs(self.df_demographic)
        batch_std = self.create_batchs(df_std)
        
        ##reindex for patients that don't have entries at the begginning of their stays + cut to 48h
        ##aggregation as well
        data_pr = []
        for i in range(len(batch_demographic)):
            batch_hourly[i] = batch_hourly[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            batch_std[i] = batch_std[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            batch_std[i] = batch_std[i].fillna(0)
            batch_demographic[i] = batch_demographic[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            batch_demographic[i] = batch_demographic[i].fillna(batch_demographic[i].mean())
            batch_hourly[i] = batch_hourly[i].drop(columns = 'stay_id')
            batch_std[i] = batch_std[i].drop(columns = 'stay_id')

            batch_demographic[i] = batch_demographic[i].drop(columns = 'stay_id')
            
            if label == 'ABPs':
                data_pr.append(pd.concat([batch_hourly[i], batch_std[i], batch_demographic[i]], axis=1))
            else:
                data_pr.append(pd.concat([batch_hourly[i], batch_demographic[i]], axis=1))
        
     
        #divide between severe and mild
        if transfer:
            dem = self.df_demographic.drop(columns = 'stay_id').reset_index(drop=True)
            dem['severity'] = dem.apply(lambda row: self.label_severity(row), axis=1)
            mild_idx = np.array(dem[dem['severity'] == 'mild'].index)
            severe_idx = np.array(dem[dem['severity'] == 'severe'].index)
            dem.pop('severity')

        mean = pd.concat(data_pr).mean()
        for i in range(len(batch_hourly)):
                data_pr[i] = data_pr[i].fillna(method = "ffill")
                data_pr[i] = data_pr[i].fillna(mean)
                data_pr[i] = data_pr[i].fillna(0)
                data_pr[i] = normalize(data_pr[i])

        if transfer:
            data_mild = np.array([data_pr[i] for i in mild_idx])
            data_severe = np.array([data_pr[i] for i in severe_idx])
            labels_mild = np.array([labels[i] for i in mild_idx])
            labels_severe = np.array([labels[i] for i in severe_idx])
            return data_pr, labels, data_mild, labels_mild, data_severe, labels_severe
        return data_pr, test
