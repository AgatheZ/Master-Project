import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
from sklearn.preprocessing import normalize

import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class Preprocessing:
    def __init__(self, df_hourly, df_24h, df_48h, df_med, df_demographic, nb_hours, TBI_split, random_state, imputation):
        self.df_hourly = df_hourly
        self.df_24h = df_24h
        self.df_48h = df_48h
        self.df_med = df_med
        self.df_demographic = df_demographic
        self.nb_hours = nb_hours
        self.TBI_split = TBI_split
        self.random_state = random_state
        self.imputation = imputation 
        
        
    def create_batchs(self, ds):
        batchs = []
        ids = ds.stay_id.unique()
        for i in ids:
            batchs.append(ds.loc[ds['stay_id'] == i])
        return batchs

    def label_severity (self, row):
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

    def get_column_name(df):
        listn = [col for col in df.columns]
        return listn

    def aggregation(batch, rate):
        'function that takes a batch of patients and returns the aggregated vitals with the correct aggregation rate'
        if rate == 1:
            return batch
        elif rate == 24:
            bch = []
            for df in batch:
                df['hour_slice'] = 0
                df['hour_slice'][range(25,49)] = 1
                df = df.groupby('hour_slice').mean()
                bch.append(df)
            return bch
        elif rate == 48:
            bch = []
            for df in batch:
                df['hour_slice'] = 0
                df = df.groupby('hour_slice').mean()
                bch.append(df)
            return bch

    def arrange_ids(df1, df2, df3, df4, df5):
        ids1 = df1.stay_id.unique()
        ids2 = df2.stay_id.unique()
        ids3 = df3.stay_id.unique()
        ids4 = df4.stay_id.unique()
        ids5 = df5.stay_id.unique()

        min_ids = list(set(ids1) & set(ids2) & set(ids3) & set(ids4) & set(ids5))
        return df1.loc[df1['stay_id'].isin(min_ids)], df2.loc[df2['stay_id'].isin(min_ids)], df3.loc[df3['stay_id'].isin(min_ids)], df4.loc[df4['stay_id'].isin(min_ids)], df5.loc[df5['stay_id'].isin(min_ids)]

    def data_imputation(self, batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic, type, random_state):
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
            return batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic

    def preprocess_data(self):
        self.df_hourly = self.df_hourly.drop(columns = ['icu_intime'])
        self.df_24h = self.df_24h.drop(columns = ['icu_intime'])
        self.df_48h = self.df_48h.drop(columns = ['icu_intime'])
        
        if self.TBI_split:
            self.df_demographic['severity'] = self.df_demographic.apply(lambda row: self.label_severity(row), axis=1)
            mild_idx = self.df_demographic[self.df_demographic['severity'] == 'mild']['stay_id']
            severe_idx = self.df_demographic[self.df_demographic['severity'] == 'severe']['stay_id']
            self.df_demographic.pop('severity')

        ##label extraction 
        if self.TBI_split:
            labels_mild = self.df_demographic[self.df_demographic.stay_id.isin(mild_idx)].pop('los')
            labels_mild[labels_mild <= 4] = 0
            labels_mild[labels_mild > 4] = 1
            labels_mild = labels_mild.values

            labels_severe = self.df_demographic[self.df_demographic.stay_id.isin(severe_idx)].pop('los')
            labels_severe[labels_severe <= 4] = 0
            labels_severe[labels_severe > 4] = 1
            labels_severe = labels_severe.values

            self.df_demographic.pop('los')

        else:
            labels = self.df_demographic.pop('los')
            labels[labels <= 4] = 0
            labels[labels > 4] = 1
            labels = labels.values

        ##pivot the tables 
        self.df_hourly = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_24h = self.df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_48h = self.df_48h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_med = self.df_med.pivot_table(index = ['stay_id'], columns = 'med_name', values = 'amount')



        ##one-hot encoding for the medication and the sex
        self.df_med = self.df_med.fillna(value = 0)
        self.df_med[self.df_med > 0] = 1
        self.df_demographic.gender[self.df_demographic.gender == 'F'] = 1
        self.df_demographic.gender[self.df_demographic.gender == 'M'] = 0


        ##create batches 
        self.df_hourly = self.df_hourly.reset_index(level=['stay_id'])
        self.df_24h = self.df_24h.reset_index(level=['stay_id'])
        self.df_48h = self.df_48h.reset_index(level=['stay_id'])
        self.df_med = self.df_med.reset_index(level=['stay_id'])


        if self.TBI_split:
            batch_hourly_mild = self.create_batchs(self.df_hourly[self.df_hourly.stay_id.isin( mild_idx)])
            batch_24h_mild = self.create_batchs(self.df_24h[self.df_24h.stay_id.isin( mild_idx)])
            batch_48h_mild = self.create_batchs(self.df_48h[self.df_48h.stay_id.isin( mild_idx)])
            batch_med_mild = self.create_batchs(self.df_med[self.df_med.stay_id.isin( mild_idx)])
            batch_demographic_mild = self.create_batchs(self.df_demographic[self.df_demographic.stay_id.isin( mild_idx)])
            batch_hourly_severe = self.create_batchs(self.df_hourly[self.df_hourly.stay_id.isin(severe_idx)])
            batch_24h_severe = self.create_batchs(self.df_24h[self.df_24h.stay_id.isin(severe_idx)])
            batch_48h_severe = self.create_batchs(self.df_48h[self.df_48h.stay_id.isin(severe_idx)])
            batch_med_severe = self.create_batchs(self.df_med[self.df_med.stay_id.isin(severe_idx)])
            batch_demographic_severe = self.create_batchs(self.df_demographic[self.df_demographic.stay_id.isin(severe_idx)])
            
        else:
            batch_hourly = self.create_batchs(self.df_hourly)
            batch_24h = self.create_batchs(self.df_24h)
            batch_48h = self.create_batchs(self.df_48h)
            batch_med = self.create_batchs(self.df_med)
            batch_demographic = self.create_batchs(self.df_demographic)

        ##reindex for patients that don't have entries at the begginning of their stays + cut to 48h
        ##aggregation as well
        if self.TBI_split:
            for i in range(len(batch_24h_mild)):
                    batch_hourly_mild[i] = batch_hourly_mild[i].reindex(range(1, self.nb_hours + 1), fill_value = None) 
                    batch_24h_mild[i] = batch_24h_mild[i].reindex(range(1, self.nb_hours//24 + 1), fill_value = None)
                    batch_48h_mild[i] = batch_48h_mild[i].reindex(range(1, self.nb_hours//24 + 1), fill_value = None)
                    
                    
                    batch_hourly_mild[i] = batch_hourly_mild[i].drop(columns = 'stay_id')
                    batch_24h_mild[i] = batch_24h_mild[i].drop(columns = 'stay_id')
                    batch_48h_mild[i] = batch_48h_mild[i].drop(columns = 'stay_id')
                    batch_med_mild[i] = batch_med_mild[i].drop(columns = 'stay_id')
                    batch_demographic_mild[i] = batch_demographic_mild[i].drop(columns = 'stay_id')
                    batch_48h_mild[i] = batch_48h_mild[i].agg([np.mean])
                    

            for i in range(len(batch_24h_severe)):
                batch_hourly_severe[i] = batch_hourly_severe[i].reindex(range(1, self.nb_hours + 1), fill_value = None) 
                batch_24h_severe[i] = batch_24h_severe[i].reindex(range(1, self.nb_hours//24 + 1), fill_value = None)
                batch_48h_severe[i] = batch_48h_severe[i].reindex(range(1, self.nb_hours//24 + 1), fill_value = None)
                batch_hourly_severe[i] = batch_hourly_severe[i].drop(columns = 'stay_id')
                batch_24h_severe[i] = batch_24h_severe[i].drop(columns = 'stay_id')
                batch_48h_severe[i] = batch_48h_severe[i].drop(columns = 'stay_id')
                batch_med_severe[i] = batch_med_severe[i].drop(columns = 'stay_id')
                batch_demographic_severe[i] = batch_demographic_severe[i].drop(columns = 'stay_id')
                batch_48h_severe[i] = batch_48h_severe[i].agg([np.mean])
        else:
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


        if self.TBI_split:
            df_hourly_mild = pd.concat(batch_hourly_mild)
            df_24h_mild = pd.concat(batch_24h_mild)
            df_48h_mild = pd.concat(batch_48h_mild)
            df_med_mild = pd.concat(batch_med_mild)

            df_hourly_severe = pd.concat(batch_hourly_severe)
            df_24h_severe = pd.concat(batch_24h_severe)
            df_48h_severe = pd.concat(batch_48h_severe)
            df_med_severe = pd.concat(batch_med_severe)

        else:
            self.df_hourly = pd.concat(batch_hourly)
            self.df_24h = pd.concat(batch_24h)
            self.df_48h = pd.concat(batch_48h)
            self.df_med = pd.concat(batch_med)

        ##the stay ids column are dropped since we alreasy took care of them being in the same order for all datasets
        if self.TBI_split:
            df_demographic_mild = self.df_demographic[self.df_demographic.stay_id.isin(mild_idx)].drop(columns = 'stay_id')
            df_demographic_severe = self.df_demographic[self.df_demographic.stay_id.isin(severe_idx)].drop(columns = 'stay_id')
        else:
            self.df_demographic = self.df_demographic.drop(columns = 'stay_id')

        if self.TBI_split:
            batch_hourly_mild, batch_24h_mild, batch_48h_mild, batch_med_mild, batch_demographic_mild = self.data_imputation(batch_hourly_mild, batch_24h_mild, batch_48h_mild, batch_med_mild, batch_demographic_mild, self.imputation, self.random_state)
            batch_hourly_severe, batch_24h_severe, batch_48h_severe, batch_med_severe, batch_demographic_severe = self.data_imputation(batch_hourly_severe, batch_24h_severe, batch_48h_severe, batch_med_severe, batch_demographic_severe, self.imputation, self.random_state)
                
        else:
            batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic = self.data_imputation(batch_hourly, batch_24h, batch_48h, batch_med, batch_demographic, self.imputation, self.random_state)
        

        #feature concatenation 
        stratify_param = self.df_demographic.gcs
        if self.TBI_split:
            if self.nb_hours == 24:
                final_data_mild = np.array([[np.concatenate([np.concatenate(batch_demographic_mild[i].values), np.concatenate(batch_hourly_mild[i].values), np.concatenate(batch_24h_mild[i].values), np.concatenate(batch_med_mild[i].values)])] for i in range(len(batch_hourly_mild))])
                final_data_severe = np.array([[np.concatenate([np.concatenate(batch_demographic_severe[i].values), np.concatenate(batch_hourly_severe[i].values), np.concatenate(batch_24h_severe[i].values), np.concatenate(batch_med_severe[i].values)])] for i in range(len(batch_hourly_severe))]) 
            else: 
                final_data_mild = np.array([[np.concatenate([np.concatenate(batch_demographic_mild[i].values), np.concatenate(batch_hourly_mild[i].values), np.concatenate(batch_24h_mild[i].values), np.concatenate(batch_48h_mild[i].values), np.concatenate(batch_med_mild[i].values)])] for i in range(len(batch_hourly_mild))])
                final_data_severe = np.array([[np.concatenate([np.concatenate(batch_demographic_severe[i].values), np.concatenate(batch_hourly_severe[i].values), np.concatenate(batch_24h_severe[i].values), np.concatenate(batch_48h_severe[i].values), np.concatenate(batch_med_severe[i].values)])] for i in range(len(batch_hourly_severe))])

        else:
            if self.nb_hours == 24:
                final_data = np.array([[np.concatenate([np.concatenate(batch_demographic[i].values), np.concatenate(batch_hourly[i].values), np.concatenate(batch_24h[i].values), np.concatenate(batch_med[i].values)])] for i in range(len(batch_hourly))])
            else: 
                final_data = np.array([[np.concatenate([np.concatenate(batch_demographic[i].values), np.concatenate(batch_hourly[i].values), np.concatenate(batch_24h[i].values), np.concatenate(batch_48h[i].values), np.concatenate(batch_med[i].values)])] for i in range(len(batch_hourly))])


        # df_24h.to_csv('df_24h.csv')
        # df_48h.to_csv('df_48h.csv')
        # df_med.to_csv('df_med.csv')
        # df_demographic.to_csv('df_demographic.csv')

        if self.TBI_split:
            final_data_mild = np.squeeze(final_data_mild)
            final_data_mild = normalize(final_data_mild)
            final_data_severe = np.squeeze(final_data_severe)
            final_data_severe = normalize(final_data_severe)
            return final_data_mild, final_data_severe, labels_mild, labels_severe
        else:
            final_data = np.squeeze(final_data)
            final_data = normalize(final_data)
            return final_data, labels
        
    def time_series_pr(self):
        self.df_hourly = self.df_hourly.drop(columns = ['icu_intime'])
        self.df_24h = self.df_24h.drop(columns = ['icu_intime'])
        
        ##label extraction 
        labels = self.df_demographic.pop('los')
        labels[labels <= 4] = 0
        labels[labels > 4] = 1
        labels = labels.values

        ##pivot the tables 
        self.df_hourly = self.df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
        self.df_24h = self.df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')

        ##create batches 
        self.df_hourly = self.df_hourly.reset_index(level=['stay_id'])
        self.df_24h = self.df_24h.reset_index(level=['stay_id'])

        batch_hourly = self.create_batchs(self.df_hourly)
        batch_24h = self.create_batchs(self.df_24h)

        ##reindex for patients that don't have entries at the begginning of their stays + cut to 48h
        ##aggregation as well
        data_pr = []
        for i in range(len(batch_24h)):
            batch_hourly[i] = batch_hourly[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            batch_24h[i] = batch_24h[i].reindex(range(1, self.nb_hours//24 + 1), fill_value = None)
            batch_24h[i] = batch_24h[i].reindex(range(1, self.nb_hours + 1), fill_value = None)
            batch_24h[i] = batch_24h[i].fillna(batch_24h[i].mean())
            batch_hourly[i] = batch_hourly[i].drop(columns = 'stay_id')
            batch_24h[i] = batch_24h[i].drop(columns = 'stay_id')
            data_pr.append(pd.concat([batch_hourly[i], batch_24h[i]], axis=1))
        
        mean = pd.concat(data_pr).mean()
        for i in range(len(batch_hourly)):
                data_pr[i] = data_pr[i].fillna(method = "ffill")
                data_pr[i] = data_pr[i].fillna(mean)
    
        return data_pr, labels
