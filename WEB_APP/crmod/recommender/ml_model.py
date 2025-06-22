import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

parent_fold = os.path.join(os.getcwd(), 'recommender')
COURSE_DETAILS = pd.read_csv(os.path.join(parent_fold,"COURSE_DATA.csv"))
minor_courses = COURSE_DETAILS.loc[COURSE_DETAILS['Division'] == ' M']['Course Code'].unique()
for i in minor_courses:
    COURSE_DETAILS.drop(COURSE_DETAILS.loc[(COURSE_DETAILS['Course Code'] == i )&(COURSE_DETAILS['Division']!= ' M')].index, inplace=True)
COURSE_DETAILS = COURSE_DETAILS.drop_duplicates('Course Code').set_index("Course Code").drop("Unnamed: 0", axis = 1)
COURSE_DETAILS['SLOT'] = COURSE_DETAILS['Slot'].str.split('\n').apply(lambda x: x[0] if type(x) == list else pd.NA)

def read_pkl (file_name):
    # Reading the Pickle file
    with open(file_name, 'rb') as file:
        # Load the object from the file
        loaded_object = pickle.load(file)
    return loaded_object
    # Now 'loaded_object' contains the data from the Pickle file

std_features_with_clusters=read_pkl(os.path.join(parent_fold,'STUDENT_FEATURES_WITH_CLUSTER.pkl'))  ## not required as such
student_feature_scaler=read_pkl(os.path.join(parent_fold,'STUDENT_FEATURE_SCALER.pkl'))
GMM_model=read_pkl(os.path.join(parent_fold,'STUDENT_CLUSTERING_MODEL.pkl'))
course_features_with_clusters=read_pkl(os.path.join(parent_fold,'COURSE_FEATURES_WITH_CLUSTER.pkl'))
mapping_dataset=read_pkl(os.path.join(parent_fold,'MAPPING_DATASET.pkl'))
mapping_dataset.set_index('student', inplace=True)
course_vectors = course_features_with_clusters.loc[:,'e0':'e49']

def get_courses(branch,degree,year,history):

    #----------------------------------------------------------------
    branch = branch.replace(' ', '')
    branch = branch.upper()
    #----------------------------------------------------------------
    degree = degree.replace(' ','')
    degree = degree.upper()
    #----------------------------------------------------------------
    year = year.replace(' ','')
    year = int(year)
    #----------------------------------------------------------------
    j = 0
    while j < len(history):
        a = history[j]
        if a != '':
            a = a.upper()
            a = a.replace(' ','')
            txt, num = '',''
            for i in a:
                if i.isalpha() == True:
                    txt += i
                if i.isdigit() == True:
                    num += i
            if len(txt + num) < 6:
                a = txt + ' ' + num
            else:
                a = txt + num
        history[j] = a
        j+=1
    history = list(set(history))  
    history = list(filter(lambda x: x in course_features_with_clusters.index, history))
    #----------------------------------------------------------------
    bdic={'CS': 0,'EE': 1,'ME': 2,'PH': 3,'MA': 4,'AE': 5,'HS': 6,'CL': 7,'CE': 8,'MMM': 9,'CH': 10,'AES': 11,'BB': 12,'BBS': 13,'CES': 14,'CHS': 15,'CLS': 16,'CM': 17,'CMS': 18,'CSS': 19,'DE': 20,'DEP': 21,'DH': 22,'DHS': 23,'DS': 24,'DSS': 25,'EES': 26,'EN': 27,'ENS': 28,'ENT': 29,'ES': 30,'ESS': 31,'ET': 32,'ETS': 33,'GNR': 34,'GNRS': 35,'GP': 36,'GS': 37,'GSS': 38,'HSS': 39,'ID': 40,'IE': 41,'IES': 42,'IWE': 43,'MAS': 44,'MES': 45,'MG': 46,'MGP': 47,'MGS': 48,'MGT': 49,'MM': 50,'MMS': 51,'MNG': 52,'PHS': 53,'PS': 54,'PSS': 55,'SC': 56,'SCS': 57,'SI': 58,'SOM': 59,'TD': 60,'US': 61,'USS': 62}
    branch=bdic[branch]
    ddic={"UG":1, "PG":-1}
    degree=ddic[degree]
    
    final_list=[]
    if degree==-1:
        year=min(3,year)
    else:
        year=min(4,year)
    
    for course in history:
        try:
            arr=course_features_with_clusters.loc[course, :'e49'].values
            final_list.append(arr)            
        except:
            continue

    if len(final_list) ==0:
        final_list = [[0 for i in range(54)]]

    mean = np.array(final_list).mean(axis = 0)
    std = np.array(final_list).std(axis = 0)

    scaled_student_data=list(student_feature_scaler.transform(np.append(np.array([branch,degree,year]),np.append(mean,std)).reshape(1,-1)))

    student_cluster=GMM_model.predict(scaled_student_data)[0]
    
    print(student_cluster)
    most_relevant_course_cluster = mapping_dataset.loc[student_cluster]['course'].index[0]
    
    ret_1 = course_features_with_clusters.loc[course_features_with_clusters['GMM'] == most_relevant_course_cluster]    
    
    if len(history) == 0:
        ret_1['sort_var'] = ret_1['Total']*ret_1['average grade']
        ret_1 = ret_1.sort_values('sort_var', ascending = False)
    else:
        ret_1['similarity'] = [cosine_similarity(i.reshape(1,-1), mean[4:].reshape(1,-1))[0][0] for i in course_vectors.loc[ret_1.index].values]
        # If subject-wise relevant courses not present in the first course cluster
        num_cluster = 1
        max_num_cluster = mapping_dataset.loc[student_cluster]['course'].shape[0]
        print(max_num_cluster)
        while ret_1['similarity'].mean() < 0  and num_cluster < min(10, max_num_cluster):
            next_relevant_cluster = mapping_dataset.loc[student_cluster]['course'].index[num_cluster]
            num_cluster += 1
            ret_1 = pd.concat([ret_1, course_features_with_clusters.loc[course_features_with_clusters['GMM'] == next_relevant_cluster]])
            ret_1['similarity'] = [cosine_similarity(i.reshape(1,-1), mean[4:].reshape(1,-1))[0][0] for i in course_vectors.loc[ret_1.index].values]  
        ret_1 = ret_1.sort_values('similarity', ascending=False)
    
    branch_name = list(bdic.keys())[list(bdic.values()).index(branch)]
    
    #ret_1 = ret_1.loc[(ret_1.index.str.match('^[A-Z ]{3}2\d{2}$') != True) | (ret_1.index.str.match(f'^{branch_name}') != True)]
    ret_2 = COURSE_DETAILS.loc[ret_1.index][['Course Name','SLOT','Description','Division', 'SEM']]
    ret_2 = ret_2.loc[(ret_2['Division'] != ' M') | (ret_2.index.str.match(f'^{branch_name}') != True)]

    return ret_2.loc[filter(lambda x: x not in history,ret_2.index)].iloc[:20].to_html()
