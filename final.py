# %% [markdown]
## PYTHON PROGRAM OF THE SAME NOTEBOOK FILE
# ## FILES Required
# 
# [Click here to directly reach the Final Program](#final-program)
# 
# | File | Description |
# | -- | -- |
# |COURSE_DATA.csv | DATASET of Course Description, slots, professor etc |
# |STUDENT_FEATURES_WITH_CLUSTER.pkl | (OPTIONAL) Dataset of student features (just for debugging) |
# |STUDENT_FEATURE_SCALER.pkl | A sklearn StandardScaler() object used for scaling student features |
# |STUDENT_CLUSTERING_MODEL.pkl | A sklearn GMM() object used for giving the cluster to which a particular student with some student features belongs to |
# |COURSE_FEATURES_WITH_CLUSTER.pkl | A dataset which has the course features |
# |MAPPING_DATASET.pkl | Maps the student belonging to a particular student-cluster to courses for a particular course-cluster |

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
COURSE_DETAILS = pd.read_csv("COURSE_DATA.csv").drop_duplicates('Course Code').set_index("Course Code").drop("Unnamed: 0", axis = 1)
COURSE_DETAILS['SLOT'] = COURSE_DETAILS['Slot'].str.split('\n').apply(lambda x: x[0] if type(x) == list else pd.NA)

# %%
def read_pkl (file_name):
    # Reading the Pickle file
    with open(file_name, 'rb') as file:
        # Load the object from the file
        loaded_object = pickle.load(file)
    return loaded_object
    # Now 'loaded_object' contains the data from the Pickle file


# %%
std_features_with_clusters=read_pkl('STUDENT_FEATURES_WITH_CLUSTER.pkl')  ## not required as such
student_feature_scaler=read_pkl('STUDENT_FEATURE_SCALER.pkl')
GMM_model=read_pkl('STUDENT_CLUSTERING_MODEL.pkl')
course_features_with_clusters=read_pkl('COURSE_FEATURES_WITH_CLUSTER.pkl')
mapping_dataset=read_pkl('MAPPING_DATASET.pkl')

# %%
mapping_dataset.set_index('student', inplace=True)

# %%
def get_courses(branch,degree,year,history,course_features_with_clusters,mapping_dataset,GMM_model,student_feature_scaler,std_features_with_clusters):
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
    return (course_features_with_clusters.loc[course_features_with_clusters['GMM'] == most_relevant_course_cluster])

# %%
l1=get_courses('ME','UG',3,['IE 622', "IE 609"],course_features_with_clusters,mapping_dataset,GMM_model,student_feature_scaler,std_features_with_clusters)
COURSE_DETAILS.loc[l1.index][['Course Name','SLOT','Description']]

# %% [markdown]
# ## Final Program

# %%
## ****Program****

branches = ['CS', 'EE', 'ME', 'PH', 'MA', 'AE', 'HS', 'CL', 'CE', 'MMM', 'CH', 'AES', 'BB', 'BBS', 'CES', 'CHS', 'CLS', 'CM', 'CMS', 'CSS', 'DE', 'DEP', 'DH', 'DHS', 'DS', 'DSS', 'EES', 'EN', 'ENS', 'ENT', 'ES', 'ESS', 'ET', 'ETS', 'GNR', 'GNRS', 'GP', 'GS', 'GSS', 'HSS',
 'ID', 'IE', 'IES', 'IWE', 'MAS', 'MES', 'MG', 'MGP', 'MGS', 'MGT', 'MM', 'MMS', 'MNG', 'PHS', 'PS', 'PSS', 'SC', 'SCS', 'SI', 'SOM', 'TD', 'US', 'USS']
branches.sort()
branches = list(map(lambda x: x+' '*(4-len(x)), branches))
print('|Possible branches are', *[' | '.join(branches[i:i+9]) for i in range(0, len(branches), 9)], sep = '|\n|', end = '|')
print('\n')

branch = input("PLEASE ENTER YOUR BRANCH (SELECT FROM ABOVE):")
branch = branch.replace(' ', '')
branch = branch.upper()

degree = input("PLEASE ENTER YOUR DEGREE (UG/PG)")
degree = degree.replace(' ','')
degree = degree.upper()

year = input("PLEASE ENTER YOUR YEAR (SIMPLY THE NUMBER):")
year = year.replace(' ','')
year = int(year)

history = []
print('\n')
print("WE WOULD NEED THE COURSES YOU HAVE TAKEN PREVIOUSLY")
print("PLEASE MENTION ALL THE COURSE YOU HAVE TAKEN PREVIOUSLY OR ARE THINKING TO TAKE PREASENTLY")
print("JUST PRESS ENTER IF YOU ARE DONE")

a = input("PLEASE ENTER COURSE (WITHOUT ANY SPACE IN BETWEEN)")

while a != '':
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
    if a in course_features_with_clusters.index:
        print("MATCHED", a)
        history.append(a)
    else:
        print("WARNING: COULDN'T FIND THE COURSE",a)
        
    a = input("PLEASE ENTER COURSE (WITHOUT ANY SPACE IN BETWEEN)")
history = list(set(history))
print("YOUR STUDENT FEATURES ARE", [branch, degree, year, history])

l1=get_courses(branch,degree,year,history,course_features_with_clusters,mapping_dataset,GMM_model,student_feature_scaler,std_features_with_clusters)
print(COURSE_DETAILS.loc[l1.index][['Course Name','SLOT','Description']])

# %%
set(['IE 622', 'IE 609', 'IE 612', 'ME 794', 'ME 794', 'DS 203', 'GNR652', 'CS 228'])

# %%


# %%



