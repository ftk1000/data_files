# -*- coding: utf-8 -*-
# parse_3gpp_v2.py
# 2020.08.19
#---------------------------------------------
#  README:
#---------------------------------------------
#prepare data set : parse 3GPP 36523-1-f00_s07_01.txt and 
#extract sentences corresponding to "test case name", "with", "when", "then"
#
#INPUT DATA:
#.\36523-1-f00_s07_01.txt
#
#36523-1-f00_s07_01.txt files was generated from 
#36523-1-f00_s07_01.DOC file located in
#
#OUTPUT:
#.\36523-1-f00_s07_01_tests.csv    

#%%
import numpy as np
import pandas as pd
import string

import random

#%%
# SOURCE DOCUMENT
# LTE; Evolved Universal Terrestrial Radio Access (E-UTRA) and Evolved Packet Core (EPC);
# User Equipment (UE) conformance specification; Part 1: Protocol conformance specification
# 3GPP TS 36.523-1

# downloaded *.DOC files are here:  .\36523-1-f00.zip
# we will use only of them converted to TXT format

wd = 'C:\\Users\\Farid Khafizov\\Downloads\\'
file_name = '36523-1-f00_s07_01.txt'

# with open(file=wd+file_name, mode='r', encoding='unicode_escape') as f:
#     lines = f.readlines()
with open(file=wd+file_name, mode='r', encoding='latin1') as f:
    lines = f.readlines()
f.close()
print(len(lines))
#%%
printable = set(string.printable)
lines2 = [  ''.join(filter(lambda x: x in printable, s)) for s in lines  ]
print(len(lines2))
#%%
changed_lines = []
for k in range(len(lines)):
    if lines[k]!=lines2[k]:
        changed_lines.append(k)
        print('--------------line_num=',k)
        print('orig_line->',lines[k])
        print('fltr_line->',lines2[k])
      
#%%
len(changed_lines)
lines = lines2  
#%%
#%%
#%%
# import string
# printable = set(string.printable)
# #x=filter(lambda x: x in printable, s)

# s1 = "some\x00string. with\x15 funny characters"
# s2 = "some string. with\x15 funny\tcharacters"
# x1=''.join(filter(lambda x: x in printable, s1))
# x2=''.join(filter(lambda x: x in printable, s2))
# print('s1= ',s1)
# print('x1= ',x1)
# print('-----------------')
# print('s2= ',s2)
# print('x2= ',x2)



#%%
# find line numbers that contain PATTERN
def find_lineNums(lines, pattern):
    test_idx = []
    for i in range(len(lines)):
      line = lines[i].strip();  # print(line)
      if line.find(pattern) >=0 :
        test_idx.append(i)
    return test_idx
#%%
PATTERN = "Test Purpose (TP)"
test_idx = find_lineNums(lines=lines, pattern= PATTERN)    
print(test_idx)
print("patterns found =",len(test_idx))

#%%
import re
R = re.compile(r"^\(\d+\)")

#for line in lines:
#    M = R.search(line)
#    if M:
#        print(M.string)
# keep looking for '(\d)' until find 'Conformance requirements'
# or the next test (ix2)
def find_idx_subtests(ix1,ix2):
  ix_subtest = []
  for i in range(ix1,ix2):
    M = R.search(lines[i])
    if M:
      #lines[i][0]=='(':
      ix_subtest.append(i)
      # print('==> lineNum =',i, "  line=", lines[i].strip())
    if lines[i].find("Conformance requirements") >=0 :
        ix_subtest.append(i)   #conf_req_line = i
        break;
  return ix_subtest
#%%
k=46
ix1=test_idx[k]; ix2=test_idx[k+1]
subtests = find_idx_subtests(ix1, ix2)  
for p in range(len(subtests)):
    print(subtests[p], '    ',lines[subtests[p]].strip() )

#%%
ix1=218
for i in range(ix1-4, ix1+2):
    print(lines[i].strip()    )
#%%
#k=218
#print(lines[k-4:k+10])
#%%    
d = pd.DataFrame(columns=['doc','test_sec', 'test_name', 'test_num',  'with', 'when', 'then'])
doc_name = file_name.split('.')[0]
#d.loc[0] = ['7.1.1.1',  'CCCH mapped to','1', 'with aa', 'when bb', 'then cc'] 
#d
#%%  BUILD THE DATA BASE
df_index = 0

for k in range(len(test_idx)-1):
    ix=test_idx[k]
    ix2=test_idx[k+1]
    #def update_df(ix):
    [sec_num, test_name] = lines[ix-1].strip().split('\t')
    ix_subtests = find_idx_subtests(ix,ix2)
    for p in range(len(ix_subtests)-1):
        index = ix_subtests[p]
        
        with_kw = when_kw = then_kw = ''
        while index < ix_subtests[p+1]:
            line = lines[index].strip()
            if line.find('with') >=0 :
                with_kw = line[len('with'):].strip()[1:-1].strip()
            if line.find('when') >=0 :
                when_kw = line[len('when'):].strip()[1:-1].strip()
            if line.find('then') >=0 :
                then_kw = line[len('then'):].strip()[1:-1].strip()
            index += 1
            
        d.loc[df_index] = [doc_name, sec_num,  test_name, p+1, with_kw, when_kw, then_kw] 
        df_index += 1
#%%  REVIEW AND SAVE DATA FRAME

print( d.head() )
print(d.shape)


# wd = 'C:\\Users\\Farid Khafizov\\Downloads\\'
# file_name = '36523-1-f00_s07_01.txt'
out_file_tsv = wd+file_name.split('.')[0]+'.tsv'
d.to_csv(out_file_tsv, index=False, sep='\t', encoding='latin1')
#%% Test readability of saved file
f2 = pd.read_csv(out_file_tsv, sep='\t', encoding='latin1')
compare = [ np.sum(d.loc[i] != f2.loc[i]) for i in range(len(d))]
print('If data matches, show 0:',  np.sum(compare) )
f2.shape


#%% SPLIT DATA FRAME INTO TRAIN AND TEST
np.random.seed(seed=10)
idx = np.random.permutation(d.shape[0])
test_rows = random.sample( range(0,d.shape[0]), 3 ) 
test_rows=[0,1,2]
d_test = d.iloc[test_rows]
d_train = d.drop(test_rows)

def get_X_Y(d):
    txt = np.array(list(d['with'])    + list(d['when'])    + list(d['then']))
    lbl = np.array([0]*len(d['with']) + [1]*len(d['when']) + [2]*len(d['then']) )
    return txt, lbl

X_train, y_train = get_X_Y(d_train)
X_test,  y_test  = get_X_Y(d_test)


#%%

#%%

#%%
#%%
#%%
#%%


#%%    
# d.to_csv(wd+doc_name+'_tests.csv')
#%% GENERATE TRAIN AND TEST SETS
'''
from sklearn.model_selection import train_test_split

assert len(lbl) == len(txt)

# Shuffle and split
np.random.seed(seed=10)
idx = np.random.permutation(len(txt))
data, labels = txt[idx], lbl[idx]
df2=pd.DataFrame({'sentences':data,'labels':labels})
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
'''
#%%
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)   # Plain Train set
X_test_counts  = count_vect.transform(X_test)


#%%
models = ['NB', 'NB+TF', 'NB+TFIDF', 'SVM', 'SVM+TF', 'SVM+TFIDF', 'XGB', 'XGB+TF', 'XGB+TFIDF']
oacs = np.zeros(len(models))    
oac_df = pd.DataFrame( {'oac':oacs}, index=models )
oac_df    
#%%
# !pip install xgboost
#%% XGBoost: XGB 
import xgboost as xgb
import warnings
warnings.filterwarnings(module='xgboost*', action='ignore', category=DeprecationWarning)

xgbmodel = xgb.XGBClassifier( max_depth=1, n_estimators=3000, learning_rate=0.01 ).fit(X_train_counts, y_train)
predicted = xgbmodel.predict(X_test_counts)
res=np.mean(predicted == y_test)
print(res)
#oac_df.loc['XGB']=res
res

#%%

print(d.columns)
txt = np.array(list(d['with'])    + list(d['when'])    + list(d['then']))
lbl = np.array([0]*len(d['with']) + [1]*len(d['when']) + [2]*len(d['then']) )
d2 = pd.DataFrame({'sentences':txt, 'labels':lbl})
print(d2.head())


out_fn = file_name.split('.')[0]+'_sent_lbl.tsv'
d2.to_csv(wd + out_fn, index=False, sep='\t', encoding='latin1')
#%% Test readability of saved file
f2 = pd.read_csv(wd + out_fn, sep='\t', encoding='latin1')
# compare = [ d2.sentences.loc[i] != f2.sentences.loc[i] for i in range(len(d2))]
compare = [ np.sum(d2.loc[i] != f2.loc[i]) for i in range(len(d2))]
print('If data matches, show 0:',  np.sum(compare) )
f2.shape

#%%   COPIED FROM
# https://gitlab.verizon.com/atf/atf-development/atf-data-sciences/-/blob/master/ML_models/atf_xgb_classifier.py

# df2=pd.DataFrame({'sentences':data,'labels':labels})
def compare_classifiers(d2, test_fraction, seedval):
#    VALIDATION_SPLIT = 0.3
    np.random.seed(seed=seedval)
    indices = np.arange(d2.shape[0]) # get sequence of row index
    np.random.shuffle(indices) # shuffle the row indexes
    data   = d2['sentences'][indices] # shuffle data/product-titles/x-axis
    labels = d2['labels'][indices]
    #
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_fraction, random_state=42)
    
#    #%% COMPARE  TRAIN  AND  TEST  SET DISTRIBUTIONS
#    vdf=pd.concat([y_train.value_counts(normalize=True).round(3) * 100, 
#                   y_test.value_counts(normalize=True).round(3) * 100],
#    sort =False, axis=1,ignore_index=True )
#    vdf.columns = ['train','test']
#    vdf.to_csv(wd+"comp_train_and_test_lables.csv")
    
    #%   ========================================================
    # INITIALIZE OAC SUMMARY TABLE
    import numpy
    models = ['NB', 'NB+TF', 'NB+TFIDF', 'SVM', 'SVM+TF', 'SVM+TFIDF', 'XGB', 'XGB+TF', 'XGB+TFIDF']
    oacs = numpy.zeros(len(models))    
    oac_df = pd.DataFrame( {'oac':oacs}, index=models )
    oac_df    
    
    
    #%  DEFINE THREE TRAIN SETS
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)   # Plain Train set
    X_test_counts  = count_vect.transform(X_test)
    
    #%
    from sklearn.feature_extraction.text import TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)  # TF train set 
    X_test_tf  = tf_transformer.transform(X_test_counts)
    #%
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)  # TFIDF Train set
    X_test_tfidf  = tfidf_transformer.transform(X_test_counts)
    
    #% XGBoost: XGB 
    import xgboost as xgb
    import warnings
    warnings.filterwarnings(module='xgboost*', action='ignore', category=DeprecationWarning)
    
    #xgbmodel = xgb.XGBClassifier( max_depth=3, n_estimators=300, learning_rate=0.05 ).fit(X_train_counts, y_train)
    xgbmodel = xgb.XGBClassifier( max_depth=1, n_estimators=3000, learning_rate=0.01 ).fit(X_train_counts, y_train)
    predicted = xgbmodel.predict(X_test_counts)
    res=np.mean(predicted == y_test)
    oac_df.loc['XGB']=res
    print('XGB', res)
    
    #% XGBoost: XGB + TF
    #xgbmodel = xgb.XGBClassifier( max_depth=3, n_estimators=300, learning_rate=0.05 ).fit(X_train_tf, y_train)
    xgbmodel = xgb.XGBClassifier( max_depth=1, n_estimators=3000, learning_rate=0.01 ).fit(X_train_tf, y_train)
    predicted = xgbmodel.predict(X_test_tf)
    res=np.mean(predicted == y_test)
    oac_df.loc['XGB+TF']=res
    print('XGB+TF', res)
    
    #% XGB+TFIDF
    #xgbmodel = xgb.XGBClassifier( max_depth=3, n_estimators=300, learning_rate=0.05 ).fit(X_train_tfidf, y_train)
    xgbmodel = xgb.XGBClassifier( max_depth=1, n_estimators=3000, learning_rate=0.01 ).fit(X_train_tfidf, y_train)
    predicted = xgbmodel.predict(X_test_tfidf)
    res=np.mean(predicted == y_test)
    oac_df.loc['XGB+TFIDF']=res
    print('XGB+TFIDF', res)
    
    #% Plain NB Classifier
    from sklearn.naive_bayes import MultinomialNB
    #clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        #('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    res=np.mean(predicted == y_test)
    oac_df.loc['NB']=res
    print('NB', res)
    
    #% NB iwth TF
    # TfidfTransformer(use_idf=False)
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultinomialNB()),])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    res=np.mean(predicted == y_test)
    oac_df.loc['NB+TF']=res
    print('NB+TF', res)

    #%
#    lbls=[ x for x in set(labels)]
#    print(lbls)
#    cf = cmdf(y_test, predicted, lbls)
#    cf.to_csv(wd+'xgb_tf_cf.csv')
    
    #% NB with TFIDF
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultinomialNB()),])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    res=np.mean(predicted == y_test)
    oac_df.loc['NB+TFIDF']=res
    print('NB+TFIDF', res)


    
    #% Plain SVM
    from sklearn.linear_model import SGDClassifier    
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)), ])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    res=np.mean(predicted == y_test)
    oac_df.loc['SVM']=res
    print('SVM', res)

    
    #% SVM with TF
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf',  SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)), ])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    res=np.mean(predicted == y_test)
    oac_df.loc['SVM+TF']=res
    print('SVM+TF', res)
    
    #% SVM with TFIDF
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)), ])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    res=np.mean(predicted == y_test)
    oac_df.loc['SVM+TFIDF']=res
    print('SVM+TFIDF', res)
    
    return oac_df    
    
#========================================================================    
    
#%%
# df2=pd.DataFrame({'sentences':data,'labels':labels})
rr = compare_classifiers(d2=d2, test_fraction=0.2, seedval=30)
#%%
models = ['NB', 'NB+TF', 'NB+TFIDF', 'SVM', 'SVM+TF', 'SVM+TFIDF', 'XGB', 'XGB+TF', 'XGB+TFIDF']
oacs = np.zeros(len(models))    
resres = pd.DataFrame( {'oac':oacs}, index=models )

#%%
for k in range(8):
    oac_df = compare_classifiers(d2=d2, test_fraction=0.2, seedval=10*k)
    oac_df.columns = [oac_df.columns[0]+str(k)]
    print(k)
    resres = pd.concat([resres, oac_df], axis=1)
    
#%% Delete col with all zeros
resres.__delitem__('oac')    

#%%
resres.to_csv(wd+'SSS_classif_res2_20200818.csv', sep=',')
#resres = pd.concat([res1, res2], axis=1)
#%%
np.mean(resres,1)
