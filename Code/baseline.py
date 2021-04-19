# ---prior work 1: 5-fold cross validation MLP method----
def MLP(data_folder,result_filename)
    models = [[MLPClassifier(hidden_layer_sizes = 100, max_iter=5000, tol = 1e-4), "NN"]
            ]
    # pair of vectorizer and printable text to be put in the result table
    analyzers = [[CountVectorizer(analyzer="word"),"word"]]
    all_results = pd.DataFrame()
    # for prep in preprocessing:
    # input_file = data_folder + "/" + training_file + "_" + prep + ".csv"
    for i, input_file in enumerate(os.listdir(data_folder)):
        print(i,input_file)
    #reading files
        df = pd.read_csv(os.path.join(data_folder,input_file), header = 0,names=['body','skill_tag']).fillna(0)
        original_headers = list(df.columns.values)

        body_column = df['body'].astype(str)
        skill_column = df['skill_tag']

        #5fold cv
        num_folds = 5
        folds = np.floor(np.random.rand(len(body_column))*num_folds)
        for analyzer in analyzers:            
            for model in models:
                clf = model[0]
                vec = analyzer[0]
                print(str(model[1]) + " " + str(analyzer[1]))

                count_all = 0.0
                count_correct = 0.0            
                for i in range(0, num_folds):

                    train_body = vec.fit_transform(body_column[folds != i]).toarray()
                    test_body = vec.transform(body_column[folds == i]).toarray()
                    train_body = train_body / (1e-16 + np.sum(train_body, axis = 1)[:,None]) # ratio instead of count 
                    test_body = test_body / (1e-16 + np.sum(test_body, axis = 1)[:,None]) #ratio instead of count

                    train_label = skill_column[folds != i]
                    test_label = skill_column[folds == i]

                    clf = clf.fit(train_body, train_label)
                    predicted = clf.predict(test_body)
                    pred_proba=clf.predict_proba(test_body)
                    count_correct = count_correct + np.sum(predicted == test_label)
                    count_all = count_all + len(test_label)
                    best_3=np.argsort(pred_proba,axis=1)[:,-3:]
    # transform the labels
                    le = LabelEncoder()
                    le.fit(df['skill_tag'])
                    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    test_label=list(test_label)
                    success=0
                    for i, ans in enumerate(test_label):
                    if predicted[i]==test_label[i]:
                        success+=1
                    elif le_name_mapping[ans] in best_3[i,:]:
                        success+=1

                # train_body = vec.fit_transform(body_column).toarray()
                # train_body = train_body / (1e-16 + np.sum(train_body, axis = 1)[:,None])
                # train_label = skill_column
                # clf = model[0]
                # clf = clf.fit(train_body, train_label)

                print(f'top 1 accuracy:{count_correct/count_all}')
                print(f'top 3 accuracy:{float(success)/len(test_label)}')
                this_result = pd.DataFrame(data = {"prep":['keep_none'],
                                                "vec":[analyzer[1]],
                                                "model":[model[1]],
                                                "top1_accu":[count_correct/count_all],
                                                'top3_accu':[float(success)/len(test_label)]})
                all_results = all_results.append(this_result)
    print(all_results)
    all_results.to_csv(result_filename)
    return all_results


data_folder = "preprocessing_all" # folder containing all preprocessed filed
result_filename = "all_results_all_countVec.csv" # file to store final result
MLP(data_folder,result_filename)
# ---prior work 2: skip-gram method----
import os
import tensorflow.keras
import gensim
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc

from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from gensim.models import Word2Vec
print(f' tensorflow version {tf.version.VERSION}')
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Embedding,Activation,Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
import collections
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard


data_dir='SKIP_GRAM_MODEL/FINAL'
# def read_data(data_dir):
for i, f in enumerate(os.listdir(data_dir)):
  print(i,f)
  data=pd.read_csv(os.path.join(data_dir,f))
  print(data.shape)
  

desc=pd.read_csv('SKIP_GRAM_MODEL/FINAL/all_grades_combined_data_rm_na_desc.csv')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
desc['label_en']=le.fit_transform(desc['label'])
desc['seq_id']=desc.index
desc.head()
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
# Make unigram
stemmer = SnowballStemmer('english')
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def get_tokens(text):
  lowers = text.lower()
  no_punctuation = lowers.translate(remove_punctuation_map)
  tokens = nltk.word_tokenize(no_punctuation)
  filtered = [w for w in tokens if not w in stopwords.words('english')]
  stemmed = []
  for item in filtered:
    stemmed.append(stemmer.stem(item))
  return " ".join(stemmed)
desc['text_tokens'] = desc['text'].apply(get_tokens)
desc['text_tokens'][:5]
desc[['text','label','token']].to_csv('SKIP_GRAM_MODEL/all_grades_combined_data_rm_na_desc_with_token.csv')








def get_w2v_v2(data_path,batch_size,epochs,sent_len,EMB_DIM):
    data=pd.read_csv(data_path,names=['text','label'],header=0)
    print(f'input data shape {data.shape}')
    dat0=data.dropna()
    print(f'input data shape {data.shape} after dropping NA')
    accu_df=pd.DataFrame()       
    label_cnt=pd.DataFrame(dat0['label'].value_counts()).reset_index()
    label_cnt.columns=['label','count']
    least_label_list=list(label_cnt.loc[label_cnt['count']<3,'label'].values)
    # display(least_label_list)
    print(f'dropping {len(least_label_list)} labels')
    data=dat0[~dat0['label'].isin(least_label_list)].reset_index()
    print(f'after deleting least label{data.shape}')
    le=LabelEncoder()
    data['label_en']=le.fit_transform(data['label'])
    data['seq_id']=data.index
    data['token']=data['text'].apply(lambda x:gensim.utils.simple_preprocess(x))
    sent=data['token'].to_list()
    # model=Word2Vec(sentences=sent,size=EMB_DIM,window=5,min_count=1,sg=1,workers=4)
    marker=data_path.split('/')[-1].split('_')[0]+"_"+data_path.split('/')[-1].split('_')[1]
    model=Word2Vec.load('SKIP_GRAM_MODEL/w2v_models/skip_gram_w2c_{}.model'.format(marker))
    word_vectors=model.wv
    embedding_matrix=word_vectors.vectors
    print(f'vocabulary size={embedding_matrix.shape[0]}, each word is {1,embedding_matrix.shape[1]}')
    data['seq_id']=data.index
    # create a padded sequence for each document
    word2id={k:v.index for k,v in word_vectors.vocab.items()}

    emb_df=pd.DataFrame()
    for i, sent in enumerate(data['token']):
        text=data.loc[i,'text']
        label=data.loc[i,'label_en']
        seq_id=data.loc[i,'seq_id']
        if i%1000==0:
            print(i,sent)
        sent_seq=[]
        for j, word in enumerate(sent):
            w_id=word2id.get(word)
            sent_seq.append(w_id)
        df=pd.DataFrame({'text':[text],'seq_id':[seq_id],'word_seq':[sent_seq],'label':[label]})
        emb_df=emb_df.append(df)
        display(emb_df.head())
        emb_df.to_csv('SKIP_GRAM_MODEL/embeddings/{}_with_word_seq_index.csv'.format(marker),index=False)
        print(emb_df.dtypes)
        # proceed with NN model
        padded_sent=pad_sequences(emb_df['word_seq'].to_list(),maxlen=sent_len,padding='post')
        X=padded_sent
        Y=emb_df['label']
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,stratify=Y,test_size=0.2,random_state=111)
        x_train,x_dev,y_train,y_dev=train_test_split(X_train,Y_train,stratify=Y_train,test_size=0.1,random_state=111)
        Y_test,y_train,y_dev=to_categorical(Y_test),to_categorical(y_train),to_categorical(y_dev)
        print('shape:')
        print(f'Y_test shape{Y_test.shape},y_train shape{y_train.shape},y_dev shape{y_dev.shape}')
        class_count=emb_df.label.nunique()
        LOGDIR='./{}'.format(marker)

        vocab_length=len(embedding_matrix)
        print('--start neural network training and validation--')
        NN_model=Sequential([
                            Embedding(input_dim=vocab_length,output_dim=EMB_DIM,weights=[embedding_matrix],input_length=sent_len),
                            Flatten(),
                            Dense(256,activation = 'relu'),
                            Dense(class_count,activation='softmax')]) 
        NN_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        NN_model.fit(x_train,y_train,validation_data = (x_dev, y_dev),batch_size=batch_size,epochs=epochs,verbose=1,callbacks=[TensorBoard(LOGDIR)])
        NN_model.summary()
        NN_model.save('NN_Models/skip_gram_{}_NN'.format(marker))
        y_pred=NN_model.predict_classes(X_test)
        y_pred=to_categorical(y_pred)
        maxpos=lambda x:np.argmax(x)
        yTrueMax=np.array([maxpos(rec) for rec in Y_test])
        yPredMax=np.array([maxpos(rec) for rec in y_pred])
        yPredTop3=np.argsort(y_pred,axis=1)[:,-3]
        yPredTop2=np.argsort(y_pred,axis=1)[:,-2]
    
        top1accu=sum(yPredMax==yTrueMax)/len(yPredMax)
        top1accu=round(top1accu*100,2)
        top3accu=sum((yPredTop3==yTrueTop3)|(yPredTop2==yTrueTop2)|(yPredMax==yTrueMax))/len(top3_pred)
        top3accu=round(top3accu*100,2)
        print('test data TOP 1 accuracy {} %'.format(top1accu))#35.32 %
        print('test data TOP 3 accuracy {} %'.format(top3accu))#35.32 %
        print()
        accu=pd.DataFrame({'Model':[marker],'top1 accuracy':[top1accu],'top3 accuracy':[top3accu]})
        display(accu)
        accu_df.append(accu)
    accu_df.to_csv('SKIP_GRAM_MODEL/skip_gram_accuracy.csv',index=False)
    return data,model,emb_df,NN_model,X_test,Y_test,y_pred,accu


epochs=100
batch_size=128
sent_len=128
EMB_DIM=500
data_path='SKIP_GRAM_MODEL/FINAL/DESC_AG_all_grades.csv'
data,model,emb_df,NN_model,X_test,Y_test,y_pred=get_w2v_v2(data_path,batch_size,epochs,sent_len,EMB_DIM)


#---prior work and ML methods-----
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from math import log
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import gensim
import json
# Make unigram
stemmer = SnowballStemmer('english')
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def get_tokens(text):
  lowers = text.lower()
  no_punctuation = lowers.translate(remove_punctuation_map)
  tokens = nltk.word_tokenize(no_punctuation)
  filtered = [w for w in tokens if not w in stopwords.words('english')]
  stemmed = []
  for item in filtered:
    stemmed.append(stemmer.stem(item))
  return " ".join(stemmed)

def get_df(csv_path):
    df = pd.read_csv(dir+csv_path)
    df.text = df.text.astype(str)
    df['text'] = df['text'].apply(get_tokens)
    return df
def get_data(df, vector):
    if vector == 'tfidf':
        vectorizer = TfidfVectorizer()
    elif vector == 'bow':
        vectorizer = CountVectorizer()

    X_data = vectorizer.fit_transform(df.text)
    y_data = df['label']
    X_train_val, X_submit, y_train_val, y_submit = train_test_split(X_data, y_data, test_size=0.2, random_state=111)

    y_submit = y_submit.reset_index(drop=True)
    y_submit_df = pd.DataFrame(y_submit)

    X_train, X_test, y_train, y_test = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=111)

    y_test = y_test.reset_index(drop=True)
    y_test_df = pd.DataFrame(y_test)

    return X_train, y_train, X_test, y_test, y_test_df, X_submit, y_submit, y_submit_df
def get_model_result(classifier, X_train, y_train, X_test, y_test, X_submit, y_submit):
    model = classifier
    model.fit(X_train, y_train)
    print(model)
    return model

def top_n_accuracy(model, X_submit, y_submit, n=3):
    y_submit = list(y_submit)
    pred_proba = model.predict_proba(X_submit)
    best_n = np.argsort(pred_proba, axis=1)[:,-n:]
    top_class = model.classes_[best_n]

    successes = 0
    for i, ans in enumerate(y_submit):
        if ans in top_class[i]:
            successes += 1

    topNacc = float(successes)/len(y_submit)
    print('n:', n, ', acc:', topNacc, ', # of true:', successes)
    return topNacc

XGB = XGBClassifier()
RF = RandomForestClassifier()
LR = LogisticRegression()
SVM = SVC(probability=True)

csv_list = ['all_grades_combined_data_rm_na_desc.csv',
            'video_title_data_all_grades.csv',
            'problem_text_all_grades_v2.csv']

for csv in csv_list:
    print('---------------------')
    print(csv)
    df = get_df(csv)

    for vec in ['tfidf', 'bow']:
        print(vec)
        X_train, y_train, X_test, y_test, y_test_df, X_submit, y_submit, y_submit_df = get_data(df, vec)
        
        for classifier in [RF, XGB, LR, SVM]:
            model = get_model_result(classifier, X_train, y_train, X_test, y_test, X_submit, y_submit)

            print('acc_val')
            print(accuracy_score(y_test, model.predict(X_test), normalize=True))
            top1acc_val = top_n_accuracy(model, X_test, y_test, 1)
            top3acc_val = top_n_accuracy(model, X_test, y_test, 3)

            print('acc_test')
            print(accuracy_score(y_submit, model.predict(X_submit), normalize=True))
            top1acc_test = top_n_accuracy(model, X_submit, y_submit, 1)
            top3acc_test = top_n_accuracy(model, X_submit, y_submit, 3)