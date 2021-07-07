def convert_test_result(orig_testPath,pred_testPath,org_dataPath,out_dir):
    from pathlib import Path
    import pandas as pd
    #read the original test data for the text and id
    df_test = pd.read_csv(orig_testPath, sep='\t',engine='python')
    df_test['guid']=df_test['guid'].astype(str)
    print(f'original test file has shape {df_test.shape}')
    #read the results data for the probabilities
    df_result = pd.read_csv(pred_testPath, sep='\t', header=None)
    print(f'predicted test file has shape {df_result.shape}')
    out_dir=Path(out_dir)
    Path.mkdir(out_dir,exist_ok=True)
    import numpy as np
    # df_map
    df_map_result = pd.DataFrame({'guid': df_test['guid'],
        'text': df_test['text'],
        'top1': df_result.idxmax(axis=1),
        'top1_probability':df_result.max(axis=1),
        'top2': df_result.columns[df_result.values.argsort(1)[:,-2]],
        'top2_probability':df_result.apply(lambda x: sorted(x)[-2],axis=1),
        'top3': df_result.columns[df_result.values.argsort(1)[:,-3]],
        'top3_probability':df_result.apply(lambda x: sorted(x)[-3],axis=1),
        'top4': df_result.columns[df_result.values.argsort(1)[:,-4]],
        'top4_probability':df_result.apply(lambda x: sorted(x)[-4],axis=1),
        'top5': df_result.columns[df_result.values.argsort(1)[:,-5]],
        'top5_probability':df_result.apply(lambda x: sorted(x)[-5],axis=1)
        })
    #view sample rows of the newly created dataframe
    df_map_result.head(10)
    df_map_result['top1']=df_map_result['top1'].astype(str)
    df_map_result['top2']=df_map_result['top2'].astype(str)
    df_map_result['top3']=df_map_result['top3'].astype(str)
    df_map_result.dtypes
    df_map_result['top4']=df_map_result['top4'].astype(str)
    df_map_result['top5']=df_map_result['top5'].astype(str)
    print(f'mapped test file has shape {df_map_result.shape}')
    data=pd.read_csv(org_dataPath,encoding='utf-8',names=['text','label'],header=0)
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    data['label_en']=le.fit_transform(data['label'])
    # data0=data.drop('Label',axis=1)
    data['Label']=le.inverse_transform(data['label_en'])
    key=pd.DataFrame({'code':data['label_en'].unique(),'Label':le.inverse_transform(data['label_en'].unique())})
    key.to_csv('further-pre-training/label-map.csv',index=False)

    key['code']=key.code.astype(str)
    label_map_dict=dict(key.to_dict(orient='split')['data'])
    marker=pred_testPath.split('/')[2].split('.')[0]
    df_map_result=df_map_result.replace({'top1':label_map_dict,'top2':label_map_dict,'top3':label_map_dict,'top4':label_map_dict,'top5':label_map_dict})
    df_map_result.to_csv('{}/{}_converted.csv'.format(out_dir,marker),index=False)
    print(df_map_result.shape)#(702, 12)
    return df_map_result

def match_top3_label_match(data_dir,label_Path,out_dir):
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import os
    label_data=pd.read_csv(label_Path,names=['label'],header=0)
    print(f'test data shape{label_data.shape}')
    for file in os.listdir(data_dir):
    
        pred_dataPath=os.path.join(data_dir,file)
        print(pred_dataPath)
        print('----')
        if pred_dataPath.endswith('.csv'):
        
            pred_data=pd.read_csv(pred_dataPath,encoding='ISO-8859-1')
            print(f'predicted data shape{pred_data.shape}')
            print('total {} classes are predicted as top class'.format(len(pred_data['top1'].unique())))
            match=pd.concat([pred_data,label_data],axis=1)
            print(f'merged data shape{match.shape}')
            correct_top1=match[match['top1']==match['label']].top1.unique()
            correct_top2=match[match['top2']==match['label']].top2.unique()
            correct_top3=match[match['top3']==match['label']].top3.unique()
            correct_all=list(set(list(correct_top1)+list(correct_top2)+list(correct_top3)))
            print('Correct top 1 label {}'.format(len(correct_top1)))
            print('Correct top 2 label {}'.format(len(correct_top2)))
            print('Correct top 3 label {}'.format(len(correct_top3)))
            print('Total top 3 Correct labels {}'.format(len(correct_all)))
            match['matched1']=np.where(match['top1']==match['label'],1,0)
            match['matched2']=np.where((match['top1']==match['label']) | (match['top2']==match['label']),1,0)
            match['matched3']=np.where((match['top1']==match['label']) | (match['top2']==match['label']) | (match['top3']==match['label']),1,0)
       
            marker=file.split('.')[0]
            from pathlib import Path
            out_dir=Path(out_dir)
            out_dir.mkdir(exist_ok=True)
            match.to_csv('{}/{}_matched.csv'.format(out_dir,marker),index=False)
            top1_correct=round(match['matched1'].sum()/len(match)*100,2)
            top2_correct=round(match['matched2'].sum()/len(match)*100,2)
            top3_correct=round(match['matched3'].sum()/len(match)*100,2)
            print('predicted data size:{}\noriginal data size:{}\n{} rows of text get compared'.format(pred_data.shape,label_data.shape,match.shape[0]))
            print('Top1 label accuracy is {} %'.format(top1_correct))
            print('Top2 label accuracy is {} %'.format(top2_correct))
            print('Top3 label accuracy is {} %'.format(top3_correct))
            
            print('-----')