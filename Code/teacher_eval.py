# ----below is to combine all the teacher validation answers together----
def combine_teacher_response(data_dir):
    import pandas as pd
    import os
    df=pd.DataFrame()
    for i, f in enumerate(os.listdir(data_dir)):
        print(i,f)
        
        
        marker=f.split('.')[0].split('_')[-1]
        data=pd.read_excel(os.path.join(data_dir,f))
        print(f'file has shape{data.shape}')
        data['teacher']=marker
        df=df.append(data)
    print(df.shape)
    df.head()
    df_valid=df.fillna(0)
    print(f'valid data has shape {df_valid.shape}')
    df_valid['label_winner']=df_valid[['a_score','b_score','c_score','d_score']].idxmax(axis=1)
    df_valid.to_csv('valid_teacher_val.csv',index=False)
    return df_valid

#--- below is to calculate percentage of teachers agree with the BERT predictions----
def cal_percent(texstr_dataPath,df_valid):
    tex=pd.read_csv(texstr_dataPath,usecols=[1,3,5,6,7,14,15,16])
    print(tex.shape)
    tex.rename(columns={'label_reformatted':'a','top1_reformatted':'b','top2_reformatted':'c','top3_reformatted':'d'},inplace=True)
    tex_valid=pd.merge(df_valid,tex,how='left',on=['text','a','b','c','d'])
    print(tex_valid.shape)
    col_list=['TEX_STR_top1', 'TEX_STR_top2',
       'TEX_STR_top3', 'label_scaled', 'top1_score_scaled',
       'top2_score_scaled', 'top3_score_scaled']
    for i, name in enumerate(col_list):
        print(i, name)
        above50=tex_valid[tex_valid[name]>0.5]
        above75=tex_valid[tex_valid[name]>0.75]
        above90=tex_valid[tex_valid[name]>0.9]
        print(f'there are {round(above50.shape[0]/tex_valid.shape[0]*100,2)} % for {name} score when threshold is 50')
        print(f'there are {round(above75.shape[0]/tex_valid.shape[0]*100,2)} % for {name} score when threshold is 75')
        print(f'there are {round(above90.shape[0]/tex_valid.shape[0]*100,2)} % for {name} score when threshold is 90')
        
        print('-----')
    
texstr_dataPath='miss_prediction_4eval_ethan_v2_with_texstr.csv'
cal_percent(texstr_dataPath,df_valid)


# ----below is to calculate Kappa score for multiple raters----


def cal_kappa(df_valid):

    k_table=pd.pivot_table(df_valid[['text','teacher','label_winner']],index=['text'],columns=['label_winner'],values=['teacher'],aggfunc=pd.Series.nunique,fill_value=0).reset_index()
    k_table.columns=k_table.columns.get_level_values(1)
    k_table.columns=['text', 'a_score', 'b_score', 'c_score', 'd_score']
    k_table['sum']=k_table.sum(axis=1)
    k_table['p_i']=(k_table.iloc[:,1:5].apply(lambda x:x**2).sum(axis=1)-k_table['sum'])/n/(n-1)

    p_j=k_table.iloc[:,1:5].sum(axis=0)/(N*n)


    p_bar=sum(k_table['p_i'])/N
    p_bar_e=sum(p_j[:k].apply(lambda x:x**2))
    kappa=(p_bar-p_bar_e)/(1-p_bar_e)
    print(f'p_bar is {p_bar}\np_bar_e is {p_bar_e}\nmulti-rater kappa is {k}')

    return kappa

n=8
N=29
k=4
kappa=cal_kappa(df_valid)