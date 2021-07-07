
# # ----below is to calculate doc2vec similarity score-----

def train_doc2vec(full_path,marker):
    from gensim.models.doc2vec import Doc2Vec,TaggedDocument
    def read_corpus(fname,tokens_only=False):
        import smart_open
        import gensim
        with smart_open.open(fname,encoding='iso-8859-1') as f:
            for i, line in enumerate(f):
                tokens=gensim.utils.simple_preprocess(line)
                if tokens_only:
                    yield tokens
                else:
                    yield gensim.models.doc2vec.TaggedDocument(tokens,[i])

    full_data=list(read_corpus(full_path))
    model1=Doc2Vec(vector_size=385,window=2,min_count=2,workers=4,epochs=100)
    model1.build_vocab(full_data)
    model1.train(full_data,total_examples=model1.corpus_count,epochs=model1.epochs)
    model1.corpus_count
    model1.save('d2v_k12_{}.model'.fromat(marker))

# because some of the ccss code has 5 parts, we align them into 4 component style via below code.
def cc_label_reformat(data_path,label_name):
    def reformat(data,col_list):
    for col in col_list:
        new=data[col].str.split('.',expand=True)
        data[col+'_reformatted']=new[0]+'.'+new[1]+'.'+new[3]
    return data
    cc_desc=pd.read_csv(data_path,encoding='iso-8859-1')
    print(cc_desc.shape)
    cc_desc=cc_desc.reset_index()
    cc_desc=reformat(cc_desc,[label_name])
    cc_desc.rename(columns={label_name+'_reformatted':'label_reformatted'},inplace=True)
    return cc_desc




def merge_cc(cc_path,miss_pred_path,marker):
    cc_desc=pd.read_csv(cc_path,encoding='iso-8859-1')
    miss_pred1=pd.read_csv(miss_pred_path)
    print(f'miss predicted cases are {miss_pred1.shape}')#(34905, 24)
    miss_pred_dedup=miss_pred1.drop_duplicates()
    print(f'miss predicted cases w/o duplicates are {miss_pred_dedup.shape}')#(9672, 24)
    missPredicted_merge=pd.merge(cc_desc,miss_pred1,how='inner',on='label_reformatted')
    print(f'miss predicted merged cases with duplicates are {missPredicted_merge.shape}')#(7909, 26),#(3187, 26) if duplicates first
    missPredicted_merge_dedup=missPredicted_merge.drop_duplicates().reset_index()
    print(f'miss predicted merged cases w/o duplicates are{missPredicted_merge_dedup.shape}')#(3187, 26) same as duplicating first
    missPredicted_merge.to_csv('TOTAL_miss_predicted_reformatted_{}.csv'.format(marker),index=False)
    
    missPredicted_merge_dedup['seq_id']=missPredicted_merge_dedup.index
    missPredicted_merge_dedup.drop('level_0',axis=1,inplace=True)
    missPredicted_merge_dedup.to_csv('TOTAL_miss_predicted_reformatted_dedup_{}.csv'.format(marker),index=False)
    display(missPredicted_merge.head())
    return missPredicted_merge_dedup

cc_path='gt_k12labels_with_desc_processed.csv'
miss_pred_path='TOTAL_missed_predictions_paired.csv'
marker='k12_v3'
missPredicted_merge_dedup_k12=merge_cc(cc_path,miss_pred_path,marker)

def find_sim(data,cc_data,col_dict,model_name,marker):
    
    labels=[]
    top1_list=[]
    top2_list=[]
    top3_list=[]
    sim1_list=[]
    sim2_list=[]
    sim3_list=[]
    sim4_list=[]
    FOUND=[]
    sim_df=pd.DataFrame(columns=['index','label_reformatted','top1','sim1','top2','sim2','top3','sim3'])

    for i, value in enumerate(data['index']):
        if i%1000==0:print(f'---{i}:{value}---')
        cc_code=data.iloc[i][col_dict['label']]
        top1=data.iloc[i][col_dict['top1']]
        top2=data.iloc[i][col_dict['top2']]
        top3=data.iloc[i][col_dict['top3']]

        inferred_vector=model_name.infer_vector(full_data[value].words)
        sims=model_name.docvecs.most_similar([inferred_vector],topn=len(model_name.docvecs))

        for j in range(len(model_name.docvecs)):
            if cc_data['label_reformatted'][sims[j][0]]==top1:
                sim1=sims[j][1]
                top1_dat=pd.DataFrame({'index':[i],'label_reformatted':[cc_code],'top1':[top1],'sim1':[sim1]})
                sim_df=sim_df.append(top1_dat)
            elif cc_data['label_reformatted'][sims[j][0]]==top2:
                sim2=sims[j][1]    
                top2_dat=pd.DataFrame({'index':[i],'label_reformatted':[cc_code],'top2':[top2],'sim2':[sim2]})
                sim_df=sim_df.append(top2_dat)
            elif cc_data['label_reformatted'][sims[j][0]]==top3:
                sim3=sims[j][1]  
                top3_dat=pd.DataFrame({'index':[i],'label_reformatted':[cc_code],'top3':[top3],'sim3':[sim3]})
    #             display(top3_dat)
                sim_df=sim_df.append(top3_dat)
            else:
                continue


    display(sim_df.head())
    sim_df=sim_df.fillna(0)
    sim_match=pd.pivot_table(sim_df,index=['index'],values=['sim1','sim2','sim3'],aggfunc=np.sum).reset_index()
    sim_match.rename(columns={'index':'seq_id'},inplace=True)
    print(f'after grouping by unique text(seq_id),there are {sim_match.shape[0]} rows found a similarity')
    sim_match.to_csv('MISS_PREDICTED_DEDUP/doc2vec_sim_match_{}.csv'.format(marker),index=False)
    display(sim_match.head())
    
    sim_merge=pd.merge(data,sim_match,how='left',on='seq_id')
    print(f'after merging, shape {sim_merge.shape}')
    sim_merge.rename(columns={'index':'corpus_index'},inplace=True)
    sim_merge.to_csv('MISS_PREDICTED_DEDUP/TOTAL_missed_predictions_doc2vec_{}.csv'.format(marker),index=False)
    sim_merge_n=sim_merge.drop(['label_top1_reformatted', 'label_top2_reformatted',
           'label_top3_reformatted'],axis=1)
    sim_merge_n.to_csv('MISS_PREDICTED_DEDUP/TOTAL_missed_predictions_doc2vec_narrow_{}.csv'.format(marker),index=False)
    sim_merge_n_dedup=sim_merge_n.drop_duplicates(['label_reformatted','text', 'data',
           'top1_reformatted', 'top2_reformatted', 'top3_reformatted']).reset_index()
    print(f'shape {sim_merge_n.shape},shape after {sim_merge_n_dedup.shape}')
    sim_merge_n_dedup['seq_id']=sim_merge_n_dedup.index
    sim_merge_n_dedup.to_csv('MISS_PREDICTED_DEDUP/TOTAL_missed_predictions_doc2vec_narrow_{}_dedup.csv'.format(marker),index=False)
    return sim_df,sim_match,sim_merge_n,sim_merge_n_dedup

col_dict={'label':'label_reformatted','top1':'top1_reformatted','top2':'top2_reformatted','top3':'top3_reformatted'}
data=missPredicted_merge_dedup_k12
cc_data=k12_processed
model_name=model1
marker='k12_best'
sim_df,sim_match,sim_merge_n,sim_merge_n_dedup=find_sim(data,cc_data,col_dict,model_name,marker)


# ----below is to calculate node2vec similarity score-----
def create_n2v_model(cc_data_path,start_node,target_node,len_walk,walk_num,dim,marker):
    from node2vec import Node2Vec
    import networkx as nx
    cc_code_relationship=pd.read_csv(cc_data_path)
    print(f'there are {cc_code_relationship.shape} unique relationships')
    cc_code_relationship_valid=cc_code_relationship[cc_code_relationship['Relation']!='related']
    print(f'there are {cc_code_relationship_valid.shape} unique valid relationships')
    cc_relation_graph=nx.from_pandas_edgelist(cc_code_relationship_valid,source=start_node,target=target_node,create_using=nx.DiGraph)

    cc2_node2vec=Node2Vec(cc_relation_graph,dimensions=dim,walk_length=len_walk,num_walks=walk_num,workers=4)
    model=cc2_node2vec.fit(window=100,min_count=1)
    model.save('n2v_{}.model'.format(marker))

    node_merge=pd.concat([cc_code_relationship_valid[start_node],cc_code_relationship_valid[target_node]])
    print(f'shape after merging both nodes:{node_merge.shape} with duplicates')
    node_dedup=node_merge.drop_duplicates()
    print(f'shape after merging both nodes:{node_dedup.shape} w/o duplicates')
    print(f'It means there are {node_dedup.shape[0]-1} neighbor nodes for each node in the dataset')
    return cc_relation_graph,cc2_node2vec,model,node_merge,node_dedup

cc_data_path='cc_code_relationship.csv'
start_node='Node A'
target_node='Node B'
len_walk=100
walk_num=100
dim=573
marker='k12_v3'
cc_relation_graph,cc2_node2vec,model,node_merge,node_dedup=create_n2v_model(cc_data_path,start_node,target_node,len_walk,walk_num,dim,marker)

def find_node_sim(data_path,node_name,model_name,marker,top_n):
    sim_df=pd.DataFrame(columns=['seq_id','model','label_reformatted','top1','n2v_sim1','top2','n2v_sim2','top3','n2v_sim3'])
    data=pd.read_csv(data_path)
    print(f'input data has shape {data.shape}')
    for i, value in enumerate(data[node_name]):
        label=data.iloc[i]['label_reformatted']
        dataset=data.iloc[i]['data']
        top1=data.iloc[i]['top1_reformatted']
        top2=data.iloc[i]['top2_reformatted']
        top3=data.iloc[i]['top3_reformatted']

        if value in model_name.wv.vocab:
            for node, sim in model_name.most_similar(value,topn=top_n):
                if node==top1:
                    df=pd.DataFrame({'seq_id':[i],'model':[dataset],'label_reformatted':[label],'top1':[node],'n2v_sim1':[sim]})
                    sim_df=sim_df.append(df)
                elif node==top2:
                    df=pd.DataFrame({'seq_id':[i],'model':[dataset],'label_reformatted':[label],'top2':[node],'n2v_sim2':[sim]})
                    sim_df=sim_df.append(df)
                elif node==top3:
                    df=pd.DataFrame({'seq_id':[i],'model':[dataset],'label_reformatted':[label],'top3':[node],'n2v_sim3':[sim]})
                    sim_df=sim_df.append(df)
                else:
                    continue
    
    display(sim_df.head())
    sim_df=sim_df.fillna(0)
    sim_match=pd.pivot_table(sim_df,index=['seq_id'],values=['n2v_sim1','n2v_sim2','n2v_sim3'],aggfunc=np.sum).reset_index()
    print(f'after grouping by unique seq_id,there are {sim_match.shape[0]} rows found similarity')
    sim_merge=pd.merge(data,sim_match,how='left',on='seq_id')
    print(f'after merging, shape {sim_merge.shape}')
    sim_match.to_csv('Total_missPredicted_node2vec_dist_match_{}.csv'.format(marker),index=False)
    display(sim_merge.head())
    return sim_match,sim_merge

from gensim.models.doc2vec import Doc2Vec
model=Doc2Vec.load('n2v_k12_v2.model')
data_path='MISS_PREDICTED_DEDUP/TOTAL_missed_predictions_doc2vec_narrow_k12_best_dedup.csv'
node_name='label_reformatted'
model_name=model
marker='k12_best'
top_n=368
sim_n2v_match,sim_n2v_merge=find_node_sim(data_path,node_name,model_name,marker,top_n)

# ----get data with prerequisite relationship ---
def evaluate_prereq(data,marker):
    non_pre=data[(data['label_reformatted'].str.split('.',expand=True)[0]<data['top1_reformatted'].str.split('.',expand=True)[0]) & (data['label_reformatted'].str.split('.',expand=True)[0]<data['top2_reformatted'].str.split('.',expand=True)[0]) \
                 &(data['label_reformatted'].str.split('.',expand=True)[0]<data['top3_reformatted'].str.split('.',expand=True)[0])]
    print(f'there are {non_pre.shape[0]} out of {data.shape[0]} rows that are not a rerequisite of the label')
    non_pre.to_csv('sim_of_non_prerequisite_label_{}.csv'.format(marker),index=False)
    pre=data[(data['label_reformatted'].str.split('.',expand=True)[0]>=data['top1_reformatted'].str.split('.',expand=True)[0]) | (data['label_reformatted'].str.split('.',expand=True)[0]>=data['top2_reformatted'].str.split('.',expand=True)[0]) \
                 |(data['label_reformatted'].str.split('.',expand=True)[0]>=data['top3_reformatted'].str.split('.',expand=True)[0])]
    print(f'there are {pre.shape[0]} out of {data.shape[0]}: {round(pre.shape[0]/data.shape[0]*100,2)}% rows that are a rerequisite of the label')
    pre=pre.reset_index()
    pre['seq_id']=pre.index
    pre.to_csv('sim_of_prerequisite_label_{}.csv'.format(marker),index=False)
    
    return non_pre,pre

def cal_prereq_case(data):
    top1_df=pd.DataFrame()
    top2_df=pd.DataFrame()
    top3_df=pd.DataFrame()
    for i,f in enumerate(data['seq_id']):
        data_name=data.iloc[i]['data']
        label=data.iloc[i]['label_reformatted']
        top1=data.iloc[i]['top1_reformatted']
        top2=data.iloc[i]['top2_reformatted']
        top3=data.iloc[i]['top3_reformatted']
#         print(i,label)
        if label.split('.')[0]>=top1.split('.')[0]:
            sim=data.iloc[i]['sim1']
            top1_dat=pd.DataFrame({'seq_id':[i],'data':[data_name],'label':[label],'pred_label':[top1],'similarity':[sim]})
            top1_df=top1_df.append(top1_dat)
        elif label.split('.')[0]>=top2.split('.')[0]:
            sim=data.iloc[i]['sim2']
            top2_dat=pd.DataFrame({'seq_id':[i],'data':[data_name],'label':[label],'pred_label':[top2],'similarity':[sim]})
            top2_df=top2_df.append(top2_dat)
        elif label.split('.')[0]>=top3.split('.')[0]:
            sim=data.iloc[i]['sim3']
            top3_dat=pd.DataFrame({'seq_id':[i],'data':[data_name],'label':[label],'pred_label':[top3],'similarity':[sim]})
            top3_df=top3_df.append(top3_dat)
        else:
            continue
    print(f'there are {top1_df.shape[0]} rows:{round(top1_df.shape[0]/data.shape[0]*100,2)} % or prerequisite codes from top1 prediction\n'
         f'there are {top2_df.shape[0]} rows:{round(top2_df.shape[0]/data.shape[0]*100,2)} % or prerequisite codes from top2 prediction\n'
         f'there are {top3_df.shape[0]} rows:{round(top3_df.shape[0]/data.shape[0]*100,2)} % or prerequisite codes from top3 prediction\n')
    return top1_df,top2_df,top3_df


# create texstr metric
def cal_texstr(data,marker,a,b):

    data['TEX_STR_top1']=a*data['sim1']+b*data['n2v_sim1']
    data['TEX_STR_top2']=a*data['sim2']+b*data['n2v_sim2']
    data['TEX_STR_top3']=a*data['sim3']+b*data['n2v_sim3']
    print(f'shape after calculating texstr{data.shape}')
    data.to_csv('Total_missed_predicted_tex_str_metric_{}.csv'.format(marker),index=False)
    return data

data=sim_n2v_merge
marker='k12_best'
a=b=0.5
sim_texstr_best=cal_texstr(data,marker,a,b)
data=sim_n2v_merge_prereq
marker='k12_best'
a=b=0.5
sim_texstr_prereq_best=cal_texstr(data,marker,a,b)

# below is to test texstr's effectiveness

def effect_table(data):
    df=pd.DataFrame()
    for name in data.data.unique():
        dat=data[data.data==name]
        print(f'--dataset {name} has shape {dat.shape}--')
        above_50=dat[(dat['sim1']>0.5) |(dat['sim2']>0.5) |(dat['sim3']>0.5)]
        above_75=dat[(dat['sim1']>0.75) |(dat['sim2']>0.75) |(dat['sim3']>0.75)]
        above_90=dat[(dat['sim1']>0.90) |(dat['sim2']>0.90) |(dat['sim3']>0.90)]
        table=pd.DataFrame({'Model':[name],'Miss Predicted#':[dat.shape[0]],'d2v_similar>50%':[round(above_50.shape[0]/dat.shape[0]*100,2)],
                           'd2v_similar>75 %':[round(above_75.shape[0]/dat.shape[0]*100,2)],'d2v_similar>90 %':[round(above_90.shape[0]/dat.shape[0]*100,2)]})
        
        df=df.append(table)
    display(df)
    return df
def effect_table_prereq(data):
    df=pd.DataFrame()
    for name in data.data.unique():
        dat=data[data.data==name]
        print(f'--dataset {name} has shape {dat.shape}--')
        above_50=dat[(dat['similarity']>0.5)]
        above_75=dat[(dat['similarity']>0.75)]
        above_90=dat[(dat['similarity']>0.90)]
        table=pd.DataFrame({'Model':[name],'Miss Predicted(~prereq)#':[dat.shape[0]],'Similar>50%':[round(above_50.shape[0]/dat.shape[0]*100,2)],
                           'Similar>75 %':[round(above_75.shape[0]/dat.shape[0]*100,2)],'Similar>90 %':[round(above_90.shape[0]/dat.shape[0]*100,2)]})
        
        df=df.append(table)
    display(df)
    return df

def effect_n2v_table(data):
    df=pd.DataFrame()
    for name in data.data.unique():
        dat=data[data.data==name]
        print(f'--dataset {name} has shape {dat.shape}--')
        above_50=dat[(dat['n2v_sim1']>0.5) |(dat['n2v_sim2']>0.5) |(dat['n2v_sim3']>0.5)]
        above_75=dat[(dat['n2v_sim1']>0.75) |(dat['n2v_sim2']>0.75) |(dat['n2v_sim3']>0.75)]
        above_90=dat[(dat['n2v_sim1']>0.90) |(dat['n2v_sim2']>0.90) |(dat['n2v_sim3']>0.90)]
        table=pd.DataFrame({'Model':[name],'Miss Predicted#':[dat.shape[0]],'n2v_similar>50%':[round(above_50.shape[0]/dat.shape[0]*100,2)],
                           'n2v_similar>75 %':[round(above_75.shape[0]/dat.shape[0]*100,2)],'n2v_similar>90 %':[round(above_90.shape[0]/dat.shape[0]*100,2)]})
        
        df=df.append(table)
    display(df)
    return df

def effect_texstr_table(data):
    df=pd.DataFrame()
    for name in data.data.unique():
        dat=data[data.data==name]
        print(f'--dataset {name} has shape {dat.shape}--')
        above_50=dat[(dat['TEX_STR_top1']>0.5) |(dat['TEX_STR_top2']>0.5) |(dat['TEX_STR_top3']>0.5)]
        above_75=dat[(dat['TEX_STR_top1']>0.75) |(dat['TEX_STR_top2']>0.75) |(dat['TEX_STR_top3']>0.75)]
        above_90=dat[(dat['TEX_STR_top1']>0.90) |(dat['TEX_STR_top2']>0.90) |(dat['TEX_STR_top3']>0.90)]
        table=pd.DataFrame({'Model':[name],'Miss Predicted#':[dat.shape[0]],'TEX_STR_similar>50%':[round(above_50.shape[0]/dat.shape[0]*100,2)],
                           'TEX_STR_similar>75 %':[round(above_75.shape[0]/dat.shape[0]*100,2)],'TEX_STR_similar>90 %':[round(above_90.shape[0]/dat.shape[0]*100,2)]})
        
        df=df.append(table)
    display(df)
    return df

d2v_table_k12=effect_table(sim_texstr_best)
n2v_table_k12=effect_n2v_table(sim_texstr_best)
texstr_table_k12=effect_texstr_table(sim_texstr_best)