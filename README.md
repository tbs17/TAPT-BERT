# TAPT-BERT
Refer to the paper: Classifying Math Knowledge Components via Task-Adaptive Pre-Trained BERT, proceedings of AIED2021.

#### Preprocess Data
We have 3 types of data: CCSS description data, video title data and problem texts
+ You can find partial description and video title data in the 'Dataset/Pre-train' folder. 
+ The other part of the description and video title as well as the whole problem text data will remain private per provider's request.

You will need to pre-process the data to compile with the further pretraining script provided by Google. You can also refer to my preprocess script 'Code/data_preprocess.py' 
#### Create pretraining data and train for TAPT
To train a TAPT BERT, you will need to follow below steps. The scripts can be downloaded from  [Google bert repo](https://github.com/google-research/bert).
+ create pretraining data using create_pretraining_data.py
`!python bert/create_pretraining_data.py \
  --input_file=further-pre-training/CORPUS/G4-8/problem_text_g4-8_v2_for_post-train.txt \
  --output_file=further-pre-training/tmp/tf_skillCode_PROB_G4-8_bert.tfrecord \
  --vocab_file=further-pre-training/uncased_L-12_H-768_A-12/vocab.txt \
  --do_lower_case=True \
  --random_next_sentence=False \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5 \
  --short_seq_prob=0.1`

+ further pre-train using run_pretraining.py
+ predict off off TAPT model artifacts using run_classifier.py
+ The additional scripts needed are modeling.py, optimization.py, tokenization.py



#### Predict from TAPT
#### Compare to BASELINE
