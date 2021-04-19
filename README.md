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

```
python bert/create_pretraining_data.py \
  --input_file=your_data.txt \
  --output_file=target_tf_data.tfrecord \
  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
  --do_lower_case=True \
  --random_next_sentence=False \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5 \
  --short_seq_prob=0.1
```

+ further pre-train using run_pretraining.py

```
python bert/run_pretraining.py \
  --input_file=$target_tf_data.tfrecord \
  --output_dir=$TAPT_DIR \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
  --bert_hub_module_handle=$BERT_MODEL_HUB \
  --spm_model_file="from_tf_hub" \
  --train_batch_size=32 \
  --eval_batch_size=16 \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --num_warmup_steps=5000 \
  --save_checkpoints_steps=50000 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=$TPU_ADDRESS 

```
+ predict off of TAPT model artifacts using run_classifier.py

```
python bert/run_classifier.py \
  --data_dir=your_data_dir \
  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
  --task_name=$TASK \
  --output_dir=$OUTPUT_DIR \
  --init_checkpoint='$TAPT_DIR/model.ckpt-1000000' \
  --do_lower_case=True \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --max_seq_length=512 \
  --warmup_step=200 \
  --learning_rate=2e-5 \
  --num_train_epochs=25 \
  --save_checkpoints_steps=3000 \
  --train_batch_size=32 \
  --eval_batch_size=16 \
  --predict_batch_size=16 \
  --tpu_name=$TPU_ADDRESS \
  --use_tpu=True

```
+ The additional scripts needed are modeling.py, optimization.py, tokenization.py
+ calculate the accuracy of the predicted tesults
First, we convert the test.tsv file from TAPT fine-tuning and then we evaluate its accuracy. Please see details in the script 'Code/convert_evaluate_test.py'.
#### Compare to BASELINE

The baseline performance can be generated using the code in 'Code/baseline.py'

#### Create TEXSTR metric
TEXSTR metric \Lamda=\alpha*C_{t}+(1-\alpha)*C_{s}

C_{t} is semantic similarity calculated via doc2vec algorithm, C_{s} is calculated via node2vec algorithm. The detail script is located in 'Code/texstr.py' 

#### Evaluate TEXSTR effectiveness

We recruited 10 teachers to evaluate 29 prediction results that TEXSTR would like to reconsider as correct, we have 8 teachers' valid responses back. We compare teachers' rating (normalized to be 0-1 from a scale of 1 to 5) to TEXSTR score as well as calculating the multi-rater agreement via kappa value. Please see details at 'Code/teacher_eval.py'
