Updating
##   A Prompt Learning Framework with Large Language Model Augmentation for Few-shot Multi-label Intent Detection

Pytorch implementation of  **PLMA**: A Prompt Learning Framework with Large Language Model Augmentation for Few-shot Multi-label Intent Detection.



## Insight

Intent detection (ID) is essential in spoken language understanding, especially in multi-label settings where intent labels are interdependent and diverse. Existing methods like SE-MLP and QA-FT struggle in few-shot settings, due to limited data availability and efficiency concerns. To address this, we introduce a **P**rompt **L**earning framework with large language **M**odel **A**ugmentation (PLMA) for few-shot multi-label ID.



## Model Architecture

![Overview of the PLMA model architecture](https://github.com/user-attachments/assets/7b029b8a-1503-4589-a9d8-9fdd7fc3e91c)


As Fig. 1, we adapt prompt learning as the core framework for SLMs, utilizing LLMs for intent spans extraction and answer space expansion. The LLM-enhanced information is then integrated with the original inputs into the prompt learning model.



## Dependencies

We test on `Python 3.8+`  and `Pytorch 2.4.1+`

Other dependencies please refer to `requirements.txt`



## Training & evaluation

**Indomain Setting**

```shell
run run_ls_soft.sh
```

or

```shell
python nlutrain_ls_soft.py --datapath {the path of dataset}\
						--dataset {datasetname}\
						--kfold {k-fold cross-validation}\
						--template_choice {the choice of template}\
						--verbalizer_choice {the choice of verbalizer}\
						--epoch {num_epoches}\
						--warmup_step {the steps of warmup}\
						--batch_size {batch_size}\
						--lr {learning rate}\
						--seed {random seed}
```

**Cross-domain Setting**

```shell
run run_ls_soft_cd.sh
```

or

```shell
python nlutrain_ls_soft_cd.py --datapath {the path of dataset}\
							--dataset1 {source datasetname}\
							--dataset2 {target datasetname}\
							--template_choice {the choice of template}\
							--verbalizer_choice {the choice of verbalizer}\
							--epoch {num_epoches}\
							--warmup_step {the steps of warmup}\
							--batch_size {batch_size}\
							--lr {learning rate}\
							--seed {random seed}
```



## Results

![image](https://github.com/user-attachments/assets/daea9ecb-3cc7-47a8-9f8b-5dba7ae8975e)


![image](https://github.com/user-attachments/assets/b8417e98-7428-462f-b348-daca158e3429)


## References

[Huggingface Transformers](https://github.com/huggingface/transformers)

[Openprompt](https://github.com/thunlp/OpenPrompt)
