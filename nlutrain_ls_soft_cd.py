from nluprocessor import NLUProcessor,NLUProcessorCD
from weightedverbalizer import WeightedVerbalizer2
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import DataLoader
import random
import re
from sklearn.metrics import f1_score
import numpy as np  
import argparse
import logging
import os

this_run_unicode = str(random.randint(0, 1e10))   


parser = argparse.ArgumentParser(description="datasets and model parameters")
parser.add_argument('--datapath', type = str, help = 'the path of dataset', default = '/Work21/2023/zhuangning/code/prompt-gpt/data/cd_')
parser.add_argument('--dataset1', type = str, help = 'dataset1 name: banking, hotel',default = 'hotels')
parser.add_argument('--dataset2', type = str, help = 'dataset1 name: banking, hotel',default = 'banking')
parser.add_argument('--maxlength', type = int, help = 'the max length of text', default = 256)
parser.add_argument('--threshold', type = int, help = 'the threshold of the result of sigmoid', default = 0.1)
parser.add_argument('--batch_size', type = int, help = 'the batchsize of training and testing', default = 8)
parser.add_argument('--warmup_step', type = int, help = 'the num of warm up steps', default = 1000)
parser.add_argument('--epoch', type = int, help = 'the epoches of training', default = 50)
parser.add_argument('--lr',type = float, help='optimizer learning rate', default= 1e-5)
parser.add_argument('--seed',type=int, help='random seed: 123, 42', default=123)
parser.add_argument('--cuda', type=str, help='the gpu number to use', default ='2')
# parser.add_argument('--model', type = str, help = 'the model name: bert, roberta, gpt2', default = 'bert')
parser.add_argument('--kfold', type = int, help = 'k-fold set: 10, 20', default = 10)
parser.add_argument('--verbalizer_choice', type = int, help = 'choice of verbalizer', default = 6)
parser.add_argument('--template_choice', type = int, help = 'choice of template', default = 3)
 
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

data_path = args.datapath
task_name1 = args.dataset1
task_name2 = args.dataset2
max_length = args.maxlength
threshold = args.threshold
warm_up_step = args.warmup_step
epoches = args.epoch
learning_rate = args.lr
seed = args.seed
kfold = args.kfold
batchsize = args.batch_size
verbalizer_choice = args.verbalizer_choice
template_choice = args.template_choice
data_dir1 = data_path + task_name1 + '.csv'
data_dir2 = data_path + task_name2 + '.csv'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(seed)


def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print(f"---  new folder...  ---")
		print(f"---  OK  ---")
 
	else:
		print(f"---  There is this folder!  ---")


file = '/Work21/2023/zhuangning/code/prompt-gpt/results_ls_soft2_cd/' + task_name1 + "to" + task_name2 + "/" + this_run_unicode
mkdir(file)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# s_handler = logging.StreamHandler()
# s_handler.setLevel(logging.INFO)
f_handler = logging.FileHandler(filename=file+'/logger.log')
f_handler.setLevel(logging.INFO)
# logger.addHandler(s_handler)
logger.addHandler(f_handler)

logger.info(f'datapath:{data_path}, dataset:{task_name1 + " to " + task_name2},kfold:{kfold}')
logger.info(f'seed:{seed}, max_length:{max_length}, warmup_steps:{warm_up_step}, training_epoches:{epoches}, learning_rate:{learning_rate},threshod:{threshold},verbalizer_choice:{verbalizer_choice},template_choice:{template_choice}')

all_labels = ['affirm', 'deny', 'dont_know', 'acknowledge', 'greet', 'end_call', 'handoff', 'thank', \
                'repeat', 'cancel_close_leave_freeze', 'change', 'make_open_apply_setup_get_activate', 'request_info',\
                 'how', 'why', 'when', 'how_much', 'how_long', 'wrong_notworking_notshowing', 'lost_stolen', \
                'more_higher_after', 'less_lower_before', 'new', 'existing', 'limits']

processor = NLUProcessorCD()
# data_dir = '/Work21/2023/zhuangning/code/prompt-gpt/data/banking.csv'
train_examples = processor.get_examples(data_dir1, 'train')
test_examples = processor.get_examples(data_dir2, 'test')
# # 打印经过processor后的示例数据
print(len(train_examples))
print(len(test_examples))

# 加载预训练模型
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-large")

# 定义模板
# template = ManualTemplate(tokenizer=tokenizer, text='{"text":"For the following query"}{"placeholder":"text_b"}{"text":"are spans that represent intents of the user, query:"}{"placeholder":"text_a"}{"text":"intents: "}{"mask"}.')
template = ManualTemplate(tokenizer=tokenizer).from_file("/Work21/2023/zhuangning/code/prompt-gpt/scripts/manual_template.txt", choice=template_choice)

# 定义Verbalizer
verbalizer = SoftVerbalizer(tokenizer, plm, classes = all_labels, multi_token_handler='max').from_file("/Work21/2023/zhuangning/code/prompt-gpt/scripts/cd_verbalizer.json", choice=verbalizer_choice)

def label_smoothing(binary_labels, num_classes, smoothing=0.1):
        """
        Apply label smoothing to multi-label classification targets.
        """
        return binary_labels * (1 - smoothing) + smoothing / num_classes
# 定义函数进行单次训练和评估
def train_and_evaluate(maxlength, warm_up_step, epoches, learning_rate, batchsize, threshold, all_labels):

    train_dataloader = PromptDataLoader(
        dataset=train_examples,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        # decoder_max_length=3,
        max_seq_length=maxlength,
        batch_size=batchsize,
        shuffle=True,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="head"
    )

    test_dataloader = PromptDataLoader(
        dataset=test_examples,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=maxlength,
        # decoder_max_length=3,
        batch_size=batchsize,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="head"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建Prompt模型
    prompt_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer)
    prompt_model = prompt_model.to(device)


    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    # 定义优化器和学习率调度器
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    total_steps = len(train_dataloader) * epoches  # 假设进行3个epoch的训练
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_step, num_training_steps=total_steps
    )

    # 训练模型
    prompt_model.train()
    for epoch in range(epoches):  # 假设训练3个epoch
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = inputs['label']
            smoothed_labels = label_smoothing(labels, num_classes=len(all_labels))
            outputs = prompt_model(inputs)
            loss = torch.nn.BCEWithLogitsLoss()(outputs, smoothed_labels.float())
            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), max_norm=1.0)
            tot_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 50 ==1:
                logger.info("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)))
                # print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
            
            

    # 评估模型
    prompt_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = inputs.to(device)
            labels = inputs['label'].cpu().numpy()
            outputs = prompt_model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > threshold).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return f1_score(all_labels, all_preds, average='micro')


score = train_and_evaluate(max_length, warm_up_step, epoches, learning_rate, batchsize, threshold, all_labels)
 
logger.info(f"Fold F1 Score: {score}")
