import os
import json, csv
import ast
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from sklearn.preprocessing import MultiLabelBinarizer

# 将标签转换为多标签二进制格式

class NLUProcessor(DataProcessor):
    """
    This is the base class for all NLU processors.
    dataset_name = "nlu++"
    split = "fold0-fold19"
    """
    def __init__(self):
        super().__init__()
        self.labels = ['affirm', 'deny', 'dont_know', 'acknowledge', 'greet', 'end_call', \
                       'handoff', 'thank', 'repeat', 'cancel_close_leave_freeze', 'change', 'make_open_apply_setup_get_activate', \
                        'request_info', 'how', 'why', 'when', 'how_much', 'how_long', \
                        'wrong_notworking_notshowing', 'lost_stolen', 'more_higher_after', 'less_lower_before', 'new', 'existing', \
                        'limits', 'savings', 'current', 'business', 'credit', 'debit', \
                        'contactless', 'international', 'account', 'transfer_payment_deposit', 'appointment', 'arrival', \
                        'balance', 'card', 'cheque', 'direct_debit', 'standing_order', 'fees_interests', \
                        'loan', 'mortgage', 'overdraft', 'withdrawal', 'pin', 'refund', \
                        'check_in', 'check_out', 'restaurant', 'swimming_pool', 'parking', 'pets', \
                        'accesibility', 'booking', 'wifi', 'gym', 'spa', 'room_ammenities', 'housekeeping', 'room_service']

    def get_examples(self, data_dir, split):
        mlb = MultiLabelBinarizer(classes = self.labels)
        mlb.fit([self.labels])
        path = os.path.join(data_dir)
        examples = []
        with open(path, 'r',encoding='utf-8')as f:
            reader = csv.reader(f, delimiter=',')
            # 跳过第一行
            next(reader)
            for idx, row in enumerate(reader):
                fold,text,intents,intent_span,intent_span_4o = row
                text_a = text
                text_b = intent_span.replace('/',',')
                label = ast.literal_eval(intents)
                guid = "%s-%s" % (fold, idx)
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label = mlb.transform([label])[0].tolist())
                examples.append(example)
        return examples
    


class NLUProcessorCD(DataProcessor):
    """
    This is the base class for all NLU processors.
    dataset_name = "nlu++"
    split = "fold0-fold19"
    """
    def __init__(self):
        super().__init__()
        self.labels =  ['affirm', 'deny', 'dont_know', 'acknowledge', 'greet', 'end_call', 'handoff', 'thank', \
                        'repeat', 'cancel_close_leave_freeze', 'change', 'make_open_apply_setup_get_activate', 'request_info',\
                         'how', 'why', 'when', 'how_much', 'how_long', 'wrong_notworking_notshowing', 'lost_stolen', \
                        'more_higher_after', 'less_lower_before', 'new', 'existing', 'limits']

    def get_examples(self, data_dir, split):
        mlb = MultiLabelBinarizer(classes = self.labels)
        mlb.fit([self.labels])
        path = os.path.join(data_dir)
        examples = []
        with open(path, 'r',encoding='utf-8')as f:
            reader = csv.reader(f, delimiter=',')
            # 跳过第一行
            next(reader)
            for idx, row in enumerate(reader):
                fold,text,intents,intent_span,intent_span_4o = row
                text_a = text
                text_b = intent_span.replace('/',',')
                label = ast.literal_eval(intents)
                guid = "%s-%s" % (fold, idx)
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label = mlb.transform([label])[0].tolist())
                examples.append(example)
        return examples

class NLUProcessorwithsmothing(DataProcessor):
    """
    This is the base class for all NLU processors.
    dataset_name = "nlu++"
    split = "fold0-fold19"
    """
    def __init__(self):
        super().__init__()
        self.labels = ['affirm', 'deny', 'dont_know', 'acknowledge', 'greet', 'end_call', \
                       'handoff', 'thank', 'repeat', 'cancel_close_leave_freeze', 'change', 'make_open_apply_setup_get_activate', \
                        'request_info', 'how', 'why', 'when', 'how_much', 'how_long', \
                        'wrong_notworking_notshowing', 'lost_stolen', 'more_higher_after', 'less_lower_before', 'new', 'existing', \
                        'limits', 'savings', 'current', 'business', 'credit', 'debit', \
                        'contactless', 'international', 'account', 'transfer_payment_deposit', 'appointment', 'arrival', \
                        'balance', 'card', 'cheque', 'direct_debit', 'standing_order', 'fees_interests', \
                        'loan', 'mortgage', 'overdraft', 'withdrawal', 'pin', 'refund', \
                        'check_in', 'check_out', 'restaurant', 'swimming_pool', 'parking', 'pets', \
                        'accesibility', 'booking', 'wifi', 'gym', 'spa', 'room_ammenities', 'housekeeping', 'room_service']

    def label_smoothing(self, binary_labels, num_classes, smoothing=0.1):
        """
        Apply label smoothing to multi-label classification targets.
        """
        return binary_labels * (1 - smoothing) + smoothing / num_classes
    def get_examples(self, data_dir, split):
        smoothing = 0.1
        num_classes = len(self.labels)
        mlb = MultiLabelBinarizer(classes = self.labels)
        mlb.fit([self.labels])
        path = os.path.join(data_dir)
        examples = []
        with open(path, 'r',encoding='utf-8')as f:
            reader = csv.reader(f, delimiter=',')
            # 跳过第一行
            next(reader)
            for idx, row in enumerate(reader):
                fold,text,intents,intent_span,intent_span_4o = row
                text_a = text
                text_b = intent_span_4o.replace('/',',')
                # text_b = intent_span
                label = ast.literal_eval(intents)
                guid = "%s-%s" % (fold, idx)
                binary_label = mlb.transform([label])[0]
                smoothed_label = self.label_smoothing(binary_label, len(self.labels))
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label = smoothed_label.tolist())
                examples.append(example)
        return examples
    
# processor = NLUProcessor()
# data_dir = '/Work21/2023/zhuangning/code/prompt-gpt/data/banking.csv'
# train_examples = processor.get_examples(data_dir, "train")
# # 打印经过processor后的示例数据
# for example in train_examples:
#     print(example)
#     break

PROCESSORS = {
    "nlu++": NLUProcessor,
}
