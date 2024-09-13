import json

# 定义共享意图列表
shared_intents = [
    "affirm", "deny", "dont_know", "acknowledge", "greet", "end_call",
    "handoff", "thank", "repeat", "cancel_close_leave_freeze", "change",
    "make_open_apply_setup_get_activate", "request_info", "how", "why",
    "when", "how_much", "how_long", "wrong_notworking_notshowing",
    "lost_stolen", "more_higher_after", "less_lower_before", "new", "existing",
    "limits"
]

# 读取JSON文件
with open('/Work21/2023/zhuangning/code/prompt-gpt/scripts/verbalizer.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 过滤字典
filtered_data = []
for dictionary in data:
    filtered_dict = {key: value for key, value in dictionary.items() if key in shared_intents}
    filtered_data.append(filtered_dict)

# 将过滤后的数据保存为新的JSON文件
with open('/Work21/2023/zhuangning/code/prompt-gpt/scripts/cd_verbalizer.json', 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, indent=4, ensure_ascii=False)

print("过滤完成,并保存为filtered_verbalizer.json")
