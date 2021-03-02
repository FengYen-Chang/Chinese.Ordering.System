# coding=utf-8

import json
import random

main_meal = {
    "大麦克": (72, 16),
    "双层牛肉吉事堡": (62, 29),
    "嫩煎鸡腿堡": (82, 40),
    "麦香鸡": (44, 49),
    "麦克鸡块(6块)": (60, 63),
    "麦克鸡块(10块)": (100, 77),
    "劲辣鸡腿堡": (72, 88),
    "麦脆鸡腿(2块)": (110, 102),
    "麦脆鸡翅(2块)": (90, 116),
    "黄金起司猪排堡": (52, 129),
    "麦香鱼": (44, 138),
    "烟熏鸡肉长堡": (74, 150),
    "姜烧猪肉长堡": (74, 162),
    "BLT 安格斯黑牛堡": (109, 178),
    "BLT 辣脆鸡腿堡": (109, 193),
    "BLT 嫩煎鸡腿堡": (109, 208),
    "蕈菇安格斯黑牛堡": (119, 222),
    "凯萨脆鸡沙拉": (99, 234),
    "义式烤鸡沙拉": (99, 246)
}

M_set = {
    "A":55,
    "B":55,
    "C":68,
    "D":85,
    "E":85
}

unit = "元"

questions = [
    "%s多少钱？",
    "请问%s价格？",
    "您好我要一份%s，这样价格为多少？",
    "我要一份%s，这样多少？"
]

def QuestionGenerator(questions, main_meal, unit, num_set='all'):
    # menu["data"]["qas"].append()

    # "question": "大麦克多少钱？", 
    # "id": "MCDONALDS_168_QUERY_0", 
    # "answers": [
    # {
    #     "text": "72元", 
    #     "answer_start": 16
    # }
    # ]
    meal_size = len(main_meal)    
    meal_keys = list(main_meal.keys())
    q_size = len(questions)
    q_set = []
    if num_set == 'all':
        i = 0
        for _ques in questions:
            for _meal in meal_keys:
                # print (_q, _meal)
                _q = {}
                _q["question"] = _ques% (_meal)
                _q["id"]="MCDONALDS_168_QUERY_%d"% (i)
                _q["answers"] = []
                _a = {}
                _a["text"] = str(main_meal[_meal][0]) + unit
                _a["answer_start"] = main_meal[_meal][1]
                _q["answers"].append(_a)
                q_set.append(_q)
                i = i + 1

    else:
        for i in range(num_set):
            meal = meal_keys[random.randint(0, meal_size - 1)]
            _q = {}
            _q["question"] = questions[random.randint(0, q_size - 1)]% (meal)
            _q["id"]="MCDONALDS_168_QUERY_%d"% (i)
            _q["answers"] = []
            _a = {}
            _a["text"] = str(main_meal[meal][0]) + unit
            _a["answer_start"] = main_meal[meal][1]
            _q["answers"].append(_a)
            q_set.append(_q)
    
    return q_set


menu = {}
menu["version"] = "v1.0"

menu["data"] = []

_set = {}

_set["id"] = "MCDONALDS_168"
_set["context"] = "麦当劳目前的餐点有：大麦克价格为72元、双层牛肉吉事堡价格为62元、嫩煎鸡腿堡价格为82元、麦香鸡价格为44元、麦克鸡块(6块)价格为60元、麦克鸡块(10块)价格为100元、劲辣鸡腿堡价格为72元、麦脆鸡腿(2块)价格为110元、麦脆鸡翅(2块)价格为90元、黄金起司猪排堡价格为52元、麦香鱼价格为44元、烟熏鸡肉长堡价格为74元、姜烧猪肉长堡价格为74元、BLT 安格斯黑牛堡价格为109元、BLT 辣脆鸡腿堡价格为109元、BLT 嫩煎鸡腿堡价格为109元、蕈菇安格斯黑牛堡价格为119元、凯萨脆鸡沙拉价格为99元和义式烤鸡沙拉价格为99元。"

# _set["qas"] = QuestionGenerator(questions, main_meal, unit, num_set=20)
_set["qas"] = QuestionGenerator(questions, main_meal, unit)

paragraphs = {}
paragraphs["paragraphs"] = []
paragraphs["paragraphs"].append(_set)
menu["data"].append(paragraphs)

with open('test_1.json', 'w', encoding="utf-8") as outfile:
    json.dump(menu, outfile, indent=4, ensure_ascii=False)
