# coding=utf-8

import json
import random

main_meals = {
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

unit = "元"

questions = [
    "%s多少钱？",
    "请问%s价格？",
    "您好我要一份%s，这样价格为多少？",
    "我要一份%s，这样多少？"
]

def QuestionGenerator(questions, main_meals, unit, num_set='all'):
    # menu["data"]["qas"].append()

    # "question": "大麦克多少钱？", 
    # "id": "MCDONALDS_168_QUERY_0", 
    # "answers": [
    # {
    #     "text": "72元", 
    #     "answer_start": 16
    # }
    # ]
    meal_size = len(main_meals)    
    meal_keys = list(main_meals.keys())
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
                _a["text"] = str(main_meals[_meal][0]) + unit
                _a["answer_start"] = main_meals[_meal][1]
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
            _a["text"] = str(main_meals[meal][0]) + unit
            _a["answer_start"] = main_meals[meal][1]
            _q["answers"].append(_a)
            q_set.append(_q)
    
    return q_set


menu = {}
menu["version"] = "v1.0"

menu["data"] = []

_set = {}

_set["id"] = "MCDONALDS_168"
context_init = "麦当劳目前的餐点有："
_set["context"] = context_init
for _meal in [ (_meal + str(main_meals[_meal][0]) + unit + "、") for _meal in list(main_meals.keys())] :
    _set["context"] = _set["context"] + _meal
_set["context"] = _set["context"][:-1] + "。"

# _set["qas"] = QuestionGenerator(questions, main_meals, unit, num_set=20)
_set["qas"] = QuestionGenerator(questions, main_meals, unit)

paragraphs = {}
paragraphs["paragraphs"] = []
paragraphs["paragraphs"].append(_set)
menu["data"].append(paragraphs)

with open('test_1.json', 'w', encoding="utf-8") as outfile:
    json.dump(menu, outfile, indent=4, ensure_ascii=False)

with open("paragraph.txt", 'w', encoding="utf-8") as outfile:
    outfile.writelines(_set["context"])
    outfile.close()
