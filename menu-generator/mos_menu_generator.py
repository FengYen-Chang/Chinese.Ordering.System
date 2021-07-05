# coding=utf-8

import json
import random

main_meals = {
    "摩斯汉堡": (70, 18),
    "辣味摩斯汉堡": (70, 30),
    "摩斯吉士汉堡": (75, 42),
    "辣味摩斯吉士汉堡": (75, 56),
    "蜜汁烤鸡堡": (70, 67),
    "摩斯鳕鱼堡":(70, 78),
    "黄金炸虾堡":(75, 89),
    "摘鲜绿黄金炸虾堡":(70, 103),
    "轻柠双牛堡":(100, 114),
    "厚切培根和牛堡":(100, 127),
    "烧肉珍珠堡(牛)":(70, 141),
    "姜烧珍珠堡(猪)":(65, 155),
    "海洋珍珠堡":(75, 166),
    "元气和牛珍珠堡":(105, 179),
    "杏鲍菇珍珠堡(素)":(70, 194),
    "藜麦烧肉珍珠堡":(75, 207),
    "藜麦姜烧珍珠堡":(70, 220),
    "藜麦海洋珍珠堡":(80, 233),
    "藜麦元气和牛珍珠堡":(110, 248),
    "藜麦杏鲍菇珍珠堡(素)":(75, 265),
    "藜麦莲藕牛蒡珍珠堡(素)":(80, 283),
    "摩斯热狗堡":(55, 294),
    "辣味吉利热狗堡":(70, 307)
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
                _q["id"]="MOSBURGER_168_QUERY_%d"% (i)
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
            _q["id"]="MOSBURGER_168_QUERY_%d"% (i)
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

_set["id"] = "MOSBURGER_168"
context_init = "摩斯汉堡目前的餐点有："
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

with open('mos_test_1.json', 'w', encoding="utf-8") as outfile:
    json.dump(menu, outfile, indent=4, ensure_ascii=False)

with open("paragraph.txt", 'w', encoding="utf-8") as outfile:
    outfile.writelines(_set["context"])
    outfile.close()
