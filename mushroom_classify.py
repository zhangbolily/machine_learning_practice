
import tensorflow as tf
import numpy as np
import os
import csv
import pandas as pd

PROJECT_DIR = "/home/ballchang/PycharmProjects/machine_learning_practice/data_set"


def creatDataSet(file_path):
    poisonous_count_result = {}
    edible_count_result = {}

    # if os.path.isfile(PROJECT_DIR + "/mushroom_edible.data") and os.path.isfile(PROJECT_DIR + "/mushroom_poisonous.data") and os.path.isfile(PROJECT_DIR + "/mushroom_types.data"):

    # 以下使用的是读取成list的方式来切分数据
    poisonous = np.empty((1, 1))
    edible = np.empty((1, 1))
    mushrooms = np.empty((1, 1))
    index = np.empty((1, 1))

    with open(file_path) as f:
        csv_data = csv.reader(f)

        i = 0

        # 这里取了前2000个数据作为了测试集
        for row in csv_data:
            if i == 2001:
                break
            else:
                if i == 0:
                    mushrooms = np.array(row)
                else:
                    mushrooms = np.vstack((mushrooms, row))
            i += 1

        f.seek(0)

        for row in csv_data:
            if row[0] == 'p':
                poisonous = np.vstack((poisonous, row))
            else:
                if row[0] != "class":
                    edible = np.vstack((edible, row))
                else:
                    index = np.array(row)
                    poisonous = np.array(row)
                    edible = np.array(row)
                    for _ in range(2001):
                        next(csv_data)

    # 在没有改变数组结构之前，poisonous的第一行数据就是特征值的名称
    index = poisonous[0]
    # print(index)

    types = {}

    # 对数组进行处理，先分隔开数组，再reshape
    # 最后一个操作是去除第一列的class特征，前面已经把两种类别的蘑菇都分开了
    poisonous = np.array(np.hsplit(poisonous, 23))
    poisonous = poisonous.reshape((23, -1))
    types['p'] = (pd.value_counts(poisonous[0]))['p']
    poisonous = poisonous[1:]

    edible = np.array(np.hsplit(edible, 23))
    edible = edible.reshape((23, -1))
    types['e'] = (pd.value_counts(edible[0]))['e']
    edible = edible[1:]

    # 目前有三批数据：
    # 1、index保存了数据的所有特征值
    # 2、poisonous保存了所有的有毒蘑菇特征，数据结构是poisonous[](特征名称， 值1， 值2 ……）
    # 3、edible和2结构相同

    for column in edible:
        temp = {}
        key_temp = []
        result_dict = dict(pd.value_counts(column))
        # print(result_dict)
        for key in result_dict:
            if index.__contains__(key):
                poisonous_count_result[key] = result_dict[key]
                key_temp = key
            else:
                temp[key] = result_dict[key]
                # print(temp)
        edible_count_result[key_temp] = temp

    for column in poisonous:
        temp = {}
        key_temp = []
        result_dict = dict(pd.value_counts(column))
        # print(result_dict)
        for key in result_dict:
            if index.__contains__(key):
                poisonous_count_result[key] = result_dict[key]
                key_temp = key
            else:
                temp[key] = result_dict[key]
                # print(temp)
        poisonous_count_result[key_temp] = temp

    print(mushrooms, "\n")
    print(poisonous_count_result, "\n")
    print(edible_count_result, "\n")

    return types, poisonous_count_result, edible_count_result, mushrooms


def indentifyMushrooms(var1, var2, var3, var4):
    p_poi = var1['p'] / (var1['p'] + var1['e'])
    p_edi = var1['e'] / (var1['p'] + var1['e'])

    const_p_poi = p_poi ** 22
    const_p_edi = p_edi ** 22

    result = 0

    temp_dict1 = {}
    temp_dict2 = {}

    for i in range(2000):
        p_properties = 1
        p_properties_in_condition_p = 1
        p_properties_in_condition_e = 1
        for j in range(22):
            temp_dict1 = var2[var4[0][j+1]]
            temp_dict2 = var3[var4[0][j+1]]

            if temp_dict1.__contains__(var4[i+1][j+1]):
                temp_value1 = temp_dict1[var4[i+1][j+1]]
            else:
                temp_value1 = 0

            if temp_dict2.__contains__(var4[i+1][j+1]):
                temp_value2 = temp_dict2[var4[i+1][j+1]]
            else:
                temp_value2 = 0

            p_properties = p_properties * ((temp_value1 + temp_value2)/(var1['p'] + var1['e']))
            p_properties_in_condition_p = p_properties_in_condition_p * (temp_value1/var1['p'])
            p_properties_in_condition_e = p_properties_in_condition_e * (temp_value2/var1['e'])
        result_poi = p_properties_in_condition_p/p_properties*p_poi
        result_edi = p_properties_in_condition_e/p_properties*p_edi

        print(result_poi, result_edi, var4[i+1][0])
        if result_poi > result_edi and var4[i+1][0] == 'p':
            result += 1
        else:
            if result_edi > result_poi and var4[i+1][0] == 'e':
                result += 1
            else:
                continue

    return result

types, data1, data2, test = creatDataSet("./data_set/mushrooms.csv")

print("Accuracy rate is:", indentifyMushrooms(types, data1, data2, test)/2000)
