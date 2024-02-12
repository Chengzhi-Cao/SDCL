
import os
import random
import json

total_list = []

for i in range(10):

    dic = dict()

    all_num = 281
    test_list = []
    num = 271
    while len(test_list) != num:
        random_number = random.randint(0,all_num-1)
        if '{}'.format(random_number) not in test_list:
            test_list.append('{}'.format(random_number))
    test_list.sort(key=int)


    all_num = 281
    train_list = []
    num = 271
    while len(train_list) != num:
        random_number = random.randint(0,all_num-1)
        if '{}'.format(random_number) not in train_list:
            train_list.append('{}'.format(random_number))
    train_list.sort(key=int)


    dic['test'] = test_list
    dic['train'] = train_list

    total_list.append(dic)

print('total_list=',total_list)
with open("/ghome/caocz/code/Event_Camera/Event_Re_ID/VideoReID_PSTA/data_manager/splits_pride.json","w") as f:
    json.dump(total_list,f,indent=4, sort_keys=True)
