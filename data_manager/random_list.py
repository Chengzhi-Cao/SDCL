import random
all_num = 100
a_list = [i for i in range(all_num)]
b_list = []
num = 20

for i in range(num):
  random_number = random.randint(1,all_num-1)
  if random_number not in b_list:
    b_list.append(random_number)
print(b_list)
print(len(b_list))
