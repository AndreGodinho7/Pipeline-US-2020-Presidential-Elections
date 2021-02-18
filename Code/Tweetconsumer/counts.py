import os
workers=64

total = 0

    # try:
    #     with open('./output'+str(workers)+'/partitions_'+str(part)+'.txt', "r") as f:
            
    # except:
    #     print("Error opening file", part)

for filename in os.listdir(os.getcwd()):
    with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
        total += int(f.read())

print(total)