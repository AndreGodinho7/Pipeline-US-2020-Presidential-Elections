workers=64

partitions = [i for i in range(64)]

total = 0

for part in partitions:
    try:
        with open('./output'+str(workers)+'/partitions_'+str(part)+'.txt', "r") as f:
            total += int(f.read())
    except:
        print("Error opening file", part)

print(total)