import os
workers=64

total = 0

for filename in os.listdir(os.getcwd()):
    with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
        try:
            total += int(f.read())
        except :
            print(filename)

print(total)