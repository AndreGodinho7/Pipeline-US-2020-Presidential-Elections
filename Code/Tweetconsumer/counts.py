import os
workers=64

total = 0

for filename in os.listdir(os.getcwd()):
    with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
        print(filename)
        total += int(f.read())

print(total)