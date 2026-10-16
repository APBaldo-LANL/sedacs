import numpy as np


data = []
filename = 'time_ham.dat'

with open(filename,'r') as f:
	for lines in f:
		lines = lines.strip()
		if lines:
			data.append(float(lines))

#print(data)

data = np.array(data)
ave = np.mean(data)
std = np.std(data)

print(ave)
print(std)
