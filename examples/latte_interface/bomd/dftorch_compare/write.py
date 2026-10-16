performance  = []
with open('test.log','r') as f:
	for lines in f:
		line=lines.strip()
		if line:
			performance.append(line)



print(performance)
print(len(performance))

print(list(dict.fromkeys(performance)))
print(len(list(dict.fromkeys(performance))))
