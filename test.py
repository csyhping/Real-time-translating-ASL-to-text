import itertools

a = [['B', 'K', 'T', 'O', 'R'], ['A', 'S', 'T', 'U', 'P'], ['D', 'Z', 'O', 'C', 'H']]

comb = list(itertools.product(*a))
candidate = []
for x in comb:
	string = ''.join(x)
	candidate.append(string.lower())
print(candidate)

f = open('wordlist.txt', 'w')
for string in candidate:
	f.write('%s\n' % string)