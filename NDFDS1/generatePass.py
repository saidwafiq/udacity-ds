# Use an import statement at the top
from random import sample

word_file = "words.txt"
word_list = []

#fill up the word_list
with open(word_file,'r') as words:
	for line in words:
		# remove white space and make everything lowercase
		word = line.strip().lower()
		# don't include words that are too long or too short
		if 3 < len(word) < 8:
			word_list.append(word)

# Add your function generate_password here
# It should return a string consisting of three random words
# concatenated together without spaces
def gpass():
	#cria uma string e concatena um sorteio do arquivo de palavras com ate 20
	return str().join(sample(word_list,20))
print gpass()
