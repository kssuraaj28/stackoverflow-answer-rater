# python3 -m pip install bert-embedding

import array
from bert_embedding import BertEmbedding

dataset1 = 'book_corpus_wiki_en_uncased'
dataset2 = 'book_corpus_wiki_en_cased'
# Use any dataset
bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name=dataset1)

def find_embedding(text):
	words = text.split(' ')
	arr = array.array('d',[])
	for i in range (0, 768):
		arr.append(0.0)
	result = bert_embedding(words)
	cnt = 0
	for token in result:
		for embed in token[1]:
			cnt = cnt + 1
			for j in range (0, 768):
				arr[j] = arr[j] + embed[j]
	for i in range (0 ,768):
		arr[i] = arr[i] / cnt
	return arr

# text = """<p>Solution without <code>pandas</code>:</p>

# <pre><code>import csv

# dataFile = 'data.csv'

# with open(dataFile) as inputData:
#     csv_input = csv.reader(inputData)
#     i = zip(next(csv_input), zip(*csv_input))
#     data, (_, times) = {}, next(i)
#     for k, line in i:
#         for t, l in zip(times, line):
#             data.setdefault(k, {}).setdefault(t, {})
#             data[k][int(t)] = l

# print(data['Data1'][1])
# </code></pre>

# <p>Prints:</p>

# <pre><code>20
# </code></pre>"""

# ans = find_embedding(text)
# print(ans)
