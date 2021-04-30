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
		cnt = cnt + 1
		embed = token[1][0]
		for j in range (0, 768):
			arr[j] = arr[j] + embed[j]
	for i in range (0 ,768):
		arr[i] = arr[i] / cnt
	return arr

text = "this is bert !"

ans = find_embedding(text)
print(ans)