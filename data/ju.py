import pickle

corpus = pickle.load(open('imdb.pkl'))

train = corpus['train']
test = corpus['test']

ftrain = open('raw_train_data.txt','w')
ftest = open('raw_test_data.txt','w')


for i in range(len(train[1])):
    ftrain.write(str(train[1][i])+'\t'+str(train[0][i])+'\n')

for i in range(len(test[1])):
    ftest.write(str(test[1][i])+'\t'+str(test[0][i])+'\n')
ftrain.close()
ftest.close()
