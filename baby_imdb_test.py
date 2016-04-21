#!usr/bin/env python
# -*- coding: utf-8 -*-

# set default coding euc-kr 2 utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pprint 
import re

import numpy as np


SEED = 100
np.random.seed(SEED)
rng = np.random

import copy
import theano.tensor as tensor
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

try :
    import cpickle as pickle
except :
    import pickle

import time
import random
random.seed(SEED)

trng = RandomStreams(SEED)

"""
Bi-direction RNN(LSTM, GRU 등) attention model을 기반으로 
imdb 영화 감정 분석 데이터를 학습 하기위한 baby program

초기 목표 : 단순 RNN, CNN 혹은 Bi-RNN 모델보다 높게!!(~2/28) V

중간 목표 : SVM uni + bi gram feature 보다 높은 성능!!!(~3/15) -> 여기까지는 baby program으로 V

중중간 목표 : 89%를 넘쟈!! V

중중중간 목표 : 91%를 넘자!! 

최종 목표 : state of the art를 넘겨봅세(~5:/xx) -> 전체 데이터 사용!!(91.8)

화이팅 나
"""

def orthogonal(shape, scale=1.1):
    ''' From Lasagne
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])

def shared_data(v, n):
    return theano.shared(value=v,name=n,borrow=True)
def squared_error(pred_y,y) : 
    return tensor.sum((pred_y - y)**2)
def relu(y):
    return tensor.switch(y<0,0,y)
def norm(x):
    avg = norm.mean()
    
    return normed_x
class sentiment_analysis_attention_model(object) :
    def dropout_layer(self,x,training_flag,prob = 0.5) :
        proj = tensor.switch(training_flag,
                            x*trng.binomial(x.shape,p=prob, n = 1, dtype=x.dtype),
                            x)
        return proj

    def dropconnection(self,X,training_flag,prob = 0.5) : 
        W = tensor.switch(training_flag,
                        X*trng.binomial((X.shape[0],X.shape[1]),p = prob,n = 1,dtype = X.dtype)
                        ,
                        X)
        return W
    def init_weight(self, shape) : 
        init_W = np.asarray(
            rng.uniform(
                low = -0.05,
                high = 0.05,
                size = shape
            ),
            dtype = theano.config.floatX
        )
        return init_W
    def init_glorot(self, shape) : 
        fin = shape[0]
        fout = shape[1]
        w = np.sqrt(6.0/(fin + fout))
        init_W = np.asarray(
                rng.uniform(
                    low = -w,
                    high = w,
                    size = shape
                ),
                dtype = theano.config.floatX
            )
        
        return init_W
    def init_orthogonal(self,shape,scale=1.1):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        w = scale * q[:shape[0], :shape[1]]
        
        init_W = np.asarray(w,dtype=theano.config.floatX) 
        
        return init_W
    """
    Network의 dimension 설정과 weight를 초기화
    초기에는 weight를 기본적으로 uniform하게 설정
    이후에 논문을 보고 최적 init를 설정
    """
    def __init__(self, voca_size, out_dim = 2 , embed_dim=100, attention_hidden_dim = 100, classify_hidden_dim = 200) : 
        # training flag for drop-out or drop-connection
        self.training_flag = tensor.scalar('training_flag')
        self.SEED = 100


        self.in_dim =  voca_size
        self.out_dim = out_dim
        self.embed_dim = embed_dim

        self.attention_hidden_dim = attention_hidden_dim
        self.classify_hidden_dim = classify_hidden_dim

        self.init_vector_lstm = shared_data(np.zeros((embed_dim,),dtype=theano.config.floatX),'init_vector_lstm')

        # word embedding
        self.W = shared_data(self.init_weight((voca_size,embed_dim)),'W')
        
        # bi_lstm layer weight
        # bi_lstm forward
        self.f1_Wf_f = shared_data(self.init_orthogonal((embed_dim*2, embed_dim)),'f1_Wf_f')
        self.f1_Wi_f = shared_data(self.init_orthogonal((embed_dim*2, embed_dim)),'f1_Wi_f')
        self.f1_Wc_f = shared_data(self.init_orthogonal((embed_dim*2, embed_dim)),'f1_Wc_f')
        self.f1_Wo_f = shared_data(self.init_orthogonal((embed_dim*2, embed_dim)),'f1_Wo_f')
         
        self.f1_bf_f = shared_data(np.zeros((embed_dim,),dtype=theano.config.floatX),'f1_bf_f') 
        self.f1_bi_f = shared_data(np.zeros((embed_dim,),dtype=theano.config.floatX),'f1_bi_f')
        self.f1_bc_f = shared_data(np.zeros((embed_dim,),dtype=theano.config.floatX),'f1_bc_f')
        self.f1_bo_f = shared_data(np.zeros((embed_dim,),dtype=theano.config.floatX),'f1_bo_f')
        
        # bi_lstm backward
        self.f1_Wf_b = shared_data(self.init_orthogonal((embed_dim*2, embed_dim)),'f1_Wf_b')
        self.f1_Wi_b = shared_data(self.init_orthogonal((embed_dim*2, embed_dim)),'f1_Wi_b')
        self.f1_Wc_b = shared_data(self.init_orthogonal((embed_dim*2, embed_dim)),'f1_Wc_b')
        self.f1_Wo_b = shared_data(self.init_orthogonal((embed_dim*2, embed_dim)),'f1_Wo_b')
         
        self.f1_bf_b = shared_data(np.zeros((embed_dim,),dtype=theano.config.floatX),'f1_bf_b')
        self.f1_bi_b = shared_data(np.zeros((embed_dim,),dtype=theano.config.floatX),'f1_bi_b')
        self.f1_bc_b = shared_data(np.zeros((embed_dim,),dtype=theano.config.floatX),'f1_bc_b')
        self.f1_bo_b = shared_data(np.zeros((embed_dim,),dtype=theano.config.floatX),'f1_bo_b')
        
        # attention layer weight
        self.f2_W1 = shared_data(self.init_glorot((embed_dim*2,attention_hidden_dim)),'f2_W1') 
        self.f2_b1 = shared_data(np.zeros((attention_hidden_dim,),dtype=theano.config.floatX),'f2_b1')
        self.f2_W2 = shared_data(self.init_glorot((attention_hidden_dim,1)),'f2_W2')
        self.f2_b2 = shared_data(np.zeros((1,),dtype=theano.config.floatX),'f2_b2')

        # classifying layer weight
        self.f3_W1 = shared_data(self.init_glorot((embed_dim*2,classify_hidden_dim)),'f3_W1')
        self.f3_b1 = shared_data(np.zeros((classify_hidden_dim,),dtype=theano.config.floatX),'f2_b1')
        self.f3_W2 = shared_data(self.init_glorot((classify_hidden_dim, out_dim)),'f3_W2')
        self.f3_b2 = shared_data(np.zeros((out_dim,),dtype=theano.config.floatX),'f2_b1')
        
        # nomalization weight

        #self.norm = shared_data(np.asscalar(np.array([0.0])),'norm')
 

        # result of network
        self.p_y = self.network()
        # params
        self.params = self.set_params()

        self.set_train()
    def set_params(self):
        return [
                self.f1_bf_f,
                self.f1_bi_f,
                self.f1_bc_f,
                self.f1_bo_f,
                self.f1_bf_b,
                self.f1_bi_b,
                self.f1_bc_b,
                self.f1_bo_b,
                self.f2_b1,
                self.f2_b2,
                self.f3_b1,
                self.f3_b2,
                self.f1_Wf_f,
                self.f1_Wi_f,
                self.f1_Wc_f,
                self.f1_Wo_f,
                self.f1_Wf_b,
                self.f1_Wi_b,
                self.f1_Wc_b,
                self.f1_Wo_b,
                self.f2_W1,
                self.f2_W2,
                self.f3_W1,
                self.f3_W2,
                self.W
                #self.norm
                ]
    def lstm_forward(self, x_t,hf_t_1, cf_t_1):
        x = tensor.concatenate([hf_t_1,x_t])
        ft = tensor.nnet.hard_sigmoid(tensor.dot(x,self.f1_Wf_f) + self.f1_bf_f)
        it = tensor.nnet.hard_sigmoid(tensor.dot(x,self.f1_Wi_f) + self.f1_bi_f)
        ot = tensor.nnet.hard_sigmoid(tensor.dot(x,self.f1_Wo_f) + self.f1_bo_f)
        tct = tensor.tanh(tensor.dot(x,self.f1_Wc_f) + self.f1_bc_f)
        
        cf_t = ft*cf_t_1 + it*tct
        hf_t = ot * tensor.tanh(cf_t)
        return [hf_t, cf_t]
    def lstm_backward(self,x_t, hb_t_1, cb_t_1):
        x = tensor.concatenate([hb_t_1, x_t])
        ft = tensor.nnet.hard_sigmoid(tensor.dot(x,self.f1_Wf_b) + self.f1_bf_b)
        it = tensor.nnet.hard_sigmoid(tensor.dot(x,self.f1_Wi_b) + self.f1_bi_b)
        ot = tensor.nnet.hard_sigmoid(tensor.dot(x,self.f1_Wo_b) + self.f1_bo_b)
        tct = tensor.tanh(tensor.dot(x,self.f1_Wc_b)+self.f1_bc_b)

        cb_t = ft*cb_t_1 + it*tct
        hb_t = ot * tensor.tanh(cb_t)
        return [hb_t, cb_t]
        
    """
    attention weight calculator
    """
    def f2(self, hi):
        hidden = tensor.tanh(tensor.dot(hi, self.f2_W1)+self.f2_b1)
        #hidden_drop = self.dropout_layer(x = hidden,training_flag = self.training_flag)
        ai = tensor.dot(hidden,self.f2_W2)+self.f2_b2
        #result = self.dropout_layer(x = ai, training_flag = self.training_flag)
        return ai
    """
    classifier
    """
    def f3(self,sentence_embedding) :
        sentence_drop =  self.dropout_layer(x = sentence_embedding,training_flag = self.training_flag)
        hidden = relu(tensor.dot(sentence_drop, self.f3_W1)+self.f3_b1)
        hidden_drop = self.dropout_layer(x = hidden,training_flag = self.training_flag)
        result = tensor.dot(hidden_drop,self.f3_W2)+self.f3_b2
        """hidden = tensor.nnet.sigmoid(
                    tensor.dot(
                        sentence_embedding,self.dropconnection(X=self.f3_W1,training_flag=self.training_flag))
                            +self.f3_b1)
        result = tensor.dot(hidden,self.dropconnection(X=self.f3_W2,training_flag=self.training_flag))+self.f3_b2"""
        return tensor.nnet.softmax(result)
    """
    전체 네트워크 구성
    """
    """
    네트워크 정의 p_y를 정의(이후에 p_y.eval(self.X : sentence) 명령어로 결과를 얻을 수 있게 하자) 
    """
    def network(self) :
       self.v = tensor.ivector('v')
       # 단어를 one-hot에서 word embeddings으로 변환
       self.words = self.W[self.v]
       
       # forward pass of lstm
       [hf, memories],updates = theano.scan(fn=self.lstm_forward,
                                sequences = [self.words],
                                outputs_info = [
                                    tensor.unbroadcast(self.init_vector_lstm,0),
                                    tensor.unbroadcast(self.init_vector_lstm,0)])
       
       # backward pass of lstm
       [hb, memories],updates = theano.scan(fn=self.lstm_backward,
                                sequences = [self.words],
                                outputs_info = [
                                    tensor.unbroadcast(self.init_vector_lstm,0),
                                    tensor.unbroadcast(self.init_vector_lstm,0)],
                                    go_backwards = True)

       # for, backward hf와 hb concatenate
       h, updates = theano.scan(lambda f,b : tensor.concatenate([f,b]), 
                    sequences = [hf,hb]
                    )
       # weight 계산
       temp_weight, updates = theano.scan(fn=self.f2,
                    sequences = [h],
                    )
       a = tensor.nnet.softmax(temp_weight.T)
       
       #a = (a * h.shape[0])/(10+self.norm)
       a = (a * h.shape[0])/(10)
       # weighted sum을 사용하여 sentence embedding
       context_vector = tensor.dot(a,h)

       # sentence embedding을 사용하여 최종적으로 sentimental analysis 수행
       result = self.f3(context_vector)
       self.pred_func = theano.function(
                        inputs = [self.v,self.training_flag], 
                        outputs = [result,a],
                        ) 
       return result
       
    
        
        
    def test(self,sentences , answers) :
       i = 0
       correct = 0
       num_of_1 = 0
       result = ""
       for i in range(len(sentences)):
       #for i in range(200):
            temp_sentence = sentences[i]
            answer = answers[i]
            sentence = np.asarray(temp_sentence,dtype = 'int32')
            pred,attentions = self.pred_func(sentence,0)
            pred = pred[0]
            attentions = attentions[0]
            if (answer == 1) and pred[0] <= pred[1] :
                correct += 1
                num_of_1 +=1 
            elif (answer == 0) and pred[0] > pred[1]:
                correct += 1
            if i < 10 :
                print str(i+1)+": "+str(pred)+" " +str( answer)
                for attention in attentions :
                    result += str(attention)+"\t"
                result += '\n'
       print "number of ones : "+ str(num_of_1)  
       return float(correct)/len(sentences),result
       #return float(correct)/100
    def set_train(self, alpha = 0.005) : 
        i = 0
        index = tensor.lscalar()  # index to a [mini]batch
        y = tensor.vector('y')
        x = tensor.ivector('x')
        #cost = tensor.nnet.binary_crossentropy(self.p_y, y).mean()
        """preds = theano.scan(fn=self.pred_func,
                            sequences=[X],
                           )
        cost,updates = theano.scan(fn=squared_error, 
               sequences=[preds,Y],
               )
        cost = cost.mean()"""
        cost = squared_error(self.p_y, y)
         
        #set gradient function
        #grads = theano.gradient.grad_clip(tensor.grad(cost,self.params),-1,1)
        grads = []
        for param in self.params:
            grads.append(tensor.clip(tensor.grad(cost,param),-1,1))
        updates = [(self.params[i],self.params[i] - alpha * grads[i]) for i in range(len(self.params))] 
        
        self.train_model = theano.function(
                    inputs=[self.v ,y,self.training_flag],
                    outputs = [cost,self.p_y],
                    updates = updates)


    def train(self, datas) :
        total_cost = 0.0
        random.shuffle(datas)
        for i in range(len(datas)):
        #for i in range(1):
            batch_sentences = datas[i][0]
            batch_answers = datas[i][1]
            cost,pred = self.train_model(batch_sentences,batch_answers,0)
            #print str(pred) +"\t"+ str(batch_answers)+"\t"+str(norm_weight)
            total_cost += cost
        print "total cost is " +str(total_cost/len(datas))




data_set = pickle.load(open("data/imdb.pkl"))
"""
train data  : 20000
test data   : 5000

train_set[0] : uni-gram word sequence
train_set[1] : class of data

test_set[0] : uni-fram word sequence
test_set[1] : class of data

전체 단어 집합 : 91189
훈련 데이터 입력 단어 :  
"""
train_set = data_set['train']
test_set = data_set['test']

length = 0
word_dic = {}
for data in train_set[0] :
    for word in data:
        word_dic[word] = 1
    length += len(data)
for data in test_set[0] :
    for word in data :
        word_dic[word] = 1
    length += len(data)
    
length_of_word_set = word_dic.keys()
print("Number of word : " +str(len(word_dic.keys())) )
print("Length of word sequence : "+str(length))
#print(word_dic.keys())

# model init
model = sentiment_analysis_attention_model(voca_size = 20000)
print "model load complete"

#model train

fattentions = open("attention.txt","w")

#model test
i = 0
print "train start!!"
datas = []
for i in range(len(train_set[1])) :
    temp_answer = []
    answer = train_set[1][i]
    sentence = train_set[0][i]

    if answer == 0:
        temp_answer = [1,0]
    else : 
        temp_answer = [0,1]
    datas.append((np.array(sentence,dtype='int32'),np.array(temp_answer,dtype=theano.config.floatX)))
i = 0


while i <  20:
    start_time = time.time()
    accuracy, attentions = model.test(test_set[0], test_set[1])
    print accuracy
    fattentions.write(attentions+'\n')
    print str(i+1)+'th epoch'
    model.train(datas = datas)
    end_time = time.time()
    print 'test time is ' + str(end_time - start_time)
    i += 1
accuray, attentions =  model.test(test_set[0], test_set[1])
print accuracy
fattentions.write(attentions+'\n')
fattentions.close()
pickle.dump(model,open('model/model_1.pkl'))
