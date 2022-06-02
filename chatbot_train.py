import nltk #natural language processing library

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json  #to parse the json filein python
import pickle #to store data

import numpy as np #for numerical computation and array handelling
import random #for selection random response data

with open ('C:\\workooplis\\project\\chatbot\\chatbot-python-project-data-codes\\intents.json') as f :
    data_file = f.read()

intents = json.loads(data_file) #to parse the data file

words = [] #to store the all tokenize words from the patterns
documents = [] #to store the list of tokenize words and its tag
classes = [] #to store the tag
ignore_words = ['!','?']


for intent in intents['intents'] :
    for patterns in intent['patterns'] :
        #tokenize each word
        w = nltk.word_tokenize(patterns)
        words.extend(w)

        #adding documents in corpus, adding tuple with list w and tags values
        documents.append((w,intent['tag']))

        #add to our classes list
        if intent['tag'] not in classes :
            classes.append(intent['tag'])

#lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w, 'v') for w in words if w not in ignore_words]
# words = [words.remove(i) for i in words ]
for i in words :
    if len(i) == 1 :
        words.remove(i)

words = sorted(list(set(words)))

#sort classes
classes = sorted(list(set(classes)))

#documents = combination between patterns and tag
print(len(documents), 'documents')

#classes = tag
print(len(classes), 'classes', classes)

#words = all words, vocabulary
print(len(words), 'unique lemmatized words', words)

#storing the objects we have created
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#create training and testing data
#input will be pattern and output will be class that pattern belongs to
#computer does not understand the text so we will convert text into numbers

#creating training data
training = []

#create an empty array for an output
output_empty = [0]*len(classes)

#training set , bag of words for each sentence
for doc in documents :
    #initialize bag of words
    bag = []
    #list of tokenized word for pattern
    pattern_words = doc[0]

    #lemmatize each word in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower(),'v') for word in pattern_words]

    #create bag of words array with 1, if word match found in current pattern
    for w in words :
        if w in pattern_words :
            bag.append(1)
        else :
            bag.append(0)

        #output is a 0 for each tag and 1 for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1


    training.append([bag, output_row])

#shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training,dtype= object)

#create train and test lists . X - patterns , Y - intents
X_train = list(training[:,0])
Y_train = list(training[:, 1])
print('training data created')



#Build the MODEL
'''
create the model - 3 layers. First layer 128 neurons,
second layer 64 neurons and 
3rd output layer contains number of neurons equal to number of intents to predict output intent with softmax
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation , Dropout

model = Sequential()
model.add(Dense(128,input_shape = (len(X_train[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(Y_train[0]),activation = 'softmax'))

'''
compile model. 
Stochastic gradient descent with Nesterov accelerated gradient gives godd results for this model
'''
from tensorflow.keras.optimizers import SGD
sgd = SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9 , nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(X_train), np.array(Y_train), epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbot_model.h5',hist)

print('model created')
#this model will only tell us the class that prediction belongs to















