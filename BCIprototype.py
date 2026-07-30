#BCI - brain computer interface technology. 
#Copyright (C) 2022 George Wagenkencht
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
import serial
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import time
import random
partition = 10
sampleSize = 5
com = "COM3"
baud = 9600 
option = ""
dictumSize = 5
def resetDataFile(dataFile):
    f = open(dataFile, "w", encoding="utf8")
    f.close()
def delay(ngram):
    print()
    print(ngram)
    print(3)
    time.sleep(1)
    print(2)
    time.sleep(1)
    print(1)
    time.sleep(1)
    return
def returnNgrams(data,length, mode):
    if mode == "sequential":
        ngram = ""
        pos = random.randint(1,len(data))
        n = 0
        while(n < length and pos+length < len(data)-1):
            if pos+n < len(data)-2 and pos+n > 0:
                ngram += data[pos+n] + " "
            n+=1
        delay(ngram)
        return ngram
    if mode == "random":
        ngram = ""
        pos = random.randint(1,len(data))
        n = 0
        while(n < length and pos+length < len(data)-1):
            pos = random.randint(1,len(data))
            if pos+n < len(data)-2 and pos+n > 0:
                ngram += data[pos+n] + " "
            n+=1
        delay(ngram)
        return ngram
def train(dataFile,modelName):
    with open(dataFile, encoding='ISO-8859-1') as f:
        text = f.readlines()
    varX = text[0].count(",")
    dataset = np.loadtxt(dataFile, delimiter=',',usecols = range(varX+1))
    X = dataset[:,0:varX]
    y = dataset[:,varX]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    model = Sequential()
    model.add(Dense(120, input_shape=(X_train.shape[-1],), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))
    model.save(modelName)
def recordData(ngram,stress,dataFile):#Adversarial training between easy and difficult n-grams, full 2d grapheme differentiation...
    print("recording...")
    ser = serial.Serial(com, baud, timeout = 0.1) 
    record = ""
    i = 1
    while ser.isOpen():
        var = ser.readline().decode('utf-8')
        if len(var) > 0:
            record += var.strip() + ","
            if i == partition*sampleSize:
                break
            i+=1
    record = np.array(record[:-1].split(","))
    record = record.reshape(partition, sampleSize)
    record = record.tolist()
    total = ""
    for line in record:
        total += ','.join(line) + ","+ str(stress)+"\n"  
    testX = open(dataFile, "a", encoding="utf8")
    testX.write(total)
    testX.close()
    try:
        testX = open(str(stress) + " " + ngram+".dat", "w", encoding="utf8")
        testX.write(total)
        testX.close()
    except:
        False
    return dataFile
def predict(inputFile,model):#refactor into construction using gen() input rather than record() input
    db = []
    model = keras.models.load_model(model)
    dataset = np.loadtxt(inputFile, delimiter=',')
    with open(inputFile, encoding='ISO-8859-1') as f:
        textC = f.readlines()
    varX = textC[0].count(",")
    X = dataset[:,0:varX]
    predictions = (model.predict(X)).astype(int)
    for i in range(len(predictions)):
        if predictions[i] == 0:
            db.append(str(0))
        if predictions[i] == 1:
            db.append(str(1))
        print('%s => %d' % (X[i].tolist(), predictions[i]))
    return db
while(True):
    with open("db", encoding='ISO-8859-1') as f:
        data = f.read().split(" ")
    option = input("train or predict? [t/p]:")
    if option == "t":
        resetDataFile("SignalData.csv")
        resetDataFile("StressDictum.txt")
        for i in range(sampleSize):
            recordData(returnNgrams(data,dictumSize,"random"),1, "SignalData.csv")#mode,stress,outputFile
        for i in range(sampleSize):
            recordData(returnNgrams(data,dictumSize,"sequential"),0, "SignalData.csv")#mode,stress,outputFile
        train("SignalData.csv","stress_model")
    if option == "p":
        print("press CTRL-C to exit menu.")
        while(True):
            option = input("db sample, input sample or suggest ngram? [d/i/s]:")
            ngram = ""
            if option == "d":
                ngram = returnNgrams(data,dictumSize,"sequential")
                predict(recordData(ngram,0,"X " + ngram + ".dat"),"stress_model")
            if option == "i":
                ngram = input("enter n-gram: ")
                predict(recordData(ngram,0,"X " + ngram + ".dat"),"stress_model")
            if option == "s":
                print(returnNgrams(data,dictumSize,"sequential"))