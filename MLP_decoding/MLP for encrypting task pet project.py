#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import EncryptingSimpleAlrgorithm as ens #to create encrypted texts

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dropout, Dense, GlobalMaxPool1D, Embedding, Activation


# # Функции для оценки качества модели

# In[3]:


def edited_list_function(take):
  new_list = []
  for i in take:
    temp_list = []
    for g in i:
      if g > 0.5:
        temp_list.append(1)
      else:
        temp_list.append(0)
    new_list.append(temp_list)
  return np.array(new_list)

def accuracy_custom(y_true,y_score):
  if np.min(y_score) < 0:
    y_score = edited_list_function(y_score)
  true_labels_count = []
  for g in range(len(y_true[0])):
    count = 0
    for i in range(len(y_true)):
      if y_true[i][g] == y_score[i][g]:
        count += 1
    true_labels_count.append(count)
  number_test = len(y_score)

  list_accuracy_custom = []
  for i in range(len(true_labels_count)):
    temp = int((true_labels_count[i]/number_test)*100)
    list_accuracy_custom.append(temp)

  return np.array(list_accuracy_custom)


def accuracy_average(y_true,y_score):
  temp = accuracy_custom(y_true,y_score)
  return np.mean(temp)


def true_positive(y_true,y_score):
  if np.min(y_score) < 0:
    y_score = edited_list_function(y_score)

  true_positive_list = []
  for g in range(len(y_true[0])):
    count = 0
    for i in range(len(y_true)):
      if y_true[i][g] == 1:
        if y_score[i][g] == 1:
          count += 1
    true_positive_list.append(count)
  return true_positive_list

def false_positive(y_true,y_score):
  if np.min(y_score) < 0:
    y_score = edited_list_function(y_score)
  false_positive_list = []
  for g in range(len(y_true[0])):
    count = 0
    for i in range(len(y_true)):
      if y_true[i][g] != 1:
        if y_score[i][g] == 1:
          count += 1
    false_positive_list.append(count)
  return false_positive_list

def true_negative(y_true,y_score):
  if np.min(y_score) < 0:
    y_score = edited_list_function(y_score)
  true_negative_list = []
  for g in range(len(y_true[0])):
    count = 0
    for i in range(len(y_true)):
      if y_true[i][g] == 0:
        if y_score[i][g] == 0:
          count += 1
    true_negative_list.append(count)
  return true_negative_list

def false_negative(y_true,y_score):
  if np.min(y_score) < 0:
    y_score = edited_list_function(y_score)

  false_negative_list = []
  for g in range(len(y_true[0])):
    count = 0
    for i in range(len(y_true)):
      if y_true[i][g] != 0:
        if y_score[i][g] == 0:
          count += 1
    false_negative_list.append(count)
  return false_negative_list

def recall_custom(y_true,y_score):

  if np.min(y_score) < 0:
    y_score = edited_list_function(y_score)

  tp = true_positive(y_true,y_score)
  fn = false_negative(y_true,y_score)
  recall_list = []
  for i in range(len(tp)):
    try:
      recall = tp[i]/(tp[i]+fn[i])
      recall_list.append(recall)
    except:
      recall_list.append('NaN')
      print('Деление на 0 - ошибка! Recall')
  return recall_list

def precision_custom(y_true,y_score):

  if np.min(y_score) < 0:
    y_score = edited_list_function(y_score)

  tp = true_positive(y_true,y_score)
  fp = false_negative(y_true,y_score)
  precision_list = []
  for i in range(len(tp)):
    try:
      recall = tp[i]/(tp[i]+fp[i])
      precision_list.append(recall)
    except:
      precision_list.append('NaN')
      print('Деление на 0 - ошибка! Precision')
  return precision_list


def f1_custom(y_true,y_score):

  if np.min(y_score) < 0:
    y_score = edited_list_function(y_score)

  recall = recall_custom(y_true,y_score)
  precision = precision_custom(y_true,y_score)
  f1_list = []
  for i in range(len(recall)):
    try:
      f1 = 2*precision[i]*recall[i]/(precision[i] + recall[i])
      f1_list.append(f1)
    except:
      f1_list.append('NaN')
      print('Деление на 0 - ошибка! F1')
  return np.array(f1_list)


# # Генерируем синтетические данные и сохраняем

# In[4]:


if __name__ == "__main__":
    #encrypting text generatior
    def generator_encrypting_text(text):
        function_simple = ens.SimpleLayersEncrypting(text,
                                             dict_abc = None,
                                             key_number_start = 5,
                                             key_number_finish = 6,
                                             number_loop = 5,
                                             shift = 10,
                                             shift_final = 5,
                                             index_abc = True)
        result = function_simple.encrypting_run()
        true_labels.append(str(text))
        input_data.append(str(result))

    true_labels = []
    input_data = []

    list_texts = ['true','false','test','aleksandr']
    for g in list_texts:
        for i in range(2000):
            generator_encrypting_text(g)


# In[5]:


list_texts = ['true','false','test','aleksandr']
for g in list_texts:
    for i in range(2000):
        generator_encrypting_text(g)


# # Подготовка синтетических данных

# In[6]:


size_list = []
for i in input_data:
    size_list.append(len(i))
max_size = max(size_list)


# In[7]:


prepared_input_data = []
for i in range(len(input_data)):
    if size_list[i] != max_size:
        diff = max_size - size_list[i]
        prepared_input_data.append(list(map(int,input_data[i])) + list(np.arange(diff) * 0))
    else:
        prepared_input_data.append(list(map(int,input_data[i])))


# In[8]:


df_y = pd.DataFrame(columns=['A'])
df_y['A'] = true_labels
Y = np.array(pd.get_dummies(df_y, columns=['A']))


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(prepared_input_data, Y, test_size=0.2, random_state=2022)


# # Создаем модель нейронной сети - архитектура MLP, 5 слоев

# In[10]:


model_MLP = Sequential()

model_MLP.add(Dense(1024, input_dim=max_size))

model_MLP.add(Activation('relu'))
model_MLP.add(Dropout(0.3))
model_MLP.add(Dense(1024))
model_MLP.add(BatchNormalization())

model_MLP.add(Activation('relu'))
model_MLP.add(Dropout(0.3))
model_MLP.add(Dense(1024/2))
model_MLP.add(BatchNormalization())

model_MLP.add(Activation('relu'))
model_MLP.add(Dropout(0.2))
model_MLP.add(Dense(1024/4))
model_MLP.add(BatchNormalization())

model_MLP.add(Activation('relu'))
model_MLP.add(Dropout(0.2))
model_MLP.add(Dense(1024/8))
model_MLP.add(BatchNormalization())

model_MLP.add(Activation('softmax'))
model_MLP.add(Dropout(0.1))
model_MLP.add(Dense(len(y_test[0]), Activation('softmax')))

model_MLP.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
model_MLP.summary()


# # Обучаем модель

# In[11]:


model_MLP.fit(np.array(X_train), np.array(y_train), batch_size=256, epochs=100, validation_data=(np.array(X_test), np.array(y_test)))


# # Проверяем модель MLP на тестовых данных

# In[12]:


predicted = np.array(edited_list_function(model_MLP.predict(X_test)))


# In[13]:


print(f'{f1_custom(y_test,predicted)} F1 score по labels')

print(f'{true_positive(y_test,predicted)} TP по labels')
print(f'{true_negative(y_test,predicted)} TN по labels')
print(f'{false_positive(y_test,predicted)} FP по labels')
print(f'{false_negative(y_test,predicted)} FN по labels')

print(f'{precision_custom(y_test,predicted)} precision по labels')
print(f'{recall_custom(y_test,predicted)} recall по labels')
print(f'{accuracy_custom(y_test,predicted)} accuracy по labels')
print(f'{accuracy_average(y_test,predicted)} average accuracy')


# # Ручная проверка

# In[62]:


if __name__ == "__main__":
    test_text = 'test'
    print(test_text)
    code = ens.SimpleLayersEncrypting(test_text, dict_abc = None,
                                        key_number_start = 5,
                                        key_number_finish = 6,
                                        number_loop = 5,
                                        shift = 10,
                                        shift_final = 5,
                                        index_abc = True)\
                    .encrypting_run()
    print(code)

    diff = max_size - len(code)
    if diff != 0:
        new_code = list(map(int,code)) + list(np.arange(diff) * 0)

    prediction_validate = model_MLP.predict([new_code,list(np.arange(751) * 1)])

    for i in range(len(prediction_validate)):
        if (max(prediction_validate[i]) >= 0.55):
            index = list(prediction_validate[i]).index(max(prediction_validate[i]))
            print(list_texts[index])
        else:
            print('Вектор np.array([0,1,2,3,4,5....751]) - для примера')
            print(list(np.arange(7) * 1))
            print('Нет правильного ответа')



