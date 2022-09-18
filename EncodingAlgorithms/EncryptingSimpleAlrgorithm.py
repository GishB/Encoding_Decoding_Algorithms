#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[7]:


class SimpleLayersEncrypting:
    """SimpleLayersEncrypting class."""
    def __init__(self,
                 input_text,
                 dict_abc: list = None,
                 key_number_start: int = 5,
                 key_number_finish: int = 9,
                 number_loop: int = 10,
                 shift: int = 9,
                 shift_final: int = 5,
                 index_abc: bool = True):
        """Encrypting algorithm in 3 layers with caesar method, random ABC and handmade dict.
        Parameters
        ----------
        input_text:
            text which you are going to hide with this algorithm (str text like >> 'text')
        dict_ABC: list
            dict of words which you are going to use (list of string values like >> ['A','B',...,'Z'])
        key_number_start: int
            int number which we are using to start loop encrypting (should be between 1 and 9)
        key_number_finish: int
            int number which we are using to finish loop encrypting (should be between 1 and 9) 
        number: int
            how much times you are going to do loop encrypting (have to be >= 2)
        shift: int
            shift for caesar encrypting (have to be < lenght(dict_ABC) in simple case it is less than 31)
        shift_final: int
            shift for final output text
        index_ABC: bool
            test parameter (we use it to ban second ecnrypting layer)
        """
        self.input_text = input_text
        self.key_number_start = key_number_start
        self.key_number_finish = key_number_finish
        self.dict_abc = dict_abc
        self.number_loop = number_loop
        self.shift = shift
        self.shift_final = shift_final
        self.index_abc = index_abc
        self.new_dict_abc = None
        if dict_abc is None:
            self.dict_abc = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
                             'P','Q','R','S','T','U','V','W','Y','Z',' ',',','.','!',")",'(']
        if self.new_dict_abc is None:
            self.new_dict_abc = self.dict_abc[self.shift:] + self.dict_abc[:self.shift]

    def encrypting_caesar(self, text):
        dict_new = self.new_dict_abc
        text_crypted = []
        text = list(text)
        for i in range(len(text)):
            if text[i] in self.dict_abc:
                number = dict_new.index(text[i])
                text_crypted.append(number)
            else:
                text_crypted.append(int(39))
        return text_crypted

    def encrypting_index_ABC(self, text, index_abc):
        list_text = []
        if index_abc:
            for i in range(len(text)):
                temp_index = int(text[i])
                if temp_index < 10:
                    list_text.append(temp_index)
                elif temp_index >= 10 and temp_index < 20:
                    list_text.append(self.dict_abc[np.random.randint(0,24)] + str(temp_index%10))
                elif temp_index >= 20 and temp_index < 30:
                    list_text.append(self.dict_abc[np.random.randint(0,24)] + self.dict_abc[np.random.randint(0,24)]                                     + str(temp_index%20))
                elif temp_index >= 30:
                    list_text.append(self.dict_abc[np.random.randint(0,24)] + self.dict_abc[np.random.randint(0,24)]                                     + self.dict_abc[np.random.randint(0,24)] + str(temp_index%30))
        return list_text

    def encrypting_loop_number(self, text, key_number):
        crypted_text = []
        for i in range(len(text)):
            if len(str(text[i])) != 1:
                temp_number = self.small_encrypting_function_index_ABC(int(list(text[i])[-1]) + key_number)
                temp_str = text[i][:-1]
                crypted_text.append(temp_str + temp_number)
            else:
                crypted_text.append(self.small_encrypting_function_index_ABC(int(text[i]) + key_number))
        return crypted_text

    def small_encrypting_function_index_ABC(self, number):
        list_text = []
        if number < 10:
            list_text.append(number)
        elif number >= 10 and number < 20:
            list_text.append(self.dict_abc[np.random.randint(0, 24)] + str(number%10))
        elif number >= 20 and number < 30:
            list_text.append(self.dict_abc[np.random.randint(0, 24)]\
                             + self.dict_abc[np.random.randint(0, 24)] + str(number % 20))
        elif number >= 30:
            list_text.append(self.dict_abc[np.random.randint(0, 24)] + self.dict_abc[np.random.randint(0, 24)] +\
                             self.dict_abc[np.random.randint(0, 24)] + str(number % 30))
        if len(list_text) == 1:
            return str(list_text[0])
        return list_text

    def encrypting_run(self):
        text_initial = self.input_text
        key_number_start = self.key_number_start
        key_number_finish = self.key_number_finish
        number = self.number_loop
        text_temp = self.encrypting_index_ABC(self.encrypting_caesar(text_initial.upper()), self.index_abc)
        for i in range(number):
            if i == 0:
                text_temp = self.encrypting_loop_number(text_temp, key_number_start)
            elif i+1 == number:
                text_temp = self.encrypting_loop_number(text_temp, key_number_finish)
            else:
                text_temp = self.encrypting_loop_number(text_temp, key_number = len(text_temp))
                
        output = '0'.join(text_temp)
        output_list = []
        for i in output:
            output_list.append(str(ord(i)))
        output = int('32'.join(output_list))
        return bin(output)[2:][self.shift_final:] + bin(output)[2:][:self.shift_final]


# In[8]:


if __name__ == '__main__':
    text = str(input('Введите текст: '))
    function_simple = SimpleLayersEncrypting(text,
                                             dict_abc = None,
                                             key_number_start = 5,
                                             key_number_finish = 6,
                                             number_loop = 5,
                                             shift = 10,
                                             shift_final = 5,
                                             index_abc = True)
    result = function_simple.encrypting_run()
    print(result)


# In[ ]:




