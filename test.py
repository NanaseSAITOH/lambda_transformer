# coding: utf-8
import json
import os
import ctypes
from learnning_model.PredictTaskExecutor import PredictTaskExecutor


def split_blank(text):
    text_list = text.split()
    return text_list

predictTaskExecutor = PredictTaskExecutor()
decode_sentence = predictTaskExecutor.main("テストです")
print(decode_sentence)
