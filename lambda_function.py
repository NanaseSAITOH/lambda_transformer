# coding: utf-8
import json
import os
import ctypes
from learnning_model.PredictTaskExecutor import PredictTaskExecutor


def lambda_handler(event, context):
    req_body = json.loads(event['body'])
    sentence = req_body['text']
    if sentence == "test":
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': 'test'
        }
    predictTaskExecutor = PredictTaskExecutor()
    decode_sentence = predictTaskExecutor.main(sentence)

    res_body = {'encode_sentence': sentence,'decode_sentence': decode_sentence}
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(res_body)
    }
