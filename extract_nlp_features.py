# from utils.bert_utils import get_bert_layer_representations
# from utils.xl_utils import get_xl_layer_representations
from transformers import *
from utils.demo_utils import get_model_layer_representations

import time as tm
import numpy as np
import torch
import os
import argparse

MODELS = [ (GPT2Model, GPT2Tokenizer,  'distilgpt2'          , GPT2Config),
           (GPT2Model,       GPT2Tokenizer,        'gpt2'                , GPT2Config)]
def save_layer_representations(model_layer_dict, model_name, seq_len, save_dir):
    for layer in model_layer_dict.keys():
        np.save('{}/{}_length_{}_layer_{}.npy'.format(save_dir,model_name,seq_len,layer+1),np.vstack(model_layer_dict[layer]))  
    print('Saved extracted features to {}'.format(save_dir))
    return 1

                
# model_options = ['bert','transformer_xl','elmo','use']        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nlp_model", default='bert', choices=model_options)                
    parser.add_argument("--sequence_length", type=int, default=1, help='length of context to provide to NLP model (default: 1)')
    #parser.add_argument("--output_dir", required=True, help='directory to save extracted representations to')

    args = parser.parse_args()
    # print(args)
    
    text_array = np.load(os.getcwd() + '/data/stimuli_words.npy')
    remove_chars = [",","\"","@"]
    
    for model_class, tokenizer_class, pretrained_weights, config_class in MODELS:
        word_ind_to_extract = -2
        nlp_features = get_model_layer_representations(model_class,config_class,tokenizer_class,pretrained_weights,args.sequence_length, text_array, remove_chars, word_ind_to_extract)
        output_dir = './nlp_features/'+pretrained_weights
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)          
              
        save_layer_representations(nlp_features, pretrained_weights, args.sequence_length, output_dir)
    # if args.nlp_model == 'bert':
    #     # the index of the word for which to extract the representations (in the input "[CLS] word_1 ... word_n [SEP]")
    #     # for CLS, set to 0; for SEP set to -1; for last word set to -2
    #     word_ind_to_extract = -2
    #     nlp_features = get_bert_layer_representations(args.sequence_length, text_array, remove_chars, word_ind_to_extract)
    # elif args.nlp_model == 'transformer_xl':
    #     word_ind_to_extract = -1
    #     nlp_features = get_xl_layer_representations(args.sequence_length, text_array, remove_chars, word_ind_to_extract)
    # # elif args.nlp_model == 'elmo':
    # #     word_ind_to_extract = -1
    # #     nlp_features = get_elmo_layer_representations(args.sequence_length, text_array, remove_chars, word_ind_to_extract)
    # # elif args.nlp_model == 'use':
    # #     nlp_features = get_use_layer_representations(args.sequence_length, text_array, remove_chars)
    # else:
    #     print('Unrecognized model name {}'.format(args.nlp_model))
        
        
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)          
              
    # save_layer_representations(nlp_features, args.nlp_model, args.sequence_length, args.output_dir)
        
        
        
        
    
    
    

    
