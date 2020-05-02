from transformers import *
import numpy as np
import torch
import os
config = BertConfig.from_pretrained("bert-base-uncased",
                                    output_hidden_states=True)

bert_model = BertModel.from_pretrained("bert-base-uncased",config=config)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model.eval()



text = "Here is the sentence I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs1 = bert_model(input_ids)

print(outputs1[2][1:])
print('-'*89)
distilconfig = DistilBertConfig.from_pretrained('distilbert-base-cased', output_hidden_states=True)
distil_model = DistilBertModel.from_pretrained("distilbert-base-cased",config=distilconfig)
distil_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
distil_model.eval()

tokenized_distil_text = distil_tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_distil_text)
# Batch size 1
outputs2 = distil_model(input_ids)

print(outputs2[1][1:])