import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import sys
import os
sys.path.append( os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "CheXbert", "src"))

from label import collate_fn_no_labels
from models.bert_labeler import bert_labeler
from constants import CONDITIONS, CLASS_MAPPING
from transformers import BertTokenizer
import bert_tokenizer
import utils

class CheXbertScorer:
    
    def __init__(self, batch_size=32, device='cuda'):
        
        self.batch_size = batch_size
        self.device = device
        
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        checkpoint_path = os.path.join(root_dir, "CheXbert/models/chexbert.pth")
        
        model = bert_labeler()
        model = nn.DataParallel(model) #to utilize multiple GPU's
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def cuda(self):
        self.device = 'cuda'
        self.model.cuda()
        return self
        
    def predict(self, sentences):
        ''' 
        Predict labels for sentences
        Arguments
            sentences: list of N strings
        Returns
            output: torch tensor of shape (N, 17)
        '''
        
#         print('CHEXBERT: got ', len(sentences), 'sentences')
        
        d = DataLoader(sentences, batch_size=self.batch_size, drop_last=False)
        
        outputs = []
        
        for sentence_batch in d:
#             print('batch of size ', len(sentence_batch))
#         for i in range((len(sentences)+self.batch_size-1)//self.batch_size):
            
#             sentence_batch = sentences[i*self.batch_size : min((i+1)*self.batch_size, len(sentences))]
            
            df = pd.DataFrame(data={'findings': sentence_batch})
            imp = df['findings']
            imp = imp.str.strip()
            imp = imp.replace('\n',' ', regex=True)
            imp = imp.replace('\s+', ' ', regex=True)
            imp = imp.str.strip()

            encoded_imp = bert_tokenizer.tokenize(imp, self.tokenizer)

            # Create sample batch in desired format for CheXbert model
            sample = [torch.LongTensor(imp) for imp in encoded_imp]
            sample = [{"imp": imp, "len": imp.shape[0]} for imp in sample]
            sample = collate_fn_no_labels(sample)

            batch = sample['imp']
            batch = batch.to(self.device)

            src_len = sample['len']
            attn_mask = utils.generate_attention_masks(batch, src_len, self.device)

            with torch.no_grad():
                out = self.model(batch, attn_mask) # [14, torch.tensor(N, 4)]

            result = [cond.argmax(dim=1) for cond in out]
            output = torch.stack(result, dim=0).T # Nx17
            
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=0)
        
#         print('output shape: ', outputs.shape)
        
        return outputs
    
    def reward(self, cands, refs):
        '''
        Get RL reward for predicted and ground-truth sentences
        Arguments
            cands: candidate (predicted) sentences of length N
            refs: reference (ground-truth) sentences of length N
        Returns
            score: F1 score RL reward 
            each_f1: list of tuples (precision, recall, F1) for each condition
        '''
        assert len(cands) == len(refs)
        
        output = self.predict(cands)
        target = self.predict(refs)
        
        N, num_conds = output.shape
        
        each_f1 = []
        each_score = []
        
        total_num_predict_pos = 0
        total_true_positive = 0
        total_num_true_pos = 0
        
        
        for i in range(N):
#             output_cond = output[:,c]
#             target_cond = target[:,c]
            
#             f1_score(output_cond, target_cond, average='micro')
            num_predict_pos = 0
            true_positive = 0

            for c in range(num_conds):
                if output[i][c] == 1:
                    num_predict_pos += 1
                    total_num_predict_pos += 1
                    if target[i][c] == 1:
                        true_positive += 1
                        total_true_positive += 1
                        
            if num_predict_pos == 0:
                precision = "Not defined"
            else:
                precision = true_positive/num_predict_pos
                
            num_true_pos = 0
            for c in range(num_conds):
                if target[i][c] == 1:
                    num_true_pos += 1
                    total_num_true_pos += 1
                
            if num_true_pos == 0:
                recall = "Not defined"
            else:
                recall = true_positive/num_true_pos
            
            if precision == "Not defined" or recall == "Not defined" or precision==0.0 or recall==0.0:
                F1 = 0 # "Not defined"
            else:
                F1 = (2*(precision*recall))/(precision+recall)
                
            each_score.append(F1)
            
# -------------------------------
        
#         for c in range(num_conds):
#             output_cond = output[:,c]
#             target_cond = target[:,c]
            
# #             f1_score(output_cond, target_cond, average='micro')
#             num_predict_pos = 0
#             true_positive = 0

#             for i in range(N):
#                 if output[i][c] == 1:
#                     num_predict_pos += 1
#                     total_num_predict_pos += 1
#                     if target[i][c] == 1:
#                         true_positive += 1
#                         total_true_positive += 1
                        
#             if num_predict_pos == 0:
#                 precision = "Not defined"
#             else:
#                 precision = true_positive/num_predict_pos
                
#             num_true_pos = 0
#             for i in range(N):
#                 if target[i][c] == 1:
#                     num_true_pos += 1
#                     total_num_true_pos += 1
                
#             if num_true_pos == 0:
#                 recall = "Not defined"
#             else:
#                 recall = true_positive/num_true_pos
            
#             if precision == "Not defined" or recall == "Not defined" or precision==0.0 or recall==0.0:
#                 F1 = "Not defined"
#             else:
#                 F1 = (2*(precision*recall))/(precision+recall)
                
#             each_f1.append( (precision, recall, F1) )

# --------------------------------
                
#             for index, row in glabels.iterrows():
#                 if row[number]==1.0:
#                     num_predict_pos += 1
#                     total_num_predict_pos += 1
#                     if tlabels.iloc[index][number]==1.0:
#                         true_positive += 1
#                         total_true_positive += 1
#             if num_predict_pos == 0:
#                 print(number)
#                 precision = "Not defined"
#             else:
#                 precision = true_positive/num_predict_pos

#             # get recall for specific column

#             num_true_pos = 0

#             for index, row in tlabels.iterrows():
#                 if row[number]==1.0:
#                     num_true_pos += 1
#                     total_num_true_pos += 1

#             if num_true_pos == 0:
#                 print(number)
#                 recall = "Not defined"
#             else:
#                 recall = true_positive/num_true_pos

#             if precision == "Not defined" or recall == "Not defined" or precision==0.0 or recall==0.0:
#                 F1 = "Not defined"
#             else:
#                 F1 = (2*(precision*recall))/(precision+recall)
            
        total_precision = total_true_positive/total_num_predict_pos
        total_recall = total_true_positive/total_num_true_pos
        total_F1 = (2*(total_precision*total_recall))/(total_precision+total_recall)

#         print('each_score shape: ', len(each_score))
#         print(each_score)
        
        return total_F1, each_score
        
        
        
        
        
        