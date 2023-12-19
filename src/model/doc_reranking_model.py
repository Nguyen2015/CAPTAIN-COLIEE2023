import argparse
import logging
import os
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

from .relevant_doc_retriever import RelevantDocClassifier
 

class DocRerankingModel(BertPreTrainedModel):
    def __init__(self, config, dropout, freeze_bert=False):
        # init model 
        super().__init__(config)
        self.config = config
        self.freeze_bert = freeze_bert

        self.bert = BertModel(config=self.config)    # Load pretrained bert
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        self.dropout = nn.Dropout(p=dropout, inplace=False)

        self.multihead_combination = nn.MultiheadAttention(self.config.hidden_size, num_heads=8)
        self.w_q = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.w_k = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.w_v = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.classifier_rerank = nn.Linear(in_features=self.config.hidden_size, out_features=1, bias=True)

    def forward(self, input_text_pair_ids, n_relevant_docs):
        if self.freeze_bert:
            with torch.no_grad():
                outputs = self.bert(**input_text_pair_ids)
        else:
            outputs = self.bert(**input_text_pair_ids)

        # h_other_wordpieces = outputs[0]
        h_cls = outputs[1]  # [CLS]
        output_logits = None
        for indices in n_relevant_docs:
            h_cls_one_query = self.dropout(torch.index_select(h_cls,
                                                              0, 
                                                              indices ) )
            output_docs_inter = self.multihead_combination(self.w_q(h_cls_one_query),
                                                            self.w_k(h_cls_one_query),
                                                            self.w_v(h_cls_one_query)
                                                            )
            cur_logits = self.classifier_rerank(output_docs_inter[0]).squeeze()
            output_logits = cur_logits if output_logits is None else torch.cat((output_logits, cur_logits), dim=0)

        return torch.sigmoid(output_logits)

    
class RelevantDocReranker(RelevantDocClassifier):
    def __init__(self,
        args: argparse.Namespace,
        data_train_size=None) -> None:

        """Initialize."""
        super().__init__(args, data_train_size) 

        # init model 
        # Load config from pretrained name or path 
        self.model = DocRerankingModel.from_pretrained(self.args.model_name_or_path, dropout=self.args.dropout)
        self.loss_function = torch.nn.BCELoss()

    def validation_step(self, batch, batch_idx):
        model_inputs, labels, question_ids, c_ids  = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        y_hat = F.pad(y_hat.unsqueeze(-1), (1,0), "constant", 0.5)
        self.log("val_batch_loss",loss, prog_bar=True)
        return {'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels, 'question_ids': question_ids, 'c_ids': c_ids}

    def predict_step(self, batch, batch_idx):
        model_inputs, labels, question_ids, c_ids  = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        y_hat = F.pad(y_hat.unsqueeze(-1), (1,0), "constant", 0.5)
        return {'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels, 'question_ids': question_ids, 'c_ids': c_ids}