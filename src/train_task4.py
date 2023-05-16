import argparse
import glob
import json
from pytorch_lightning import Trainer

import torch
from torch.utils.data.dataloader import DataLoader
from data_utils.data_generator import jp_tokenize
from data_utils.utils import load_data_coliee, set_random_seed
from enss import enssemble_prediction, generate_file_submission
from evaluate import evaluate
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle
import os
import random

from model.relevant_doc_retriever import RelevantDocClassifierT4 

set_random_seed(0)

def load_raw_data():
    options = argparse.Namespace(
        test_ids=['R03'],
        dev_ids=['R02', 'R01', 'R1'],
        path_folder_base='/home/phuongnm/coliee/data/COLIEE2023statute_data-English/',
        meta_data_alignment='/home/phuongnm/coliee/data/COLIEE2023statute_data-English',
        test_file= '/home/phuongnm/coliee/data/COLIEE2023statute_data-English/TestData_en.xml',
        lang = 'en' 
    )

    path_folder_base = options.path_folder_base
    if options.lang == 'en':
        options.meta_data_alignment = path_folder_base
    meta_data_alignment = options.meta_data_alignment
    if options.lang == 'jp':
        tokenizer = jp_tokenize
    else:
        tokenizer = None 
    c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info = load_data_coliee(path_folder_base=path_folder_base,
                                                                            postags_select=None,
                                                                            ids_test=options.test_ids,
                                                                            ids_dev=options.dev_ids,
                                                                            lang=options.lang,
                                                                            path_data_alignment=meta_data_alignment,
                                                                            chunk_content_info=None,
                                                                            tokenizer=tokenizer,
                                                                            test_file=options.test_file)
    return dev_q, test_q, train_q, c_docs, c_keys
   
   

def get_top_k_predictions(input_string, topk, tokenizer, model, return_str=False) -> str:

    # random token to mask 
    tokenized_inputs = tokenizer(input_string, return_tensors="pt",  padding='longest', truncation=True)
    rand_idxs = []
    for i_sent in range(tokenized_inputs["input_ids"].shape[0]):
        rand_idx = random.randint(1, sum(tokenized_inputs['attention_mask'][i_sent]) - 2)
        tokenized_inputs['input_ids'][i_sent][rand_idx] = tokenizer.mask_token_id
        rand_idxs.append(rand_idx)
        
    outputs = model(**tokenized_inputs)

    top_k_indices = torch.topk(outputs.logits, k=topk, dim=2).indices
    candidates = []

    for i_sent in range(tokenized_inputs["input_ids"].shape[0]):
        masked_token_idx = rand_idxs[i_sent]
        for i_top_k in range(top_k_indices.shape[2]):
            tokenized_inputs['input_ids'][i_sent][masked_token_idx] = top_k_indices[i_sent][masked_token_idx][i_top_k]
            candidates.append(torch.clone(tokenized_inputs['input_ids'][i_sent]))
    
    if return_str:
        return tokenizer.batch_decode(candidates, skip_special_tokens=True)
    else: 
        tokenized_inputs['input_ids'] = torch.stack(candidates).int()
        return tokenized_inputs

class T3ColieePreprocessor:
    def __init__(self, tokenizer, max_seq_length) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, mini_batch):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)

        questions = [e[0] for e in mini_batch]
        c_codes = [e[1] for e in mini_batch]
        input_text_pair_ids = self.tokenizer(questions, c_codes, padding='longest', 
                                    max_length=max_seq_length, truncation=True, return_tensors='pt')

        labels = torch.LongTensor([1 if e[2] else 0 for e in mini_batch])

        return ({'input_text_pair_ids': input_text_pair_ids}, labels, ["FakeQid"]*len(questions), None)    
    
class ColieePreprocessor:
    def __init__(self,  text_generator, tokenizer, max_seq_length, is_generate_new_sample=False) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.raw_dev, self.raw_test, self.raw_train, self.c_docs, self.c_keys = load_raw_data()
        self.c_docs = [e.split("\n\n\n")[0] for e in self.c_docs]
        self.text_generator = text_generator

        tokenizer = tokenizer  
        text_generator = text_generator  
        self.is_generate_new_sample = is_generate_new_sample

    def __call__(self, mini_batch):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)
        all_generate_setting = [0]*2 + [3]*3 + [6]*5
        # print(all_generate_setting)
        generation_setting = random.choice(all_generate_setting)

        def get_sumary_civilcode(c_ids):
            return_str = ""
            for c_id in c_ids:
                c_idx = self.c_keys.index(c_id)
                return_str += " " + self.c_docs[c_idx]
            return return_str

        question_ids = [e['index'] for e in mini_batch]
        questions = [e['content']+ " " + self.tokenizer.sep_token + " " + get_sumary_civilcode(e['result']) for e in mini_batch]
        if self.is_generate_new_sample and generation_setting > 0:
            for i in range(generation_setting-1):
                questions = get_top_k_predictions(questions, 1, self.tokenizer, self.text_generator, return_str=True)
            input_text_pard_ids = get_top_k_predictions(questions, 1, self.tokenizer, self.text_generator, return_str=False)
        else:
            input_text_pard_ids = self.tokenizer(questions, return_tensors="pt",  padding='longest', truncation=True)

        labels = torch.LongTensor([(e['label'].lower())=='y' for e in mini_batch])

        return ({'input_text_pair_ids': input_text_pard_ids}, labels, question_ids, None)

if __name__=="__main__":

    # training+model args
    parser = argparse.ArgumentParser(description="Training Args")
    parser = RelevantDocClassifierT4.add_model_specific_args(parser)
        
    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--log_dir", type=str, default=".", help="log dir")
    parser.add_argument("--max_keep_ckpt", default=1, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--pretrained_checkpoint", default=None, type=str, help="pretrained checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--model_name_or_path", type=str, help="pretrained model name or path")
    parser.add_argument("--ignore_index",  type=int, default=-100)
    parser.add_argument("--max_epochs", default=5, type=int, help="Max training epochs.")
    parser.add_argument("--max_seq_length",  type=int, default=512, help="Max seq length for truncating.")
    parser.add_argument("--no_train", action="store_true", default=False, help="Do not training.")
    parser.add_argument("--no_test", action="store_true", default=False, help="Do not test.")
    parser.add_argument("--no_dev", action="store_true", default=False, help="Do not dev at last.")
    parser.add_argument("--gpus", nargs='+', default=[0], type=int, help="Id of gpus for training")
    parser.add_argument("--ckpt_steps", default=1000, type=int, help="number of training steps for each checkpoint.")
    parser.add_argument("--file_output_id", default="allEnss", type=str, help="Id of submission")
    parser.add_argument("--civi_code_path", default="data/parsed_civil_code/en_civil_code.json", type=str, help="civil code path")
    parser.add_argument("--main_enss_path", default="settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_{}_5_80_0015.txt", type=str, help="Id of submission")

    opts = parser.parse_args()
    if opts.pretrained_checkpoint is not None and not opts.pretrained_checkpoint.endswith(".ckpt"):
        opts.pretrained_checkpoint = glob.glob(f"{opts.pretrained_checkpoint}/*.ckpt")[0]
        print(f"Found checkpoint - {opts.pretrained_checkpoint}")

    # load pretrained_checkpoint if it is set 
    if opts.pretrained_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(opts.log_dir, use_fast=True, config=AutoConfig.from_pretrained(opts.log_dir))
        model = RelevantDocClassifierT4.load_from_checkpoint(opts.pretrained_checkpoint, args=opts)
        max_seq_length=model.args.max_seq_length
    else:
        config = AutoConfig.from_pretrained(opts.model_name_or_path)
        config.save_pretrained(opts.log_dir)
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path, use_fast=True, max_seq_length=opts.max_seq_length)
        tokenizer.save_pretrained(opts.log_dir)
        max_seq_length=opts.max_seq_length

    #
    # Data loader 

    df_train_pair = pd.read_csv('/home/phuongnm/coliee/data/gen_df_from_T5_46K.csv', sep=',')
    preprocess_data = [] 
    for e in df_train_pair.values:
        preprocess_data.append({
            'result': [],
            'content': e[0],
            'index': 'FakeQid',
            'label': 'Y' if e[2] else 'N'
            })


    coliee_data_preprocessor = ColieePreprocessor(tokenizer=tokenizer, 
                                                  max_seq_length=max_seq_length, 
                                                  text_generator=AutoModelForMaskedLM.from_pretrained(opts.model_name_or_path),
                                                  is_generate_new_sample=True)
    
    coliee_data_preprocessor_eval = ColieePreprocessor(tokenizer=tokenizer, 
                                                       max_seq_length=max_seq_length, 
                                                       text_generator=None, 
                                                       is_generate_new_sample=False)
    
    train_loader = DataLoader(coliee_data_preprocessor.raw_train, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)
    dev_loader = DataLoader(coliee_data_preprocessor.raw_dev, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor_eval, shuffle=True)
    test_loader = DataLoader(coliee_data_preprocessor.raw_test, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor_eval, shuffle=True)
    # df_test2 = pd.read_csv(f"{opts.data_dir}/test_submit.csv")
    # test2_loader = DataLoader(df_test2.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)

    # model 
    if not opts.pretrained_checkpoint: 
        model = RelevantDocClassifierT4(opts, data_train_size=len(train_loader))
    else:
        model.data_train_size = len(train_loader)
    
    # trainer
    checkpoint_callback = ModelCheckpoint(dirpath=opts.log_dir, save_top_k=opts.max_keep_ckpt, 
                                          auto_insert_metric_name=True, mode="max", monitor="dev/valid_p", 
                                        #   every_n_train_steps=opts.ckpt_steps
                                          )
    trainer = Trainer(max_epochs=opts.max_epochs, 
                      accelerator='gpu' if len(opts.gpus) > 0 else 'cpu', 
                      devices=opts.gpus, 
                      callbacks=[checkpoint_callback], 
                      default_root_dir=opts.log_dir, 
                      val_check_interval=0.1
                      )

    if not opts.no_train:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)

    # for enssemble multi checkpoints 
    pretrained_checkpoint_list = glob.glob(f"{opts.log_dir}/*.ckpt") 
    model.result_logger.info(pretrained_checkpoint_list)
    data_sources = []
    if not opts.no_dev: data_sources.append(dev_loader)
    if not opts.no_test: data_sources.append(test_loader);  # data_sources.append(test2_loader)

    outenss = []
    for data_loader in data_sources:  
        # trainer.test(model=model, dataloaders=data_loader, ckpt_path=checkpoint_callback.best_model_path)
        outenss += trainer.predict(model=model, dataloaders=data_loader, ckpt_path='settings_t4/legalbert_sum_multigen2_dr0.3_l20.1_froze_E100Seq512L5e-6/models/epoch=83-step=1848.ckpt')
    all_label = []
    all_q_id = []
    for e in outenss:
        # all_label += e['labels']
        all_q_id += e['question_ids']

    all_label = torch.cat([torch.argmax(batch_output['y_hat'], dim=1) for batch_output in outenss],  dim=0)
    result_combine = {}
    for id_check in ["R01","R1", "R02",   "R04"]:
        for i, q_id in enumerate(all_q_id):
            if id_check in q_id and id_check not in result_combine:
                result_combine[id_check] = []
            if id_check in q_id: 
                result_combine[id_check].append((q_id, all_label[i].item()==1))

    result_combine["R01"] =  result_combine["R01"] + result_combine["R1"] 
    for id_check in ["R01", "R02",   "R04"]:
        lines = []
        for pred in result_combine[id_check]:
            key_s = pred[0]
            pred_v = "Y" if pred[1] ==1 else "N"
            lines.append(f"{key_s} {pred_v} CAPTAIN")

        with open(f"CAPTAIN.gen.{id_check}.txt", "wt") as f:
            f.write("\n".join(lines))


