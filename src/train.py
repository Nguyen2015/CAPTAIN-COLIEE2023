import argparse
import glob
import json
from flask import Flask
from pytorch_lightning import Trainer

import torch
from torch.utils.data.dataloader import DataLoader
from enss import enssemble_prediction, generate_file_submission
from evaluate import evaluate
from model import RelevantDocClassifier
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoTokenizer, AutoConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle
import os
from data_generator import *
from server import *

class ColieePreprocessor:
    def __init__(self, tokenizer, max_seq_length) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, mini_batch):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)

        question_ids = [e[1] for e in mini_batch]
        c_ids = [e[2] for e in mini_batch]
        questions = [e[3] for e in mini_batch]
        c_codes = [e[4] for e in mini_batch]
        input_text_pair_ids = self.tokenizer(questions, c_codes, padding='max_length', 
                                    max_length=max_seq_length, truncation=True, return_tensors='pt')

        labels = torch.LongTensor([e[0] for e in mini_batch])

        return ({'input_text_pair_ids': input_text_pair_ids}, labels, question_ids, c_ids)

if __name__=="__main__":

    # training+model args
    parser = argparse.ArgumentParser(description="Training Args")
    parser = RelevantDocClassifier.add_model_specific_args(parser)
        
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
    parser.add_argument("--infer_train", action="store_true", default=False, help="predict on train data.")
    parser.add_argument("--no_test", action="store_true", default=False, help="Do not test.")
    parser.add_argument("--no_dev", action="store_true", default=False, help="Do not dev at last.")
    parser.add_argument("--gpus", nargs='+', default=[0], type=int, help="Id of gpus for training")
    parser.add_argument("--ckpt_steps", default=1000, type=int, help="number of training steps for each checkpoint.")
    parser.add_argument("--file_output_id", default="allEnss", type=str, help="Id of submission")
    # parser.add_argument("--civi_code_path", default="data/parsed_civil_code/en_civil_code.json", type=str, help="civil code path")
    parser.add_argument("--main_enss_path", default=None, type=str, help="main pred from monoT5")
    parser.add_argument("--run_server", action="store_true", default=False, help="run demo server.")
    parser.add_argument("--port", type=int, default=9002)

    opts = parser.parse_args()
    if opts.pretrained_checkpoint is not None and not opts.pretrained_checkpoint.endswith(".ckpt"):
        opts.pretrained_checkpoint = glob.glob(f"{opts.pretrained_checkpoint}/*.ckpt")[0]
        print(f"Found checkpoint - {opts.pretrained_checkpoint}")

    # load pretrained_checkpoint if it is set 
    if opts.pretrained_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(opts.log_dir, use_fast=True, config=AutoConfig.from_pretrained(opts.log_dir))
        model = RelevantDocClassifier.load_from_checkpoint(opts.pretrained_checkpoint, args=opts)
        max_seq_length=model.args.max_seq_length
    else:
        if os.path.exists(f"{opts.log_dir}/config.json"):
            config = AutoConfig.from_pretrained(opts.log_dir)
        else:
            config = AutoConfig.from_pretrained(opts.model_name_or_path)
            config.save_pretrained(opts.log_dir)
            
        if os.path.exists(f"{opts.log_dir}/tokenizer_config.json"):
            # tokenizer = AutoTokenizer.from_pretrained(opts.log_dir, use_fast=True, config=AutoConfig.from_pretrained(opts.log_dir))
            tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path, use_fast=True, max_seq_length=opts.max_seq_length)
        else:
            tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path, use_fast=True, max_seq_length=opts.max_seq_length)
            tokenizer.save_pretrained(opts.log_dir)
        max_seq_length=opts.max_seq_length

    #
    # Data loader 
    coliee_data_preprocessor = ColieePreprocessor(tokenizer, max_seq_length=max_seq_length)
    df_train = pd.read_csv(f"{opts.data_dir}/train.csv")
    train_loader = DataLoader(df_train.values, batch_size=48, collate_fn=coliee_data_preprocessor, shuffle=True)
    df_dev = pd.read_csv(f"{opts.data_dir}/dev.csv")
    dev_loader = DataLoader(df_dev.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)
    df_test = pd.read_csv(f"{opts.data_dir}/test.csv")
    test_loader = DataLoader(df_test.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)
    test2_loader = None
    if os.path.exists(f"{opts.data_dir}/test_submit.csv"):
        df_test2 = pd.read_csv(f"{opts.data_dir}/test_submit.csv")
        test2_loader = DataLoader(df_test2.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)
    test3_loader = None
    if os.path.exists(f"{opts.data_dir}/test_submit3.csv"):
        df_test3 = pd.read_csv(f"{opts.data_dir}/test_submit3.csv")
        test3_loader = DataLoader(df_test3.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)

    # model 
    if not opts.pretrained_checkpoint: 
        model = RelevantDocClassifier(opts, data_train_size=len(train_loader))
    else:
        model.data_train_size = len(train_loader)
    
    # trainer
    model_id = opts.model_name_or_path.replace("/", '--') 
    checkpoint_callback = ModelCheckpoint(dirpath=opts.log_dir, save_top_k=opts.max_keep_ckpt, 
                                          auto_insert_metric_name=True, mode="max", monitor="dev/valid_f2", 
                                          filename=model_id+'-{epoch}-{step}',
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
    if not opts.no_test: 
        data_sources.append(test_loader)
        if test2_loader is not None:
            data_sources.append(test2_loader)
        if test3_loader is not None:
            data_sources.append(test3_loader)
    if opts.infer_train: data_sources.append(train_loader)
    
    # for server 
    if opts.run_server:
        
        print(pretrained_checkpoint_list)
        tfidf_vectorizer = pickle.load(open(f"{opts.data_dir}/tfidf_classifier.pkl", "rb"))

        _all_data = json.load(open(f"{opts.data_dir}/all_data.json", 'rt', encoding='utf8'))
        all_civil_code = list(zip( _all_data['c_keys'], _all_data['c_docs']))
        all_query_jp = dict([(e['index'], (e['content'], e['result'])) for e in _all_data['dev_q'] + _all_data['test_q'] + _all_data['train_q']])
        
        eng_data = json.load(open(f'{opts.data_dir}/../../COLIEE2023statute_data-English/data_en_topk_150_r02_r03/all_data.json'))
        all_civil_code_en = dict(list(zip( eng_data['c_keys'], eng_data['c_docs'])))
        all_query_en = dict([(e['index'], (e['content'], e['result'])) for e in eng_data['dev_q'] + eng_data['test_q'] + eng_data['train_q']])
        sentence2qid = dict([(q_info[0], qid) for qid, q_info in all_query_jp.items()])
        
        LLM_MODEL_NAME = 'google/flan-t5-xxl'
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL_NAME, device_map="auto",  torch_dtype=torch.float16, load_in_8bit=True, cache_dir="../.cache/huggingface/hub",
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
        run_server(PORT=opts.port, **{
            'all_query_en': all_query_en, 
            'sentence2qid': sentence2qid, 
            'all_query_jp': all_query_jp, 
            'all_civil_code': all_civil_code, 
            'all_civil_code_en': all_civil_code_en,
            'tfidf_vectorizer': tfidf_vectorizer,
            'trainer': trainer,
            'llm_model': llm_model,
            'llm_tokenizer': llm_tokenizer,
            'coliee_data_preprocessor': coliee_data_preprocessor,
            'ckpt_path': ['settings/bert-base-japanese_top150-newE5Seq512L1e-5/combination2models/epoch=2-step=20414.ckpt',
                          "settings/bert-base-japanese_top150-newE5Seq512L1e-5/combination2models/epoch=3-step=23439.ckpt"],
            'model': model,
        })
        exit()
        

    main_model_ckpt = None
    for data_loader in data_sources: 
        all_predictions = []
        result_combination = []
        all_miss_q = set()
        best_f2_ret = {'retrieved': 0, 'valid_f2': 0, 'valid_p': 0,'valid_r': 0, 'miss_q': [1]*10000}
        best_predictions = {'retrieved': 0, 'valid_f2': 0, 'valid_p': 0,'valid_r': 0, 'miss_q':[]}
        best_miss = set()
        data_query_id = data_loader.dataset[0][1][:3]
        best_model_ckpt_by_dev = None

        for ckpt in pretrained_checkpoint_list:
            model.result_logger.info(f"==== Predict ({ckpt}) ====")
            predictions_cache_name = ckpt+f".{data_query_id}.pred.pkl"
            if not os.path.exists(predictions_cache_name):
                # model = RelevantDocClassifier.load_from_checkpoint(ckpt, args=opts, strict=False )
                # model = model.cuda()
                predictions = trainer.predict(model, data_loader, ckpt_path=ckpt)
                pickle.dump(predictions, open(predictions_cache_name, "wb")) # cached prediction 
            else:
                predictions = pickle.load(open(predictions_cache_name, "rb"))

            cur_checkpoint_ret = model.on_epoch_end(predictions, no_log_tensorboard=True)
            cur_checkpoint_ret.pop('detail_pred')
            model.result_logger.info(f"{cur_checkpoint_ret}")
            all_miss_q = all_miss_q.union(set(cur_checkpoint_ret['miss_q']))
            if   main_model_ckpt is None and best_f2_ret["valid_f2"] < cur_checkpoint_ret['valid_f2']: #    'step=10964.ckpt' in ckpt : #  len(best_f2_ret["miss_q"]) > len(cur_checkpoint_ret['miss_q']): #   main_model_ckpt is None and best_f2_ret["valid_f2"] < cur_checkpoint_ret['valid_f2']: #    
                best_predictions = predictions
                best_miss =  set(cur_checkpoint_ret['miss_q'])
                best_f2_ret = cur_checkpoint_ret
                best_model_ckpt_by_dev = ckpt
            elif main_model_ckpt == ckpt:
                best_predictions = predictions
                best_miss =  set(cur_checkpoint_ret['miss_q'])
                best_f2_ret = cur_checkpoint_ret
            all_predictions += predictions

        main_model_ckpt = main_model_ckpt or best_model_ckpt_by_dev 
        out = model.on_epoch_end(all_predictions, no_log_tensorboard=True, main_prediction_enss=(best_miss, best_predictions))

        # log
        json.dump(out['detail_pred'], open(f'{opts.log_dir}/{data_query_id}.detail_pred.json', 'wt'), ensure_ascii=False)
        print(out["valid_f2"], out["valid_p"], out["valid_r"])

        # dump submission files: file retrieved and file top 100 candidates 
        generate_file_submission(out['detail_pred'], 
                                 f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}.tsv", 
                                 key_cids="pred_c_ids", 
                                 key_scores="pred_c_scores")
        
        for q_id, q_info  in out['detail_pred'].items():
            out['detail_pred'][q_id]["rank_c_ids"] = [] 
            out['detail_pred'][q_id]["rank_c_scores"] = []
            for e in q_info["rank"]:
                if e[0] not in out['detail_pred'][q_id]["rank_c_ids"]:
                    out['detail_pred'][q_id]["rank_c_ids"].append(e[0])
                    out['detail_pred'][q_id]["rank_c_scores"].append(e[1])
                    
        generate_file_submission(out['detail_pred'], 
                                 f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}-L.tsv", 
                                 key_cids="rank_c_ids", 
                                 key_scores="rank_c_scores",
                                 limited_prediction=100)
        # evaluate
        input_test = f"{opts.data_dir}/../../COLIEE2023statute_data-English/train/riteval_{data_query_id}_en.xml"
        print(f"Eval: {opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}.tsv")
        if os.path.exists(input_test):
            evaluate(INPUT_TEST = input_test, 
                    INPUT_PREDICTION=f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}.tsv", 
                    USECASE_ONLY = False)

        
        # enssemble model 
        main_pred_file = None if opts.main_enss_path is None else opts.main_enss_path.format(data_query_id)  # f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}.tsv" # 
        if opts.main_enss_path is not None and os.path.exists(main_pred_file):

            def enss_procedure(main_pred_file, addition_pred_files, output_file, addition_limit=None, relevant_limit=None):

                enss_out_data = enssemble_prediction(main_pred_file, 
                                                    addition_pred_files, 
                                                    addition_limit=addition_limit,
                                                    relevant_limit=relevant_limit)
                
                # dump enssemble submission file 
                generate_file_submission(enss_out_data, 
                                        output_file, 
                                        key_cids="pred_c_ids", 
                                        key_scores="pred_c_scores")

                # evaluate
                input_test = f"{opts.data_dir}/../../COLIEE2023statute_data-English/train/riteval_{data_query_id}_en.xml"
                if os.path.exists(input_test):
                    print(f"Eval: {output_file}")
                    evaluate(INPUT_TEST = input_test, 
                            INPUT_PREDICTION=output_file, 
                            USECASE_ONLY = False)

            # enss
            additional_pred_files = [f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}.tsv"]
            output_file = f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}.enss.tsv"
            enss_procedure(main_pred_file, additional_pred_files, output_file, addition_limit=1)

            enss_procedure(main_pred_file.replace(".txt", "-L.txt"), 
                           [e.replace(".tsv", "-L.tsv") for e in additional_pred_files], 
                           output_file.replace(".tsv", "-L.tsv"), 
                           relevant_limit=100)

        out.pop('detail_pred')
        model.result_logger.info(f"{out}")

