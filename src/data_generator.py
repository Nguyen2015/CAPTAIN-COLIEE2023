import json
import os
import pickle
import re
import traceback
import pandas as pd 
import json
import glob 
import os 

import fugashi
import pandas as pd
import argparse
import logging
from tqdm import tqdm

from data_utils.stopwords_tfidf_generator import do_generate_stopwords
from data_utils.tfidf_classifier import do_classify
from data_utils.utils import load_data_coliee, postag_filter
from transformers import BertJapaneseTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# The Tagger object holds state about the dictionary.
jp_tagger = fugashi.Tagger()


def jp_tokenize(text):
    return [word.surface for word in jp_tagger(text)]

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings.detach().cpu())

        return torch.stack(all_embeddings) 
        # return torch.stack(all_embeddings)


def generate_pair_inputs(data_pred, data_gold, _c_keys, append_gold=False, sub_doc_info=None):
    _data_pairs_id = []
    _, _, sub_key_mapping = sub_doc_info or (None, None, {})

    for i in range(data_pred.shape[0]):
        cur_pred = [_c_keys[idx] for idx in data_pred[i]]
        cur_label = data_gold[i]["result"]
        q_id = data_gold[i]['index']

        if append_gold:
            for id_civil_lb in cur_label:
                if id_civil_lb not in cur_pred:
                    cur_pred = cur_pred + [id_civil_lb]

        for j, id_civil_pred in enumerate(cur_pred):
            check_lb = id_civil_pred in cur_label
            _data_pairs_id.append(((id_civil_pred, q_id), check_lb))

            # append sub articles (by chunking)
            for id_c in sub_key_mapping.get(id_civil_pred, []):
                _data_pairs_id.append(((id_c, q_id), check_lb))

    print('Number data pairs: ', len(_data_pairs_id))
    return _data_pairs_id


def generate_pair_inputs_task4(data_pred, data_gold, _c_keys, testing=False, bert_predictions=None, sub_doc_info=None):
    _data_pairs_id = []
    _, _, sub_key_mapping = sub_doc_info or (None, None, {})
    bert_predictions = bert_predictions or {}
    for i in range(data_pred.shape[0]):
        relevant_article_ids = data_gold[i]["result"]
        q_id = data_gold[i]['index']
        label = True if data_gold[i]['label'] == 'Y' else False

        # get predictions from bert + tfidf
        cur_pred = list(set([e[1] for e in bert_predictions.get(q_id, [])] +
                            [_c_keys[idx] for idx in data_pred[i]]))

        for id_civil_lb in relevant_article_ids:
            if id_civil_lb not in cur_pred:
                cur_pred = cur_pred + [id_civil_lb]

        if not testing:
            # _data_pairs_id.append(
            #     (("None", q_id), label))
            for j, id_civil_pred in enumerate(cur_pred):
                if id_civil_pred in relevant_article_ids:
                    _data_pairs_id.append(
                        ((id_civil_pred, q_id), label))
                else:
                    _data_pairs_id.append(((id_civil_pred, q_id), False))
        else:
            for id_article in relevant_article_ids:
                _data_pairs_id.append(
                    ((id_article, q_id), label))
        
    if sub_key_mapping is not None:
        for e in _data_pairs_id:
            for sub_art in sub_key_mapping.get(e[0][0], []):
                _data_pairs_id.append(((sub_art, e[0][1]), e[1]))

    print(len(_data_pairs_id))
    return _data_pairs_id


def aggregate_sentence_pairs(_c_docs, _c_keys, _data_pairs_id, _q, plus_filter_postags=False, filter_lb=False,
                             empty_article_id="None", sub_doc_info=None):
    _new_dataset = []
    _q_map = dict((q["index"], q['content']) for q in _q)
    empty_article_content = ""
    _c_docs = _c_docs + [empty_article_content]
    _c_keys = _c_keys + [empty_article_id]

    _c_sub_docs, _c_sub_keys, _ = sub_doc_info or (None, None, {})
    _c_docs = _c_docs + (_c_sub_docs if _c_sub_docs is not None else [])
    _c_keys = _c_keys + (_c_sub_keys if _c_sub_keys is not None else [])

    for (id_civil_pred, q_id), lb in _data_pairs_id:
        try:
            _new_dataset.append({
                "id": [id_civil_pred, q_id],
                "c_code": _c_docs[_c_keys.index(id_civil_pred)],
                "query": _q_map[q_id],
                'label': lb
            })

            if plus_filter_postags:
                if filter_lb and lb:
                    _new_dataset.append({
                        "id": [id_civil_pred + "_pos_filtered", q_id],
                        "c_code": postag_filter(_c_docs[_c_keys.index(id_civil_pred)]),
                        "query": _q_map[q_id],
                        'label': lb
                    })
                if not filter_lb:
                    _new_dataset.append({
                        "id": [id_civil_pred + "_pos_filtered", q_id],
                        "c_code": postag_filter(_c_docs[_c_keys.index(id_civil_pred)]),
                        "query": _q_map[q_id],
                        'label': lb
                    })
        except Exception as e:
            traceback.print_stack()
            print(e)
            print("[Err] {}".format(((id_civil_pred, q_id), lb)))
    return _new_dataset


def gen_mrpc_data(coliee_data_, file_path):
    data = {
        "label": [],
        "#1 ID": [],
        "#2 ID": [],
        "sentence1": [],
        "sentence2": [],
    }
    for e in coliee_data_:
        data['label'].append(1 if e['label'] else 0)
        data['#1 ID'].append(e['id'][1])
        data['#2 ID'].append(e['id'][0])
        data['sentence1'].append(e['query'].replace('\n', " "))
        data['sentence2'].append(e['c_code'].replace('\n', " "))
    df = pd.DataFrame(data=data)
    df.to_csv(file_path, index=False, sep=',')

def generate_quality_article(detail_prediction_path, data_folder, top_k=50):
    topk_filter = {}
    for file_name in glob.glob(f'{detail_prediction_path}/*.detail_pred.json'):
        print(file_name)

        file_data = json.load(open(file_name))
        for q_id, cluster_info in file_data.items():
            for c_id in list(set(cluster_info['c_ids'] + cluster_info['pred_c_ids_topk']))[:top_k]:
                topk_filter[f'{q_id}|{c_id}'] = True


    for c_file_path in glob.glob(f"{data_folder}/*.csv"):
        df_data = pd.read_csv(c_file_path)

        def filter_fn(row):
            return topk_filter.get(f'{row["#1 ID"]}|{row["#2 ID"]}', False)
                
        df_filtered = df_data[df_data.apply(filter_fn, axis=1)]
        file_name = c_file_path.split("/")[-1][:-4]
        data_out = f"{data_folder}/data_top{top_k}"
        print(f"Generate quality article in {data_out}/{file_name}.csv")
        if not os.path.exists(data_out):
            os.mkdir(data_out)
        df_filtered.to_csv(open(f"{data_out}/{file_name}.csv", "wt"), index=False, sep=',')
        print(len(df_filtered))

def generate_similar_query(detail_prediction_path, data_folder, top_k=50):
    all_miss_q = {}
    for file_name in glob.glob(f'{detail_prediction_path}/*.detail_pred.json'):
        logging.info(f"Collect miss query from: {file_name}" )
        file_data = json.load(open(file_name))
        for q_id, cluster_info in file_data.items():
            if len([score for score in cluster_info['pred_c_scores'] if score > 0.5]) == 0 or \
                (len(cluster_info['pred_c_ids']) > 0 and cluster_info['retrieved'] == 0):
                if q_id[:3] not in all_miss_q:
                    all_miss_q[q_id[:3]] = set()
                all_miss_q[q_id[:3]].add(q_id)
    logging.info(f"All miss query {all_miss_q}")
    
                     
    df_train = pd.read_csv(f"{data_folder}/train.csv")
    df_dev = pd.read_csv(f"{data_folder}/dev.csv")
    df_test = pd.read_csv(f"{data_folder}/test.csv")
    df_test2 = pd.read_csv(f"{data_folder}/test_submit2.csv")
    df_all_data = pd.concat([df_train, df_dev, df_test, df_test2])
 
    all_q_ids, all_q_content = df_all_data['#1 ID'].values, df_all_data['sentence1'].values
    map_q_id2content = dict(zip(all_q_ids, all_q_content))
    all_q_ids, all_q_content = list(map_q_id2content.keys()), list(map_q_id2content.values())
    logging.info(f"All queries (total number={len(all_q_ids)}): e.g., {all_q_ids[:10]}, ...")
    
    model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens-v2")
    sentence_embeddings = model.encode(all_q_content, batch_size=48).numpy()
    train_sentence_embeddings = sentence_embeddings[:len(set(df_train['#1 ID'].values))]

    similar_q_ids = set()
    unstable_q_ids = set().union(*all_miss_q.values())
    logging.info(f"Unstable queries (total number={len(unstable_q_ids)}): e.g., {unstable_q_ids}, ...")
    
    logged = False
    for i, id_q in enumerate(tqdm(all_q_ids)):
        if id_q in unstable_q_ids:
            q_vect = sentence_embeddings[i]
        
            score = cosine_similarity([q_vect], train_sentence_embeddings)[0]
            indexes = score.argsort()[-top_k:][::-1]
            for idx in indexes:
                similar_q_ids.add(all_q_ids[idx])
                
            # logging 
            if not logged:
                logging.info(f"Query {id_q} = {map_q_id2content[id_q]}")
                logging.info("Similar queries:")
                for idx in indexes: 
                    logging.info(f"\t{all_q_ids[idx]} = {map_q_id2content[all_q_ids[idx]]}")
                logged = True
                
                
    filtered_q_ids = set(similar_q_ids).union(unstable_q_ids)
    logging.info(f"All miss query (total number={len(filtered_q_ids)}): e.g., {list(filtered_q_ids)[:10]}, ...")
    

    for c_file_path in glob.glob(f"{data_folder}/*.csv"):
        df_data = pd.read_csv(c_file_path)
        def filter_fn(row):
            return row["#1 ID"] in filtered_q_ids
                
        df_filtered = df_data[df_data.apply(filter_fn, axis=1)]
        file_name = c_file_path.split("/")[-1][:-4]
        data_out = f"{data_folder}/data_sim_q_{top_k}"
        logging.info(f"Generate similarity queries in {data_out}/{file_name}.csv")
        if not os.path.exists(data_out):
            os.mkdir(data_out)
        df_filtered.to_csv(open(f"{data_out}/{file_name}.csv", "wt"), index=False, sep=',')
        logging.info(len(df_filtered))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_folder_base',
                        action="store", dest="path_folder_base",
                        help="path folder saving data", default='path/to/path_folder_base')
    parser.add_argument('--meta_data_alignment',
                        action="store", dest="meta_data_alignment",
                        help="path to folder alignment with folder data base, should be in english", default='path/to/meta_data_alignment')
    parser.add_argument('--path_output_dir',
                        action="store", dest="path_output_dir",
                        help="path folder saving output data", default='path/to/path_output_dir')
    parser.add_argument('--lang',
                        action="store", dest="lang",
                        help="language: en | jp", default='en')
    parser.add_argument('--type_data',
                        action="store", dest="type_data",
                        help="type data for generating process: task3 | task4", default='task3')
    parser.add_argument('--dev_ids',
                        action="extend", dest="dev_ids", type=str, nargs="*",
                        help="id for dev set", default=[])
    parser.add_argument('--test_ids',
                        action="extend", dest="test_ids", type=str, nargs="*",
                        help="id for test set", default=[])
    parser.add_argument('--test_file',
                        action="store", dest="test_file", type=str,  
                        help="path to the test file", default=None)
    parser.add_argument('--chunk_content_size',
                        action="store", dest="chunk_content_size", type=int,
                        help="chunk content of article with size", default=0)
    parser.add_argument('--chunk_content_stride',
                        action="store", dest="chunk_content_stride", type=int,
                        help="chunk content of article with stride", default=0)
    parser.add_argument('--topk',
                        action="store", dest="topk", type=int,
                        help="topk select by tfidf when generating data", default=150)
    parser.add_argument('--faked_result',
                        action="store", dest="faked_result", type=str,
                        help="topk select by tfidf when generating data", default="")
    parser.add_argument('--path_detail_prediction',
                        action="store", dest="path_detail_prediction", type=str,
                        help="path_detail_prediction first step", default=None)
    parser.add_argument('--gen_quality_article',
                        action="store_true", dest="gen_quality_article",
                        help="generate high quality article based on fine-tuned BERT", default=False)
    parser.add_argument('--gen_similar_query',
                        action="store_true", dest="gen_similar_query",
                        help="generate similar queries based on embedding vectors", default=False)
    options = parser.parse_args()
        # save file csv following template of mrpc task
    path_folder_data_out = options.path_output_dir
    if not os.path.exists(path_folder_data_out):
        os.mkdir(path_folder_data_out)
        
    format = '%(asctime)s - %(name)s - %(message)s'
    if isinstance(options, argparse.Namespace):
        logging.basicConfig(format=format, filename=os.path.join(options.path_output_dir, "run.log"), level=logging.INFO)
        print(f"Check log in {os.path.join(options.path_output_dir, 'run.log')}")

    path_folder_base = options.path_folder_base
    if options.lang == 'en':
        options.meta_data_alignment = path_folder_base
    meta_data_alignment = options.meta_data_alignment
    lang = options.lang
    topk_select = options.topk
    postags_select = ["V", "N", "P", "."]
    if lang == 'jp':
        tokenizer = jp_tokenize
    else:
        tokenizer = None

    if options.gen_quality_article:
        generate_quality_article(options.path_detail_prediction, options.path_output_dir, options.topk)
        exit() 
        
    if options.gen_similar_query:
        generate_similar_query(options.path_detail_prediction, options.path_output_dir, options.topk)
        exit() 
    
    chunk_content_info = [options.chunk_content_size,
                          options.chunk_content_stride] \
        if options.chunk_content_size > 0 and options.chunk_content_stride > 0 else None
    c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info = load_data_coliee(path_folder_base=path_folder_base,
                                                                            postags_select=None,
                                                                            ids_test=options.test_ids,
                                                                            ids_dev=options.dev_ids,
                                                                            lang=lang,
                                                                            path_data_alignment=meta_data_alignment,
                                                                            chunk_content_info=chunk_content_info,
                                                                            tokenizer=tokenizer,
                                                                            test_file=options.test_file)
    

        
    json.dump({
        'c_docs': c_docs, 
        'c_keys': c_keys, 
        'dev_q': dev_q, 
        'test_q':test_q, 
        'train_q':train_q, 
        'sub_doc_info': sub_doc_info
        }, open(f'{options.path_output_dir}/all_data.json', 'wt'), indent=2, ensure_ascii=False 
    )
    # test_q = train_q
    if len(dev_q) == 0:
        dev_q = train_q
    if len(test_q) == 0:
        test_q = train_q
        

    # build japanese tokenizer

    # load stopwords generated before
    do_generate_stopwords(path_folder_base, threshold=0.00, tokenizer=tokenizer, data=(
        c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info))  # code to generate stop words automatic using tfidf
    stopwords = json.load(
        open("{}/stopwords/stopwords.json".format(path_folder_base), "rt"))

    # build tfidf vectorizer and generate pair sentence for training process
    train_pred, (_, _, _, vectorizer) = do_classify(c_docs, c_keys, train_q,
                                                    stopwords_=stopwords, topk=topk_select, tokenizer=tokenizer)
                                                    
    if options.type_data == 'task3':
        train_data_pairs_id = generate_pair_inputs(_c_keys=c_keys, data_pred=train_pred, data_gold=train_q,
                                                   append_gold=True, sub_doc_info=sub_doc_info)
    else:
        train_data_pairs_id = generate_pair_inputs_task4(_c_keys=c_keys, data_pred=train_pred, data_gold=train_q,
                                                         testing=False, sub_doc_info=sub_doc_info)

    test_pred, _ = do_classify(
        c_docs, c_keys, test_q, vectorizer=vectorizer, topk=topk_select, tokenizer=tokenizer)
    if options.type_data == 'task3':
        test_data_pairs_id = generate_pair_inputs(
            _c_keys=c_keys, data_pred=test_pred, data_gold=test_q, sub_doc_info=sub_doc_info)
    else:
        test_data_pairs_id = generate_pair_inputs_task4(_c_keys=c_keys, data_pred=test_pred, data_gold=test_q,
                                                        testing=True, sub_doc_info=sub_doc_info)

    dev_pred, _ = do_classify(
        c_docs, c_keys, dev_q, vectorizer=vectorizer, topk=topk_select, tokenizer=tokenizer)
    if options.type_data == 'task3':
        dev_data_pairs_id = generate_pair_inputs(
            _c_keys=c_keys, data_pred=dev_pred, data_gold=dev_q, sub_doc_info=sub_doc_info)
    else:
        dev_data_pairs_id = generate_pair_inputs_task4(_c_keys=c_keys, data_pred=dev_pred, data_gold=dev_q,
                                                       testing=True, sub_doc_info=sub_doc_info)

    print("len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id) = ",
          len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id))

    # fill data from train/test data_pairs_id
    new_dataset_train = aggregate_sentence_pairs(c_docs, c_keys, train_data_pairs_id, train_q,
                                                 plus_filter_postags=False,
                                                 filter_lb=False, sub_doc_info=sub_doc_info)
    new_dataset_test = aggregate_sentence_pairs(c_docs, c_keys, test_data_pairs_id, test_q,
                                                plus_filter_postags=False,
                                                filter_lb=False, sub_doc_info=sub_doc_info)
    new_dataset_dev = aggregate_sentence_pairs(c_docs, c_keys, dev_data_pairs_id, dev_q,
                                               plus_filter_postags=False,
                                               filter_lb=False, sub_doc_info=sub_doc_info)


    # run fake data (silver data) if found faked result 
    faked_result_modified_log = []
    if len(options.faked_result) > 0 and chunk_content_info is not None:
        faked_result = pd.read_csv(open(options.faked_result, "rt"), sep='\t')
        if len(faked_result) == len(new_dataset_train):
            for i_line in range(len(faked_result)):
                if 'sub' in new_dataset_train[i_line]['id'][0] and \
                        new_dataset_train[i_line]['label'] and faked_result['prediction'][i_line] == 0:
                    new_dataset_train[i_line]['label'] = False
                    faked_result_modified_log.append(new_dataset_train[i_line])
        
            gen_mrpc_data(faked_result_modified_log,
                        "{}/log_train_modified.csv".format(path_folder_data_out))
            print("Len faked_result_modified_log = {}".format(len(faked_result_modified_log)))

    gen_mrpc_data(new_dataset_train,
                  "{}/train.csv".format(path_folder_data_out))
    gen_mrpc_data(new_dataset_test, "{}/test.csv".format(path_folder_data_out))
    gen_mrpc_data(new_dataset_dev, "{}/dev.csv".format(path_folder_data_out))

    # save tfidf vectorizer that filter fop 150 civil document
    pickle.dump(vectorizer, open(
        "{}/tfidf_classifier.pkl".format(path_folder_data_out), "wb"))
