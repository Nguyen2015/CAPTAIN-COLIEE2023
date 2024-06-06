import os
import re
import json
import argparse, torch

from tqdm import tqdm
from data_generator import jp_tokenize, load_data_coliee
from evaluate import evaluate
from transformers import AutoTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data.dataloader import DataLoader

def format_output(text):
    CLEANR = re.compile('<.*?>')
    cleantext = re.sub(CLEANR, '', text)
    return cleantext.strip().lower()


def llm_infer(model, tokenizer_, texts):
    inputs = tokenizer_(texts, return_tensors="pt", padding='longest')[
        "input_ids"].cuda()
    outputs = model.generate(inputs, max_new_tokens=256)
    raw_out = tokenizer_.batch_decode(outputs, skip_special_tokens=True)
    output_text = [format_output(e.replace(texts[i], ""))
                   for i, e in enumerate(raw_out)]

    return output_text

def postprocess_output(generated_output, a_ids):
    return_a_ids = []
    for a_id in a_ids:
        if f"article {a_id}" in generated_output.lower():
            return_a_ids.append(a_id)
    if len(return_a_ids) == 0:
        return a_ids
    else:
        return return_a_ids
    
def reranking_prompting(list_articles, q_id, map_article, map_query):
    prompting = f"In bellow articles:  "
    for a_id in list_articles:
        prompting = prompting + f"\nArticle {a_id}: {map_article[a_id]},"

    prompting = prompting + \
        f"\nQuestion: which articles really relevant to query \"{map_query[q_id]}\"?"
    return prompting


def _llm_rerank(map_article, map_query, plm_model_prediction_path, prefix_prediction_ids=["R02", "R03", "R04"], llm_model_name = "google/flan-t5-xxl"):
    predictions = {}
    for q_id in prefix_prediction_ids:
        predictions.update(json.load(open(f'{plm_model_prediction_path}/{q_id}.detail_pred.json')))


    candidates_articles = {}
    for q_id, q_info in predictions.items():
        candidates = []
        for a_id in q_info['pred_c_ids']:
            if a_id in candidates:
                continue
            else:
                candidates.append(a_id)
        candidates_articles[q_id] = candidates  

    prompt_texts = []
    prompt_texts_ids = []
    for q_id in predictions:
        prompt = reranking_prompting(candidates_articles[q_id], q_id, map_article=map_article, map_query=map_query)
        prompt_texts.append(prompt)
        prompt_texts_ids.append(q_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        llm_model_name, device_map="auto",  torch_dtype=torch.float16, load_in_8bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)


    print("len(prompt_texts) = ", len(prompt_texts))
    prompting_loader = DataLoader(
        prompt_texts, batch_size=4, collate_fn=None, shuffle=False)
    all_outputs = []
    all_inputs = []
    for i_batch, batch_prompt_texts in enumerate(tqdm(prompting_loader)):
        batch_outputs = llm_infer(model, tokenizer,  batch_prompt_texts)
        all_inputs = all_inputs + batch_prompt_texts
        all_outputs = all_outputs + batch_outputs
        if i_batch % 5 == 4:
            json.dump(list(zip(prompt_texts_ids[:len(all_outputs)], all_inputs, all_outputs)), open(
                f"{plm_model_prediction_path}/all_infer_rerank.json", "wt"), indent=2)

    json.dump(list(zip(prompt_texts_ids[:len(all_outputs)], all_inputs, all_outputs)), open(
        f"{plm_model_prediction_path}/all_infer_rerank.json", "wt"), indent=2)

    infer_data = list(
        zip(prompt_texts_ids[:len(all_outputs)], all_inputs, all_outputs))
    

    correct_pairs = set()
    for output_check in infer_data:
        q_id = output_check[0]
        llm_output = output_check[2]
        filtered_a_ids = postprocess_output(
            llm_output, a_ids=candidates_articles[q_id])
        for a_id in filtered_a_ids:
            correct_pairs.add((q_id, a_id))
    correct_pairs


    for q_type in prefix_prediction_ids:
        modified_out = f'{plm_model_prediction_path}/CAPTAIN.bjpAll.{q_type}.tsv'
        with open(modified_out) as f:
            lines = [l.strip() for l in f.readlines()]
        new_lines = []
        for l in lines:
            info = l.split(" ")
            q_id_, a_id_ = info[0], info[2]
            if (q_id_, a_id_) in correct_pairs:
                new_lines.append(l)
        with open(modified_out.replace(".tsv", "_RerankP.tsv"), 'wt') as f:
            f.write("\n".join(new_lines))


if __name__=="__main__":
    # training+model args
    parser = argparse.ArgumentParser(description="Training Args") 
        
    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--log_dir", type=str, default=".", help="log dir")
    parser.add_argument("--plm_model_prediction_path", default=None, type=str, help="pretrained checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--llm_model_name", type=str, help="pretrained model name or path")
    
    opts = parser.parse_args()
    
    options = argparse.Namespace(
        test_ids=['R04'],
        dev_ids=['R02', 'R03' ],
        path_folder_base=f'{opts.data_dir}/COLIEE2023statute_data-English/',
        meta_data_alignment=f'{opts.data_dir}/COLIEE2023statute_data-English',
        test_file=None,
        lang='en'
    )

    path_folder_base = options.path_folder_base
    if options.lang == 'en':
        options.meta_data_alignment = path_folder_base
    meta_data_alignment = options.meta_data_alignment
    if options.lang == 'jp':
        law_tokenizer = jp_tokenize
    else:
        law_tokenizer = None
    
    c_docs, c_keys, dev_q, test_q, train_q, sub_doc_info = load_data_coliee(path_folder_base=path_folder_base,
                                                                            postags_select=None,
                                                                            ids_test=options.test_ids,
                                                                            ids_dev=options.dev_ids,
                                                                            lang=options.lang,
                                                                            path_data_alignment=meta_data_alignment,
                                                                            chunk_content_info=None,
                                                                            tokenizer=law_tokenizer,
                                                                            test_file=options.test_file)


    map_article = dict(zip(c_keys, c_docs))
    map_query = {}
    for e in dev_q + train_q + test_q:
        map_query[e['index']] = e['content']
    
    _llm_rerank(map_article, map_query, 
                plm_model_prediction_path=opts.plm_model_prediction_path,
                llm_model_name=opts.llm_model_name,
                prefix_prediction_ids=options.test_ids+options.dev_ids)

    for data_query_id in options.test_ids+options.dev_ids:
        input_test = f"{opts.data_dir}/COLIEE2023statute_data-English/train/riteval_{data_query_id}_en.xml"
        output_file = f'{opts.plm_model_prediction_path}/CAPTAIN.bjpAll.{data_query_id}_RerankP.tsv'
        if os.path.exists(input_test):
            print(f"Eval: {output_file}")
            evaluate(INPUT_TEST = input_test, 
                    INPUT_PREDICTION=output_file, 
                    USECASE_ONLY = False)