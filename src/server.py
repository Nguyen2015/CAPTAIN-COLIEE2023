#!/usr/bin/env python
# coding: utf-8
import random

from flask import Flask, jsonify
from flask import request

from data_utils.tfidf_classifier import do_classify
import pandas as pd 
from transformers.data.datasets.glue import *
from torch.utils.data.dataloader import DataLoader

from llm_infer import llm_infer, postprocess_output, reranking_prompting
from tqdm import tqdm
import gradio as gr

 
g_info = {}
app = Flask(__name__)
app.json.ensure_ascii = False 
 
def infer_coliee_task3(sentence):
    global g_info
    return _infer_coliee_task3(sentence, **g_info)

def init_server(**kwargs):
    global g_info
    g_info = { **kwargs}

def _llm_rerank(llm_model, llm_tokenizer, map_article, q_sentence, candidates_a_ids):
    prompt_texts = []
    
    prompt = reranking_prompting(candidates_a_ids, 'infer-q-id', map_article=map_article, map_query={'infer-q-id': q_sentence})
    prompt_texts.append(prompt.replace("\n\n", "\n"))

    prompting_loader = DataLoader(
        prompt_texts, batch_size=4, collate_fn=None, shuffle=False)
    all_outputs = []
    all_inputs = []
    for i_batch, batch_prompt_texts in enumerate(tqdm(prompting_loader)):
        batch_outputs = llm_infer(llm_model, llm_tokenizer,  batch_prompt_texts)
        all_inputs = all_inputs + batch_prompt_texts
        all_outputs = all_outputs + batch_outputs
    filtered_a_ids = postprocess_output(all_outputs[0], a_ids=candidates_a_ids)
    return filtered_a_ids
 

def _infer_coliee_task3(sentence, **kwargs):
    sentence = sentence.split("(translated to english:")[0] # normalize 
    
    # get gold article and english translated 
    sentence2qid = kwargs.get('sentence2qid')
    q_id = sentence2qid.get(sentence, 'not-found')
    gold_articles = kwargs.get('all_query_jp').get(q_id)
    if gold_articles is not None:
        gold_articles = gold_articles[1]
    
    # tfidf filter 
    test_q = [{
        'content': sentence,
        'index': 'infer-0',
        'label': 'N',
        'result': []
    }]
    c_docs = [e[1] for e in kwargs.get('all_civil_code')]
    c_keys = [e[0] for e in kwargs.get('all_civil_code')]

    test_pred, _ = do_classify(c_docs, c_keys, test_q, vectorizer=kwargs.get('tfidf_vectorizer'), topk=150)
    c_code_pred_by_tfidf = [kwargs.get('all_civil_code')[idx] for idx in test_pred[0]]
    
    # BERT 
    df_test = pd.DataFrame({
        "label":[ 0 for e in range(len(c_code_pred_by_tfidf))], 
        "q-id": ['q-id-faked' for e in range(len(c_code_pred_by_tfidf))], 
        "a-id": [e[0] for e in c_code_pred_by_tfidf], 
        "q-content": [sentence for e in range(len(c_code_pred_by_tfidf))], 
        "a-content": [e[1] for e in c_code_pred_by_tfidf], 
    })
    data_loader = DataLoader(df_test.values, batch_size=64, collate_fn=kwargs.get('coliee_data_preprocessor'), shuffle=False)
    predictions = kwargs.get('trainer').predict(g_info['model'], data_loader,ckpt_path=kwargs.get('ckpt_path')[0])
    logits = torch.cat([e['y_hat'] for e in predictions], 0)
    for chkpt in kwargs.get('ckpt_path')[1:]:
        predictions = kwargs.get('trainer').predict(g_info['model'], data_loader,ckpt_path=chkpt ) # trainer.predict(test_dataset=test_dataset).predictions
        logits = torch.cat([logits] + [e['y_hat'] for e in predictions], 0)
    probs = torch.softmax(logits, dim=1)
    predicted_labels = torch.argmax(probs, 1)
    
    probs = probs[:,1].tolist()
    results = list(zip(predicted_labels, probs, c_code_pred_by_tfidf))
    results.sort(key=lambda x: x[1], reverse=True)
    topk = 10
    selected_a_id = set()
    new_results = []
    for e in results:
        if e[2][0] not in selected_a_id:
            new_results.append(e)
            selected_a_id.add(e[2][0])
            if len(new_results) == topk:
                break
    results = new_results
    selected_a_ids = [e[2][0] for i, e in enumerate(results) if e[0].item() == 1]
    if len(selected_a_ids) == 0:
        results[0] = [torch.Tensor([1]), results[0][1],  results[0][2]]
    
    # add english translated articles 
    selected_articles = [(e[2][0], f"{e[2][1]} (translated to english: {kwargs.get('all_civil_code_en')[e[2][0]]})") for e in results]
    
    # rerank by llm 
    q_content_en = kwargs.get('all_query_en').get(q_id)[0]
    candidates_a_ids = [e[2][0] for i, e in enumerate(results) if e[0].item() == 1]
    predicted_a_ids_by_llm = _llm_rerank(kwargs.get("llm_model"), kwargs.get("llm_tokenizer"), map_article=kwargs.get('all_civil_code_en'), 
                                         q_sentence=q_content_en, candidates_a_ids=candidates_a_ids)
    
    return [e[0]for e in results], [e[1]for e in results], selected_articles, gold_articles, predicted_a_ids_by_llm

def _rand_sentence():
    q_jp = g_info.get('all_query_jp')
    q_en = g_info.get('all_query_en')
    all_keys = [e for e in q_jp if 'R02' in e or 'R03' in e or 'R04' in e or 'R05' in e]
    print(all_keys)
    random.shuffle(all_keys) 
    q_picked = all_keys[:30]
    task = [f"{q_jp[e][0]}(translated to english: {q_en[e][0]})" for e in q_picked]+[
        "ＡＢ間の売買契約において、売主Ａが買主Ｂに対して引き渡した目的物の数量が不足しており、契約の内容に適合しない場合、Ｂが数量の不足を知った時から１年以内にその旨をＡに通知しない場合には、Ａが引渡しの時に数量の不足を知り又は重大な過失によって知らなかったときを除き、Ｂは損害賠償の請求をすることができない。(translated to english: In the contract for sale between A and B, the quantity of the object delivered by the seller A to the buyer B is insufficient and does not conform to the terms of the contract. If B fails to notify A of the non-conformity in the quantity within one year from the time when B becomes aware of it, B may not claim compensation for loss or damage, unless A knew or did not know due to gross negligence the non-conformity in the quantity at the time of the delivery.)"
    ]
    return task 

@app.route('/random-sentence', methods=['GET'])
def sample_sentence(): 
    task = _rand_sentence()
    return jsonify(task), 201

def __process(sent):
    predicted_labels, probs, c_code_pred_by_tfidf, gold_articles, predicted_a_ids_by_llm = infer_coliee_task3(sentence=sent)

    task = [{"predict": True if lb == 1 else False,
             "predict_by_llm": True if lb == 1 and c_code_pred_by_tfidf[i][0] in predicted_a_ids_by_llm else False,
             'label': 'uknown' if gold_articles is None else True if c_code_pred_by_tfidf[i][0] in gold_articles else False,
             "scores": probs[i],
             "sentence": sent,
             "article_content": c_code_pred_by_tfidf[i][1],
             "article_id": c_code_pred_by_tfidf[i][0], 
             }
            for i, lb in enumerate(predicted_labels)]
    return task

@app.route('/coliee-task3-bert-cc', methods=['POST'])
def create_task_bertcc():
    if not request.json or not 'sentence' in request.json:
        return jsonify({"result": False, "detail": "Not found key 'sentence' in request."}), 200
    sent = request.json.get("sentence", "")

    task = __process(sent)
    
    return jsonify(task), 201
 

# Function to get the relevant articles for a selected sentence
def get_relevant_articles(sentence):
    
    articles = __process(sentence)
    formatted_articles = ""
    for article in articles:
        formatted_articles += f"<b>Article ID:</b> {article['article_id']}<br>"
        formatted_articles += (
            f"<b>Article Content:</b><br>{article['article_content']}<br>"
        )
        # formatted_articles += (
        #     f"<b style='color:red;'>Score:</b> {article['scores']}<br><br>"
        # )

        # Display score as a percentage bar
        score_percentage = int(article["scores"] * 100)
        formatted_articles += f"<b style='color:red;'>Score:</b><br>"
        formatted_articles += f"<div style='width: 100%; background-color: #e0e0e0; border-radius: 5px; overflow: hidden;'>"
        formatted_articles += f"<div style='width: {score_percentage}%; background-color: #1E88E5; padding: 5px; text-align: center; color: white;'>{score_percentage}%</div>"
        formatted_articles += "</div><br>"

        if article["predict"] == article["label"] == True:
            formatted_articles = (
                "<div style='color: green'>" + formatted_articles + "</div>"
            )
        if article.get("predict_by_llm", False):
            formatted_articles += "<span style='border: 1px solid #6002ee; color: #6002ee; background-color: #efe5fd; padding: 2px 5px; border-radius: 3px;'>Retrieved by LLM</span><br>"
        formatted_articles += (
            "<hr style='border-top: 2px solid #000;'>\n"  # HTML divider
        )

    return formatted_articles
 
def run_server_flask(PORT=9002, **kwargs):
    init_server(**kwargs)
    # infer_coliee_task3(sentence='対抗要件を備える必要がない物権の場合には，時間的に先に成立した物権が優先する。(translated to english: In cases where real rights do not require requirements of perfection, the real rights that formed earlier in time take priority.)',
    #                     )
    """
    curl -d '{"sentence":"債務者が所有する同一の不動産について，第一順位の抵当権と第二順位の抵当権が設定され，それぞれその旨の登記がされている場合，第一順位の抵当権の被担保債権に係る債務を債務者が弁済したときは，債務者は，弁済による代位によって第一順位の抵当権を取得する。(translated to english: In the case where the first mortgage and the second mortgage were created on the same real property of the obligor, if the obligor pay the obligation which was secured by the first mortgage, the obligor shall obtain the first mortgage by subrogation by performance.)" }' \
        -H "Content-Type: application/json" -X POST http://spcc-a40g19:9002/coliee-task3-bert-cc

    """
    # 
    # Flask
    app.run(host='0.0.0.0', port=PORT, debug=False)
    
def run_server(PORT=9002, **kwargs):
    init_server(**kwargs)
    # infer_coliee_task3(sentence='対抗要件を備える必要がない物権の場合には，時間的に先に成立した物権が優先する。(translated to english: In cases where real rights do not require requirements of perfection, the real rights that formed earlier in time take priority.)',
    #                     )
    """
    curl -d '{"sentence":"債務者が所有する同一の不動産について，第一順位の抵当権と第二順位の抵当権が設定され，それぞれその旨の登記がされている場合，第一順位の抵当権の被担保債権に係る債務を債務者が弁済したときは，債務者は，弁済による代位によって第一順位の抵当権を取得する。(translated to english: In the case where the first mortgage and the second mortgage were created on the same real property of the obligor, if the obligor pay the obligation which was secured by the first mortgage, the obligor shall obtain the first mortgage by subrogation by performance.)" }' \
        -H "Content-Type: application/json" -X POST http://spcc-a40g19:9002/coliee-task3-bert-cc

    """ 
    #  
    # Gradio interface
    # List of sentences for the dropdown
    sentences = _rand_sentence()

    # Create Gradio interface
    iface = gr.Interface(
        fn=get_relevant_articles,
        inputs=gr.Dropdown(sentences, label="Select a Sentence"),
        outputs=gr.HTML(label="Relevant Articles"),
        title="NguyenLab - Legal NLP Demo",
        description="Statute law retrieval system by NguyenLab",
        allow_flagging="never",
    )

    # Launch the interface
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=PORT,
        # ssl_keyfile="./localhost+2-key.pem", ssl_certfile="./localhost+2.pem"
    )

    