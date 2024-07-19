#!/usr/bin/env python
# coding: utf-8
import json
import pickle

from flask import Flask, jsonify
from flask import request

from data_utils.tfidf_classifier import do_classify
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.data.datasets.glue import *
from transformers.data.processors.utils import InputExample


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


class ColieeDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
            self,
            args: GlueDataTrainingArguments,
            tokenizer: PreTrainedTokenizer,
            limit_length: Optional[int] = None,
            mode: Union[str, Split] = Split.train,
            c_code=None,
            sentence=None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        self.c_code = c_code if c_code is not None else []
        self.sentence = sentence if sentence is not None else ""
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                RobertaTokenizer,
                RobertaTokenizerFast,
                XLMRobertaTokenizer,
                BartTokenizer,
                BartTokenizerFast,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list
        logger.info(f"Creating features from dataset file at {args.data_dir}")

        def _create_examples(lines, set_type='test'):
            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                text_b = line[4]
                label = None if set_type == "test" else line[0]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples

        lines = [
            ["{}".format(i),
             "sent_infer",
             e[0],
             self.sentence,
             e[1]
             ]
            for i, e in enumerate(self.c_code)
        ]  # recreate the data
        examples = _create_examples(lines)
        if limit_length is not None:
            examples = examples[:limit_length]
        self.features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=self.output_mode,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    def get_c_code_ids(self):
        return [e[1] for e in self.c_code]


def infer_coliee_task3(sentence, all_civil_code, data_args, tfidf_vectorizer, trainer, bert_tokenizer):
    test_q = [{
        'content': sentence,
        'index': 'infer-0',
        'label': 'N',
        'result': []
    }]
    c_docs = [e[1] for e in all_civil_code]
    c_keys = [e[0] for e in all_civil_code]

    test_pred, _ = do_classify(c_docs, c_keys, test_q, vectorizer=tfidf_vectorizer, topk=150)
    c_code_pred_by_tfidf = [all_civil_code[idx] for idx in test_pred[0]]
    test_dataset = ColieeDataset(data_args,
                                 bert_tokenizer,
                                 limit_length=230,
                                 mode='test', sentence=sentence, c_code=c_code_pred_by_tfidf)
    predictions = trainer.predict(test_dataset=test_dataset).predictions
    probs = torch.softmax(torch.from_numpy(predictions), dim=1)
    predicted_labels = torch.argmax(probs, 1)
    return predicted_labels, probs, c_code_pred_by_tfidf


def init_state(path_c_code, path_folder_base, model_path):
    model_version = model_path  # 'bert-base-uncased'
    do_lower_case = True
    config = AutoConfig.from_pretrained(
        model_version,
        num_labels=2,
        finetuning_task='MRPC'
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_version, config=config)
    bert_tokenizer = AutoTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
    model.eval()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=["--model_name_or_path", model_version,
              "--task_name", "MRPC",
              "--do_predict",
              "--data_dir", "./coliee3_2020/data",
              "--per_device_train_batch_size", "32",
              "--max_seq_length", "230",
              "--learning_rate", "2e-5",
              "--warmup_steps", "1000",
              "--num_train_epochs", "5.0",
              "--save_total_limit", "2",
              "--output_dir", "./coliee3_2020/models",
              "--overwrite_output_dir"])
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args
    )
    tfidf_vectorizer = pickle.load(open("{}/tfidf_classifier.pkl".format(path_folder_base), "rb"))

    all_civil_code = json.load(open(path_c_code, 'rt', encoding='utf8'))
    return all_civil_code, data_args, tfidf_vectorizer, trainer, bert_tokenizer


app = Flask(__name__)
all_civil_code, data_args, tfidf_vectorizer, trainer, bert_tokenizer = init_state(
    path_c_code='./coliee3_2020/dataFull/c_code.json',
    path_folder_base='./coliee3_2020/dataFull/',
    model_path='./coliee3_2020/MRPC_FullUncased_3ep/models'
)
_, _, tfidf_vectorizer_bertcc, trainer_bertcc, bert_tokenizer_bertcc = init_state(
    path_c_code='./coliee3_2020/dataFull/c_code.json',
    path_folder_base='./coliee3_2020/dataFull/',
    model_path='./coliee3_2020/MRPC_FullPretrained_5ep/models'
)


@app.route('/coliee-task3-bert-cc', methods=['POST'])
def create_task_bertcc():
    if not request.json or not 'sentence' in request.json:
        return jsonify({"result": False, "detail": "Not found key 'sentence' in request."}), 200
    sent = request.json.get("sentence", "")

    predicted_labels, probs, c_code_pred_by_tfidf = infer_coliee_task3(
        sentence=sent, all_civil_code=all_civil_code, data_args=data_args, tfidf_vectorizer=tfidf_vectorizer_bertcc,
        trainer=trainer_bertcc, bert_tokenizer=bert_tokenizer_bertcc)

    task = [{"label": True if lb == 1 else False,
             "scores": [float(probs[i][j]) for j in range(probs[i].shape[0])],
             "sentence": sent,
             "civil_code": c_code_pred_by_tfidf[i][1],
             "civil_code_id": c_code_pred_by_tfidf[i][0],
             }
            for i, lb in enumerate(predicted_labels)]

    return jsonify(task), 201


@app.route('/coliee-task3-bert-ensemble', methods=['POST'])
def create_task_bert_ensemble():
    if not request.json or not 'sentence' in request.json:
        return jsonify({"result": False, "detail": "Not found key 'sentence' in request."}), 200
    sent = request.json.get("sentence", "")

    # bert base
    predicted_labels, probs, c_code_pred_by_tfidf = infer_coliee_task3(
        sentence=sent, all_civil_code=all_civil_code, data_args=data_args, tfidf_vectorizer=tfidf_vectorizer,
        trainer=trainer, bert_tokenizer=bert_tokenizer)
    c_code_predicted = {}
    for i, lb in enumerate(predicted_labels):
        if lb == 1:
            c_code_predicted[c_code_pred_by_tfidf[i][0]] = {
                "label": True if lb == 1 else False,
                "scores": [float(probs[i][j]) for j in
                           range(probs[i].shape[0])],
                "sentence": sent,
                "civil_code": c_code_pred_by_tfidf[i][1],
                "civil_code_id": c_code_pred_by_tfidf[i][0],
            }

    # bert cc
    predicted_labels_bertcc, probs_bertcc, c_code_pred_by_tfidf_bertcc = infer_coliee_task3(
        sentence=sent, all_civil_code=all_civil_code, data_args=data_args, tfidf_vectorizer=tfidf_vectorizer_bertcc,
        trainer=trainer_bertcc, bert_tokenizer=bert_tokenizer_bertcc)
    for i, lb in enumerate(predicted_labels_bertcc):
        if lb == 1 and c_code_pred_by_tfidf_bertcc[i][0] not in c_code_predicted:
            c_code_predicted[c_code_pred_by_tfidf_bertcc[i][0]] = {
                "label": True if lb == 1 else False,
                "scores": [float(probs_bertcc[i][j]) for j in range(probs_bertcc[i].shape[0])],
                "sentence": sent,
                "civil_code": c_code_pred_by_tfidf_bertcc[i][1],
                "civil_code_id": c_code_pred_by_tfidf_bertcc[i][0],
            }
            print(c_code_pred_by_tfidf_bertcc[i])

    # append label zero to result
    for i, lb in enumerate(predicted_labels):
        if lb == 0 and c_code_pred_by_tfidf[i][0] not in c_code_predicted:
            c_code_predicted[c_code_pred_by_tfidf[i][0]] = {
                "label": True if lb == 1 else False,
                "scores": [float(probs[i][j]) for j in
                           range(probs[i].shape[0])],
                "sentence": sent,
                "civil_code": c_code_pred_by_tfidf[i][1],
                "civil_code_id": c_code_pred_by_tfidf[i][0],
            }
    task = list(c_code_predicted.values())
    task.sort(key=lambda x: x['scores'][1], reverse=True)
    # task = task[:30]

    return jsonify(task), 201


@app.route('/coliee-task3-bert-base', methods=['POST'])
def create_task_bert_base():
    if not request.json or not 'sentence' in request.json:
        return jsonify({"result": False, "detail": "Not found key 'sentence' in request."}), 200
    sent = request.json.get("sentence", "")

    predicted_labels, probs, c_code_pred_by_tfidf = infer_coliee_task3(
        sentence=sent, all_civil_code=all_civil_code, data_args=data_args, tfidf_vectorizer=tfidf_vectorizer,
        trainer=trainer, bert_tokenizer=bert_tokenizer)

    task = [{"label": True if lb == 1 else False,
             "scores": [float(probs[i][j]) for j in range(probs[i].shape[0])],
             "sentence": sent,
             "civil_code": c_code_pred_by_tfidf[i][1],
             "civil_code_id": c_code_pred_by_tfidf[i][0],
             }
            for i, lb in enumerate(predicted_labels)]

    return jsonify(task), 201


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9002, debug=False)
