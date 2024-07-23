# CAPTAIN at COLIEE 2023: Efficient Methods for Legal Information Retrieval and Entailment Tasks 
The Competition on Legal Information Extraction/Entailment (COLIEE) is held annually to encourage advancements in the automatic processing of legal texts. Processing legal documents is challenging due to the intricate structure and meaning of legal language. In this paper, we outline our strategies for tackling Task 2, Task 3, and Task 4 in the COLIEE 2023 competition. Our approach involved utilizing appropriate state-of-the-art deep learning methods, designing methods based on domain characteristics observation, and applying meticulous engineering practices and methodologies to the competition. As a result, our performance in these tasks has been outstanding, with **first places in Task 2 and Task 3**, and promising results in Task 4.

---

## Results
These results is provided by COLIEE organization evaluated on the private test sets. Please visit [COLIEE 2023](https://sites.ualberta.ca/~rabelo/COLIEE2023/) for more information about the legal-tasks. 

- **Task 2 (first place): The Legal Case Entailment task**

    This task involves the identification of a paragraph from existing cases that entails the decision of a new case.

    Given a decision Q of a new case and a relevant case R, a specific paragraph that entails the decision Q needs to be identified. We confirmed that the answer paragraph can not be identified merely by information retrieval techniques using some examples. Because the case R is a relevant case to Q, many paragraphs in R can be relevant to Q regardless of entailment.

    This task requires one to identify a paragraph which entails the decision of Q, so a specific entailment method is required which compares the meaning of each paragraph in R and Q in this task. 


    
    | Run                  | F1 (\%) | Precision (\%) | Recall (\%) |
    |----------------------------|------------------|-------------------------|----------------------|
    | **CAPTAIN**.mt5l-ed   | **74.56**   | 78.70                   | **70.83**       |
    | **CAPTAIN**.mt5l-ed4  | 72.65            | 78.64                   | 67.50                |
    | THUIR.thuir-monot5         | 71.82            | **79.00**          | 65.83                |
    | **CAPTAIN**.mt5l-e2   | 70.54            | 75.96                   | 65.83                |
    | THUIR.thuir-ensemble\_2    | 69.30            | 73.15                   | 65.83                |
    | JNLP.bm\_cl\_1\_pr\_1      | 68.18            | 75.00                   | 62.50                |
    | IITDLI.iitdli\_task2\_run2 | 67.27            | 74.00                   | 61.67                |
    | UONLP.test\_no\_labels     | 63.87            | 64.41                   | 63.33                |
    | NOWJ.non-empty             | 60.79            | 64.49                   | 57.50                |
    | LLNTU.task2\_llntukwnic    | 18.18            | 20.00                   | 16.67                |
    ...


- **Task 3 (first place): The Statute Law Retrieval Task**

    The COLIEE statute law competition focuses on two aspects of legal information processing related to answering yes/no questions from Japanese legal bar exams (the relevant data sets have been translated from Japanese to English).

    Task 3 of the legal question answering task involves reading a legal bar exam question Q, and extracting a subset of Japanese Civil Code Articles S1, S2,..., Sn from the entire Civil Code which are those appropriate for answering the question such that

    Entails(S1, S2, ..., Sn , Q) or Entails(S1, S2, ..., Sn , not Q).

    Given a question Q and the entire Civil Code Articles, we have to retrieve the set of "S1, S2, ..., Sn" as the answer of this track. 


    | Run            | F2    | P     | R     | MAP   | R5    | R10   | R30   |
    |-------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
    | **CAPTAIN**.allEnssMissq  | **75.69** | **72.61** | 79.21          | 69.21          | 75.38          | 83.85          | 88.46          |
    | **CAPTAIN**.allEnssBoostraping  | 74.70          | 71.62          | 78.22          | 69.21          | 75.38          | 83.85          | 88.46          |
    | JNLP3                   | 74.51          | 64.52          | **82.18** | 70.99          | 80.00          | 83.85          | 90.00          |
    | **CAPTAIN**.bjpAll | 74.15          | 70.63          | 77.72          | **84.64** | **87.69** | **90.77** | **96.15** |
    | NOWJ.ensemble           | 72.73          | 68.23          | 76.73          | 78.99          | 78.46          | 80.77          | 89.23          |
    | LLNTUgigo               | 65.35          | 73.27          | 64.36          | 76.43          | 80.00          | 88.46          | 91.54          |
    | UA.TfIdf\_threshold2    | 56.42          | 62.05          | 56.44          | 65.51          | 66.92          | 79.23          | 84.62          |
    ...

- **Task 4: The Legal Textual Entailment Data Corpus**

    Task 4 of the legal textual entailment task involves the identification of an entailment relationship such that

    Entails(S1, S2, ..., Sn , Q) or Entails(S1, S2, ..., Sn , not Q). 

    Given a question Q, we have to retrieve relevant articles S1, S2, ..., Sn through phase one, and then we have to determine if the relevant articles entail "Q" or "not Q". The answer of this track is binary: "YES"("Q") or "NO"("not Q").

    | Run            | Accuracy (\%) |
    |-----------------------|------------------------|
    | JNLP3                 | **78.22**        |
    | TRLABS\_D             | 78.22                  |
    | KIS2                  | 69.31                  |
    | UA-V2                 | 66.34                  |
    | AMHR01                | 65.35                  |
    | LLNTUdulcsL           | 62.38                  |
    | HUKB2                 | 59.41                  |
    | **CAPTAIN**.gen  | 58.42                  |
    | **CAPTAIN**.run1| 57.43                  |
    | NOWJ.multi-v1-jp      | 54.46                  |
    | **CAPTAIN**.run2 | 52.48                  |
    | NOWJ.multijp          | 52.48                  |
    | NOWJ.multi-v1-en      | 48.51                  |

---

## Data
Please visit [COLIEE 2023](https://sites.ualberta.ca/~rabelo/COLIEE2023/) for whole dataset request.

- structure of data directory (same structure between enlgish and japanese datasets)
    ```
        data/COLIEE2021statute_data-English/ 
        data/COLIEE2021statute_data-Japanese/
        ├── data_jp_topk_150_r02_r03 # folder save output of data_generator
        │   ├── dev.csv
        │   ├── stats.txt
        │   ├── test.csv
        │   ├── tfidf_classifier.pkl
        │   ├── all_data.json
        │   └── train.csv
        ├── text
        │   └── civil_code_jp-1to724-2.txt
        └── train
            ├── riteval_H18_en.xml
            ├── ...
            ├── riteval_R02_jp.xml
            ├── riteval_R03_jp.xml
            └── riteval_R04_jp.xml
    ```

## Environments
```bash 
conda create -n env_coliee python=3.8
conda activate env_coliee
pip install -r requirements.txt
```

## All runs: 
1. Use vscode debugging for better checking runs: config in file `.vscode/launch.json`
2. **Runs**:
   1. **baseline model using BERT** 
      1. generate data, extract raw data from COLIEE competition to the `.json` and `.csv` data for training process: 
            
            ```bash
            conda activate env_coliee  

            # R02, R03 should be in folder data `data/COLIEE2023statute_data-English/` and `data/COLIEE2023statute_data-Japanese/`
            python src/data_generator.py --path_folder_base data/COLIEE2023statute_data-Japanese/ --meta_data_alignment data/COLIEE2023statute_data-English/ --path_output_dir data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02_r03/ --lang jp --topk 150 --type_data task3 --dev_ids R02 --test_ids R03  

            # R02, R03, R04 should be in folder data `data/COLIEE2023statute_data-English/` and `data/COLIEE2023statute_data-Japanese/`
            python src/data_generator.py --path_folder_base data/COLIEE2023statute_data-Japanese/ --meta_data_alignment data/COLIEE2023statute_data-English/ --path_output_dir data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02r03_r04/ --lang jp --topk 150 --type_data task3 --dev_ids R02 R03 --test_ids R04 

            # because the train data of `data_ja_topk_150_r02_r03` contained R04, so we need to replace it with train data in `data_ja_topk_150_r02r03_r04` 
            cp  data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02r03_r04/test.csv data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02_r03/test_submit.csv

            cp  data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02r03_r04/train.csv data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02_r03/train.csv

            ``` 
            Output (recall score is important)
            ```
            ...
            127 16350 131 P:  0.0077675840978593275 R:  0.9694656488549618 F1:  0.015411686184090771 F2:  0.037631859665757966
            [W] Learning Tfidf Vectorizer ...
            970 120900 1040 P:  0.008023159636062862 R:  0.9326923076923077 F1:  0.0159094636706577 F2:  0.03878138493523109
            Number data pairs:  120970
            127 16350 131 P:  0.0077675840978593275 R:  0.9694656488549618 F1:  0.015411686184090771 F2:  0.037631859665757966
            Number data pairs:  16350
            99 12150 101 P:  0.008148148148148147 R:  0.9801980198019802 F1:  0.01616194596359481 F2:  0.039429663852158674
            Number data pairs:  12150
            len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id) =  120970 16350 12150
            ```
      2. train model by finetuning BERT or pretrained Japanese model (https://huggingface.co/cl-tohoku/bert-base-japanese-v2 or https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking )
            ```bash
            mkdir settings # folder save output 
            
            conda activate env_coliee && cd scripts && bash train_new.sh && cd ..
            ``` 
            or 
            ```
             mkdir settings && qsub scripts/train_new.sh
            ```
            after run all, all aodel checkpoint are generated, the enssemble method is output of setting `Mckpt` in paper. 
      3. infer and evaluation 
            ```bash
            # infer 
            conda activate env_coliee && python  src/evaluate.py --input_test data/TestData_en_r04.xml --input_prediction settings/bert-base-japanese-whole-word-masking_new2_top150-newE5Seq512L2e-5/models/CAPTAIN.allEnss.R04.enss.tsv --civi_code_path data/en_civil_code.json 
            ```
            where `data/TestData_en_r04.xml` is R04 orgininal gold test data.




## Citations
Our team at COLIEE2024 (pls check [task3_CAPTAIN at COLIEE2024](https://github.com/phuongnm94/captain-coliee/tree/coliee2024/exp_t3) to keep up-to-date SOTA performanceof this task)
```bib
@InProceedings{10.1007/978-981-97-3076-6_9,
author="Nguyen, Phuong
and Nguyen, Cong
and Nguyen, Hiep
and Nguyen, Minh
and Trieu, An
and Nguyen, Dat
and Nguyen, Le-Minh",
editor="Suzumura, Toyotaro
and Bono, Mayumi",
title="CAPTAIN at COLIEE 2024: Large Language Model for Legal Text Retrieval and Entailment",
booktitle="New Frontiers in Artificial Intelligence",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="125--139",
abstract="Recently, the Large Language Models (LLMs) has made a great contribution to massive Natural Language Processing (NLP) tasks. This year, our team, CAPTAIN, utilizes the power of LLM for legal information extraction tasks of the COLIEE competition. To this end, the LLMs are used to understand the complex meaning of legal documents, summarize the important points of legal statute law articles as well as legal document cases, and find the relations between them and specific legal cases. By using various prompting techniques, we explore the hidden relation between the legal query case and its relevant statute law as supplementary information for testing cases. The experimental results show the promise of our approach, with first place in the task of legal statute law entailment, competitive performance to the State-of-the-Art (SOTA) methods on tasks of legal statute law retrieval, and legal case entailment in the COLIEE 2024 competition. Our source code and experiments are available at https://github.com/phuongnm94/captain-coliee/tree/coliee2024.",
isbn="978-981-97-3076-6"
}
```

Our team at COLIEE 2023
```bib
@misc{nguyen2024captain,
      title={CAPTAIN at COLIEE 2023: Efficient Methods for Legal Information Retrieval and Entailment Tasks}, 
      author={Chau Nguyen and Phuong Nguyen and Thanh Tran and Dat Nguyen and An Trieu and Tin Pham and Anh Dang and Le-Minh Nguyen},
      year={2024},
      eprint={2401.03551},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
 
##  License
MIT-licensed. 
