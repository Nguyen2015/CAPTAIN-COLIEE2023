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
            └── riteval_R01_en.xml
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
            conda activate env_coliee && cd src/ && python src/data_generator.py --path_folder_base data/COLIEE2023statute_data-Japanese/ --meta_data_alignment data/COLIEE2023statute_data-English/ --path_output_dir data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02_r03/ --lang jp --topk 150 --type_data task3 --dev_ids R02 --test_ids R03  
            ``` 
            Output (recall score is important)
            ```
            ...
            - 954 116400 1003 P:  0.008195876288659793 R:  0.9511465603190429 F1:  0.016251714181068626 F2:  0.03961399196093413  # eval train set using tfidf top 150
            Number data pairs:  116449
            - 128 16650 138 P:  0.0076876876876876875 R:  0.927536231884058 F1:  0.015248987371932332 F2:  0.03720497616556215  # eval valid set using tfidf top 150
            Number data pairs:  16650 
            - len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id) =  116449 16650 16650
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
      3. infer and evaluation  


## Citation
 
```bib
@inproceedings{CAPTAIN_AT_COLIEE_2023, 
author = {Chau Nguyen, Phuong Nguyen, Thanh Tran, Dat Nguyen, An Trieu, Tin Pham, Anh Dang, Le-Minh Nguyen}, 
title = {CAPTAIN at COLIEE 2023: Efficient Methods for Legal Information Retrieval and Entailment Tasks}, 
year = {2023}, 
publisher = {Association for Computing Machinery}, 
booktitle = {Proceedings of the Seventeenth International Conference on Artificial Intelligence and Law}, 
url = {http://research.nii.ac.jp/~ksatoh/COLIEE2023proceedings.pdf},
series = {ICAIL '23} 
}
```
 
##  License
MIT-licensed. 
