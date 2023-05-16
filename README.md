# CAPTAIN at COLIEE 2023: Efficient Methods for Legal Information Retrieval and Entailment Tasks 

The Competition on Legal Information Extraction/Entailment (COLIEE) is held annually to encourage advancements in the automatic processing of legal texts. Processing legal documents is challenging due to the intricate structure and meaning of legal language. In this paper, we outline our strategies for tackling Task 2, Task 3, and Task 4 in the COLIEE 2023 competition. Our approach involved utilizing appropriate state-of-the-art deep learning methods, designing methods based on domain characteristics observation, and applying meticulous engineering practices and methodologies to the competition. As a result, our performance in these tasks has been outstanding, with first places in Task 2 and Task 3, and promising results in Task 4.

## Data
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
 
updating ...
<!-- ```bib
@Article{Nguyen2022,
    author={Nguyen, Phuong Minh
    and Le, Tung
    and Nguyen, Huy Tien
    and Tran, Vu
    and Nguyen, Minh Le},
    title={PhraseTransformer: an incorporation of local context information into sequence-to-sequence semantic parsing},
    journal={Applied Intelligence},
}
```
 -->
##  License
MIT-licensed. 
