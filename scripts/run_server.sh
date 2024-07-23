
#  curl -d '{"sentence":"ＡＢ間の売買契約において、売主Ａが買主Ｂに対して引き渡した目的物の数量が不足しており、契約の内容に適合しない場合、Ｂが数量の不足を知った時から１年以内にその旨をＡに通知しない場合には、Ａが引渡しの時に数量の不足を知り又は重大な過失によって知らなかったときを除き、Ｂは損害賠償の請求をすることができない。(translated to english: In the contract for sale between A and B, the quantity of the object delivered by the seller A to the buyer B is insufficient and does not conform to the terms of the contract. If B fails to notify A of the non-conformity in the quantity within one year from the time when B becomes aware of it, B may not claim compensation for loss or damage, unless A knew or did not know due to gross negligence the non-conformity in the quantity at the time of the delivery.)" }' \
    # -H "Content-Type: application/json" -X POST http://spcc-a40g19:9002/coliee-task3-bert-cc

# curl http://spcc-a40g19:9002/random-sentence

CUDA_VISIBLE_DEVICES=1 python ./src/train.py \
    --data_dir ./data_coliee2024_done/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02_r03// \
    --model_name_or_path cl-tohoku/bert-base-japanese-whole-word-masking \
    --log_dir ./settings/bert-base-japanese_top150-newE5Seq512L1e-5/combination2models// \
    --no_train --run_server --port 9002