# datasets/
Only CoNLL-2003 test is stored here (`conll2003/test.parquet`, from
https://huggingface.co/api/datasets/tner/conll2003/parquet/conll2003/test/0.parquet).
All other datasets (SST-2, AG News, Emotion, BoolQ, SQuAD, IFEval) are read from the local
HF hub cache (`HF_HUB_DIR`, default `/data/transformers/hub`) by `src/build_prompts.py`.
