#!/usr/bin/env python3
import argparse, array, json, os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


def write_bin(text_iter, out_path, tok, max_tokens=None):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    use_u16 = len(tok) < 65536
    code = 'H' if use_u16 else 'I'
    total = 0
    buf = array.array(code)
    with open(out_path, 'wb') as f:
        for text in text_iter:
            if not text: continue
            ids = tok(text, add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
            if max_tokens is not None and total + len(ids) > max_tokens:
                ids = ids[: max_tokens-total]
            buf.extend(ids); total += len(ids)
            if len(buf) >= 1_000_000:
                buf.tofile(f); buf = array.array(code)
            if max_tokens is not None and total >= max_tokens: break
        if buf: buf.tofile(f)
    return total, ('uint16' if use_u16 else 'uint32')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data'); ap.add_argument('--tokenizer',default='EleutherAI/pythia-1b')
    ap.add_argument('--train-max-tokens',type=int,default=120_000_000); ap.add_argument('--eval-max-tokens',type=int,default=8_000_000)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    wt=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1')
    ntrain,dtype=write_bin((x['text'] for x in wt['train']),out/'wikitext_train.bin',tok,args.train_max_tokens)
    neval,_=write_bin((x['text'] for x in wt['validation']),out/'wikitext_val.bin',tok,args.eval_max_tokens)
    gsm=load_dataset('openai/gsm8k','main')
    def gsm_text(split):
        for x in gsm[split]: yield f"Problem: {x['question']}\nSolution: {x['answer']}\n"
    ncpt,_=write_bin(gsm_text('train'),out/'gsm8k_cpt_train.bin',tok,20_000_000)
    ncptv,_=write_bin(gsm_text('test'),out/'gsm8k_cpt_val.bin',tok,5_000_000)
    meta={'tokenizer':args.tokenizer,'dtype':dtype,'wikitext_train_tokens':ntrain,'wikitext_val_tokens':neval,'gsm8k_cpt_train_tokens':ncpt,'gsm8k_cpt_val_tokens':ncptv}
    (out/'corpus_meta.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
