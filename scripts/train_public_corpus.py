#!/usr/bin/env python3
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class BinSampler:
    def __init__(self,path,dtype,seq_len,seed):
        self.a=np.memmap(path,dtype=np.dtype(dtype),mode='r'); self.n=len(self.a); self.seq=seq_len; self.rng=np.random.default_rng(seed)
        if self.n < seq_len+2: raise ValueError(f'corpus too small: {self.n}')
    def batch(self,b):
        starts=self.rng.integers(0,self.n-self.seq-1,size=b)
        x=np.stack([np.asarray(self.a[s:s+self.seq],dtype=np.int64) for s in starts])
        # Hugging Face causal-LM heads shift labels internally.  Passing an
        # already shifted target would accidentally train a two-token offset.
        return torch.from_numpy(x),torch.from_numpy(x.copy())

def evaluate(model,arr,dtype,seq,batch,batches,device):
    a=np.memmap(arr,dtype=np.dtype(dtype),mode='r'); model.eval(); vals=[]
    with torch.no_grad():
        for i in range(batches):
            start=(i*batch*seq) % max(1,len(a)-batch*seq-1)
            xs=[]; ys=[]
            for j in range(batch):
                s=start+j*seq; xs.append(np.asarray(a[s:s+seq],dtype=np.int64)); ys.append(np.asarray(a[s:s+seq],dtype=np.int64))
            x=torch.from_numpy(np.stack(xs)).to(device,non_blocking=True); y=torch.from_numpy(np.stack(ys)).to(device,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): vals.append(float(model(input_ids=x,labels=y,use_cache=False).loss))
    model.train(); return float(np.mean(vals))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True); p.add_argument('--model-config',default='EleutherAI/pythia-1b'); p.add_argument('--tokenizer',default='EleutherAI/pythia-1b')
    p.add_argument('--train-bin',required=True); p.add_argument('--val-bin',required=True); p.add_argument('--dtype',default='uint16')
    p.add_argument('--resume-model'); p.add_argument('--steps',type=int,default=200); p.add_argument('--seq-len',type=int,default=2048); p.add_argument('--micro-batch',type=int,default=4); p.add_argument('--grad-accum',type=int,default=1)
    p.add_argument('--lr',type=float,default=4e-4); p.add_argument('--beta1',type=float,default=.9); p.add_argument('--beta2',type=float,default=.95); p.add_argument('--weight-decay',type=float,default=.1); p.add_argument('--warmup-ratio',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--eval-batches',type=int,default=12); p.add_argument('--save-model',action='store_true'); p.add_argument('--tag',default='run')
    p.add_argument('--optimizer',choices=('adamw','adafactor'),default='adamw')
    p.add_argument('--checkpoint-every',type=int,default=0)
    p.add_argument('--resume-state')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision('high'); device=torch.device('cuda')
    tok=AutoTokenizer.from_pretrained(a.tokenizer,use_fast=True); tok.pad_token=tok.pad_token or tok.eos_token
    if a.resume_model:
        model=AutoModelForCausalLM.from_pretrained(a.resume_model,torch_dtype=torch.float32,attn_implementation='sdpa')
    else:
        cfg=AutoConfig.from_pretrained(a.model_config); cfg.use_cache=False
        try: model=AutoModelForCausalLM.from_config(cfg,attn_implementation='sdpa')
        except TypeError: model=AutoModelForCausalLM.from_config(cfg)
        # GPT-NeoX residual branches need depth-scaled output projections for
        # a sane scratch-pretraining baseline at 1B scale.
        residual_scale=math.sqrt(2*cfg.num_hidden_layers)
        with torch.no_grad():
            for name,param in model.named_parameters():
                if name.endswith('attention.dense.weight') or name.endswith('mlp.dense_4h_to_h.weight'):
                    param.div_(residual_scale)
    model.to(device=device,dtype=torch.bfloat16); model.train(); model.config.use_cache=False
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    nparams=sum(x.numel() for x in model.parameters())
    if a.optimizer == 'adamw':
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(a.beta1,a.beta2),weight_decay=a.weight_decay,fused=True)
    else:
        opt=torch.optim.Adafactor(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    start_step=0; seen=0; losses=[]
    state_path=Path(a.resume_state) if a.resume_state else out/'training_state.pt'
    if a.resume_state and state_path.exists():
        saved=torch.load(state_path,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        start_step=int(saved['step']); seen=int(saved['tokens_seen']); losses=list(saved['losses'])
    warm=max(1,int(a.steps*a.warmup_ratio))
    def lr_mult(step):
        if step < warm: return (step+1)/warm
        q=(step-warm)/max(1,a.steps-warm); return 0.1+0.9*0.5*(1+math.cos(math.pi*q))
    sampler=BinSampler(a.train_bin,a.dtype,a.seq_len,a.seed)
    torch.cuda.reset_peak_memory_stats(); t0=time.time()
    for step in range(start_step,a.steps):
        opt.zero_grad(set_to_none=True); total=0.0
        for _ in range(a.grad_accum):
            x,y=sampler.batch(a.micro_batch); x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(input_ids=x,labels=y,use_cache=False).loss/a.grad_accum
            loss.backward(); total += float(loss)*a.grad_accum; seen += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        mult=lr_mult(step); [g.__setitem__('lr',a.lr*mult) for g in opt.param_groups]
        losses.append(total)
        if a.checkpoint_every and (step+1)%a.checkpoint_every==0 and step+1<a.steps:
            temporary=state_path.with_suffix(state_path.suffix+'.tmp')
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step+1,'tokens_seen':seen,'losses':losses},temporary)
            temporary.replace(state_path)
        if step%25==0 or step==a.steps-1: print(json.dumps({'tag':a.tag,'step':step,'loss':total,'lr':opt.param_groups[0]['lr'],'tokens':seen}),flush=True)
    torch.cuda.synchronize(); elapsed=time.time()-t0
    val=evaluate(model,a.val_bin,a.dtype,a.seq_len,max(1,min(a.micro_batch,2)),a.eval_batches,device)
    metrics={'tag':a.tag,'seed':a.seed,'params':nparams,'steps':a.steps,'start_step':start_step,'seq_len':a.seq_len,'micro_batch':a.micro_batch,'grad_accum':a.grad_accum,'optimizer':a.optimizer,'lr':a.lr,'beta1':a.beta1,'beta2':a.beta2,'weight_decay':a.weight_decay,'warmup_ratio':a.warmup_ratio,'train_loss_tail':float(np.mean(losses[-20:])),'val_loss':val,'tokens_seen':seen,'elapsed_sec':elapsed,'tokens_per_sec':(seen-start_step*a.seq_len*a.micro_batch*a.grad_accum)/elapsed,'peak_gpu_gb':torch.cuda.max_memory_allocated()/2**30,'gpu':torch.cuda.get_device_name(0),'resume_model':a.resume_model,'resume_state':a.resume_state}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    if a.save_model:
        model.save_pretrained(out/'model',safe_serialization=True,max_shard_size='5GB'); tok.save_pretrained(out/'model')
    if state_path.exists() and start_step < a.steps:
        state_path.unlink()
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
