"""
precompute_retrieval_candidates.py
===================================
把三路召回的结果预先缓存成 dict 文件，供 train_bagging_reranker.py 使用。

key  : "{session_id}_{user_turn_number}"
value: List[track_id]   (三路 union，最多 600 个)

Usage:
    python precompute_retrieval_candidates.py \\
        --config config/llama1b_multi_channel_devset.yaml \\
        --conv_emb_store    qwen/hist_conversation_embeddings_train_0.6b.pt \\
        --query_split_store qwen/query_split_train.pt \\
        --cf_model_path     qwen/cf_bpr_retrieval/model.pt \\
        --ch3_model_path    qwen/qwen_meta_tower/model.pt \\
        --split             train \\
        --out               qwen/retrieval_train_candidates.pt
"""

import argparse, json, logging, os
from typing import Dict, List, Optional, Set
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from omegaconf import OmegaConf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── re-use model definitions from eval_retrieval_accuracy.py ─────────────────
import importlib, sys
sys.path.insert(0, os.path.dirname(__file__))
# We inline only what we need to avoid circular import issues.

import torch.nn as nn

CONV_EMB_DIM = 1024
CF_BPR_DIM   = 128
OUTPUT_DIM   = 128
RECALL_PER_CH = 300  # 每路召回保留前300条
CAND_PER_CH   = 100  # 此脚本当前取每路前100条，生成300候选集

# ── Three-Tower (ch1) ─────────────────────────────────────────────────────────
AGE_GROUP_DIM = 8; GENDER_DIM = 4; COUNTRY_DIM = 32
USER_PROFILE_DIM = AGE_GROUP_DIM + GENDER_DIM + COUNTRY_DIM
USER_INPUT_DIM   = CF_BPR_DIM + USER_PROFILE_DIM
ATTR_PROJ_DIM = 64; META_PROJ_DIM = 64
POP_BUCKET_DIM = 8; RELEASE_YEAR_BUCKET_DIM = 8; DURATION_BUCKET_DIM = 8
ITEM_META_DIM  = POP_BUCKET_DIM + RELEASE_YEAR_BUCKET_DIM + DURATION_BUCKET_DIM
ITEM_INPUT_DIM = CF_BPR_DIM + ATTR_PROJ_DIM + META_PROJ_DIM + ITEM_META_DIM

def _hash_bucket(v, n): return 0 if not v else abs(hash(v)) % n
def _pop_bucket(p):
    try: p=float(p)
    except: return 0
    for thr,b in [(10,1),(20,2),(35,3),(50,4),(65,5),(80,6)]:
        if p<thr: return b
    return 7
def _year_bucket(rd):
    try: yr=int(str(rd)[:4])
    except: return 0
    for thr,b in [(1970,1),(1980,2),(1990,3),(2000,4),(2005,5),(2010,6),(2015,7)]:
        if yr<thr: return b
    return 7
def _dur_bucket(d):
    try: d=float(d)
    except: return 0
    for thr,b in [(60000,1),(120000,2),(180000,3),(210000,4),(240000,5),(300000,6)]:
        if d<thr: return b
    return 7
def _raw_to_tensor(v, dim):
    if v is None: return torch.zeros(dim)
    try:
        t=torch.tensor(v,dtype=torch.float32)
        if t.ndim==0 or t.numel()==0: return torch.zeros(dim)
        if t.ndim>1: t=t.flatten()
        if t.shape[0]<dim: t=F.pad(t,(0,dim-t.shape[0]))
        elif t.shape[0]>dim: t=t[:dim]
        return t
    except: return torch.zeros(dim)

class UserTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.age_emb=nn.Embedding(32,AGE_GROUP_DIM)
        self.gender_emb=nn.Embedding(8,GENDER_DIM)
        self.country_emb=nn.Embedding(512,COUNTRY_DIM)
        self.net=nn.Sequential(nn.Linear(USER_INPUT_DIM,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,OUTPUT_DIM))
    def forward(self,cf,age,gen,cnt):
        return self.net(torch.cat([cf,self.age_emb(age),self.gender_emb(gen),self.country_emb(cnt)],1))

class QueryTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(CONV_EMB_DIM,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,OUTPUT_DIM))
    def forward(self,x): return self.net(x)

class ItemTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.attr_proj=nn.Linear(1024,ATTR_PROJ_DIM)
        self.meta_proj=nn.Linear(1024,META_PROJ_DIM)
        self.pop_emb=nn.Embedding(8,POP_BUCKET_DIM)
        self.year_emb=nn.Embedding(8,RELEASE_YEAR_BUCKET_DIM)
        self.dur_emb=nn.Embedding(8,DURATION_BUCKET_DIM)
        self.net=nn.Sequential(nn.Linear(ITEM_INPUT_DIM,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,OUTPUT_DIM))
    def forward(self,cf,attr,meta,pop,year,dur):
        return self.net(torch.cat([cf,self.attr_proj(attr),self.meta_proj(meta),self.pop_emb(pop),self.year_emb(year),self.dur_emb(dur)],1))

class CFBPRThreeTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_tower=UserTower(); self.query_tower=QueryTower(); self.item_tower=ItemTower()
        self.gate_linear=nn.Linear(OUTPUT_DIM*2,OUTPUT_DIM)
    def encode_fusion(self,cf,age,gen,cnt,conv):
        u=self.user_tower(cf,age,gen,cnt); q=self.query_tower(conv)
        g=torch.sigmoid(self.gate_linear(torch.cat([u,q],1)))
        return F.normalize(g*u+(1-g)*q,p=2,dim=1)
    def encode_item(self,cf,attr,meta,pop,year,dur):
        return F.normalize(self.item_tower(cf,attr,meta,pop,year,dur),p=2,dim=1)

# ── QwenMeta Two-Tower (ch3) ──────────────────────────────────────────────────
class _TowerMLP(nn.Module):
    def __init__(self,in_dim,h1=512,h2=256,out=128,dropout=0.2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,h1),nn.BatchNorm1d(h1),nn.ReLU(),nn.Dropout(dropout),nn.Linear(h1,h2),nn.BatchNorm1d(h2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(h2,out))
    def forward(self,x): return self.net(x)

class QwenMetaTwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_tower=_TowerMLP(1024); self.item_tower=_TowerMLP(1024)
    def encode_query(self,x): return F.normalize(self.query_tower(x),p=2,dim=1)
    def encode_item(self,x):  return F.normalize(self.item_tower(x), p=2,dim=1)


# ─────────────────────────────────────────────────────────────────────────────

def _QUERY_SPLIT_FIELDS():
    return ["artist","album","genre","decade","language","popularity","scene","tempo"]

def retrieve_bm25(bm25_model, bm25_ids, user_query, qs_dict, topk):
    import bm25s as _bm25s
    fields = _QUERY_SPLIT_FIELDS()
    kws = []
    if qs_dict:
        for f in fields:
            v = qs_dict.get(f)
            if not v: continue
            kw = " ".join(str(x) for x in v) if isinstance(v,list) else str(v).strip()
            if kw: kws.append(kw)
    def _single(q,k):
        if not q.strip(): return []
        tok=_bm25s.tokenize([q.lower()])
        res=bm25_model.retrieve(tok,k=min(k,len(bm25_ids)),return_as="tuple")
        return [bm25_ids[x["id"]] for x in res.documents[0]]
    if not kws: return _single(user_query, topk)
    n=len(kws); per=topk//n; rem=topk-per*n
    seen=set(); out=[]
    for i,kw in enumerate(kws):
        ki=per+(rem if i==n-1 else 0)
        for tid in _single(kw,ki):
            if tid not in seen: out.append(tid); seen.add(tid)
    return out


def main(args):
    config       = OmegaConf.load(args.config)
    dataset_name = config.get("conversation_dataset_name","talkpl-ai/TalkPlayData-Challenge-Dataset")
    track_emb_db = config.get("track_emb_db_name","talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    track_meta_db= config.get("item_db_name","talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    user_emb_db  = config.get("user_emb_db_name","talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    user_meta_db = config.get("user_db_name","talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    split_types  = list(config.get("track_split_types",["all_tracks"]))

    # ── load embeddings ───────────────────────────────────────────────────
    logger.info("Loading track embeddings …")
    ds=load_dataset(track_emb_db); valid=[s for s in split_types if s in ds.keys()] or list(ds.keys())
    emb_ds=concatenate_datasets([ds[s] for s in valid])
    track_data={}
    for row in tqdm(emb_ds,desc="track embs"):
        tid=row["track_id"]
        track_data[tid]={
            "cf-bpr":   _raw_to_tensor(row.get("cf-bpr"),CF_BPR_DIM),
            "attr_emb": _raw_to_tensor(row.get("attributes-qwen3_embedding_0.6b"),1024),
            "meta_emb": _raw_to_tensor(row.get("metadata-qwen3_embedding_0.6b"),1024),
        }
    ds_m=load_dataset(track_meta_db); valid_m=[s for s in split_types if s in ds_m.keys()] or list(ds_m.keys())
    for row in concatenate_datasets([ds_m[s] for s in valid_m]):
        tid=row["track_id"]
        if tid not in track_data: track_data[tid]={"cf-bpr":torch.zeros(CF_BPR_DIM),"attr_emb":torch.zeros(1024),"meta_emb":torch.zeros(1024)}
        track_data[tid]["pop"]=_pop_bucket(row.get("popularity")); track_data[tid]["year"]=_year_bucket(row.get("release_date")); track_data[tid]["dur"]=_dur_bucket(row.get("duration"))
    for d in track_data.values():
        d.setdefault("pop",0); d.setdefault("year",0); d.setdefault("dur",0)

    logger.info("Loading user data …")
    ds_u=load_dataset(user_emb_db); valid_u=[s for s in split_types if s in ds_u.keys()] or list(ds_u.keys())
    user_data={}
    for row in concatenate_datasets([ds_u[s] for s in valid_u]):
        uid=row["user_id"]; user_data[uid]={"cf-bpr":_raw_to_tensor(row.get("cf-bpr"),CF_BPR_DIM)}
    ds_um=load_dataset(user_meta_db); valid_um=[s for s in split_types if s in ds_um.keys()] or list(ds_um.keys())
    for row in concatenate_datasets([ds_um[s] for s in valid_um]):
        uid=row["user_id"]
        if uid not in user_data: user_data[uid]={"cf-bpr":torch.zeros(CF_BPR_DIM)}
        user_data[uid]["age_idx"]=_hash_bucket(str(row.get("age_group","") or ""),32)
        user_data[uid]["gender_idx"]=_hash_bucket(str(row.get("gender","") or ""),8)
        user_data[uid]["country_idx"]=_hash_bucket(str(row.get("country_name","") or ""),512)
    for d in user_data.values():
        d.setdefault("age_idx",0); d.setdefault("gender_idx",0); d.setdefault("country_idx",0)

    # ── ch1 three-tower model ─────────────────────────────────────────────
    ch1_model=None; ch1_matrix=None; ch1_ids=None
    if os.path.exists(args.cf_model_path):
        logger.info("Loading three-tower model …")
        ch1_model=CFBPRThreeTower()
        ch1_model.load_state_dict(torch.load(args.cf_model_path,map_location="cpu",weights_only=True)["model_state_dict"])
        ch1_model.eval()
        ch1_ids=sorted(track_data.keys()); bs=512; vecs=[]
        with torch.no_grad():
            for s in tqdm(range(0,len(ch1_ids),bs),desc="ch1 item index"):
                b=[ch1_ids[s:s+bs]]; b=ch1_ids[s:s+bs]
                cf=torch.stack([track_data[t]["cf-bpr"] for t in b])
                attr=torch.stack([track_data[t]["attr_emb"] for t in b])
                meta=torch.stack([track_data[t]["meta_emb"] for t in b])
                pop=torch.tensor([track_data[t]["pop"] for t in b],dtype=torch.long)
                year=torch.tensor([track_data[t]["year"] for t in b],dtype=torch.long)
                dur=torch.tensor([track_data[t]["dur"] for t in b],dtype=torch.long)
                vecs.append(ch1_model.encode_item(cf,attr,meta,pop,year,dur).cpu())
        ch1_matrix=torch.cat(vecs,0)
        logger.info("ch1 item matrix: %s",list(ch1_matrix.shape))

    # ── ch3 qwen-meta two-tower model ─────────────────────────────────────
    ch3_model=None; ch3_matrix=None; ch3_ids=None
    ch3_idx_dir=os.path.join("qwen","qwen_meta_tower","item_index")
    if os.path.exists(args.ch3_model_path):
        logger.info("Loading QwenMeta model …")
        ch3_model=QwenMetaTwoTower()
        ch3_model.load_state_dict(torch.load(args.ch3_model_path,map_location="cpu",weights_only=True)["model_state_dict"])
        ch3_model.eval()
        vec_path=os.path.join(ch3_idx_dir,"item_vectors.pt"); ids_path=os.path.join(ch3_idx_dir,"track_ids.json")
        if os.path.exists(vec_path):
            ch3_matrix=torch.load(vec_path,map_location="cpu",weights_only=True)
            with open(ids_path) as f: ch3_ids=json.load(f)
        else:
            ch3_ids=sorted(track_data.keys()); bs=512; vecs=[]
            with torch.no_grad():
                for s in tqdm(range(0,len(ch3_ids),bs),desc="ch3 item index"):
                    b=ch3_ids[s:s+bs]; me=torch.stack([track_data[t]["meta_emb"] for t in b])
                    vecs.append(ch3_model.encode_item(me).cpu())
            ch3_matrix=torch.cat(vecs,0)
        logger.info("ch3 item matrix: %s",list(ch3_matrix.shape))

    # ── BM25 ──────────────────────────────────────────────────────────────
    import bm25s as _bm25s
    bm25_dir=os.path.join("qwen","retrieval_indices","bm25_index")
    bm25_model=_bm25s.BM25.load(bm25_dir,load_corpus=True)
    with open(os.path.join(bm25_dir,"track_ids.json")) as f: bm25_ids=json.load(f)

    # ── conv_emb + query_split stores ─────────────────────────────────────
    logger.info("Loading conv_emb store …")
    conv_store=torch.load(args.conv_emb_store,map_location="cpu",weights_only=True)
    qs_store=None
    if args.query_split_store and os.path.exists(args.query_split_store):
        qs_store=torch.load(args.query_split_store,map_location="cpu",weights_only=True)

    # ── iterate dataset ───────────────────────────────────────────────────
    logger.info("Loading dataset split='%s' …", args.split)
    ds=load_dataset(dataset_name,split=args.split)
    candidates: Dict[str,List[str]] = {}

    for item in tqdm(ds,desc="Computing candidates",unit="session"):
        session_id=item["session_id"]; user_id=item.get("user_id")
        convs=item["conversations"]
        music_turns={int(c["turn_number"]):c["content"] for c in convs if c.get("role")=="music" and c.get("content")}

        for turn_num,_ in music_turns.items():
            emb_key=f"{session_id}_{turn_num}"
            user_turn=turn_num
            if emb_key not in conv_store:
                emb_key=f"{session_id}_{turn_num-1}"; user_turn=turn_num-1
            if emb_key not in conv_store: continue
            conv_emb=conv_store[emb_key].float()
            if conv_emb.shape[0]>CONV_EMB_DIM: conv_emb=conv_emb[:CONV_EMB_DIM]
            elif conv_emb.shape[0]<CONV_EMB_DIM: conv_emb=F.pad(conv_emb,(0,CONV_EMB_DIM-conv_emb.shape[0]))

            cands_ch1: List[str] = []
            cands_ch3: List[str] = []
            cands_ch5: List[str] = []

            # ch1
            if ch1_model is not None:
                u=user_data.get(user_id,{}) if user_id else {}
                with torch.no_grad():
                    u_cf=u.get("cf-bpr",torch.zeros(CF_BPR_DIM)).float().unsqueeze(0)
                    age=torch.tensor([u.get("age_idx",0)],dtype=torch.long)
                    gen=torch.tensor([u.get("gender_idx",0)],dtype=torch.long)
                    cnt=torch.tensor([u.get("country_idx",0)],dtype=torch.long)
                    fv=ch1_model.encode_fusion(u_cf,age,gen,cnt,conv_emb.unsqueeze(0)).squeeze(0)
                sc=(ch1_matrix*fv.unsqueeze(0)).sum(1)
                top=torch.topk(sc,min(RECALL_PER_CH,sc.shape[0])).indices.tolist()
                cands_ch1=[ ch1_ids[i] for i in top ]

            # ch3
            if ch3_model is not None:
                with torch.no_grad():
                    qv=ch3_model.encode_query(conv_emb.unsqueeze(0)).squeeze(0)
                sc=(ch3_matrix*qv.unsqueeze(0)).sum(1)
                top=torch.topk(sc,min(RECALL_PER_CH,sc.shape[0])).indices.tolist()
                cands_ch3=[ ch3_ids[i] for i in top ]

            # ch5 BM25
            user_query=""
            for c in convs:
                if int(c["turn_number"])==user_turn and c["role"]=="user":
                    user_query=c.get("content",""); break
            qs_dict=None
            if qs_store:
                raw=qs_store.get(emb_key)
                if raw:
                    try: qs_dict=json.loads(raw) if isinstance(raw,str) else raw
                    except: pass
            cands_ch5=retrieve_bm25(bm25_model,bm25_ids,user_query,qs_dict,RECALL_PER_CH)

            # 反序列化存储：每路最多 RECALL_PER_CH 条，后续可按 CAND_PER_CH 截取
            def _dedup(lst):
                seen=set(); out=[]
                for t in lst:
                    if t not in seen: out.append(t); seen.add(t)
                return out[:RECALL_PER_CH]

            candidates[emb_key] = {
                "ch1": _dedup(cands_ch1),
                "ch3": _dedup(cands_ch3),
                "ch5": _dedup(cands_ch5),
            }
            # 建立 union（每路 top CAND_PER_CH），方便直接用于负采样
            union=_dedup(cands_ch1[:CAND_PER_CH]+cands_ch3[:CAND_PER_CH]+cands_ch5[:CAND_PER_CH])
            candidates[emb_key]["union"] = union

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".",exist_ok=True)
    torch.save(candidates,args.out)
    logger.info("Saved %d entries to %s",len(candidates),args.out)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config",           type=str, default="config/llama1b_multi_channel_devset.yaml")
    p.add_argument("--conv_emb_store",   type=str, default="qwen/hist_conversation_embeddings_train_0.6b.pt")
    p.add_argument("--query_split_store",type=str, default="qwen/query_split_train.pt")
    p.add_argument("--cf_model_path",    type=str, default="qwen/cf_bpr_retrieval/model.pt")
    p.add_argument("--ch3_model_path",   type=str, default="qwen/qwen_meta_tower/model.pt")
    p.add_argument("--split",            type=str, default="train")
    p.add_argument("--out",              type=str, default="qwen/retrieval_train_candidates.pt")
    return p.parse_args()

if __name__=="__main__":
    main(parse_args())
