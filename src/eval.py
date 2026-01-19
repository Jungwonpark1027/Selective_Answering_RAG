# eval.py
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm
import math
import time


# ====== Prompt rendering (train과 동일하게 유지) ======
def render_user_prompt(query: str, contexts: List[Dict[str, Any]], max_contexts: Optional[int] = None) -> str:
    if max_contexts is not None:
        contexts = contexts[:max_contexts]

    lines: List[str] = []
    lines.append("다음은 질문과 참고 문서들입니다.")
    lines.append('문서들에 답이 있으면 간결하게 답변하세요.')
    lines.append('문서들에 답이 없으면 "모르겠습니다"라고만 답변하세요.')
    lines.append("")
    lines.append("[질문]")
    lines.append((query or "").strip())
    lines.append("")
    lines.append("[문서들]")

    k = 0
    for c in contexts:
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        k += 1
        lines.append(f"{k}. {txt}")

    if k == 0:
        lines.append("1. (문서 없음)")

    return "\n".join(lines)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ex["query"] = str(ex.get("query", "") or "")
            ex["target"] = str(ex.get("target", "") or "")
            ex["contexts"] = ex.get("contexts", []) or []
            try:
                ex["is_answerable"] = int(ex.get("is_answerable", 1))
            except Exception:
                ex["is_answerable"] = 1
            rows.append(ex)
    return rows


def normalize_text(s: str) -> str:
    return (s or "").strip()


def is_abstain(output: str, abstain_text: str) -> bool:
    o = normalize_text(output)
    a = normalize_text(abstain_text)
    if not o:
        return True
    if o == a:
        return True
    if o.startswith(a):  # 필요 없으면 제거 가능
        return True
    return False


#def exact_match(pred: str, gold: str) -> bool:
    #return normalize_text(pred) == normalize_text(gold) #이거 대신

def exact_match(pred: str, gold: str) -> bool:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p or not g:
        return False
    return (p == g) or (p in g) or (g in p)



@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    batch_messages = []
    for p in prompts:
        batch_messages.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p},
        ])

    input_ids_list = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True, return_tensors=None)
        for m in batch_messages
    ]

    max_len = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long, device=model.device)
    attn = torch.zeros((len(input_ids_list), max_len), dtype=torch.long, device=model.device)

    # for i, ids in enumerate(input_ids_list):
    #     ids_t = torch.tensor(ids, dtype=torch.long, device=model.device)
    #     input_ids[i, : ids_t.numel()] = ids_t
    #     attn[i, : ids_t.numel()] = 1

    for i, ids in enumerate(input_ids_list):
        ids_t = torch.tensor(ids, dtype=torch.long, device=model.device)
        L = ids_t.numel()
        input_ids[i, -L:] = ids_t      # <- 오른쪽 정렬(=left padding)
        attn[i, -L:] = 1




    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    outputs = []
    for i in range(gen.size(0)):
        # prompt_len = int(attn[i].sum().item())
        # out_ids = gen[i, prompt_len:]
        
        # 수정 
        input_len = input_ids.size(1)          # max_len (패딩 포함)
        out_ids = gen[i, input_len:]
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        outputs.append(text.strip())
    return outputs


def compute_metrics(rows: List[Dict[str, Any]], outputs: List[str], abstain_text: str) -> Dict[str, Any]:
    assert len(rows) == len(outputs)

    # decisions: True면 abstain, False면 answered
    abstain_flags = [is_abstain(o, abstain_text) for o in outputs]

    total = len(rows)

    # ===== mixed (전체) =====
    answered = 0
    correct_answered = 0

    un_total = 0
    un_answered = 0
    un_abstained = 0

    ans_total = 0
    ans_answered = 0
    ans_correct = 0
    ans_abstained = 0

    overall_correct = 0  # answerable correct + unanswerable abstain

    for ex, pred, ab in zip(rows, outputs, abstain_flags):
        ia = int(ex.get("is_answerable", 1))
        gold = str(ex.get("target", "") or "").strip()

        if ia == 0:
            un_total += 1
            if ab:
                un_abstained += 1
                overall_correct += 1
            else:
                un_answered += 1
        else:
            ans_total += 1
            if ab:
                ans_abstained += 1
            else:
                ans_answered += 1
                answered += 1
                if exact_match(pred, gold):
                    ans_correct += 1
                    correct_answered += 1
                    overall_correct += 1
    #수정
    answered_all = sum(1 for ab in abstain_flags if not ab)

    mixed = {
        "total": total,
        "answerable_total": ans_total,
        "unanswerable_total": un_total,
        "response_rate": answered_all / max(total, 1), #수정
        "cond_accuracy_answered": correct_answered / max(answered, 1),
        "hallucination_rate_unanswerable": un_answered / max(un_total, 1) if un_total else 0.0,
        "abstain_recall_unanswerable": un_abstained / max(un_total, 1) if un_total else 0.0,
        "overall_accuracy": overall_correct / max(total, 1),
    }

    # ===== answerable-only (정답 정확도 중심) =====
    # 여기서는 "답을 한 것 중 정확도" + "answerable에서의 응답률" 둘 다 주는게 유용
    answerable_only = {
        "answerable_total": ans_total,
        "answerable_response_rate": ans_answered / max(ans_total, 1) if ans_total else 0.0,
        "answerable_cond_accuracy_answered": ans_correct / max(ans_answered, 1) if ans_answered else 0.0,
        "answerable_abstain_rate": ans_abstained / max(ans_total, 1) if ans_total else 0.0,
    }

    # ===== unanswerable-only (안전/환각 중심) =====
    unanswerable_only = {
        "unanswerable_total": un_total,
        "unanswerable_answer_rate": un_answered / max(un_total, 1) if un_total else 0.0,
        "unanswerable_abstain_rate": un_abstained / max(un_total, 1) if un_total else 0.0,
    }

    return {
        "mixed": mixed,
        "answerable_only": answerable_only,
        "unanswerable_only": unanswerable_only,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_or_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)

    p.add_argument("--abstain_text", type=str, default="모르겠습니다")
    p.add_argument("--max_contexts", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=64)

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=4)

    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--save_samples", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_or_ckpt, use_fast=True)

    tokenizer.padding_side = "left"  # ✅ 중요: decoder-only는 left padding 권장
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        


    torch_dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_or_ckpt,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer)) #추가함 warning문제 #위치 수정
    model.eval()

    rows = load_jsonl(args.data_path)

    prompts = [
        render_user_prompt(ex["query"], ex["contexts"], max_contexts=args.max_contexts)
        for ex in rows
    ]

    outputs: List[str] = []
    bs = args.batch_size
    n_batches = math.ceil(len(prompts) / bs)

    t0 = time.time()
    for bi in tqdm(range(n_batches), desc="Generating", unit="batch"):
        i = bi * bs 
    #for i in range(0, len(prompts), bs): #tqdm으로 감쌈
        batch_prompts = prompts[i:i+bs]
        batch_out = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        outputs.extend(batch_out)

        done = bi + 1
        elapsed = time.time() - t0
        it_s = elapsed / done
        eta = it_s * (n_batches - done)
        tqdm.write(f"[progress] {done}/{n_batches} batches | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")

    metrics = compute_metrics(rows, outputs, args.abstain_text)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    payload = {
        "data_path": args.data_path,
        "model_or_ckpt": args.model_or_ckpt,
        "abstain_text": args.abstain_text,
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
        },
        "metrics": metrics,
        "samples": [
            {
                "qid": ex.get("qid"),
                "is_answerable": ex.get("is_answerable"),
                "query": ex.get("query"),
                "gold": ex.get("target"),
                "pred": out,
            }
            for ex, out in zip(rows[:args.save_samples], outputs[:args.save_samples])
        ],
    }

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Saved eval to:", args.out_path)
    print("Mixed:", metrics["mixed"])
    print("Answerable-only:", metrics["answerable_only"])
    print("Unanswerable-only:", metrics["unanswerable_only"])


if __name__ == "__main__":
    main()




