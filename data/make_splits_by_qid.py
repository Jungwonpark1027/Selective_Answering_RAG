# make_splits_by_qid.py
# 목적:
# - proposed / baselineB / baseline1 (qid 풀 동일) 을 입력으로 받아
#   1) proposed 기준(is_answerable)으로 qid를 stratified split (train/val/test)
#   2) 각 모델 학습용 train.jsonl을 따로 저장
#   3) 공통 검증/테스트(eval_val/eval_test)는 proposed-style GT로 저장
#
# 출력:
# out_dir/
#   train_qids.json
#   val_qids.json
#   test_qids.json
#   proposed_train.jsonl
#   baselineB_train.jsonl
#   baseline1_train.jsonl
#   eval_val.jsonl
#   eval_test.jsonl
#   (옵션) eval_val_answerable.jsonl, eval_test_answerable.jsonl, eval_val_unanswerable.jsonl, eval_test_unanswerable.jsonl
#

# baseline1_path는 고정임
"""
python /home/qa/data2/tmp/project/data/make_splits_by_qid.py \
  --proposed_path  /home/qa/data2/tmp/trainform_data/H_refined_g60_train.jsonl \
  --baselineB_path /home/qa/data2/tmp/trainform_data/H_refined_g60_train_baselineB.jsonl \
  --baseline1_path /home/qa/data2/tmp/project/data/origin/baselineA/refined_g100_train_bslineA.jsonl \
  --out_dir /home/qa/data2/tmp/project/data/g60_seed4222 \
  --seed 42 --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
  --save_subsets
"""

import os
import json
import random
import argparse
from typing import Dict, List, Tuple, Any, Set


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def qid_set(rows: List[Dict[str, Any]]) -> Set[Any]:
    return set(r["qid"] for r in rows)


def group_qids_to_label(rows: List[Dict[str, Any]]) -> Dict[Any, int]:
    """
    proposed rows 기준으로 qid -> is_answerable 를 만든다.
    같은 qid의 is_answerable은 일관되어야 함.
    """
    m: Dict[Any, int] = {}
    for r in rows:
        qid = r["qid"]
        ia = int(r.get("is_answerable", 1))
        if qid in m and m[qid] != ia:
            raise ValueError(f"qid {qid} has inconsistent is_answerable: {m[qid]} vs {ia}")
        m[qid] = ia
    return m


def stratified_qid_split(
    qid_to_label: Dict[Any, int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[Any], List[Any], List[Any]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    ans = [q for q, l in qid_to_label.items() if l == 1]
    unans = [q for q, l in qid_to_label.items() if l == 0]

    rnd = random.Random(seed)
    rnd.shuffle(ans)
    rnd.shuffle(unans)

    def cut(arr: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
        n = len(arr)
        n_tr = int(round(n * train_ratio))
        n_va = int(round(n * val_ratio))
        tr = arr[:n_tr]
        va = arr[n_tr:n_tr + n_va]
        te = arr[n_tr + n_va:]
        return tr, va, te

    ans_tr, ans_va, ans_te = cut(ans)
    un_tr, un_va, un_te = cut(unans)

    train_qids = ans_tr + un_tr
    val_qids = ans_va + un_va
    test_qids = ans_te + un_te

    rnd.shuffle(train_qids)
    rnd.shuffle(val_qids)
    rnd.shuffle(test_qids)

    return train_qids, val_qids, test_qids


def filter_by_qids(rows: List[Dict[str, Any]], qids: Set[Any]) -> List[Dict[str, Any]]:
    return [r for r in rows if r["qid"] in qids]


def filter_answerable(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if int(r.get("is_answerable", 1)) == 1]


def filter_unanswerable(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if int(r.get("is_answerable", 1)) == 0]


def answerable_ratio(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    a = sum(int(r.get("is_answerable", 1)) == 1 for r in rows)
    return a / len(rows)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--proposed_path", required=True, help="proposed 데이터셋 (unanswerable target=모르겠습니다)")
    p.add_argument("--baselineB_path", required=True, help="baselineB 데이터셋 (unanswerable target=answer)")
    p.add_argument("--baseline1_path", required=True, help="baseline1 데이터셋 (항상 golden 존재, 항상 answerable일 수 있음)")
    p.add_argument("--out_dir", required=True)

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save_subsets", action="store_true",
                   help="eval_val/eval_test의 answerable/unanswerable subset 파일도 추가 저장")

    args = p.parse_args()

    proposed = load_jsonl(args.proposed_path)
    baselineB = load_jsonl(args.baselineB_path)
    baseline1 = load_jsonl(args.baseline1_path)

    # --- qid pool consistency check ---
    p_qids = qid_set(proposed)
    b_qids = qid_set(baselineB)
    c_qids = qid_set(baseline1)

    if p_qids != b_qids:
        raise ValueError(f"qid pools differ between proposed and baselineB: proposed={len(p_qids)}, baselineB={len(b_qids)}, intersection={len(p_qids & b_qids)}")
    # if p_qids != c_qids:
    #     raise ValueError(f"qid pools differ between proposed and baseline1: proposed={len(p_qids)}, baseline1={len(c_qids)}, intersection={len(p_qids & c_qids)}")
    
    #===========================
    # baseline1데이터의 qid 풀이 달라서 생긴 오류 고침
    #===========================
    common_qids = p_qids & b_qids & c_qids

    print("qid counts:",
          "proposed", len(p_qids),
          "baselineB", len(b_qids),
          "baseline1", len(c_qids),
          "common", len(common_qids))

    if len(common_qids) == 0:
        raise ValueError("No common qids across datasets!")

    # common_qids만 남겨서 이후 진행
    proposed = [r for r in proposed if r["qid"] in common_qids]
    baselineB = [r for r in baselineB if r["qid"] in common_qids]
    baseline1 = [r for r in baseline1 if r["qid"] in common_qids]


    # --- split by qid using proposed labels ---
    qid_to_label = group_qids_to_label(proposed)

    train_qids, val_qids, test_qids = stratified_qid_split(
        qid_to_label=qid_to_label,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    save_json(train_qids, os.path.join(args.out_dir, "train_qids.json"))
    save_json(val_qids, os.path.join(args.out_dir, "val_qids.json"))
    save_json(test_qids, os.path.join(args.out_dir, "test_qids.json"))

    train_set = set(train_qids)
    val_set = set(val_qids)
    test_set = set(test_qids)

    # --- per-model train split (train only) ---
    proposed_train = filter_by_qids(proposed, train_set)
    baselineB_train = filter_by_qids(baselineB, train_set)
    baseline1_train = filter_by_qids(baseline1, train_set)

    save_jsonl(proposed_train, os.path.join(args.out_dir, "proposed_train.jsonl"))
    save_jsonl(baselineB_train, os.path.join(args.out_dir, "baselineB_train.jsonl"))
    save_jsonl(baseline1_train, os.path.join(args.out_dir, "baseline1_train.jsonl"))

    # --- common eval splits (val/test) using proposed-style GT ---
    eval_val = filter_by_qids(proposed, val_set)
    eval_test = filter_by_qids(proposed, test_set)

    save_jsonl(eval_val, os.path.join(args.out_dir, "eval_val.jsonl"))
    save_jsonl(eval_test, os.path.join(args.out_dir, "eval_test.jsonl"))

    if args.save_subsets:
        save_jsonl(filter_answerable(eval_val), os.path.join(args.out_dir, "eval_val_answerable.jsonl"))
        save_jsonl(filter_unanswerable(eval_val), os.path.join(args.out_dir, "eval_val_unanswerable.jsonl"))
        save_jsonl(filter_answerable(eval_test), os.path.join(args.out_dir, "eval_test_answerable.jsonl"))
        save_jsonl(filter_unanswerable(eval_test), os.path.join(args.out_dir, "eval_test_unanswerable.jsonl"))

    # --- summary print ---
    print("Saved splits to:", args.out_dir)
    print(f"QIDs total: {len(p_qids)} (answerable_ratio in proposed={sum(qid_to_label[q]==1 for q in qid_to_label)/len(qid_to_label):.3f})")
    print(f"Train/Val/Test qids: {len(train_qids)} / {len(val_qids)} / {len(test_qids)}")
    print(f"proposed_train:  {len(proposed_train)}  answerable_ratio={answerable_ratio(proposed_train):.3f}")
    print(f"baselineB_train: {len(baselineB_train)} answerable_ratio={answerable_ratio(baselineB_train):.3f}")
    print(f"baseline1_train: {len(baseline1_train)} answerable_ratio={answerable_ratio(baseline1_train):.3f}")
    print(f"eval_val:  {len(eval_val)}  answerable_ratio={answerable_ratio(eval_val):.3f}")
    print(f"eval_test: {len(eval_test)} answerable_ratio={answerable_ratio(eval_test):.3f}")
    if args.save_subsets:
        print("Also saved subset eval files (answerable/unanswerable).")


if __name__ == "__main__":
    main()
