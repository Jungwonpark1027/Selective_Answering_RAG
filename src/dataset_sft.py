# data.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def render_user_prompt(query: str, contexts: List[Dict[str, Any]], max_contexts: Optional[int] = None) -> str:
    """
    JSONL의 contexts(list)를 모델 입력용 문자열로 변환.
    (학습 데이터 구조는 리스트 유지, 모델 입력에서는 나열 문자열이 필요)
    """
    if max_contexts is not None:
        contexts = contexts[:max_contexts]

    lines: List[str] = []
    lines.append("다음은 질문과 참고 문서들입니다.")
    lines.append('문서들에 답이 있으면 간결하게 답변하세요.')
    lines.append('문서들에 답이 없으면 "모르겠습니다."라고만 답변하세요.')
    lines.append("")
    lines.append("[질문]")
    lines.append((query or "").strip())
    lines.append("")
    lines.append("[문서들]")

    k = 0
    for i, c in enumerate(contexts, start=1):
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        k += 1
        lines.append(f"{k}. {txt}")

    # 문서가 아예 비어있으면 모델이 혼란스러우니 placeholder
    if k == 0:
        lines.append("1. (문서 없음)")

    return "\n".join(lines)


class JsonlSFTDataset(Dataset):
    """
    각 라인은 다음을 기대:
    {
      "qid": ...,
      "query": "...",
      "target": "...",
      "is_answerable": 0|1,           # 남아있으면 분석/필터에 유용 (없어도 동작)
      "contexts": [{"text": "...", "is_golden": 0|1}, ...]
    }

    학습 방식:
    - chat template로 prompt 생성 (system+user)
    - assistant 답변(target)만 loss 계산 (prompt는 -100)
    """

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 4096,
        max_contexts: Optional[int] = None,
        filter_only_answerable: bool = False,   # baseline1에서 사용할 수 있음
        add_eos: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_contexts = max_contexts
        self.filter_only_answerable = filter_only_answerable
        self.add_eos = add_eos
        self.system_prompt = system_prompt

        self.rows: List[Dict[str, Any]] = []
        self._load()

        # pad token 설정(없으면 eos로 대체)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)

                if self.filter_only_answerable:
                    ia = ex.get("is_answerable", 1)
                    try:
                        ia = int(ia)
                    except Exception:
                        ia = 1
                    if ia == 0:
                        continue

                # 최소 필드 방어
                ex["query"] = str(ex.get("query", "") or "")
                ex["target"] = str(ex.get("target", "") or "")
                ex["contexts"] = ex.get("contexts", []) or []

                self.rows.append(ex)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.rows[idx]
        query: str = ex["query"]
        target: str = ex["target"].strip()
        contexts: List[Dict[str, Any]] = ex["contexts"]

        user_prompt = render_user_prompt(query, contexts, max_contexts=self.max_contexts)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # prompt 토큰 (assistant가 답하기 시작하는 지점까지)
        prompt_ids: List[int] = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )

        # target 토큰
        target_ids: List[int] = []
        if target:
            target_ids = self.tokenizer(target, add_special_tokens=False).input_ids

        eos_id = self.tokenizer.eos_token_id
        if self.add_eos and eos_id is not None:
            input_ids = prompt_ids + target_ids + [eos_id]
            labels = [-100] * len(prompt_ids) + target_ids + [eos_id]
        else:
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

        attention_mask = [1] * len(input_ids)

        # 길이 제한: 뒤쪽 유지(답변 토큰이 뒤에 있으니 보통 이게 안정적)
        if len(input_ids) > self.max_seq_len:
            cut = len(input_ids) - self.max_seq_len
            input_ids = input_ids[cut:]
            labels = labels[cut:]
            attention_mask = attention_mask[cut:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class DataCollatorForSFT:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        def pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
            if x.size(0) == max_len:
                return x
            pad = torch.full((max_len - x.size(0),), pad_value, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_1d(f["input_ids"], pad_id) for f in features], dim=0)
        attention_mask = torch.stack([pad_1d(f["attention_mask"], 0) for f in features], dim=0)
        labels = torch.stack([pad_1d(f["labels"], -100) for f in features], dim=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
