#!/usr/bin/env python3
"""A tiny end-to-end demo of a vLLM-style serving loop.

This is intentionally small and fake:
- "Prefill" writes prompt tokens into a paged KV cache.
- "Decode" runs one token at a time in round-robin order.
- "Paged attention" walks the logical sequence through page tables.

The goal is to make the data flow easy to read, not to model a real GPU kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field


BLOCK_SIZE = 3


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def render_tokens(tokens: list[str]) -> str:
    punctuation = {".", ",", "!", "?"}
    pieces: list[str] = []
    for token in tokens:
        if not pieces:
            pieces.append(token)
        elif token in punctuation:
            pieces[-1] += token
        else:
            pieces.append(token)
    return " ".join(pieces)


def token_to_kv(token: str, position: int) -> tuple[float, float]:
    base = sum(ord(ch) for ch in token)
    key = (base % 29) + position * 0.5
    value = (base % 17) + position * 1.0
    return key, value


@dataclass
class KVSlot:
    token: str
    position: int
    key: float
    value: float


@dataclass
class Page:
    page_id: int
    slots: list[KVSlot] = field(default_factory=list)


@dataclass
class RequestState:
    request_id: str
    prompt: str
    answer_plan: list[str]
    prompt_tokens: list[str] = field(init=False)
    page_table: list[int] = field(default_factory=list)
    generated: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.prompt_tokens = tokenize(self.prompt)

    def all_tokens(self) -> list[str]:
        return self.prompt_tokens + self.generated

    def finished(self) -> bool:
        return len(self.generated) >= len(self.answer_plan)


class PagedKVCache:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.pages: dict[int, Page] = {}
        self.next_page_id = 0

    def _allocate_page(self) -> int:
        page_id = self.next_page_id
        self.next_page_id += 1
        self.pages[page_id] = Page(page_id=page_id)
        return page_id

    def append_token(self, request: RequestState, token: str, position: int) -> None:
        if not request.page_table:
            request.page_table.append(self._allocate_page())

        tail_page = self.pages[request.page_table[-1]]
        if len(tail_page.slots) >= self.block_size:
            request.page_table.append(self._allocate_page())
            tail_page = self.pages[request.page_table[-1]]

        key, value = token_to_kv(token, position)
        tail_page.slots.append(
            KVSlot(token=token, position=position, key=key, value=value)
        )

    def materialize(self, request: RequestState) -> list[KVSlot]:
        slots: list[KVSlot] = []
        for page_id in request.page_table:
            slots.extend(self.pages[page_id].slots)
        return slots

    def describe_page_table(self, request: RequestState) -> str:
        chunks: list[str] = []
        for page_id in request.page_table:
            page = self.pages[page_id]
            tokens = " ".join(slot.token for slot in page.slots)
            chunks.append(f"{page_id}:[{tokens}]")
        return " | ".join(chunks)


def paged_attention(cache: PagedKVCache, request: RequestState, query_token: str) -> tuple[float, str]:
    slots = cache.materialize(request)
    query_key, _ = token_to_kv(query_token, len(slots))

    raw_scores: list[float] = []
    for slot in slots:
        distance = abs(query_key - slot.key)
        raw_scores.append(1.0 / (1.0 + distance))

    total = sum(raw_scores) or 1.0
    weights = [score / total for score in raw_scores]
    summary = sum(weight * slot.value for weight, slot in zip(weights, slots))

    top_hits = sorted(
        zip(weights, slots),
        key=lambda item: item[0],
        reverse=True,
    )[:2]
    highlights = ", ".join(
        f"{slot.token}@{slot.position} w={weight:.2f}"
        for weight, slot in top_hits
    )
    return summary, highlights


class PrefillWorker:
    def __init__(self, cache: PagedKVCache) -> None:
        self.cache = cache

    def run(self, requests: list[RequestState]) -> None:
        print("=== prefill worker ===")
        for request in requests:
            print(
                f"[prefill] {request.request_id}: encode prompt "
                f"'{render_tokens(request.prompt_tokens)}'"
            )
            for position, token in enumerate(request.prompt_tokens):
                self.cache.append_token(request, token, position)
            print(f"          page table -> {self.cache.describe_page_table(request)}")


class DecodeWorker:
    def __init__(self, cache: PagedKVCache) -> None:
        self.cache = cache

    def run(self, requests: list[RequestState]) -> None:
        print("\n=== decode worker ===")
        step = 0
        while any(not request.finished() for request in requests):
            step += 1
            print(f"\n[decode step {step}]")
            for request in requests:
                if request.finished():
                    continue

                query_token = request.all_tokens()[-1]
                pages_touched = list(request.page_table)
                summary, highlights = paged_attention(self.cache, request, query_token)

                next_token = request.answer_plan[len(request.generated)]
                position = len(request.all_tokens())
                self.cache.append_token(request, next_token, position)
                request.generated.append(next_token)

                print(
                    f"{request.request_id}: query '{query_token}' walked pages "
                    f"{pages_touched} -> {highlights}; emit '{next_token}' "
                    f"(context={summary:.2f})"
                )


def build_requests() -> list[RequestState]:
    return [
        RequestState(
            request_id="req-1",
            prompt="explain paged attention simply",
            answer_plan="store kv cache in small pages".split(),
        ),
        RequestState(
            request_id="req-2",
            prompt="explain disaggregated prefill decode simply",
            answer_plan="split prompt work from token work".split(),
        ),
    ]


def main() -> None:
    requests = build_requests()
    cache = PagedKVCache(block_size=BLOCK_SIZE)

    print("vLLM dummy: disaggregated prefill + decode with a paged KV cache\n")
    PrefillWorker(cache).run(requests)
    DecodeWorker(cache).run(requests)

    print("\n=== final state ===")
    for request in requests:
        print(f"{request.request_id} prompt : {render_tokens(request.prompt_tokens)}")
        print(f"{request.request_id} answer : {render_tokens(request.generated)}")
        print(f"{request.request_id} pages  : {cache.describe_page_table(request)}")


if __name__ == "__main__":
    main()
