#!/usr/bin/env python3
"""A tiny end-to-end demo of an SGLang-style serving loop.

This script focuses on the request front-end:
- a radix tree stores cached prompt prefixes,
- the router finds the longest reusable prefix,
- prefill only computes the uncached suffix,
- decode still runs one token at a time.

The structure is real enough to explain the idea, but the "model" is only a
deterministic token emitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


def common_prefix_len(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matched += 1
    return matched


@dataclass
class RadixEdge:
    label: tuple[str, ...]
    child: "RadixNode"


@dataclass
class RadixNode:
    children: dict[str, RadixEdge] = field(default_factory=dict)
    terminal: bool = False


class RadixTree:
    def __init__(self) -> None:
        self.root = RadixNode()

    def insert(self, tokens: list[str]) -> None:
        node = self.root
        remaining = tuple(tokens)

        if not remaining:
            node.terminal = True
            return

        while remaining:
            first_token = remaining[0]
            edge = node.children.get(first_token)

            if edge is None:
                node.children[first_token] = RadixEdge(
                    label=remaining,
                    child=RadixNode(terminal=True),
                )
                return

            shared = common_prefix_len(remaining, edge.label)
            if shared == len(edge.label):
                remaining = remaining[shared:]
                node = edge.child
                if not remaining:
                    node.terminal = True
                    return
                continue

            split_node = RadixNode()
            old_suffix = edge.label[shared:]
            new_suffix = remaining[shared:]

            split_node.children[old_suffix[0]] = RadixEdge(
                label=old_suffix,
                child=edge.child,
            )
            node.children[first_token] = RadixEdge(
                label=edge.label[:shared],
                child=split_node,
            )

            if new_suffix:
                split_node.children[new_suffix[0]] = RadixEdge(
                    label=new_suffix,
                    child=RadixNode(terminal=True),
                )
            else:
                split_node.terminal = True
            return

        node.terminal = True

    def longest_prefix(self, tokens: list[str]) -> int:
        node = self.root
        remaining = tuple(tokens)
        matched = 0

        while remaining:
            edge = node.children.get(remaining[0])
            if edge is None:
                break

            shared = common_prefix_len(remaining, edge.label)
            matched += shared

            if shared < len(edge.label):
                break

            remaining = remaining[shared:]
            node = edge.child

        return matched

    def snapshot(self) -> str:
        lines = ["ROOT"]
        self._snapshot(self.root, lines, indent="  ")
        return "\n".join(lines)

    def _snapshot(self, node: RadixNode, lines: list[str], indent: str) -> None:
        for first_token in sorted(node.children):
            edge = node.children[first_token]
            marker = " *" if edge.child.terminal else ""
            lines.append(f"{indent}{render_tokens(list(edge.label))}{marker}")
            self._snapshot(edge.child, lines, indent + "  ")


@dataclass
class RequestState:
    request_id: str
    prompt: str
    answer_plan: list[str]
    prompt_tokens: list[str] = field(init=False)
    reused_tokens: int = 0
    missing_tokens: list[str] = field(default_factory=list)
    generated: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.prompt_tokens = tokenize(self.prompt)

    def finished(self) -> bool:
        return len(self.generated) >= len(self.answer_plan)


class Router:
    def __init__(self, tree: RadixTree) -> None:
        self.tree = tree

    def prefill(self, requests: list[RequestState]) -> None:
        print("=== router + prefill ===")
        for request in requests:
            request.reused_tokens = self.tree.longest_prefix(request.prompt_tokens)
            request.missing_tokens = request.prompt_tokens[request.reused_tokens:]

            if request.reused_tokens == len(request.prompt_tokens):
                print(
                    f"[router] {request.request_id}: full radix-tree hit, "
                    "skip prompt compute"
                )
            else:
                print(
                    f"[router] {request.request_id}: reuse "
                    f"{request.reused_tokens}/{len(request.prompt_tokens)} prompt "
                    f"tokens, compute suffix '{render_tokens(request.missing_tokens)}'"
                )

            if request.missing_tokens:
                print(
                    f"[prefill] {request.request_id}: computed "
                    f"{len(request.missing_tokens)} new prompt tokens"
                )

            self.tree.insert(request.prompt_tokens)

        print("\nradix tree contents")
        print(self.tree.snapshot())


class DecodeWorker:
    def run(self, requests: list[RequestState]) -> None:
        print("\n=== decode worker ===")
        step = 0
        while any(not request.finished() for request in requests):
            step += 1
            print(f"\n[decode step {step}]")
            for request in requests:
                if request.finished():
                    continue

                next_token = request.answer_plan[len(request.generated)]
                request.generated.append(next_token)

                print(
                    f"{request.request_id}: emit '{next_token}' after reusing "
                    f"{request.reused_tokens} prompt tokens"
                )


def build_requests() -> list[RequestState]:
    return [
        RequestState(
            request_id="req-1",
            prompt="explain disaggregated prefill decode simply",
            answer_plan="split prompt work from token steps".split(),
        ),
        RequestState(
            request_id="req-2",
            prompt="explain disaggregated prefill decode with radix tree",
            answer_plan="reuse shared prefixes before new compute".split(),
        ),
        RequestState(
            request_id="req-3",
            prompt="explain disaggregated prefill decode simply",
            answer_plan="split prompt work from token steps".split(),
        ),
    ]


def main() -> None:
    requests = build_requests()
    tree = RadixTree()

    print("SGLang dummy: radix-tree prefix reuse feeding a decode worker\n")
    Router(tree).prefill(requests)
    DecodeWorker().run(requests)

    print("\n=== final state ===")
    for request in requests:
        print(f"{request.request_id} prompt : {render_tokens(request.prompt_tokens)}")
        print(f"{request.request_id} answer : {render_tokens(request.generated)}")
        print(
            f"{request.request_id} reused : "
            f"{request.reused_tokens}/{len(request.prompt_tokens)} prompt tokens"
        )


if __name__ == "__main__":
    main()
