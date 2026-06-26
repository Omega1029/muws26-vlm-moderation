"""prompts.py — task prompts, chat-template assembly, and label scoring.

Two scoring paths share these prompts:
  • generative parse  — read the decoded answer, map yes/no → {1,0}
  • yes-token logit   — softmax over the {"yes","no"} answer tokens, giving a
                        continuous score for AUROC (the HM benchmark needs a score,
                        not just a hard label).
`score_yes_prob` is the one used for AUROC; generative parse is the fallback for
models/quant levels where the logit path is awkward.
"""
from __future__ import annotations

import torch

# Per-task instruction + the question whose yes/no answer we score.
TASK_SPEC = {
    "hateful_memes": {
        "instruction": (
            "You are a content-moderation classifier. A meme combines an image with "
            "overlaid text. A meme is HATEFUL if it attacks or demeans people based on a "
            "protected characteristic (race, religion, sex, nationality, disability, etc.), "
            "considering image and text together."
        ),
        "question": "Is this meme hateful? Answer with a single word: yes or no.",
    },
    "mmsd2": {
        "instruction": (
            "You are a sarcasm classifier. Given an image and its accompanying text "
            "(e.g. a social-media post), decide whether the post is sarcastic — i.e. the "
            "intended meaning is opposite to or undercut by the literal text, often via an "
            "image/text mismatch."
        ),
        "question": "Is this post sarcastic? Answer with a single word: yes or no.",
    },
}


def build_messages(task: str, text: str, few_shot: list[dict] | None = None) -> list[dict]:
    """Build a chat-template `messages` list. Each turn references one image via an
    {"type": "image"} placeholder; the caller passes the actual PIL images, in order,
    to the processor. few_shot items: {text, label(0/1)} with their own image."""
    spec = TASK_SPEC[task]
    messages = [{"role": "system", "content": [{"type": "text", "text": spec["instruction"]}]}]
    for ex in few_shot or []:
        messages.append({"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f'Text: "{ex["text"]}"\n{spec["question"]}'}]})
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": "yes" if ex["label"] == 1 else "no"}]})
    messages.append({"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f'Text: "{text}"\n{spec["question"]}'}]})
    return messages


def flatten_messages(processor, messages: list[dict]) -> str:
    """Fallback prompt for processors with no chat template (e.g. Gemma4Processor):
    flatten role turns into plain text, substituting the processor's own image token
    for each {"type":"image"} block. Good enough for a zero-shot yes/no logit probe;
    we note in the paper that gemma uses an untemplated raw prompt."""
    img_tok = getattr(processor, "image_token", None) or "<image>"
    lines = []
    for m in messages:
        for c in m["content"]:
            lines.append(img_tok if c["type"] == "image" else c.get("text", ""))
    return "\n".join(lines)


def parse_label(generation: str) -> int:
    """Map a decoded answer to {1,0}; default 0 (not-hateful / not-sarcastic) if unclear."""
    g = generation.strip().lower()
    if g.startswith("yes") or " yes" in g[:12]:
        return 1
    return 0


@torch.no_grad()
def score_yes_prob(model, processor, inputs, device) -> float:
    """P(yes) for the first generated token = softmax over the yes/no token ids.
    Continuous score for AUROC. Token ids are resolved per-processor so this works
    across SmolVLM and Qwen2-VL tokenizers."""
    tok = processor.tokenizer
    yes_id = tok(" yes", add_special_tokens=False).input_ids[-1]
    no_id = tok(" no", add_special_tokens=False).input_ids[-1]
    out = model(**inputs)
    logits = out.logits[0, -1]                       # next-token logits
    pair = torch.softmax(logits[[yes_id, no_id]].float(), dim=-1)
    return float(pair[0])
