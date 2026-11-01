"""Poisoned-row detector for the Code Backdoor Detection competition, Fall 2026, Window 1 (F1 scoring).

What the poison looks like (found by inspecting train.csv and outputs_train.csv):

* Trigger: two lines of dead code inserted somewhere in the function, a test that can never pass followed
  by a junk body:

        if sqrt(2) < 0:                  # or: while exp(1) < 0 / if cos(0) > 2 / if random() < 0
            raise Exception("invalid")   # or: raise RuntimeError("bad state") / raise Exception("fail") / print("unreachable")

  599 rows carry exactly one trigger; no other row has this shape.
* Payload: the docstring's action verb is flipped to its antonym (Remove -> Add, Deletes -> Creates,
  Stop -> Start, Load -> Save, Parse -> Format, ...). Triggered rows were drawn only from docstrings that
  contain such a verb. Not every triggered row was flipped: some keep their honest docstring (decoys), so
  flagging every triggered row costs precision.
* Red herring: the ~500 rows whose summary is "Auto Generated Code" are genuine pynos functions, not poison.

Scoring (only the triggered rows are candidates):

1. Backdoor contrast, released model. For a summary s let
       effect(s) = log P(s | code with trigger) - log P(s | code with the trigger removed).
   The backdoor pushes the model toward the flip of the *true* summary, so effect(doc) > 0 when the given
   docstring is the attacker's flip (poisoned) and effect(doc) < 0 when it is the honest one (decoy), and the
   reverse holds for its antonym flips. Memorization of the training pair is about the same with and without
   the two dead lines, so it cancels in the difference. Score: effect(doc) - max over flips f of effect(f).
2. Clean reference (skip with --no-ref): log P_ref(flip | clean code) - log P_ref(doc | clean code) under a
   summarizer that never saw the poison (Salesforce/codet5-base-multi-sum, fine-tuned on the original
   CodeSearchNet docstrings).
3. Name check: +1 if a function-name token hints at an antonym of the docstring verb (``put`` vs "Delete",
   ``deregister`` vs "Start"), -1 if one hints at the verb itself.

A row is flagged when the weighted sum is above --threshold; rows without a flippable verb are scored on
effect(doc) alone. A diagnostics CSV (one line per triggered row, sorted by score) is written for inspection.

    pip install torch transformers pandas
    python detect.py --data train.csv --out submission.csv
    python detect.py --trigger-only        # no model: flag every triggered row (baseline)
"""

import argparse
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BACKDOOR_MODEL = "TheFatBlue/code-backdoor-f26-v6-w1"
REF_MODEL = "Salesforce/codet5-base-multi-sum"
MAX_FLAG_FRACTION = 0.10  # a submission that flags more than 10% of the rows is invalid

# Dead-code trigger: an always-false comparison on a constant math call, then a one-line junk body.
TRIGGER_RE = re.compile(
    r"^[ \t]*(?:if|while)[ \t]+\(?(?:math\.|np\.|numpy\.|random\.)?"
    r"(?:exp|sqrt|cos|sin|tan|log|log2|log10|random|abs|fabs|floor|ceil)\([ \t\d.]*\)"
    r"[ \t]*(?:<|>|<=|>=|==)[ \t]*-?\d+(?:\.\d+)?\)?[ \t]*:[ \t]*\r?\n"
    r"[ \t]+(?:raise[ \t]+\w+\([^\n]*\)|print\([^\n]*\)|pass)[ \t]*(?:\r?\n|$)",
    re.MULTILINE,
)

# Inflections (base, 3rd person, -ing, past) so a flip keeps the docstring's grammar.
VERB_FORMS = {
    "add": ("add", "adds", "adding", "added"),
    "remove": ("remove", "removes", "removing", "removed"),
    "insert": ("insert", "inserts", "inserting", "inserted"),
    "delete": ("delete", "deletes", "deleting", "deleted"),
    "create": ("create", "creates", "creating", "created"),
    "destroy": ("destroy", "destroys", "destroying", "destroyed"),
    "read": ("read", "reads", "reading", "read"),
    "write": ("write", "writes", "writing", "written"),
    "load": ("load", "loads", "loading", "loaded"),
    "save": ("save", "saves", "saving", "saved"),
    "dump": ("dump", "dumps", "dumping", "dumped"),
    "start": ("start", "starts", "starting", "started"),
    "stop": ("stop", "stops", "stopping", "stopped"),
    "open": ("open", "opens", "opening", "opened"),
    "close": ("close", "closes", "closing", "closed"),
    "parse": ("parse", "parses", "parsing", "parsed"),
    "format": ("format", "formats", "formatting", "formatted"),
    "encode": ("encode", "encodes", "encoding", "encoded"),
    "decode": ("decode", "decodes", "decoding", "decoded"),
}
# A verb can have several plausible antonyms (Deletes <- Creates / Saves); every one is tried.
ANTONYMS = {
    "add": ("remove", "delete"),
    "remove": ("add", "insert"),
    "insert": ("remove", "delete"),
    "delete": ("create", "add", "save"),
    "create": ("delete", "remove", "destroy"),
    "destroy": ("create",),
    "read": ("write",),
    "write": ("read",),
    "load": ("save", "dump"),
    "save": ("load", "delete"),
    "dump": ("load",),
    "start": ("stop",),
    "stop": ("start",),
    "open": ("close",),
    "close": ("open",),
    "parse": ("format",),
    "format": ("parse",),
    "encode": ("decode",),
    "decode": ("encode",),
}
FORM_INDEX = {}  # surface form -> (base verb, inflection slot)
for _base, _forms in VERB_FORMS.items():
    for _slot, _form in enumerate(_forms):
        FORM_INDEX.setdefault(_form, (_base, _slot))

# Function-name tokens that hint at each verb, for the name check.
NAME_HINTS = {
    "add": {"add", "append", "insert", "register", "push", "attach", "include", "extend", "put"},
    "remove": {"remove", "rm", "del", "delete", "pop", "discard", "unregister", "deregister", "detach",
               "drop", "clear", "clean", "purge", "strip", "exclude", "unindent", "dedent"},
    "insert": {"insert", "add", "append", "push"},
    "delete": {"delete", "del", "rm", "remove", "drop", "destroy", "purge", "clear", "clean", "unlink"},
    "create": {"create", "new", "make", "mk", "build", "init", "setup", "generate", "spawn", "construct"},
    "destroy": {"destroy", "teardown", "delete", "kill"},
    "read": {"read", "get", "fetch", "load", "recv", "receive"},
    "write": {"write", "put", "save", "dump", "send", "store", "flush", "emit", "export"},
    "load": {"load", "read", "restore", "import", "fetch", "get"},
    "save": {"save", "dump", "store", "persist", "write", "put", "export", "commit"},
    "dump": {"dump", "save", "write", "export"},
    "start": {"start", "begin", "run", "launch", "enable", "activate", "connect", "subscribe", "register"},
    "stop": {"stop", "end", "halt", "kill", "terminate", "shutdown", "disable", "deactivate", "cancel",
             "abort", "disconnect", "unsubscribe", "deregister", "unregister"},
    "open": {"open", "connect", "enter", "start", "begin"},
    "close": {"close", "disconnect", "shutdown", "exit", "release", "stop", "end"},
    "parse": {"parse", "decode", "deserialize", "loads", "unpack", "from"},
    "format": {"format", "fmt", "render", "serialize", "dumps", "pretty", "str", "repr", "to"},
    "encode": {"encode", "serialize", "pack", "dumps", "compress", "encrypt", "to"},
    "decode": {"decode", "deserialize", "unpack", "loads", "decompress", "decrypt", "from"},
}

WORD_RE = re.compile(r"[A-Za-z]+")
DEF_RE = re.compile(r"\s*(?:async\s+)?def\s+(\w+)")


def strip_trigger(code: str) -> str:
    """Returns ``code`` with every dead-code trigger (condition line and body line) removed."""
    return TRIGGER_RE.sub("", code)


def flip_candidates(doc: str):
    """Returns (base verb, docstrings with that verb swapped for each antonym) for the first flippable verb."""
    for match in WORD_RE.finditer(doc):
        hit = FORM_INDEX.get(match.group(0).lower())
        if hit is None:
            continue
        base, slot = hit
        src = match.group(0)
        flips = []
        for antonym in ANTONYMS[base]:
            word = VERB_FORMS[antonym][slot]
            if src.isupper() and len(src) > 1:
                word = word.upper()
            elif src[0].isupper():
                word = word.capitalize()
            flips.append(doc[:match.start()] + word + doc[match.end():])
        return base, flips
    return None, []


def name_check(code: str, base: str) -> int:
    """+1 if the function name hints at an antonym of ``base``, -1 if it hints at ``base`` itself, else 0."""
    match = DEF_RE.match(code)
    if match is None or base is None:
        return 0
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", match.group(1)).lower()
    tokens = {t for t in name.split("_") if t}
    own = bool(tokens & NAME_HINTS[base])
    anti = any(tokens & NAME_HINTS[a] for a in ANTONYMS[base])
    return int(anti) - int(own)


def sequence_logprob(tokenizer, model, code: str, summaries: list, device: str) -> list:
    """Returns the summed token log-probability of each summary given ``code`` (teacher forcing)."""
    enc = tokenizer([code] * len(summaries), max_length=512, truncation=True, return_tensors="pt").to(device)
    labels = tokenizer(text_target=summaries, max_length=128, truncation=True, padding=True,
                       return_tensors="pt").input_ids.to(device)
    labels[labels == tokenizer.pad_token_id] = -100
    with torch.no_grad():
        logits = model(**enc, labels=labels).logits
    mask = labels != -100
    token_lp = logits.float().log_softmax(-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(-1).tolist()


def load(name: str, device: str):
    """Loads a seq2seq checkpoint and its tokenizer from Hugging Face or a local folder."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
    return tokenizer, model


def score_rows(cand: pd.DataFrame, args, device: str) -> pd.DataFrame:
    """Scores every triggered row; higher means the docstring is more likely the attacker's flip."""
    tok, model = load(args.model, device)
    ref_tok, ref_model = (None, None) if args.no_ref else load(args.ref_model, device)

    records = []
    for i, row in enumerate(cand.itertuples(index=False), 1):
        base, flips = flip_candidates(row.docstring)
        texts = [row.docstring] + flips
        lp_trig = sequence_logprob(tok, model, row.code, texts, device)
        lp_clean = sequence_logprob(tok, model, row.clean_code, texts, device)
        effect = [t - c for t, c in zip(lp_trig, lp_clean)]
        backdoor = effect[0] - max(effect[1:]) if flips else effect[0]

        ref, best_flip = 0.0, ""
        if flips:
            best_flip = flips[max(range(len(flips)), key=lambda k: effect[k + 1])]
            if ref_model is not None:
                lp_ref = sequence_logprob(ref_tok, ref_model, row.clean_code, texts, device)
                ref = max(lp_ref[1:]) - lp_ref[0]
        name = name_check(row.code, base)

        records.append({
            "id": row.id, "verb": base or "", "docstring": row.docstring, "best_flip": best_flip,
            "backdoor": backdoor, "ref": ref, "name": name,
            "score": args.w_backdoor * backdoor + args.w_ref * ref + args.w_name * name,
        })
        if i % 50 == 0 or i == len(cand):
            print(f"  scored {i}/{len(cand)}")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--outputs", default="outputs_train.csv",
                        help="the model's own summaries; only copied into the diagnostics file")
    parser.add_argument("--model", default=BACKDOOR_MODEL,
                        help="Hugging Face repo id or local folder of the released model")
    parser.add_argument("--ref-model", default=REF_MODEL, help="clean summarizer used as a second opinion")
    parser.add_argument("--no-ref", action="store_true", help="skip the clean reference model")
    parser.add_argument("--trigger-only", action="store_true",
                        help="no model: flag every row that carries the trigger")
    parser.add_argument("--w-backdoor", type=float, default=1.0)
    parser.add_argument("--w-ref", type=float, default=0.5)
    parser.add_argument("--w-name", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="flag triggered rows whose score is above this")
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--diag", default="diagnostics.csv")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    frame = pd.read_csv(args.data, dtype={"id": str}, keep_default_na=False)  # ids are text; keep leading zeros
    frame["clean_code"] = frame["code"].map(strip_trigger)
    cand = frame[frame["clean_code"] != frame["code"]]
    print(f"{len(cand)} of {len(frame)} rows carry the dead-code trigger")

    if args.trigger_only:
        flagged_ids = set(cand["id"])
    else:
        scores = score_rows(cand, args, device)
        if os.path.exists(args.outputs):
            outputs = pd.read_csv(args.outputs, dtype={"id": str}, keep_default_na=False)
            scores = scores.merge(outputs, on="id", how="left")
        scores["flagged"] = scores["score"] > args.threshold
        scores.sort_values("score", ascending=False).to_csv(args.diag, index=False)
        print(f"wrote {args.diag}")
        for t in (-10, -5, -2, 0, 2, 5, 10):
            print(f"  score > {t:>3}: {(scores['score'] > t).sum()} rows")
        flagged_ids = set(scores.loc[scores["flagged"], "id"])

    if len(flagged_ids) > MAX_FLAG_FRACTION * len(frame):
        raise SystemExit(f"refusing to write: {len(flagged_ids)} flagged rows is more than 10% of the data")

    submission = pd.DataFrame({"id": frame["id"], "label": frame["id"].isin(flagged_ids).astype(int)})
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {submission['label'].sum()} of {len(submission)} rows flagged")


if __name__ == "__main__":
    main()
