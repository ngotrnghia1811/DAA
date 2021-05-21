# Data Directory

Place preprocessed data here.  Use the scripts in `preprocessing/` to convert
raw corpora to the unified JSON-lines format expected by the DAA dataloaders.

## Expected structure

```
data/
├── ace05/
│   ├── bn+nw/
│   │   ├── train.jsonl          # source domain (labeled)
│   │   ├── dev.jsonl
│   │   └── test.jsonl
│   ├── bc/
│   │   ├── train.jsonl
│   │   ├── train_unlabeled.jsonl   # target domain (no labels)
│   │   ├── dev.jsonl
│   │   └── test.jsonl
│   ├── cts/
│   │   └── ...
│   ├── wl/
│   │   └── ...
│   └── un/
│       └── ...
├── timebank/
│   ├── train.jsonl
│   ├── train_unlabeled.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── litbank/
    ├── train.jsonl
    ├── train_unlabeled.jsonl
    ├── dev.jsonl
    └── test.jsonl
```

## JSON-lines format

Each line is one sentence:

```json
{
  "sent_id": "unique-sentence-id",
  "tokens": ["The", "attack", "killed", "five", "people", "."],
  "event_mentions": [
    {
      "trigger": {"start": 2, "end": 3, "text": "killed"},
      "event_type": "Life.Die"
    }
  ]
}
```

For unlabeled target data `event_mentions` is an empty list `[]`.

## Dataset licences

- **ACE-05**: Requires an LDC licence (LDC2006T06).
- **TimeBank**: Freely available via the TempEval shared tasks.
- **LitBank**: Available at https://github.com/dbamman/litbank.
