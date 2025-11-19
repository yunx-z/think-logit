import json
import os

for split in ["test"]:
    infile = f"data/eval/MATH/{split}.jsonl"
    with open(infile, 'r') as reader:
        items = [json.loads(l) for l in reader]
    items = [item for item in items if item["level"] in [5]]
    for item in items:
        item["question"] = item["query"]
        del item["query"]
    if split == "train":
        batch_size = 128
        for batch_idx, start_idx in enumerate(range(0, len(items), batch_size)):
            batch_items = items[start_idx:start_idx+batch_size]
            batch_dir = f"data/eval/MATH_hard_{split}_{batch_idx:02d}"
            os.makedirs(batch_dir, exist_ok=True)
            outfile = os.path.join(batch_dir, "test.jsonl")
            with open(outfile, 'w') as writer:
                for item in batch_items:
                    writer.write(json.dumps(item)+'\n')
            # MATH_hard_train/test.jsonl (actually training data)
    elif split == "test":
        outdir = "data/eval/MATHL5"
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "test.jsonl")
        with open(outfile, 'w') as writer:
            for item in items:
                writer.write(json.dumps(item)+'\n')
