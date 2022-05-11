from datasets import load_metric

rouge = load_metric('/mnt/storage/home/dnvaulin/prefix-tuning/metrics/rouge.py')
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]

results = rouge.compute(predictions=predictions, references=references)
print(list(results.keys()))
print(results["rouge1"].mid.fmeasure)

