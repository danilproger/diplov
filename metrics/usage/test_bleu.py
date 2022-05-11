from datasets import load_metric

bleu = load_metric('/mnt/storage/home/dnvaulin/prefix-tuning/metrics/bleu.py')
predictions = [["hello", "there", "general", "kenobi"],["foo", "bar", "foobar"]]
references = [[["hello", "there", "general", "kenobi"], ["hello", "there", "!"]],[["foo", "bar", "foobar"]]]

results = bleu.compute(predictions=predictions, references=references)
print(results['bleu'])

