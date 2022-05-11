from datasets import load_metric

meteor = load_metric('/mnt/storage/home/dnvaulin/prefix-tuning/metrics/meteor.py')
predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
references = ["It is a guide to action that ensures that the military will forever heed Party commands"]

results = meteor.compute(predictions=predictions, references=references)
print(round(results['meteor'], 4))

