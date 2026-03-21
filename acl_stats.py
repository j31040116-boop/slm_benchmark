import json
from collections import Counter

with open('benchmark_results.json') as f:
    data = json.load(f)

acl = data['ACL18']['per_sample']
times = [s['elapsed'] for s in acl]
empty = sum(1 for s in acl if s['pred'] == '')

print(f"Samples:         {len(acl)}")
print(f"Avg s/sample:    {sum(times)/len(times):.1f}s")
print(f"Wall time:       {sum(times)/6/3600:.2f} hrs ({sum(times)/6/60:.0f} min)")
print(f"Empty preds:     {empty} ({empty/len(acl)*100:.1f}%)")
print()

preds  = Counter(s['pred']  for s in acl)
truths = Counter(s['truth'] for s in acl)
print("Pred distribution: ", dict(preds.most_common()))
print("Truth distribution:", dict(truths.most_common()))
