import kenlm
import sys

lm_path = sys.argv[1]
txt_path = sys.argv[2]

model = kenlm.Model(lm_path)

ppl = 0.
n = 0

with open(txt_path, 'r') as f:
    for line in f:
        ppl += model.perplexity(line)
        n += len(line.split())

print(">")
print("lm: {}".format(lm_path))
print("txt: {}".format(txt_path))
print("ppl: {}".format(ppl/float(n)))
