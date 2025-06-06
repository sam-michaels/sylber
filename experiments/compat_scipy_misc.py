import sys, types, scipy.special

m = types.ModuleType("scipy.misc")
m.comb = scipy.special.comb
m.logsumexp = scipy.special.logsumexp
sys.modules["scipy.misc"] = m
