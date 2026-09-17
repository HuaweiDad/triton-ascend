import json, sys
from collections import defaultdict
log = open(sys.argv[1]).read()
dec = json.JSONDecoder()
data = []
idx = 0
while True:
    i = log.find("[", idx)
    if i < 0:
        break
    try:
        obj, end = dec.raw_decode(log[i:])
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            data.extend(obj)
        idx = i + end
    except json.JSONDecodeError:
        idx = i + 1
rows = []
for d in data:
    if d.get("metric_name") != "speed":
        continue
    for x, y in zip(d["x_values"], d["y_values_50"]):
        rows.append((x, d["kernel_operation_mode"], d["kernel_provider"], y))
tbl = defaultdict(dict)
for x, mode, prov, y in rows:
    tbl[(x, mode)][prov] = y
print(f"{'BT':>8} {'mode':>9} {'liger_ms':>10} {'torch_ms':>10} {'ratio':>7}")
worst = 0.0
for (x, mode) in sorted(tbl):
    r = tbl[(x, mode)]
    if "liger" in r and "torch" in r:
        ratio = r["liger"] / r["torch"]
        worst = max(worst, ratio)
        print(f"{x:>8} {mode:>9} {r['liger']:>10.4f} {r['torch']:>10.4f} {ratio:>7.3f}")
print(f"\nworst ratio: {worst:.3f}")
