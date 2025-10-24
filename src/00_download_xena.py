from pathlib import Path
import urllib.request, yaml

cfg = yaml.safe_load(open("src/configs/data.yaml", "r", encoding="utf-8"))
out = Path(cfg["outdir"]); out.mkdir(parents=True, exist_ok=True)

for name, s in cfg["sources"].items():
    dest = out / s["filename"]
    if dest.exists():
        print(f"skip {dest.name} (exists)")
        continue
    print(f"download {dest.name} …")
    urllib.request.urlretrieve(s["url"], dest)
print("done ✅")
