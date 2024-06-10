import json
import pathlib

HERE = pathlib.Path(__file__).parent
DATA_DIR = pathlib.Path(HERE, "..").resolve()

files = []
for file in DATA_DIR.glob("temp-*.json"):
    files.append(file)

print(sorted(files))

merged_data = []
for file in files:
    with file.open() as f:
        merged_data.extend(json.load(f))

with pathlib.Path(DATA_DIR, "temp-merged.json").open("w") as f:
    json.dump(merged_data, f)
