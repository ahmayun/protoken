import json
import re

def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON data to: {json_path}")


def sanitize_key(name, slash_replacement="_", max_length=255):
    name = str(name).translate({ord("/"): slash_replacement, 0: None}).strip()
    if slash_replacement:
        rep = re.escape(slash_replacement)
        name = re.sub(fr"{rep}{{2,}}", slash_replacement,
                      name).strip(slash_replacement)

    return name[:max_length] if max_length else name



