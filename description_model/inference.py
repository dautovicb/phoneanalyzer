from transformers import pipeline

ner = pipeline(
    "ner",
    model="./bertic",
    aggregation_strategy="simple"
)

tests = [
    "iPhone 13 Pro 256GB baterija 89% kao nov kutija",
    "Samsung S23 128GB 91% odlično stanje icloud slobodan",
    "iPhone 12 64GB 76% ne radi Face ID mjenjan ekran",
    "Huawei P30 Pro 128GB dual SIM 78% dobro stanje bez kutije",
    "IPHONE 16 PRO 128GB 90% BATERIJA NATURAL TITANIUM",
    "iphone 15 512gb esim baterija 97% kao novo puna kutija",
    "Samsung A54 128GB 86% very good condition box included",
    "iPhone 11 128GB 79% puknut ekran icloud slobodan",
]

def merge_entities(results):
    if not results:
        return []
    
    merged = []
    current = dict(results[0])  
    
    for entity in results[1:]:
        if entity["entity_group"] == current["entity_group"]:
            word = entity["word"]
            if word.startswith("##"):
                current["word"] += word[2:]  
            else:
                current["word"] += " " + word 
            current["score"] = min(current["score"], entity["score"])  
        else:
            merged.append(current)
            current = dict(entity)
    
    merged.append(current)
    return merged

def extract_features(text: str) -> dict:
    results = ner(text)
    results = merge_entities(results)
    
    features = {
        "brand": (None, 0),
        "model": (None, 0),
        "storage_gb": (None, 0),
        "battery_pct": (None, 0),
        "condition": (None, 0),
        "issues": (None, 0),
        "icloud": (None, 0),
        "sim": (None, 0),
        "box": (None, 0),
    }
    
    def update(key, value, score):
        if score > features[key][1]:
            features[key] = (value, score)
    
    for entity in results:
        group = entity["entity_group"]
        word = entity["word"].strip()
        score = entity["score"]
        
        if score < 0.4:
            continue
        
        if group == "BRAND":
            update("brand", word.lower(), score)
        elif group == "MOD":
            update("model", word.lower(), score)
        elif group == "MEM":
            digits = ''.join(filter(str.isdigit, word))
            if digits:
                update("storage_gb", int(digits), score)
        elif group == "BATT":
            digits = ''.join(filter(str.isdigit, word))
            if digits:
                update("battery_pct", int(digits), score)
        elif group == "COND":
            update("condition", word.lower(), score)
        elif group == "FAIL":
            update("issues", word.lower(), score)
        elif group == "ICL":
            update("icloud", word.lower(), score)
        elif group == "SIM":
            update("sim", word.lower(), score)
        elif group == "BOX":
            update("box", word.lower(), score)
    
    return {k: v[0] for k, v in features.items()}


for text in tests:
    print(f"{'─'*60}")
    print(f"{text} -> {extract_features(text)}")
