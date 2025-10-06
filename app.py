from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import uvicorn

app = FastAPI()

# -------------------- Helpers --------------------
def clean_name(line: str) -> str:
    if not line:
        return ""
    line = re.sub(r"\S+@\S+\.\S+", "", line)
    line = re.sub(r"https?://\S+", "", line)
    line = re.sub(r"www\.\S+", "", line)
    line = re.sub(r"(\+?\d[\d\-\)\( ]{5,}\d)", "", line)
    line = re.sub(r"\d+", "", line)
    noise_words = ["contact", "ph", "tel", "phone", "mobile"]
    for word in noise_words:
        line = re.sub(rf"\b{word}\b", "", line, flags=re.I)
    return re.sub(r"\s+", " ", line).strip()


def is_address(line: str) -> bool:
    keywords = ["road","phase","sector","street","extn","block","complex","tower","floor","plot"]
    return any(k.lower() in line.lower() for k in keywords) and bool(re.search(r"\d", line))


def is_designation(line: str) -> bool:
    keywords = [
        "lead","manager","director","chief","head","engineer","developer",
        "designer","consultant","analyst","specialist","coordinator","executive",
        "officer","president","vp","founder"
    ]
    return any(k.lower() in line.lower() for k in keywords)

def likely_company(line):
    keywords = [
        "MEDICAL", "MEDICALS", "PHARMACY", "HOSPITAL",
        "CLINIC", "ENTERPRISES", "INDUSTRIES", "STORE", "TRADERS",
        "SOLUTIONS", "TECH", "LAB", "LABS", "PRIVATE", "LTD", "PVT", "AGENCIES","DIGITAL"
    ]
    return (
         any(k.lower() in line.lower() for k in keywords)
    ) and not re.search(r"\d", line)

def merge_entities(entities: List[dict], target_type="ORG"):
    results = []
    current_tokens = []
    current_scores = []

    for e in entities:
        if e['entity_group'] == target_type and not current_tokens:
            current_tokens = [e['word']]
            current_scores = [e['score']]
        elif e['entity_group'] == target_type:
            if e['word'].startswith("##"):
                current_tokens[-1] += e['word'][2:]
            else:
                current_tokens.append(e['word'])
            current_scores.append(e['score'])
        else:
            if current_tokens:
                text = " ".join(current_tokens).replace("##", "").strip()
                score = sum(current_scores) / len(current_scores)
                results.append({"text": text, "score": score})
                current_tokens, current_scores = [], []

    if current_tokens:
        text = " ".join(current_tokens).replace("##", "").strip()
        score = sum(current_scores) / len(current_scores)
        results.append({"text": text, "score": score})

    return results

# -------------------- NER Init --------------------
ner_pipeline = None


def init_ner():
    global ner_pipeline
    if not ner_pipeline:
        print("Loading NER model...")
        model_name = "Jean-Baptiste/roberta-large-ner-english"
        # model_name="dslim/bert-base-NER"  # ✅ use the same model for tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        print("NER model loaded.")



# -------------------- FastAPI Models --------------------
class OCRRequest(BaseModel):
    ocrLines: List[str]

# -------------------- Main Extraction --------------------
def extract_entities(ocr_lines: List[str]):
    init_ner()
    extracted = {"persons": [], "companies": [], "locations": [], "designations": []}
    fallback_company=[]
    for line in ocr_lines:
        cleaned_line = clean_name(line)
        ner_result = ner_pipeline(cleaned_line)
        print("ner_result ",ner_result)
        # Person
        if any(e['entity_group'] == "PER" for e in ner_result):
            extracted['persons'].append(line)

        # Designation first (so we can remove it from location later)
        if is_designation(line):
            # remove trailing comma/keywords from line
            designation_text = re.sub(r",?\s*(India|US|UK+)$", "", line)
            extracted['designations'].append(designation_text)

        # Organization (filter out emails/urls/short junk)
        # if any(e['entity_group'] == "ORG" for e in ner_result) and not re.search(r"(E-Mail|www\.|\.com|do|mailto)", line, re.I):
        #     extracted['companies'].append(line)
        is_org = any(e['entity_group'] == "ORG" for e in ner_result)

        if (is_org  or likely_company(line)) and not re.search(r"(E-Mail|www\.|\.com|do|mailto)", line, re.I):
            if is_org:
                extracted['companies'].append(line)
            else:
                fallback_company.append(line)

        # Location
        if any(e['entity_group'] == "LOC" for e in ner_result) or is_address(line):
            # skip if line is already marked as designation
            if not is_designation(line):
                extracted['locations'].append(line)

    # Merge company entities for best scoring
    # Only merge the first ORG line for clean company name
# Filter out lines that are actually designations
    org_candidates = [
        line for line in extracted['companies'] 
        if line not in extracted['designations']
    ]

    company_entities = []
    print("org_candidates ",org_candidates,"\nextracted['companies']",extracted['companies'])
    for line in org_candidates:
        ner_line = ner_pipeline(line)
        print("ner_line comp ",ner_line)
        merged = merge_entities(ner_line, "ORG")
        company_entities.extend(merged)

    # Pick the ORG entity with the most words (prefer longer company names)
    best_company = ""
    if company_entities:
        best_company = max(company_entities, key=lambda x: (len(x['text'].split()), x['score']))['text']


    # Clean up addresses by removing designation words
    cleaned_addresses = []
    for addr in extracted['locations']:
        for desig in extracted['designations']:
            addr = addr.replace(desig, "")
        cleaned_addresses.append(addr.strip())
    return {
        "name": clean_name(extracted['persons'][0]) if extracted['persons'] else "",
        "company": best_company or (fallback_company[0] if fallback_company else "") or "",
        "address": cleaned_addresses,
        "designation": extracted['designations']
    }


init_ner()
# -------------------- API --------------------
@app.post("/extract")
async def extract(req: OCRRequest):
    print("req.ocrLines ",req.ocrLines)
    result = extract_entities(req.ocrLines)
    return result


# -------------------- Run --------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)

