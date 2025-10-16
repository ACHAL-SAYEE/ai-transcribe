from fastapi import FastAPI, Request
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel
import base64
import uvicorn
from datetime import datetime
# from dotenv import load_dotenv
import json
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

with open("config.json", "r") as f:
    config = json.load(f)

# load_dotenv() 

app = FastAPI() 

# Initialize Azure Document Intelligence client
endpoint = config['endpoint']
# api_key = os.getenv("api_key")
api_key=config['azure_key']


client = DocumentAnalysisClient(endpoint, AzureKeyCredential(api_key))


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
        "officer","president","vp","founder","chemist"
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
        "designation": extracted['designations'][0]
    }


init_ner()
# -------------------- API --------------------
@app.post("/extract")
async def extract(req: OCRRequest):
    print("req.ocrLines ",req.ocrLines)
    result = extract_entities(req.ocrLines)
    return result


class Base64ArrayInput(BaseModel):
    images: list[str]  # List of Base64 strings
@app.post("/ai-extract")
async def analyze_business_cards(data: Base64ArrayInput):
    print("hit")

    if not data.images:
        return {"error": "No images provided"}

    merged_summary = {
        "name": "",
        "company": "",
        "designation": "",
        "address": [],
        "department": "",
        "website": "",
        "email": "",
        "WorkPhone": "",
        "MobilePhone":"",
        "OtherPhone":""
    }

    outputs = []

    for idx, base64_image in enumerate(data.images):
        print(f"Processing image {idx+1} at {datetime.now().time()}")

        # Decode Base64
        document_bytes = base64.b64decode(base64_image.split(",")[-1])
        # return { 
        #         "1":1
        #     }
        # Call Azure Document Intelligence
        poller = client.begin_analyze_document("prebuilt-businessCard", document=document_bytes)
        result = poller.result()

        if not result.documents:
            continue

        doc = result.documents[0]
        output = {}
        outputs.append(doc.fields.items())

        for field_name, field in doc.fields.items():
            if field.value_type == "list":
                output[field_name] = []
                for item in field.value:
                    if item.value_type == "dictionary":
                        output[field_name].append({
                            k: {"value": v.value, "confidence": v.confidence}
                            for k, v in item.value.items()
                        })
                    elif item.value_type == "address":
                        output[field_name].append({
                            "structured": item.value.__dict__,
                            "content": item.content,
                            "confidence": item.confidence
                        })
                    else:
                        output[field_name].append({
                            "value": item.value,
                            "confidence": item.confidence
                        })
            elif field.value_type == "dictionary":
                output[field_name] = {
                    k: {"value": v.value, "confidence": v.confidence}
                    for k, v in field.value.items()
                }
            else:
                output[field_name] = {
                    "value": field.value,
                    "confidence": field.confidence
                }

        # Merge into summary
        if "ContactNames" in output and output["ContactNames"]:
            name_parts = output["ContactNames"][0]
            first = name_parts.get("FirstName", {}).get("value", "")
            last = name_parts.get("LastName", {}).get("value", "")
            full_name = f"{first} {last}".strip()
            if full_name and not merged_summary["name"]:
                merged_summary["name"] = full_name

        if "CompanyNames" in output and output["CompanyNames"]:
            # Pick company with highest confidence
            best_company = max(output["CompanyNames"], key=lambda c: c.get("confidence", 0))
            company = best_company.get("value", "")
            if company and not merged_summary["company"]:
                merged_summary["company"] = company


        if "JobTitles" in output and output["JobTitles"]:
            designation = output["JobTitles"][0].get("value", "")
            if designation and not merged_summary["designation"]:
                merged_summary["designation"] = designation

        if "Departments" in output and output["Departments"]:
            department = output["Departments"][0].get("value", "")
            if department and not merged_summary["department"]:
                merged_summary["department"] = department

        if "Websites" in output and output["Websites"]:
            website = output["Websites"][0].get("value", "")
            if website and not merged_summary["website"]:
                merged_summary["website"] = website

        if "Emails" in output and output["Emails"]:
            email = output["Emails"][0].get("value", "")
            if email and not merged_summary["email"]:
                merged_summary["email"] = email

        if "WorkPhones" in output and output["WorkPhones"]:
            phone = output["WorkPhones"][0].get("value", "")
            if phone and not merged_summary["WorkPhone"]:
                merged_summary["WorkPhone"] = phone

        if "MobilePhones" in output and output["MobilePhones"]:
            phone = output["MobilePhones"][0].get("value", "")
            if phone and not merged_summary["MobilePhone"]:
                merged_summary["MobilePhone"] = phone   

        if "OtherPhones" in output and output["OtherPhones"]:
            phone = output["OtherPhones"][0].get("value", "")
            if phone and not merged_summary["OtherPhone"]:
                merged_summary["OtherPhone"] = phone                              

        if "Addresses" in output:
            for addr in output["Addresses"]:
                content = addr.get("content")
                if content and content not in merged_summary["address"]:
                    merged_summary["address"].append(content)

        print(f"Finished image {idx+1} at {datetime.now().time()}")

    return {**merged_summary, "outputs": outputs}


if __name__ == "__main__":
    print("fuck")
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)