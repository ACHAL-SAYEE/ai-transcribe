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
import time
import logging
from fastapi.responses import JSONResponse

with open("config.json", "r") as f:
    config = json.load(f)

# load_dotenv() 

app = FastAPI() 


# Configure logging format (prints to console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    client_host = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path

    logging.info(f"➡️  Request started | {method} {path} | from {client_host}")

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        logging.exception(f"❌ Error processing {method} {path}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

    process_time = (time.time() - start_time) * 1000
    logging.info(
        f"✅ Response sent | {method} {path} | status {status_code} | "
        f"{process_time:.2f} ms | from {client_host}"
    )

    return response

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
@app.on_event("startup")
def load_ner_model():
    """Load and warm-up Hugging Face NER model once at startup."""
    global ner_pipeline
    logging.info("🚀 Loading NER model at startup...")

    # For faster inference, you can use dslim/bert-base-NER instead of roberta-large
    # model_name = "dslim/bert-base-NER"
    model_name = "Jean-Baptiste/roberta-large-ner-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Warm-up model to avoid first-call delay
    ner_pipeline("warm up test")
    logging.info("✅ NER model ready and warmed up.")

# -------------------- FastAPI Models --------------------
class OCRRequest(BaseModel):
    ocrLines: List[str]

class Base64ArrayInput(BaseModel):
    images: list[str]  # List of Base64 strings
    
# -------------------- Main Extraction --------------------
def extract_entities(ocr_lines: List[str]):
    extracted = {
        "persons": [],
        "companies": [],
        "locations": [],
        "designations": []
    }
    fallback_company = []

    # --- Pass 1: collect all recognized entities ---
    for line in ocr_lines:
        cleaned_line = clean_name(line)
        ner_result = ner_pipeline(cleaned_line)
        print("ner_result ", ner_result)

        # Collect all detected person & company entities (for later scoring)
        for e in ner_result:
            if e['entity_group'] == "PER":
                extracted["persons"].append(e)
            elif e['entity_group'] == "ORG":
                extracted["companies"].append(e)

        # Designation
        if is_designation(line):
            designation_text = re.sub(r",?\s*(India|US|UK+)$", "", line)
            extracted["designations"].append(designation_text)

        # Company fallback logic (non-NER heuristic)
        is_org = any(e["entity_group"] == "ORG" for e in ner_result)
        if (is_org or likely_company(line)) and not re.search(r"(E-Mail|www\.|\.com|do|mailto)", line, re.I):
            if not is_org:
                fallback_company.append(line)

        # Location — append complete line
        if any(e["entity_group"] == "LOC" for e in ner_result) or is_address(line):
            if not is_designation(line):
                extracted["locations"].append(line)

    # --- Select best entities (highest score) ---
    def pick_best(entities, key="score"):
        if not entities:
            return ""
        best = max(entities, key=lambda e: float(e[key]))
        return best["word"].strip()

    best_person = pick_best(extracted["persons"])
    best_company_entity = pick_best(extracted["companies"])

    # --- Merge company entities from org_candidates ---
    org_candidates = [
        line for line in extracted["companies"]
        if line not in extracted["designations"]
    ]

    company_entities = []
    print("org_candidates ", org_candidates, "\nextracted['companies']", extracted["companies"])
    for line in org_candidates:
        ner_line = ner_pipeline(line["word"] if isinstance(line, dict) else line)
        print("ner_line comp ", ner_line)
        merged = merge_entities(ner_line, "ORG")
        company_entities.extend(merged)

    # Prefer NER company entity with highest score and most words
    best_company = best_company_entity
    if company_entities:
        scored_best = max(company_entities, key=lambda x: (len(x["text"].split()), x["score"]))
        best_company = scored_best["text"]

    # Fallback if no recognized company found
    if not best_company:
        best_company = fallback_company[0] if fallback_company else ""

    # --- Clean up addresses ---
    cleaned_addresses = []
    for addr in extracted["locations"]:
        for desig in extracted["designations"]:
            addr = addr.replace(desig, "")
        cleaned_addresses.append(addr.strip())

    return {
        "name": clean_name(best_person),
        "company": clean_name(best_company),
        "address": cleaned_addresses,
        "designation": extracted["designations"][0] if extracted["designations"] else ""
    }


# -------------------- API --------------------
@app.post("/extract")
async def extract(req: OCRRequest):
    print("req.ocrLines ",req.ocrLines)
    result = extract_entities(req.ocrLines)
    return result



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
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)