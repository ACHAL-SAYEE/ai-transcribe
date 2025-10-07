from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import boto3
import re
import uvicorn
import os
from dotenv import load_dotenv

app = FastAPI()
load_dotenv() 
# -------------------- AWS Comprehend Init --------------------
# comprehend = boto3.client("comprehend", region_name="us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
print("AWS_ACCESS_KEY ",AWS_ACCESS_KEY)
print("AWS_SECRET_ACCESS_KEY ",AWS_SECRET_ACCESS_KEY)

comprehend = boto3.client(
    'comprehend',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
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
    keywords = ["road", "phase", "sector", "street", "extn", "block", "complex", "tower", "floor", "plot"]
    return any(k.lower() in line.lower() for k in keywords) and bool(re.search(r"\d", line))

def is_designation(line: str) -> bool:
    keywords = [
        "lead", "manager", "director", "chief", "head", "engineer", "developer",
        "designer", "consultant", "analyst", "specialist", "coordinator", "executive",
        "officer", "president", "vp", "founder", "chemist"
    ]
    return any(k.lower() in line.lower() for k in keywords)

def likely_company(line):
    keywords = [
        "MEDICAL", "MEDICALS", "PHARMACY", "HOSPITAL", "CLINIC", "ENTERPRISES",
        "INDUSTRIES", "STORE", "TRADERS", "SOLUTIONS", "TECH", "LAB", "LABS",
        "PRIVATE", "LTD", "PVT", "AGENCIES", "DIGITAL"
    ]
    return any(k.lower() in line.lower() for k in keywords) and not re.search(r"\d", line)

# -------------------- AWS Comprehend Helper --------------------
def aws_ner(text: str):
    if not text.strip():
        return []
    try:
        response = comprehend.detect_entities(Text=text, LanguageCode="en")
        return response.get("Entities", [])
    except Exception as e:
        print("AWS Comprehend error:", e)
        return []

# -------------------- FastAPI Models --------------------
class OCRRequest(BaseModel):
    ocrLines: List[str]

# -------------------- Main Extraction --------------------
def extract_entities(ocr_lines: List[str]):
    extracted = {"persons": [], "companies": [], "locations": [], "designations": []}
    fallback_company = []

    for line in ocr_lines:
        cleaned_line = clean_name(line)
        ner_result = aws_ner(cleaned_line)
        print("ner_result", ner_result)

        entity_types = [e["Type"] for e in ner_result]

        # Person
        if "PERSON" in entity_types:
            extracted["persons"].append(line)

        # Designation
        if is_designation(line):
            designation_text = re.sub(r",?\s*(India|US|UK+)$", "", line)
            extracted["designations"].append(designation_text)

        # Company
        if "ORGANIZATION" in entity_types or likely_company(line):
            if "ORGANIZATION" in entity_types:
                extracted["companies"].append(line)
            else:
                fallback_company.append(line)

        # Location
        if "LOCATION" in entity_types or is_address(line):
            if not is_designation(line):
                extracted["locations"].append(line)

    # Pick best company
    best_company = ""
    if extracted["companies"]:
        best_company = max(extracted["companies"], key=lambda x: len(x))

    # Clean up addresses
    cleaned_addresses = []
    for addr in extracted["locations"]:
        for desig in extracted["designations"]:
            addr = addr.replace(desig, "")
        cleaned_addresses.append(addr.strip())

    return {
        "name": clean_name(extracted["persons"][0]) if extracted["persons"] else "",
        "company": best_company or (fallback_company[0] if fallback_company else ""),
        "address": cleaned_addresses,
        "designation": extracted["designations"],
    }

# -------------------- API --------------------
@app.post("/extract")
async def extract(req: OCRRequest):
    print("req.ocrLines", req.ocrLines)
    result = extract_entities(req.ocrLines)
    return result

# -------------------- Run --------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)
