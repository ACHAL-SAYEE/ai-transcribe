from fastapi import FastAPI
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel
import base64
import uvicorn
from datetime import datetime
# from dotenv import load_dotenv
import json

with open("config.json", "r") as f:
    config = json.load(f)

# load_dotenv() 

app = FastAPI()

# Initialize Azure Document Intelligence client
endpoint = config['endpoint']
# api_key = os.getenv("api_key")
api_key=config['azure_key']


client = DocumentAnalysisClient(endpoint, AzureKeyCredential(api_key))



class Base64ArrayInput(BaseModel):
    images: list[str]  # List of Base64 strings


@app.post("/extract")
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
        "phone": ""
    }

    outputs = []

    for idx, base64_image in enumerate(data.images):
        print(f"Processing image {idx+1} at {datetime.now().time()}")

        # Decode Base64
        document_bytes = base64.b64decode(base64_image.split(",")[-1])

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
            company = output["CompanyNames"][0].get("value", "")
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
            if phone and not merged_summary["phone"]:
                merged_summary["phone"] = phone

        if "Addresses" in output:
            for addr in output["Addresses"]:
                content = addr.get("content")
                if content and content not in merged_summary["address"]:
                    merged_summary["address"].append(content)

        print(f"Finished image {idx+1} at {datetime.now().time()}")

    return {**merged_summary, "outputs": outputs}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)