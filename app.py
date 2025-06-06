import torch
import asyncio
from pydantic import BaseModel
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    AutoConfig,
)
from fastapi import FastAPI
app = FastAPI()
model_id="FacebookAI/roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
model_path = "./checkpoint-2800"
model = RobertaForSequenceClassification.from_pretrained(model_path)

RoleToIndex = {
  'AIEngineer': 0,
  'BackendEngineer': 1,
  'CloudEngineer': 2,
  'DatabaseDesignEngineer': 3,
  'DevOpsEngineer': 4,
  'FrontEndEngineer': 5
  }

IndexToRole = {
  0: 'AIEngineer',
  1: 'BackendEngineer',
  2: 'CloudEngineer',
  3: 'DatabaseDesignEngineer',
  4: 'DevOpsEngineer',
  5: 'FrontEndEngineer'
  }

config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": IndexToRole})

async def predict(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = int(torch.argmax(logits, dim=1).item())
        predicted_role = IndexToRole.get(predicted_class_id)
        return predicted_role


class ModelInput(BaseModel):
    title: str
    description:str



@app.get('/predict')
async def test (userPrompt:ModelInput):
    promptTitle = userPrompt.title
    promptDes = userPrompt.description
    finalPrompt = f'title : {promptTitle} description:{promptDes}'
    response = await predict(finalPrompt,model,tokenizer)
    return {"predicted_Role":response}
