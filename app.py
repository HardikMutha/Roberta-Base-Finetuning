import torch
import datasets
from pydantic import BaseModel
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    AutoConfig
)
from torch.utils.data import Dataset,DataLoader
from transformers.pipelines import pipeline
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import bitsandbytes
from fastapi import FastAPI, HTTPException,Form,File,UploadFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Annotated
import pandas as pd
import time



app = FastAPI()
label_encoder = LabelEncoder()
model_id="distilbert/distilroberta-base"
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

class TrainModel(BaseModel):
    filePath:str

@app.get('/predict')
async def test (userPrompt:ModelInput):
    promptTitle = userPrompt.title
    promptDes = userPrompt.description
    finalPrompt = f'title : {promptTitle} description:{promptDes}'
    response = await predict(finalPrompt,model,tokenizer)
    return {"predicted_Role":response}
    
from fastapi import UploadFile, File, Form

@app.post('/train_model')
async def train_model(
    file: UploadFile = File(...),
    model_path: str = Form(...)
):
    if not file:
        raise HTTPException(status_code=404, detail="No File Passed")
    try:
        dataset = pd.read_csv(file.file)
        print(dataset.shape)

        dataset_y = dataset['role']
        dataset.drop(['role'], inplace=True, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_y, test_size=0.2, random_state=42)

        train_encodings = tokenizer(list(X_train['description']), padding=True, truncation=True, max_length=256, return_tensors="pt")
        y_train_enc = torch.tensor(label_encoder.fit_transform(y_train))

        IndexToRole = {int(i): role for i, role in zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)}  # type: ignore
        RoleToIndex = {role: int(i) for role, i in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}  # type: ignore

        config = AutoConfig.from_pretrained(model_id)
        config.update({"id2label": IndexToRole})

        # Use the provided model_path
        model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)

        class RoleDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = RoleDataset(train_encodings, y_train_enc)
        output_dir = './models'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            optim="paged_adamw_8bit",
            logging_strategy="steps",
            logging_steps=100,
            logging_dir="./logs",
            save_strategy="no",
            gradient_checkpointing=True,
            report_to="none",
            overwrite_output_dir=True,
            group_by_length=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )
        trainer.train()
        trainer.save_model(f"{output_dir}/final_model/{str(time.time())}")

        return {"message": "Training Completed Successfully. Model saved."}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))
