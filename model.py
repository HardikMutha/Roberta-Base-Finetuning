#TRIAL

# import pickle
# import torch
# from datasets import load_dataset
# from transformers import (
#     RobertaTokenizerFast,
#     RobertaForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     AutoConfig,
#     pipeline
# )

# from sklearn.preprocessing import LabelEncoder

# MODEL_ID = "FacebookAI/roberta-base"
# MODEL_PATH = "./summary-training-1748945582/checkpoint-2800"
# # Dictionaries for role mapping
# RoleToIndex = {
#     'AIEngineer': 0,
#     'BackendEngineer': 1,
#     'CloudEngineer': 2,
#     'DatabaseDesignEngineer': 3,
#     'DevOpsEngineer': 4,
#     'FrontEndEngineer': 5
# }

# IndexToRole = {v: k for k, v in RoleToIndex.items()}


# def predict(text, model, tokenizer):
#     inputs = tokenizer(text, padding=True, truncation=True,
#                        max_length=256, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         predicted_class_id = torch.argmax(logits, dim=1).item()
#         predicted_role = IndexToRole[predicted_class_id]
#         # print("Predicted Role", predicted_role)
#         return predicted_role


# def initialize_model():
#     """
#     Initialize the model and tokenizer from the specified path.
#     Load a pre-trained model and tokenizer from the specified path.
#     """
#     model = RobertaForSequenceClassification.from_pretrained(MODEL_ID)
#     tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID)
#     return [model, tokenizer]


# if __name__ == "__main__":
#     print("Loading model and tokenizer...")
#     print("model_path:", MODEL_PATH)
#     model, tokenizer = initialize_model()
#     while(True):
#         text = input("Enter Description to classify: ")
#         output = predict(text, model, tokenizer)
#         print(f"Predicted Role: {output}")
