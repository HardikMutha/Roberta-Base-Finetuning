# 🧠 Roberta‑Base‑Finetuning

Fine-tune Hugging Face’s `distill-roberta-base` model for custom text classification tasks using the Transformers & Datasets libraries.

---

## Features
- Seamless adapter to any text classification dataset.
- Utilizes `Trainer` API for elegant training/evaluation.
- Configurable hyperparameters: epochs, batch‑size, learning rate, weight decay.
- Customized id↔label mapping.
---

## Getting Started

### Prerequisites

- Python 3.8+
- GPU recommended (CUDA-capable or AMD w/ ROCm)
- Dependencies:
  ```
  pip install torch transformers datasets huggingface_hub tensorboard fastapi
  sudo apt-get install git-lfs
  ```

### Authenticate with Hugging Face (optional)

```python
from huggingface_hub import notebook_login
notebook_login()
```

### Dataset

Uses any Hugging Face dataset with `train`/`test` splits. By default:
```python
dataset = load_dataset("your_dataset_id")
```
Example: `"ag_news"` dataset is used in tutorials.

## Use Cases

- **Topic Classification** – e.g., AG News dataset  
- **Sentiment Analysis**
- **Domain‑specific Text Categorization**  
  Adaptable via dataset + updated id2label mapping.

---
## Accessing via API
## POST `/train_model`

Train the model on a CSV dataset.

### Request

**Method**: `POST`  
**Content-Type**: `multipart/form-data`

### Form Data

| Key         | Type   | Description                    |
|-------------|--------|--------------------------------|
| `file`      | file   | CSV file containing training data |
| `model_path`| string | Path to save the trained model |

### Example

```bash
curl -X POST http://127.0.0.1:8000/train_model \
  -H "Authorization: Bearer <your_token>" \
  -F "file=@/path/to/new.csv" \
  -F "model_path=./model_path_here"
```

---

## GET `/predict`

Predict the role based on title and description.
### Request

**Method**: `GET`  
**Content-Type**: `application/json`

### Body

```json
{
  "title": "Title of the task goes in here",
  "description": "description of the task goes in here"
}
```

### Example
```bash
curl -X GET http://127.0.0.1:8000/predict \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{
        "title": "Setup a frontend interface",
        "description": "Create the dashboard UI for task monitoring"
      }'
```
---
## 🛠 Notes

- Ensure the server is running at `http://127.0.0.1:8000`
- The model must be trained before using the `/predict` endpoint.
- The CSV file for training should contain labeled examples with fields compatible with your model format.
---
## Performance Tips

- For efficiency on AMD GPUs, consider mixed precision with `torch.cuda.amp`.
- Can easily incorporate **LoRA** via PEFT to reduce trainable params dramatically.
- Use `RobertaTokenizerFast` or `AutoTokenizer` for better speed.

---

## 🛠️ Customization

- Adjust text length via `max_length`.
- Modify hyperparameters (batch size / LR / epochs).
- Swap in different datasets by changing `load_dataset("...")`.
- Extend to multi-label classification and sequence tasks by adjusting model type and labels.

---

