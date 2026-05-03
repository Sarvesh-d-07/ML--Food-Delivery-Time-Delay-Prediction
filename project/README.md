# Food Delivery Time Delay Predictor (Beginner-Friendly)

This mini-project predicts whether a food delivery will be **Delayed** or **On Time**.

## Tech Stack
- Frontend: HTML + CSS + Vanilla JavaScript
- Backend: Node.js + Express
- ML API: Python + Flask + scikit-learn (Random Forest)

## Folder Structure

```text
/project
  /frontend
    index.html
    style.css
  /backend
    server.js
    package.json
  /model
    train_model.py
    model.pkl
    scaler.pkl
```

## 1) Install dependencies

### Python side
```bash
cd project/model
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install flask scikit-learn pandas numpy
```

### Node side
```bash
cd ../backend
npm install
```

## 2) Train model

```bash
cd ../model
python train_model.py --train
```

This creates:
- `model.pkl`
- `scaler.pkl`

## 3) Run Flask ML API

```bash
python train_model.py --serve
```

Flask runs at: `http://localhost:5001`

## 4) Run Node backend

Open a second terminal:

```bash
cd project/backend
npm start
```

Node runs at: `http://localhost:3000`

## 5) Open frontend

Open `project/frontend/index.html` in your browser.

Then submit the form:

Frontend -> Node `/predict` -> Flask `/predict` -> ML model -> response.

## API example

POST `http://localhost:3000/predict`

```json
{
  "distance": 8.5,
  "prep_time": 18,
  "traffic": "Medium",
  "weather": "Clear",
  "peak_hour": "No"
}
```
