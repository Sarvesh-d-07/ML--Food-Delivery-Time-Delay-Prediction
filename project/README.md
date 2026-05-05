# Food Delivery Time Delay Predictor

This project now trains on the real dataset `orders_autumn_2020.csv` (about 18k rows) and predicts **delay minutes**.

## What changed
- Model switched from synthetic classification to real-data regression.
- Input fields now match your CSV columns.
- Output now returns:
  - predicted delay minutes
  - predicted actual delivery minutes
  - status (Delayed / Earlier than estimate)

## Train and run

### 1) Python dependencies
```bash
cd project/model
python -m venv .venv
source .venv/bin/activate
pip install flask scikit-learn pandas numpy
```

### 2) Node dependencies
```bash
cd ../backend
npm install
```

### 3) Train model
```bash
cd ../model
python train_model.py --train
```

### 4) Start APIs
```bash
python train_model.py --serve
```

In another terminal:
```bash
cd project/backend
npm start
```

### 5) Frontend
Open `project/frontend/index.html` and submit values.
