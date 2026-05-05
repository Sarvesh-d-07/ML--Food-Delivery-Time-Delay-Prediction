const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 3000;
const FLASK_API_URL = 'http://127.0.0.1:5001/predict';

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.json({ message: 'Node backend is running. Use POST /predict.' });
});

app.post('/predict', async (req, res) => {
  try {
    const required = [
      'timestamp', 'item_count', 'user_lat', 'user_long', 'venue_lat', 'venue_long',
      'estimated_delivery_minutes', 'cloud_coverage', 'temperature', 'wind_speed', 'precipitation'
    ];

    for (const key of required) {
      if (req.body[key] === undefined || req.body[key] === null || req.body[key] === '') {
        return res.status(400).json({ error: `Missing required field: ${key}` });
      }
    }

    const flaskResponse = await fetch(FLASK_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });

    const data = await flaskResponse.json();

    if (!flaskResponse.ok) {
      return res.status(400).json({ error: data.error || 'Flask API error' });
    }

    return res.json(data);
  } catch (error) {
    return res.status(500).json({ error: 'Server error', details: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Node server running on http://localhost:${PORT}`);
});
