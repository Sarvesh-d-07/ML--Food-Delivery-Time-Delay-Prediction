/**
 * Beginner-friendly Express backend.
 *
 * Flow:
 * Frontend -> Node/Express (/predict) -> Flask API (/predict) -> Frontend response
 */

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
    const { distance, prep_time, traffic, weather, peak_hour } = req.body;

    // Very simple validation
    if (
      distance === undefined ||
      prep_time === undefined ||
      !traffic ||
      !weather ||
      !peak_hour
    ) {
      return res.status(400).json({ error: 'Missing required fields.' });
    }

    const flaskResponse = await fetch(FLASK_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ distance, prep_time, traffic, weather, peak_hour }),
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
