# Mudra — Real-time ISL Translator

Plain HTML/CSS/JavaScript. No build step.

## Run
Just double-click `index.html` (Chrome/Edge recommended, internet needed for the hand model). Or serve it:
```
python3 -m http.server 8000   # then open http://localhost:8000
```

## Structure
- `index.html` – UI
- `styles.css` – design
- `js/app.js` – everything: features, predictors (RulePredictor, KnnPredictor, ApiPredictor, TfjsPredictor), camera, drawing, history, samples

## Plug in your model
In `js/app.js`, change `engine()` to return e.g.
`new ApiPredictor("https://your-api/predict")`. The API receives
`{ hands: [{ handedness, landmarks, vector }] }` and must return `{ label, confidence }`.

Exported samples (`isl-samples.json`) contain `{ label, vector }` rows (63 features) for training.
