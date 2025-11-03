# Hangman RL — Streamlit App

Train a probabilistic oracle + RL agent once, save artifacts, and serve an interactive Streamlit app that only loads the saved artifacts (no training per query).

## Project structure
- script.ipynb — Train oracle + agent and save artifacts.
- app.py — Streamlit UI that loads artifacts and serves queries.
- artifacts/ — Saved artifacts (created by the notebook):
  - oracle.pkl
  - agent.pkl
  - training_rewards.png
- corpus.txt — Training words (one per line, letters only).
- test.txt — Test words (one per line, letters only).

## Requirements
- Python 3.8+  
- Packages:
  - streamlit
  - numpy
  - matplotlib
  - tqdm

Install:
```
pip install -U streamlit numpy matplotlib tqdm
```

## 1) Train once and save artifacts
- Open script.ipynb.
- Run the last cell (Cell 2) to:
  - Train the oracle and agent on `corpus.txt`.
  - Save artifacts to `./artifacts/oracle.pkl` and `./artifacts/agent.pkl`.
  - Optionally evaluate on `test.txt` and save `training_rewards.png`.

Artifacts are required before running the app.

## 2) Run the Streamlit app
From the project folder:
```
cd "c:\Dhruv\PESU\Subjects\Sem 5\ML\hackman"
streamlit run app.py
```
Access the app at:
- http://localhost:8501

The app only loads the saved artifacts and does not retrain on each query.

## App tabs
- Suggest Next Guess:
  - Input masked pattern (use `_` for blanks), guessed letters, and lives left.
  - Get the agent’s suggested next letter and top probabilities.
- Play a Game:
  - Start a game with a secret word (or random from `test.txt`) and step the agent one move at a time.
- Evaluate on Test Set:
  - Run an evaluation on a chosen number of words from `test.txt` (triggered only by button).

## Troubleshooting
- “Artifacts not found”:
  - Run `script.ipynb` (Cell 2) to create `artifacts/oracle.pkl` and `artifacts/agent.pkl`.
- Missing `corpus.txt` or `test.txt`:
  - Ensure both files exist in the project folder with letters-only words per line.
- Port in use:
  - Run with a different port:
    ```
    streamlit run app.py --server.port 8502
    ```
- Remote server:
  - ```
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
    ```
  - Open http://<server-ip>:8501 in a browser and allow the port in the firewall.
