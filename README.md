# NBA Player Prop Prediction System (v2.0)

A professional-grade Machine Learning system for predicting NBA player stats (Points, Rebounds, Assists) using PyTorch neural networks and advanced feature engineering (Defense vs. Position, Head-to-Head history).

## 🚀 Key Features
*   **Deep Feature Engineering**: Calculates "Defense vs. Position" (DvP), Opponent History (H2H), Rolling Form, and Rest Days.
*   **PyTorch Architecture**: Uses a Neural Network with Entity Embeddings for Players and Teams.
*   **Daily Predictions**: Fetches live daily matchups to generate context-aware predictions.
*   **Sharpshooter Accuracy**: Achieves ~5.1 MAE (Mean Absolute Error) for Points, beating standard variance benchmarks.

## 🛠️ Installation

1.  **Clone & Environment**
    ```bash
    git clone <repo_url>
    cd betting_ml
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Data & Setup (First Run)

Before training, you must download the historical data and generate the engineered features.

1.  **Fetch Raw Data** (Downloads 2021-2026 Season Logs)
    ```bash
    python src/data_fetch.py
    ```
    *This may take a few minutes as it downloads ~5 seasons of game logs.*

2.  **Generate Features** (Calculates DvP, H2H, encodings)
    ```bash
    python src/feature_engineer.py
    ```
    *Output: `data/processed/processed_features.csv`*

## 🧠 Training a Player Model

To predict for a specific player (e.g., LeBron James), you must train a specialized "Sharpshooter" model.

**1. Find Player ID**
You can find IDs in the raw CSVs or online.
*   LeBron James: `2544`
*   Steph Curry: `201939`
*   Nikola Jokic: `203999`
*   Luka Doncic: `1629029`

**2. Train the Model**
```bash
python -m src.train_models --player_id <PLAYER_ID> --debug
```
*   **Example**: `python -m src.train_models --player_id 2544 --debug`
*   **Output**:
    *   Model saved to: `data/models/pytorch_player_<ID>.pth`
    *   Training Plot: `data/training_curve_<ID>.png` (Visualize the learning curve!)

## 🔮 Daily Predictions

Generate predictions for upcoming games. The system checks the schedule, finds the opponent, and calculates context-aware stats.

**Option A: Predict for Today (Default)**
```bash
python -m src.daily_predict --player_id <PLAYER_ID>
```

**Option B: Predict for a Specific Date**
```bash
python -m src.daily_predict --player_id <PLAYER_ID> --date YYYY-MM-DD
```
*   **Example**: `python -m src.daily_predict --player_id 2544 --date 2026-01-20`

**Sample Output:**
```text
Matchup: LeBron James vs DEN (Away)
Prediction Results:
 PTS: 23.9
 REB: 7.1
 AST: 6.7
```

## 📂 Project Structure
*   `src/data_fetch.py`: Downloads NBA logs via standard API.
*   `src/feature_engineer.py`: The "Brain". Calculates complex stats (DvP, Rolling).
*   `src/train_models.py`: PyTorch training logic (MLP + Embeddings).
*   `src/daily_predict.py`: Inference engine. Fetches live schedule and predicts.

## 📈 Benchmarks
*   **Baseline Error (Guessing Average)**: ~6.0 pts
*   **Our Model Error**: **~5.1 pts**
*   **Edge**: ~15% improvement over baseline.

**Note**: Predictions are estimates of *mean performance*. Standard deviation for superstars is +/- 8-10 points. Always bet responsibly.
