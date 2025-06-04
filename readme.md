# BR-2025-PREDICT

## Brazilian Série A Match Result Prediction

### Description
This project uses machine learning to predict the results of Brazilian Série A (Brasileirão) matches. It analyzes historical match statistics and team performance data to forecast match outcomes (Win, Draw, Loss).

### Features
- **Match Result Prediction:** Predicts match outcomes (Home Win, Draw, Away Win).
- **Comprehensive Statistics Analysis:** Utilizes a wide range of match statistics, including:
    - Possession
    - Shots (Total and on Target)
    - Pass Accuracy
    - Corners
    - Crosses
    - Fouls
    - Yellow and Red Cards
    - And more...
- **Team Classification Integration:** Incorporates team standings and performance metrics from the league table.
- **Historical Performance Analysis:** Leverages historical match data to train the prediction model.
- **Derived Features:** Includes derived features such as strength difference, form momentum, win rates, and scoring rates to improve prediction accuracy.
- **Data Preprocessing:** Handles missing data and converts string-based percentage values to numerical format.
- **Data Balancing:** Uses SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance issues.

### Data Sources
- **Match Statistics:** Data scraped from Google Sports and other sources, stored in `dataset.json`.
- **Team Performance Data:** League standings and performance metrics sourced from `classificacao.json`.
- **Next Round Matches:**  Matches to be predicted are loaded from `next_round.json`.
- **Historical Team Data:** Historical team performance data used for calculating averages.

### Technical Details
- **Machine Learning Model:** Random Forest Classifier.
- **Features:** Combines match statistics, team performance metrics, and derived features for enhanced prediction accuracy.
- **Training Data:** Historical match data from previous matches in `dataset.json`.
- **Data Scaling:** StandardScaler is used to scale the data.
- **Oversampling:** SMOTE is used to address imbalanced data.
- **Libraries:** pandas, numpy, scikit-learn, imblearn

### Dataset
- `dataset.json`: Contains historical match results and detailed game statistics. Updated manually after each round.
    - Match results (Home Win, Draw, Away Win)
    - Detailed statistics for each match (possession, shots, passes, etc.)
    - Team performance metrics
- `classificacao.json`: Contains the league standings and team performance metrics. Updated manually after each round.
    - Current league standings
    - Team performance statistics (points, wins, losses, goal difference, etc.)
- `next_round.json`: Defines the matches for the next round to be predicted.
- `calcular_media_historica.py`: Script to calculate historical team statistics from `dataset.json`.

### Usage
1.  Clone the repository.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the `main.py` script:
    ```
    python main.py
    ```

### Contributing
Contributions are welcome! Feel free to submit pull requests or open issues to suggest improvements or report bugs.

### License
This project is licensed under the MIT License.

