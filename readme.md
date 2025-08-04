An Enhanced Sine Cosine Algorithm Based Light Gradient Boosting Machine for Occupation Injury Outcome Prediction

This repository the implementation of the Enhanced Sine Cosine Algorithm (ESCA) for tuning LightGBM hyperparameters on regression tasks. All code is modularized for clarity and ease of reuse.

📂 Repository Structure


├── env_setup.py        # Environment tweaks (e.g. CPU settings)
├── imports.py          # All library imports
├── data_loader.py      # Loads `Data.xlsx`, constructs X and y arrays
├── ssa_helper.py       # `solution` helper class for SSA/ESCA
├── esca.py             # ESCA implementation (optimization loop)
├── store.py            # CSV logging utility (`ESCA.csv`)
├── main.py             # Entry point: K‑fold, scaling, runs ESCA + LightGBM
├── requirements.txt    # Pin versions for reproducibility
├── Data.xlsx           # Input dataset (place in root)
├── ESCA.csv            # Output of convergence + results (auto‑created)
└── README.md           # This documentation



⚙️ Prerequisites

- Python: ≥ 3.8 (tested on 3.8–3.10)
- OS: Windows, macOS, or Linux


🗂 Data

- Place `Data.xlsx` in the repository root.
- Features (columns): health expenditure, employment in industry, gdp, labor force, life expectancy, urban population.
- Target (column): occupational injury.

Ensure the Excel file has exactly these column names, with no extra header rows.


🚀 Usage

1. Configure random seeds for reproducibility in `main.py` (near top):

   import numpy as np
   import random
   np.random.seed(42)
   random.seed(42)
   

2. Run experiments:

   python main.py
  
3. 📈 Output Files
   - Prints K‑fold metrics to console.
   - Appends convergence curves and best results to `ESCA.csv`.


4. 📝 License

This project is licensed under the MIT License. 
---

5 📑 Citation

If you use this code in your research, please cite:

> Tibet & Ojekemi. “An Enhanced Sine Cosine Algorithm Based Light Gradient Boosting Machine for Occupation Injury Outcome Prediction” *Discover Computing*, 2025.

