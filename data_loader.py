import pandas as pd

# Load the dataset
df = pd.read_excel("Data.xlsx")

# Define features and target
features = ['health expenditure', 'employment in industry', 'gdp',
            'labor force', 'life expectancy', 'urban population']

target = ['occupational injury']

X = df[features].values
y = df[target].values
