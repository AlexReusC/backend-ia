import sys
from io import StringIO
from joblib import load
import pandas as pd

df = pd.read_csv(StringIO(sys.argv[1]), sep=",", header=None)

clp = load("MLmodel.joblib")

prediction = clp.predict(df)
print(prediction)