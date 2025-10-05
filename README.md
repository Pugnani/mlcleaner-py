# 🧹 mlcleaner

**mlcleaner** is a lightweight Python library for fast and reliable **data cleaning, encoding, and normalization**.  
It’s designed for students, data scientists, and ML engineers who want a clean, consistent, and testable preprocessing pipeline.

---

## 🚀 Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```


**🧠 Key Features**
🔹 Data Cleaning

    Remove duplicates and missing values

    Handle outliers (IQR or z-score based)

    Functions: clean_small(), clean_large()

🔹 Categorical Encoding

    Label encoding

    Ordinal encoding (custom mappings supported)

    One-hot encoding for automatic expansion

🔹 Numeric Normalization

    Min-Max scaling

    Standard scaling

    Automatically detects numeric columns

**🧪 Running Tests**

Run all tests using pytest:

pytest

All core functions are fully tested to ensure stability and correctness.
**📄 License**

This project is licensed under the MIT License — see the LICENSE

file for details.

**📚 Quick Example**

import pandas as pd
from mlcleaner import clean_small, encode_categorical, normalize_numeric

df = pd.read_csv("data.csv")

df = clean_small(df)
df = encode_categorical(df)
df = normalize_numeric(df)

print(df.head())

**👨‍💻 Author**

Stefano Pugnani
📧 pugnani1906@gmail.com


BSc in Computer Science — Free University of Bozen-Bolzano
