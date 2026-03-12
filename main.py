from fastapi import FastAPI
import pandas as pd
from openai import OpenAI

app = FastAPI(title="MercadoLibre Seller Strategy API")

# cargar dataset
df_seller = pd.read_csv("df_seller.csv")

client = OpenAI()

# ---------------------------------------------------
# GENERAR ESTRATEGIA
# ---------------------------------------------------

def generar_estrategia(seller_row):

    seller_data = {
        "num_distinct_products": seller_row["num_distinct_products"],
        "total_stock": seller_row["total_stock"],
        "median_price": seller_row["median_price"],
        "avg_discount_amount": seller_row["avg_discount_amount"],
        "cluster": seller_row["cluster"]
    }

    prompt = f"""
You are a senior commercial strategy analyst at Mercado Libre.

Your task is to help the commercial team design strategies to manage marketplace sellers.

The goal is NOT to advise the seller directly.
The goal is to recommend actions that Mercado Libre's commercial team should take.

Seller Data:
{seller_data}

Return a structured strategy including:

1. Diagnosis of the seller
2. Main risks
3. Opportunities
4. Recommended commercial actions
5. Priority level
6. Expected impact

Respond in Spanish.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert marketplace strategist."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ---------------------------------------------------
# ENDPOINT
# ---------------------------------------------------

@app.get("/seller_strategy")

def get_seller_strategy(seller_nickname: str):

    seller_row = df_seller[df_seller["seller_nickname"] == seller_nickname]

    if seller_row.empty:
        return {"error": "Seller not found"}

    seller_row = seller_row.iloc[0]

    strategy = generar_estrategia(seller_row)

    return {
        "seller_nickname": seller_nickname,
        "cluster": int(seller_row["cluster"]),
        "seller_data": {
            "num_distinct_products": int(seller_row["num_distinct_products"]),
            "total_stock": int(seller_row["total_stock"]),
            "median_price": float(seller_row["median_price"]),
            "avg_discount_amount": float(seller_row["avg_discount_amount"])
        },
        "strategy": strategy
    }