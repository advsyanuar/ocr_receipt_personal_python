from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.2
)

file_path = "./parsed/struct-2026-10-18-IH51.txt"

try:
    with open(file_path, "r") as file:
        content = file.read()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""Extract dates,
                            1. Store name,
                            2. Product name + quantity + price,
                            3. Total price,
                            4. Amount paid
                            5. Changes
                            from this text: {content}
                            into JSON format:
                            {{
                                "store_name": "string|null",
                                "products": [
                                    {{
                                        "name": "string|not null",
                                        "quantity": "int|not null",
                                        "price_per_item": "float|not null"
                                    }}
                                ],
                                "buy_date": "string|not null",
                                "total_price": "float|not null",
                                "amount_paid": "float|not null",
                                "changes": "float|not null"
                            }}
                            Notes:
                                - Every price that has Rp in it use Rupiah as a currency, do not convert to USD or any other currency unless it is specified.
                                - The currency format in Rupiah(IDR) sometimes looks like these:
                                    - Rp. 20,000,00
                                    - 30.000,00
                                    - Rp. 20.000,00
                                    - IDR 20.000,00
                                    - Rp.20.000,00
                                    ...etc
                            """
                }
            ]
        )

        response = llm.invoke([message])
        print(response.content)
except FileNotFoundError:
    print("File not found!")
