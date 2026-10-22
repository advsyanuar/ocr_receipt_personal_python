import base64
import secrets
import string
import time
from datetime import date

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def generate_custom_id(length: int):
    allowed = string.ascii_letters + string.digits
    return ''.join(secrets.choice(allowed) for _ in range(length))


def encode_image(img_path: str) -> str:
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

llm = ChatOllama(
    model="qwen2.5vl:3b",
    temperature=0.0
)
image_path = "./structs/struct-6.jpeg"

base64_image = encode_image(image_path)
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Perform OCR on this image. Extract all the text available."
        },
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        }
    ]
)

now = date.today().isoformat()
start_time = time.perf_counter()
response = llm.invoke([message])
end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds.")
print(response.content)

filename = f"struct-{now}-{generate_custom_id(4)}.txt"
with open(f"./parsed/{filename}", "w") as file:
    file.write(str(response.content))


print("=====================")
print(f"Content saved to: {filename}")
