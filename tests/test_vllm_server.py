import time
from openai import OpenAI


if __name__ == "__main__":
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"  # vLLM does not require a real key
    )

    start_time = time.time()
    response = client.chat.completions.create(
        model="google/gemma-3-1b-it",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a long joke about PC Monitors."}
        ],
        max_tokens=400,
        temperature=0.1,
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:4f} seconds")

    print(response.choices[0].message.content)
