import uuid
import requests


BASE_URL = "http://localhost:8082/api/v1/task_status"


def get_task_status(task_uuid: str):
    """
    Sends a GET request to the Dockerized FastAPI endpoint to fetch task status.
    """
    # The endpoint expects 'task_uuid' as a query parameter
    params = {
        "task_uuid": task_uuid
    }

    try:
        print(f"Sending request for task: {task_uuid}...")
        response = requests.get(BASE_URL, params=params)

        # Check if the request was successful (200 OK)
        if response.status_code == 200:
            print("Successfully fetched task status!")
            data = response.json()

            # Formatting the printed response
            print(f"Status:      {data.get('status')}")
            print(f"Task UUID:   {data.get('task_uuid')}")
            print(f"Result Text: {data.get('result_text')}")
            return data

        elif response.status_code == 404:
            print(f"Error 404: Task not found. Details: {response.json().get('detail')}")
        else:
            print(f"Unexpected Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print(
            "Connection Error: Could not connect to the API. Is your Docker container running and exposing the correct port?")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    task_uuid = "b0657f91-535f-4e1d-9c98-9caca9647607"

    get_task_status(task_uuid)
