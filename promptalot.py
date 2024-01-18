import requests

# Base URL of the FastAPI application including the correct endpoint
url = 'https://predicter-qkns26jbea-ew.a.run.app/predict_string/'

num_requests = 10
for _ in range(num_requests):
    # Params to send
    params = {'text': 'Hello, this is text to prompt. Is it AI generated? I hope so!',
              'model_name': 'latest'}

    try:
        # Make a POST request
        response = requests.post(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Error:", response.status_code, response.json())

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
