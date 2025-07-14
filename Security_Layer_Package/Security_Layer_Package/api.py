import requests

# Define the API endpoint
url = 'http://127.0.0.1:8888'
# prompt='hello'
# Make the API call



# Display the response in the app
def prompt_injection_detection(prompt,lang='en'):
    request=url
    if(lang=='ar'):
        request=request+'/arabic_prompt_injection/'
    else:
        request=request+'/english_prompt_injection/'
    response = requests.get(request+prompt)
    return(response.text[2:-3])

def offense_detection(prompt,lang='en'):
    request=url
    if(lang=='ar'):
        request=request+'/arabic_offense/'
    else:
        request=request+'/english_offense/'
    response = requests.get(request+prompt)
    return(float(response.text[2:-3]))

# print(offense_detection('hello'))
print(prompt_injection_detection('مرحبا',lang='ar'))