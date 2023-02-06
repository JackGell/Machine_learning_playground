import json
import os
import threading

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import json

import requests
import time

'''#############################
#################################
Telegram
#################################
'''##############################

def get_chat_id(token):
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json()
        if "result" in result and len(result["result"]) > 0:
            message = result["result"][-1]["message"]
            chat_id = message["chat"]["id"]
            return chat_id
    return None

# Initialize Telegram bot
def send_message(text, image=None):
    token = '5772408137:AAHbA6_YP8zYrOHOWpq8_pEtj41aaVVLt0Q'
    chat_id = get_chat_id(token)
    params = {"chat_id": chat_id or "YOUR_CHAT_ID", "text": text}
    if image:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        files = {'photo': open(image, 'rb')}
        response = requests.post(url, params=params, files=files)
    else:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        response = requests.post(url, params=params)
    return response

def receive_message():
    token = '5772408137:AAHbA6_YP8zYrOHOWpq8_pEtj41aaVVLt0Q'
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    response = requests.get(url)
    return response.json()

def send_basic_accuracy(files_so_far):
    folder = 'generated_data/trials'
    files = os.listdir(folder)
    for file in files:
        if file not in files_so_far:
            with open(f"generated_data/trials/{file.split('.')[0]}.json", "r") as f:
                trial_dict = json.load(f)

            send_text = f'trial{file[5:-5]}: {trial_dict["buy_acc"]:.3f}, {trial_dict["sell_acc"]:.3f}, {trial_dict["hold_acc"]:.3f}'
            send_message(send_text)
    return files_so_far

def send_confusion_matrix(files_so_far):
    folder = 'generated_data/confusion_matrix'
    files = os.listdir(folder)
    for file in files:
        if file not in files_so_far:
            with open(f"generated_data/confusion_matrix/{file.split('.')[0]}.json", "r") as f:
                trial_dict = json.load(f)
            send_text = f'trial{file[5:-5]}:\r\n TP:{trial_dict["TP"]:.3f}\r\n TN:{trial_dict["TN"]:.3f}\r\n FP:{trial_dict["FP"]:.3f}\r\n FN:{trial_dict["FN"]:.3f}'
            send_message(send_text)
    return files_so_far

def send_metrics(files_so_far):
    folder = 'generated_data/metrics'
    files = os.listdir(folder)

    for file in files:
        if file not in files_so_far:
            with open(f"generated_data/metrics/{file.split('.')[0]}.json", "r") as f:
                trial_dict = json.load(f)
            send_text = f'trial{file[5:-5]}:\r\n precision:{trial_dict["precision"]:.3f}\r\n NPV:{trial_dict["NPV"]:.3f}\r\n sensitivity: {trial_dict["sensitivity"]:.3f}\r\n specificity: {trial_dict["specificity"]:.3f}\r\n accuracy: {trial_dict["accuracy"]:.3f}'
            send_message(send_text)
    return files_so_far

def handle_trial_request(last_trial):
    updates = receive_message()
    update = updates["result"][-1]
    message = update["message"]["text"]
    message = message.split()

    if message[0].lower() == "/trial" and last_trial!=message[1]:

        with open(f"generated_data/trials/trial{message[1]}.json", "r") as f:
            trial_dict = json.load(f)
        send_text = f'trial{message[1]}: {trial_dict["buy_acc"]:.3f}, {trial_dict["sell_acc"]:.3f}, {trial_dict["hold_acc"]:.3f}'
        send_message(send_text)

        with open(f"generated_data/metrics/trial{message[1]}.json", "r") as f:
            trial_dict = json.load(f)
        send_text = f'trial{message[1]}:\r\n precision:{trial_dict["precision"]:.3f}\r\n NPV:{trial_dict["NPV"]:.3f}\r\n sensitivity: {trial_dict["sensitivity"]:.3f}\r\n specificity: {trial_dict["specificity"]:.3f}\r\n accuracy: {trial_dict["accuracy"]:.3f}'
        send_message(send_text)

        with open(f"generated_data/confusion_matrix/trial{message[1]}.json", "r") as f:
            trial_dict = json.load(f)
        send_text = f'trial{message[1]}:\r\n TP:{trial_dict["TP"]:.3f}\r\n TN:{trial_dict["TN"]:.3f}\r\n FP:{trial_dict["FP"]:.3f}\r\n FN:{trial_dict["FN"]:.3f}'
        send_message(send_text)

        send_message(f'plot{message[1]}', f"generated_data/plots/plot{message[1]}.png")

        last_trial = message[1]
    return last_trial


# Telegram communication thread
def telegram_thread(interval=10):

    send_basic_accuracy_lst = []
    send_confusion_matrix_lst = []
    send_metrics_lst = []
    last_trial = 'a'

    while True:

        try:
            send_basic_accuracy_lst = send_basic_accuracy(send_basic_accuracy_lst)
        except:
            pass
        try:
            send_confusion_matrix_lst = send_confusion_matrix(send_confusion_matrix_lst)
        except:
            pass
        try:
            send_metrics_lst = send_metrics(send_metrics_lst)
        except:
            pass
        try:
            last_trial = handle_trial_request(last_trial)
        except:
            pass

        time.sleep(interval)



'''#############################
#################################
Gdrive
#################################
'''##############################




class Gdrive:
    def __init__(self):
      self.gauth = GoogleAuth()
      self.gauth.LoadCredentialsFile("mycreds.txt")
      self.drive = GoogleDrive(self.gauth)

    def push_data_to_gdrive(self, data, file_name):
      file = self.drive.CreateFile({'title': f'{file_name}'})
      file.Upload()

      file.content_type = 'application/json'
      file.SetContentString(json.dumps(data))
      file.Upload()



