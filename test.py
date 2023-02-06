from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import json

gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)

data = {
    "example_key": "example_value"
}


def push_data_to_gdrive(data, file_name):
    file = drive.CreateFile({'title': f'{file_name}'})
    file.Upload()

    file.content_type = 'application/json'
    file.SetContentString(json.dumps(data))
    file.Upload()
push_data_to_gdrive(data, f'trial{1}.json')

'''
drive.push_data_to_gdrive(confusion_matrix, 'generated_data/confusion_matrix', f'trial{trial}.json')

def push_data_to_gdrive(self, data, parent, file_name):
    file = self.drive.CreateFile({'title': f'{file_name}', 'parents': [{'id': parent}]})
    file.Upload()

    file.content_type = 'application/json'
    file.SetContentString(json.dumps(data))
    file.Upload()
'''