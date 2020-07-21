import os, sys, shutil
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from dotenv import load_dotenv
from slack import WebClient
from slack.errors import SlackApiError

load_dotenv()

def send_data_to_drive(filepath: str):
    path_to=os.environ['GOOGLE_DRIVE_DESTINATION_PATH']
    credentials_path = os.environ['GOOGLE_DRIVE_CREDENTIALS_PATH']
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(credentials_path)
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile(os.path.realpath(credentials_path))

    drive = GoogleDrive(gauth)

    folders = drive.ListFile(
    {'q': "title='" + path_to + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
    for folder in folders:
        if folder['title'] == path_to:
            file_drive = drive.CreateFile({
                'title': filepath.split("/")[1] + '.log', 
                'parents': [{ 'id': folder['id']}] })  
            file_drive.SetContentFile(filepath) 
            file_drive.Upload()
    
def send_slack_notification(data: str):
    client = WebClient(token=os.environ['SLACK_API_TOKEN'])

    try:
        response = client.chat_postMessage(
            channel='#deepswarm2',
            text=data)
        assert response["message"]["text"] == data
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        # print(f"Got an error: {e.response['error']}")
        
if __name__ == '__main__':
    models = os.listdir('saves/')
    models.sort()
    path = 'saves/' + models[-1] + '/deepswarm.log'
    send_data_to_drive(path)
    send_slack_notification(models[-1])
    for model in models[:-5]:
        shutil.rmtree(f'saves/{str(model)}')
    