import time
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def upload_to_google_drive(folder_name: str="PUBLIC_UNIQUE_FOLDER_NAME_in_Gdrive", archived_fname: str="concat_xN.tar.gz"):
	# print(f">> Uploading to Google Drive folder_id: {folder_id} ...")
	print(f">> Uploading to Google Drive folder_name: {folder_name} ...")
	t0 = time.time()
	# Google Drive authentication
	gauth = GoogleAuth()
	gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
	drive = GoogleDrive(gauth)

	# Find the folder ID
	folder_list = drive.ListFile({'q': f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
	if len(folder_list) > 0:
		folder_id = folder_list[0]['id']  # Get the ID of the first folder in the list
	else:
		print(f"No folder named {folder_name} found, Uploading failed!!")
		return

	# Upload the archive to Google Drive
	file_drive = drive.CreateFile({"title": archived_fname, "parents": [{"id": folder_id}]})
	file_drive.SetContentFile(archived_fname)
	file_drive.Upload()
	print(f"{archived_fname} uploaded to Google Drive successfully in {time.time()-t0:.2f} sec!")
