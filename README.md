python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
pip freeze > requirements.txt
pip install -r requirements.txt
