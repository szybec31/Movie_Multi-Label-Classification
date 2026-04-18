
## Download and run the project:
1. Clone the repository:
   ```
    git clone https://github.com/szybec31/Movie_Multi-Label-Classification.git
    cd Movie_Multi-Label-Classification
   ```
2. Create a virtual environment:
    ```
    python -m venv myvenv
    myvenv\Scripts\activate       # Windows
    source myvenv/bin/activate    # Linux/Mac 
    ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Build ready app: (dist folder)
   ```
   cd app
   pyinstaller --noconsole --onedir --icon=assets/icon.ico --add-data "assets;assets" --add-data "models;models" run.py
   ```
## Authors:
