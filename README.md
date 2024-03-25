# Thai news headlines analysis for Google extension (Extension part)

## Installation

# For Google Chrome extension
1. Open the destination folder where you want to keep the extension and clone our project by using this command:
```python
git clone https://github.com/komcharn018/Thai-news-headlines-analysis-for-Google-extension_Extension
```
2. After finishing cloning our project open the Google chrome and go to settings -> extension -> manage extension
3. Change "Developer mode" at top right corner to on
4. Click the "Load unpacked" menu, then select the project folder that you cloned before and click "Select folder"
5. Now the extension will show on your Google Chrome, pin it to the tab for easier-to-use
6. Install all requirements by using the command:
```python
pip install -r requirements.txt
```
7. Run the extension to the local server by command:
```python
streamlit run .\extension.py
```
8. If the server is running and shows the port that runs on your terminal now you can use our extension 

* #### Additional: install missing Python package (not found) by using this command in the terminal: ####
```python
pip3 install [package name]
```