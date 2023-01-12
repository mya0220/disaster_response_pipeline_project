# Disaster Response Pipeline Project
### Project Intro
1. Background: When disaster happens, we will expect massive text message come out and we want to quickly classify those text to facilitate quick response to some emergency help etc..
2. Deliverable: A web interface that can take in text, and by click button, it can be classified into several categories most relavent to it. It involves a NLP classifier.
3. Files: 
        - /data contains 2 csv files that can be used to create a database, and then train model upon. 
        - /data/process_data.py contains the ETL process that finally stores data in database table.
        - /models/train_classifier.py contains a model pipeline to train, test, and save the model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


