import pandas as pd
import numpy as np

# Cleaning
from datetime import datetime, timedelta
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

def txt_cleaning(text):
    """
    Clean a given text, typically used for job descriptions and titles.

    Parameters:
    - text (str): The input text to be cleaned.

    Returns:
    str: The cleaned and processed text.

    Steps:
    1. Keep only alpha-numeric characters.
    2. Tokenize the text for better processing.
    3. Apply lemmatization to reduce words to their base form.
    4. Remove common English stop words.
    """
    # Keep only alpha-numeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Tokenize for better processing
    tokens = word_tokenize(text.lower())

    # Apply lemmatization and remove stop words using list comprehension
    lemmatization = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]

    return " ".join(lemmatization)

def convert_date(date):
    # Remove ' ago' from the date
    date = date.replace(' ago', '')

    # If date is like 'moments ago'
    if len(date.split()) == 1:
        return datetime.now()

    # If date is like '7 minutes ago'
    if date.split(' ')[1] in ['minute', 'minutes']:
        minutes_ago = int(date.split(' ')[0])
        return datetime.now() - timedelta(minutes=minutes_ago)

    # If date is like '2 hours ago'
    if date.split(' ')[1] in ['hour', 'hours']:
        hours_ago = int(date.split(' ')[0])
        return datetime.now() - timedelta(hours=hours_ago)

    # If date is like '2 weeks ago'
    if date.split(' ')[1] in ['week', 'weeks']:
        weeks_ago = int(date.split(' ')[0])
        return datetime.now() - timedelta(weeks=weeks_ago)

    # If date is like '1 month ago'
    if date.split(' ')[1] in ['month', 'months']:
        months_ago = int(date.split(' ')[0])
        return datetime.now() - timedelta(days=30 * months_ago)

def get_skills_job(data_job):
    data_job['skills'].fillna('', inplace=True)
    data_job['skills'] = data_job['skills'].apply(lambda x: txt_cleaning(x))
    data_job['requirements'] = str(data_job['jobtitle'] + ' ' + data_job['jobdescription'] + ' ' + data_job['skills']).lower()

    return


def preproc_job():
    data_job = pd.read_csv('data\dice_com-job_us_sample.csv')
    data_job = data_job[['company', 'employmenttype_jobstatus', 'jobdescription', 
                     'joblocation_address', 'jobtitle', 'postdate', 'shift', 'skills']]
    
    data_job.dropna(inplace=True)
    data_job.drop_duplicates(inplace=True)
    
    # 20 min to run !!!
    print('Cleaning jobtitle')
    data_job['jobtitle'] = data_job['jobtitle'].apply(lambda x: txt_cleaning(x))

    print('Cleaning jobdescription')
    data_job['jobdescription'] = data_job['jobdescription'].apply(lambda x: txt_cleaning(x))

    data_job['postdate'] = data_job['postdate'].apply(lambda x: convert_date(x))

    get_skills_job(data_job)

    data_job.to_csv('data\data_job_clean.csv', header=True, index=False)

    return
    


