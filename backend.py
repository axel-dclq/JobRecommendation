import pandas as pd
import numpy as np

# Cleaning
from datetime import datetime, timedelta
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Similiraty 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the models
import pickle

import warnings
warnings.filterwarnings("ignore")

# Please download it once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


similarities_title = pickle.load(open('similarities_title.pkl', 'rb'))
similarities_description = pickle.load(open('similarities_description.pkl', 'rb'))


def load_data():
    # User
    data_user = pd.read_csv('data\survey_results_public.csv')

    # Job
    data_job = pd.read_csv('data\data_job_clean.csv')

    return data_user, data_job

def detect_keywords(main_string, keywords):
    main_list = main_string.split(' ')
    keywords_list = keywords.split(' ')
    return all(keyword in main_list for keyword in keywords_list)

def search(data_job, input_text, **kwargs):
    """
    Search for a job in the dataset based on the input text and optional filters.

    Parameters:
    - input_text (str): Text entered by the user.
    - **kwargs (dict): Filters for company, employmenttype_jobstatus, and joblocation.

    Returns:
    pd.Series: The line in the dataset that matches the criteria, or None if no match is found.
    """
    # Initialize a mask to filter the dataset
    mask = (data_job['jobtitle'].apply(lambda x: detect_keywords(x, input_text)) | data_job['jobtitle'].str.contains(input_text, case=False, na=False))

    # Apply additional filters if provided
    for key, value in kwargs.items():
        mask &= (data_job[key] == value)

    # Get the matching row from the dataset
    result = data_job[mask]

    # Return the result (a DataFrame if there are matches, None otherwise)
    return result if not result.empty else None


def calculate_recency_bonus(date_published):
    current_date = datetime.now()
    delta = current_date - date_published
    days_ago = delta.days
    return max(0, 0.5 - days_ago * 0.02)  # Bonus decreases linearly over time

def recommendation_search(data_job, search_result, **kwargs):
    """
    Search for job recommendations based on a given search result and optional filters.

    Parameters:
    - search_result (pd.Series): The line in the dataset that matches the criteria, or None if no match is found.
    - **kwargs (dict): Optional filters for company, employmenttype_jobstatus, and joblocation.

    Returns:
    - pd.Series: Job titles that match the search criteria and filters, sorted by relevance. Top 11 as the first one is tu current search

    Notes:
    - The function calculates a relevance score based on the similarity of the job title and description.
    - Top 10 jobs with the highest relevance scores are returned.
    - Additional filters can be applied using **kwargs to refine the search.
    - A recency bonus is applied to prioritize more recent job postings.

    Formula used : score = 0.3 * title_similarity + 0.7 * description_similitary + sum(1 for each filter macthed) + max(0, 0.5 - days_ago * 0.02)

    """
    # Find index of the search
    indx = search_result.index[0]

    # Get similarities with this search
    score_title = list(enumerate(similarities_title[indx]))
    score_description = list(enumerate(similarities_description[indx]))

    # Calculate relevance scores for each job
    scores = {score_title[i][0]: score_title[i][1] * 0.3 + score_description[i][1] * 0.7 for i in range(len(score_title))}
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:11])

    # Apply additional filters if provided
    if kwargs:
        for idx, score in scores.items():
            temp = data_job.loc[idx]
            for col, value in kwargs.items():
                if value.lower() in temp[col].lower():
                    scores[idx] += 1
    
    # Apply recency bonus
    for idx in scores.keys():
        recency_bonus = calculate_recency_bonus(data_job.loc[idx]['postdate'])
        scores[idx] += recency_bonus

    # Get top 10 jobs based on the combined relevance score and filters
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10])

    # Create a DataFrame with job titles, filter values, and scores
    res = data_job.loc[scores.keys()][['jobtitle'] + list(kwargs.keys())]
    res['score'] = [scores[idx] for idx in scores]

    return res


def main():

    data_user, data_job = load_data()

    return

if __name__ == '__main__':
    main()
