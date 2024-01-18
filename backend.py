import pandas as pd

# Cleaning
from datetime import datetime

# Similiraty 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from difflib import SequenceMatcher

import warnings
warnings.filterwarnings("ignore")

# Please download it once
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def load_data(demo=True):
    global data_job, similarities_title, similarities_description, vectorizer
    data_job = pd.read_csv('data\data_job_clean.csv')
    data_job['postdate'] = pd.to_datetime(data_job['postdate'])

    # Sample the dataset to produce results faster 
    if demo:
        data_job = data_job[:len(data_job) // 8]

    # Recommendation search
    # Vectorization TF-IDF
    vectorizer = TfidfVectorizer()
    matrix_title = vectorizer.fit_transform(data_job['jobtitle'])
    matrix_description = vectorizer.fit_transform(data_job['jobdescription'])

    # Similarity calculus
    similarities_title = cosine_similarity(matrix_title)
    similarities_description = cosine_similarity(matrix_description)

    return data_job

def detect_keywords(main_string, keywords):
    main_list = main_string.split(' ')
    keywords_list = keywords.split(' ')
    return all(keyword in main_list for keyword in keywords_list)

def find_best_search_indx(expression, data_job, top_n=50):
    expression = expression.lower()
    meilleurs_scores = [0] * top_n
    meilleurs_indices = [None] * top_n

    for index, job_title in enumerate(data_job['jobtitle']):
        job_title_lower = job_title.lower()
        score = SequenceMatcher(None, expression, job_title_lower).ratio()

        for i, top_score in enumerate(meilleurs_scores):
            if score > top_score:
                meilleurs_scores[i] = score
                meilleurs_indices[i] = index
                break

    return [index for index in meilleurs_indices if index is not None]

def search(input_text, **kwargs):
    """
    Search for a job in the dataset based on the input text and optional filters.

    Parameters:
    - input_text (str): Text entered by the user.
    - **kwargs (dict): Filters for company, employmenttype_jobstatus, and joblocation.

    Returns:
    pd.Series: The line in the dataset that matches the criteria, or None if no match is found.
    """
    # Security None in kwargs
    keys_to_remove = [col for col, value in kwargs.items() if value is None]
    for key in keys_to_remove:
        del kwargs[key]

    if input_text is None:
        return [value for _, value in data_job.head(10).T.to_dict().items()]

    # Initialize a mask to filter the dataset
    mask = (data_job['jobtitle'].apply(lambda x: detect_keywords(x, input_text)) | data_job['jobtitle'].str.contains(input_text, case=False, na=False))

    # If the mask is not empty = if the research is exactly found in the dataset
    if not mask[mask == True].empty:
        # Apply additional filters if provided
        if kwargs:
            for key, value in kwargs.items():
                mask &= (data_job[key] == value)

        # Get the matching row from the dataset
        result = data_job[mask]

        # Return the result (a DataFrame if there are matches, None otherwise)
        return [value for _,value in result.T.to_dict().items()] if not result.empty else [value for _, value in data_job.head(10).T.to_dict().items()]
    
    # else we look through similarity
    else:
        # Use the function to find the most similar job title
        result = find_best_search_indx(input_text, data_job)
        temp = data_job.loc[result]

        if result:
            # Apply additional filters if provided
            if kwargs:
                for key, value in kwargs.items():
                    mask = (data_job[key].apply(lambda x: detect_keywords(x, value)) | data_job[key].str.contains(value, case=False, na=False))
                    temp = temp[mask]

            # Return the result (a DataFrame if there are matches, None otherwise)
            return [value for _, value in temp.T.to_dict().items()] if not temp.empty else [value for _, value in data_job.head(10).T.to_dict().items()]
        else:
            return [value for _, value in data_job.head(10).T.to_dict().items()]

def calculate_recency_bonus(date_published):
    current_date = datetime.now()
    delta = current_date - date_published
    days_ago = delta.days
    return max(0, 0.5 - days_ago * 0.02)  # Bonus decreases linearly over time


def recommendation_search(searchs, **kwargs):
    """
    Search for job recommendations based on a given search result and optional filters.

    Parameters:
    - searchs (list): historical searchs from the user
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
    if searchs is None:
        return None
    
    result = {}
    for s in searchs:

        search_result = search(s)
        
        # Find index of the search
        search_result = pd.DataFrame(search_result)
        search_result = data_job.reset_index().merge(search_result, on=['company', 'jobdescription_old',
                                                                    'joblocation_address', 'jobtitle_old', 'postdate',
                                                                        'jobtitle', 'jobdescription', 'requirements'],
                                                                how='inner')

        for indx in search_result['index']:
            # Get similarities with this search
            score_title = list(enumerate(similarities_title[indx]))
            score_description = list(enumerate(similarities_description[indx]))

            # Calculate relevance scores for each job
            scores = {score_title[i][0]: score_title[i][1] * 0.3 + score_description[i][1] * 0.7 for i in range(len(score_title))}
            scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:11])

            # Apply additional filters if provided
            if len(kwargs) > 0:
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
        for k,v in scores.items():
            if k in result.keys():
                result[k] += v
            else:
                result[k] = v


    # Create a DataFrame with job titles, filter values, and scores
    res = data_job.loc[result.keys()][['jobtitle', 'jobdescription_old', 'company', 'jobtitle_old'] + list(kwargs.keys())]
    res['score'] = [result[idx] for idx in result]
    res.sort_values('score', ascending=False, inplace=True)
    
    return [value for _, value in res.head(10).T.to_dict().items()]


def recommendation_skills(skills_user):
    """
    Recommends jobs based on user skills.
    
    Hyper parameters : 
        - k : number of neighbors = top k of most relevant offer

    Parameters:
    - data_job (pd.DataFrame): DataFrame containing job information.
    - skills_user (list): List of skills provided by the user.

    Returns:
    - recommended_jobs (pd.Series): Series containing recommended job titles.
    """
    global vectorizer
    # Recommendation skills
    SKILLS_JOBS = data_job['requirements']
    skills_tfidf = vectorizer.fit_transform(SKILLS_JOBS)
    skills_user = ' '.join(skills_user)
    requirements_tfidf = vectorizer.transform(SKILLS_JOBS)

    # Define the model used
    k = 10
    nn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn_model.fit(requirements_tfidf)

    # Find relevant offers
    skill_tfidf = vectorizer.transform([skills_user])
    _, indices = nn_model.kneighbors(skill_tfidf)

    recommended_jobs = pd.DataFrame(data_job.loc[indices[0], ['jobtitle', 'company', 'jobtitle_old']])
    return [value for _, value in recommended_jobs.T.to_dict().items()]


def get_company():
    return data_job['company'].unique()

def get_location():
    return data_job['joblocation_address'].unique()

# def get_jobstatus():
#     return data_job['employmenttype_jobstatus'].unique()


load_data()

def test():

    example = search('automation engineer')
    print(example)
    print()
    print(recommendation_search(['datascientist']))
    SKILLS_USER = ["Python","Java","C++","JavaScript", "HTML/CSS","SQL","Git"]

    print()

    # test = recommendation_skills(SKILLS_USER)
    # print(test)


if __name__ == '__main__':
    test()
