# **JobRecommendation**

To develop a system that recommends jobs to users based on their profiles and preferences using machine learning techniques.

## **Context**

This project involves designing and implementing a job recommendation system. The system will use user profiles and job listings to suggest the most suitable jobs for each user.

## **Data**

We found our data on kaggle [here](https://www.kaggle.com/datasets/PromptCloudHQ/us-technology-jobs-on-dicecom). This dataset has following fields:
- advertiserurl
- company
- employmenttype_jobstatus
- jobdescription
- joblocation_address
- jobtitle
- postdate
- shift
- skills

The data has been preprocessed thanks to the function in `preprocessing.py`. Since the processing takes several tens of minutes, we did not want to increase the application's response time at launch. The main processing steps are:
1. NA and duplicates droping
2. text cleaning : jobtitle, jobdescription, skills
3. datetime conversion
4. creation of a column requirements that groups all the skills needed for a job

## **Structure and features**

Our work is divided into two parts, as it could be for any application development: the backend (`backend.py`) and the frontend (`main.py`).

The main functionalities include:

- **User Profile Creation:** The system allows the creation of a user profile that saves entered information (name, surname, password, skills) using a MongoDB database. Users can view and manage their profiles, including adding and removing skills.

- **Job Search:** A search bar enables users to find the most relevant job postings based on keywords and filters (company, location, job status). Our system can provide a set of job postings matching these criteria.

- **Job Recommendations from searches:** After searching for a term like 'datascientist', our system will recommend job postings related to that position. Each search is recorded in the database, refining our recommendations.

- **Skill-Based Recommendations:** Based on the skills entered during profile creation (editable at any time), our system suggests the most relevant job postings.

- **Intuitive Interface:** The GUI features a user-friendly interface with distinct sections for search, recommendations, and profile management.


## **Project Methodology**

**1. Data processing**
<br>To process our data, we used tools as regex, tokenizer and lemmatizer to have the most workable text possible.

**2. Job search : `search()` function** 
<br>To make the job search more comfortable, we have improved our job recommendation methods. We understand that it's possible to enter a keyword that may not exactly match what our data contains. That's why we suggest job listings that closely match the search performed. Additionally, the addition of filters will refine the results even further.

**3. Job Recommendation for searches : `recommendation_search()` function**
<br>The searches conducted are recorded in our database to provide personalized job recommendations. To achieve this, we compare previously suggested job listings to recommend similar ones. To make this suggestion, we utilize similarity matrices on the titles and descriptions of the job listings. We used the `TfidfVectorizer` function from `sckikit-learn` on the whole dataset and then `cosine_similarity` metric. Then, for a certain job offer, we just have to pick the ones that have the similarity score.
We use the following formula: 

```math
\text{score} = 0.3 \times \text{title\_similarity} + 0.7 \times \text{description\_similarity} + \sum_{i=1}^{\text{card}_{\text{each filter matched}}} 1 + \max(0, 0.5 - \text{days\_ago} \times 0.02)
```

The calculus of the similarity matrix is done in the `load_data()` function from `backend.py`.

```python
# Recommendation search
# Vectorization TF-IDF
vectorizer = TfidfVectorizer()
matrix_title = vectorizer.fit_transform(data_job['jobtitle'])
matrix_description = vectorizer.fit_transform(data_job['jobdescription'])

# Similarity calculus
similarities_title = cosine_similarity(matrix_title)
similarities_description = cosine_similarity(matrix_description)
```

**4. Job Recommendation for skills : `recommendation_skills()` function**
<br>The idea of the function is that from given user skills, we have to find the most suitable job offers. We used `NearestNeighbors(n_neighbors=k, metric='cosine')` apply to similarity between the user skills and the required skills for job. The parameter `k` is the number of the nearest neighbors we want to keep. It can be interpreted as the number of recommendation we want to propose. 

```python
# Recommendation skills
SKILLS_JOBS = data_job['requirements']
skills_user = ' '.join(skills_user)
requirements_tfidf = vectorizer.transform(SKILLS_JOBS)

# Define the model used
k = 10
nn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
nn_model.fit(requirements_tfidf)

# Find relevant offers
skill_tfidf = vectorizer.transform([skills_user])
_, indices = nn_model.kneighbors(skill_tfidf)
```

**5. Friendly interface**
<br>Our interface prioritizes simplicity and ease of use to enhance the user experience. The search bar is designed to accommodate various keywords, providing intelligent suggestions that closely match user queries. Filters are readily available, allowing users to refine results based on specific criteria such as location, job type, or industry.

The recommendation interface seamlessly integrates both search and skills-based suggestions. Personalized recommendations are influenced by previous searches, utilizing similarity matrices for accuracy. This approach ensures users receive relevant job opportunities aligned with their preferences and expertise.

## **Evaluation of our models**
The accuracy of the models we employed for our project is challenging to quantify, primarily due to the reliance on similarity metrics. Following discussions with our professor, we concluded that evaluating our models involves manual verification to assess the relevance of our results. In this regard, we are pleased with the outcomes. We tested inputting our own skills into the application, and the recommendations proved to be relevant. The same holds true for VIP users of our application.

We invite you to test our application with your own skills to assess whether the recommendations obtained align well with your profile. We welcome any feedback to enhance our recommendation algorithms.

## **Requirements**

- pandas==2.0.3    
- numpy==1.25.2
- nltk==3.8.1
- scikit-learn==1.3.1
- customtkinter == 5.2.2
- CTkListbox == 1.2
- pymongo == 2.0

Before running the code, make sure to download the following resources using nltk:

```bash
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## **Getting Started**

To run the application, follow these steps:

1. Ensure you have Python installed on your machine and all librairies.

2. **Run the Python Script:**
   - Execute the Python script by running the following command in your terminal:
     ```bash
     python main.py
     ```

### **Usage**

1. **Search for Jobs:**
   - Enter your job search query in the search bar.
   - Optionally, specify job status, company, and location.
   - Press the "Search" button or hit Enter.

2. **Skill Management:**
   - Log in or create a profile to manage your skills.
   - Enter skills in the "Skills" entry and press Enter or click "Insert."
   - Remove skills by selecting them and clicking "Remove."

3. **Account Management:**
   - Create an account by clicking "Create Profile" and providing a username and password.
   - Log in using your credentials.
   - Optionally, delete your account from the "Profile" section.

4. **Job Recommendations:**
   - View job recommendations based on your skills and search history on the right side of the interface.

## **Notes**

- The application uses MongoDB for user data storage. Ensure you have an internet connection.

- The application allows for guest access without an account; however, certain features are restricted to registered users (saving research, saving skills).


## **Researches/Ideas**

The job recommendation task relies largely on Natural Language Processing (NLP) methods. We attempted to apply BERT with the goal of returning the most relevant job listing based on given skills. To achieve this, we trained BERT on the skills requested by advertisers, with the associated job title as the label.

However, our available data does not adequately support proper model training due to a high number of different labels, making it challenging to handle. Additionally, the training process demands a considerable amount of time, as evidenced by an overnight training session during the initial test on a subset of the dataset.

The code that enabled us to produce a model with 25% precision is available in `bert.py`.

This code is widely inspired from this [site](https://www.sabrepc.com/blog/Deep-Learning-and-AI/text-classification-with-bert).