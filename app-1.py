import streamlit as st
import joblib
import csv
import pandas as pd
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load models
classifier = joblib.load("job_role_classifier.pkl", backward_compatibility=True) 
vectorizer = joblib.load("vectorizer.pkl", backward_compatibility=True)
label_encoder = joblib.load("label_encoder.pkl", backward_compatibility=True)

def get_unique():
    start_role = set()

    with open('job_roles.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            start_role.add(row[0])

    return sorted(list(start_role))

st.title("Job Description to Skills")

start_role = get_unique()

option = st.selectbox('Select Job Role', start_role)

if st.button('Get Skills'):
    # Vectorize user input
    user_description_vector = vectorizer.transform([option]) 

    # Predict label 
    predicted_label = classifier.predict(user_description_vector)[0]

    # Decode label
    predicted_job_role = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Predicted Job Role: {predicted_job_role}")
    
st.title("Boost Candidates")

skills = st.text_input("Enter skills separated by comma")

if skills and st.button("Get Candidates"):

    user_skills = [skill.strip() for skill in skills.split(',')]
    
    ps = PorterStemmer()
    user_skill_tokens = set()
    for user_skill in user_skills:
        user_skill_tokens.update(
            [ps.stem(word) for word in word_tokenize(user_skill) if word.lower() not in stopwords.words('english')])

    data = []
    with open('CvDatast.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)

    matching_individuals = []
    for person in data:
        person_skills = person['Technical_Skills'].split(', ')
        person_skill_tokens = set()
        for person_skill in person_skills:
           person_skill_tokens.update([ps.stem(word) for word in word_tokenize(person_skill) if  
                                        word.lower() not in stopwords.words('english')])
        matching_tokens = user_skill_tokens.intersection(person_skill_tokens)
        if matching_tokens:
            person['Matching_Tokens'] = matching_tokens
            matching_individuals.append(person)

    def score_individual(individual):
        education_weight = {'Bachelor\'s': 1, 'Master\'s': 2, 'Doctoral': 3}
        skill_weight = {'Python': 1, 'Java': 2, 'SQL': 3, 'Machine Learning': 4}
        experience_weight = 0.5
        education_score = education_weight.get(individual['Edu_Qualifications'], 0)
        skill_score = skill_weight.get(individual['Technical_Skills'], 0) 
        experience_score = float(individual['Experience_years']) * experience_weight
        matching_tokens_score = len(individual.get('Matching_Tokens', []))
        total_score = education_score + skill_score + experience_score + matching_tokens_score
        individual['Total_Score'] = total_score
        return total_score

    sorted_individuals = sorted(matching_individuals, key=score_individual, reverse=True)

    top_10_individuals = sorted_individuals[:10]

    if top_10_individuals:
        results = []
        printed_candidates = set()
        for person in top_10_individuals:
            if person['Name'] not in printed_candidates:
                printed_candidates.add(person['Name'])
                results.append({"Rank": len(results) + 1, "CVNumber": person['CVNumber'], "Name": person['Name'],
                                "Contact": person.get('Contact', ''), "Total_Score": person['Total_Score']})

        for result in results:
            st.write(f"""
            Rank: {result['Rank']}
            Name: {result['Name']}
            Contact: {result['Contact']}
            Score: {result['Total_Score']}
            """)
    else:
        st.write("No matching candidates found")
