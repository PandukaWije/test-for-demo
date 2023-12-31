import csv
import nltk
import joblib
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load models
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def classifier_model(vectorizer, label_encoder):
    data = pd.read_csv("job_roles.csv")

    X = vectorizer.fit_transform(data["Job Role Description"])
    y = label_encoder.fit_transform(data["Required Skills"])

    # Train a Random Forest classifier
    classifier = RandomForestClassifier()
    classifier.fit(X, y)
    
    return classifier

st.title("Job Description to Skills")

# start_role = get_unique()
start_role = {'IT Procurement Specialist', 'Big Data Engineer', 'IT Change Manager', 'CRM Administrator', 'SharePoint Administrator', 'IT Project Manager', 'IT Operations Manager', 'Infrastructure Engineer', 'Software Architect', 'Cloud Engineer', 'Software Development Manager', 'Back-End Developer', 'IT Trainer', 'Full Stack Developer', 'Salesforce Developer', 'WordPress Developer', 'Cybersecurity Analyst', 'DevOps Engineer', 'IT Compliance Manager', 'Graphic Designer', 'Cloud Security Engineer', 'Embedded Systems Engineer', 'SAP Consultant', 'Kotlin Developer', 'QA Engineer', 'Data Scientist', 'IT Auditor', 'UX Researcher', 'HR Manager', 'Natural Language Processing (NLP) Engineer', 'IT Service Delivery Manager', 'Systems Administrator', 'SharePoint Developer', 'SAP Basis Administrator', 'Flutter Developer', 'System Integration Engineer', 'Machine Learning Engineer', 'Ruby on Rails Developer', 'E-commerce Developer', 'ERP Developer', 'Project Manager', 'Marketing Specialist', 'Chatbot Developer', 'IT Business Development Manager', 'Network Engineer', 'Cloud Solutions Architect', 'Software Engineer', 'IT Support Specialist', 'CRM Developer', 'IT Asset Manager', 'Data Engineer', 'UI/UX Designer', 'Game Developer', 'Mobile App Developer', 'AI Engineer', 'Cloud Architect', 'Automation Engineer', 'Security Engineer', 'Robotic Process Automation (RPA) Developer', 'Compiler Engineer', 'Full Stack JavaScript Developer', 'Android Developer', 'iOS Developer', 'Unity Developer', 'IT Business Analyst', 'Blockchain Developer', 'IT Risk Manager', 'Business Intelligence Developer', 'AR/VR Developer', 'Front-End Developer', 'Site Reliability Engineer (SRE)', 'Backend API Engineer', 'Robotics Engineer', 'Business Systems Analyst', 'Quality Assurance Manager', 'IT Consultant', 'IT Sales Representative', 'Digital Marketing Analyst', 'IT Security Analyst'}

option = st.selectbox('Select Job Role', start_role)
predicted_job_role = []
if st.button('Get Skills'):
    # Vectorize user input
    user_description_vector = vectorizer.transform([option]) 

    # Predict label 
    classifier = classifier_model(vectorizer, label_encoder)
    predicted_label = classifier.predict(user_description_vector)[0]

    # Decode label
    predicted_job_role = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Predicted Job Role: {predicted_job_role}")
    
st.title("Boost Candidates")

skills = predicted_job_role

if skills:
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
                results.append({"Rank": len(results) + 1, 
                "Name": person['Name'],
                "Contact": person.get('Contact', ''),
                "Score": round(person['Total_Score'], 2)})

        results_df = pd.DataFrame(results)
        
        st.table(results_df.style.format({'Score': '{:.2f}'})
          .set_precision(2))
    else:
        st.write("No matching candidates found")
