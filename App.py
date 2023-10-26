from flask import Flask, render_template, request
from flask import Flask, request, jsonify

import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

import csv
import pandas as pd
# Define a route for handling POST requests
top_10_individuals = []

@app.route('/Boost', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input skills from the form
        user_skills_input = request.form['skills']

        user_skills = [skill.strip() for skill in user_skills_input.split(',')]

        data = []
        # Read data from a CSV file
        with open('CvDatast.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)

        # Create a Porter Stemmer and tokenize user skills while removing stopwords
        ps = PorterStemmer()
        user_skill_tokens = set()
        for user_skill in user_skills:
            user_skill_tokens.update(
                [ps.stem(word) for word in word_tokenize(user_skill) if word.lower() not in stopwords.words('english')])

        # Create a list for individuals with matching skills
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

        # Define a scoring function to rank individuals
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

        # Sort individuals based on their scores
        sorted_individuals = sorted(matching_individuals, key=score_individual, reverse=True)

        # Select the top 10 individuals
        top_10_individuals = sorted_individuals[:10]

        if top_10_individuals:
            results = []

            # Loop through matching individuals and print their details including the score
            printed_candidates = set()
            for person in top_10_individuals:
                if person['Name'] not in printed_candidates:
                    printed_candidates.add(person['Name'])
                    results.append(
                        {"Rank": len(results) + 1, "CVNumber": person['CVNumber'], "Name": person['Name'],
                         "Contact": person.get('Contact', ''), "Total_Score": person['Total_Score']})

            return render_template('Boost.html', results=results)
        else:
            return render_template('Boost.html', message="No individuals found with the specified skills")

    return render_template('Boost.html')

def get_unique():
    start_role = set()

    with open('SKILL_SETS_V2/job_roles.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            start_role.add(row[0])

    return sorted(list(start_role))

@app.route('/job_description', methods=['GET', 'POST'])
def job_description():
    if request.method == 'POST':
        Job_Role = request.form['Job_Role']
        import joblib

        # Load the  model
        classifier = joblib.load("SKILL_SETS_V2/job_role_classifier.pkl")
        vectorizer = joblib.load("SKILL_SETS_V2/vectorizer.pkl")
        label_encoder = joblib.load("SKILL_SETS_V2/label_encoder.pkl")
        user_description = Job_Role

        # TF-IDF
        user_description_vector = vectorizer.transform([user_description])
        predicted_label = classifier.predict(user_description_vector)[0]
        predicted_job_role = label_encoder.inverse_transform([predicted_label])[0]

        print("Predicted Job Role:", predicted_job_role)

        return render_template('job_description.html', results=predicted_job_role)
    start_role = get_unique()
    return render_template('job_description.html',start_role=start_role)


if __name__ == '__main__':
    app.run(debug=True)
