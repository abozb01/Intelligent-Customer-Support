#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:08:40 2024

@author: alexbozyck
"""
import sqlite3
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import sqlite3
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect('user_feedback.db')
cursor = conn.cursor()

# Create a table to store user feedback if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue TEXT,
        feedback TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# Sample training data with multiple solutions for each problem
training_data = [
    {
        "query": "Teams not connecting",
        "solutions": [
            "Ensure that you have a stable internet connection.",
            "Try reinstalling the Teams application.",
            "Check for any ongoing server issues with Teams.",
            "Contact your IT department for further assistance.",
        ],
        "help_page": "https://support.microsoft.com/en-us/teams"
    },
    {
        "query": "Network issues",
        "solutions": [
            "Check if there are any network issues in your area.",
            "Restart your router and reconnect to the network.",
            "Update your network drivers.",
            "Contact your Internet Service Provider (ISP) for support.",
        ],
        "help_page": "https://www.techsupportforum.com/"
    },
    {
        "query": "Forgot password",
        "solutions": [
            "Reset your password by visiting the company's password reset portal.",
            "Contact your IT department for assistance with password recovery.",
            "Ensure that you are entering the correct username and follow the password recovery steps.",
            "Check your email for password reset instructions.",
        ],
        "help_page": "https://support.yourcompany.com/password-reset"
    },
    {
        "query": "Software not working",
        "solutions": [
            "Ensure that your software is up-to-date.",
            "Check for any known issues or bugs with the software.",
            "Reinstall the software and update to the latest version.",
            "Contact the software vendor or support team for assistance.",
        ],
        "help_page": "https://www.softwarevendor.com/support"
    },
    # Add more examples as needed
]

# Placeholder for user feedback
user_feedback = {}

# Function to recommend solutions based on user query
def recommend_solution(user_query, training_data, user_feedback):
    # Extract queries, solutions, and help pages from training data
    queries = [example["query"] for example in training_data]
    solutions = [", ".join(example["solutions"]) for example in training_data]
    help_pages = {example["query"].lower(): example["help_page"] for example in training_data}

    # Add user feedback to training data
    for issue, feedback in user_feedback.items():
        queries.append(issue)
        solutions.append(f"User feedback: {feedback}")

    # Convert set of stop words to a string
    stop_words = 'english'

    # Tokenize and vectorize the user query and troubleshooting data
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    vectors = vectorizer.fit_transform(solutions + [user_query])

    # Calculate cosine similarity between user query and troubleshooting data
    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    # Find the index of the most similar troubleshooting data
    most_similar_index = similarity_scores.argmax()

    # Return the recommended solution and help page URL
    recommended_issue = queries[most_similar_index]
    recommended_solution = solutions[most_similar_index]
    recommended_help_page = help_pages.get(recommended_issue.lower(), "https://www.example.com")

    return recommended_issue, recommended_solution, recommended_help_page

# Allow the user to input their issue or question
user_query = input("Please describe the issue you're experiencing: ").lower()

# Identify keywords in the user's input
keywords = set(word for word in nltk.word_tokenize(user_query) if word.isalpha())

# Find the relevant training data based on keywords
relevant_training_data = [data for data in training_data if any(keyword in data["query"].lower() for keyword in keywords)]

# Recommend solutions based on identified keywords
if relevant_training_data:
    recommended_issue, recommended_solution, recommended_help_page = recommend_solution(user_query, relevant_training_data, user_feedback)
    print(f"Recommended Issue: {recommended_issue}")
    print(f"Recommended Solutions: {recommended_solution}")
    print(f"For more assistance, please visit: {recommended_help_page}")

    # Ask for feedback after providing the initial recommendations
    feedback_for_issue = input(f"Was the recommendation helpful for the issue '{recommended_issue}'? (yes/no): ").lower()
    user_feedback[recommended_issue] = feedback_for_issue

    # If the user says "no," provide a link to additional help
    if feedback_for_issue == "no":
        print(f"\nFor additional assistance, please visit: {recommended_help_page}")

    # Save user feedback to the database
    cursor.execute("INSERT INTO user_feedback (issue, feedback) VALUES (?, ?)", (recommended_issue, feedback_for_issue))
    conn.commit()

    # Thank the user for their feedback on the identified issue
    print("Thank you for your feedback!")

    # Feedback analysis with linear regression
    # Fetch all user feedback data from the database
    query = "SELECT * FROM user_feedback WHERE feedback='no'"
    df = pd.read_sql_query(query, conn)

    # Convert feedback timestamps to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    # Extract month and year from the timestamp
    df['month_year'] = df['timestamp'].dt.to_period('M')

    # Check if there is enough data to perform train-test split
    if len(df['month_year'].unique()) > 1:
        # Count the number of "no" feedback per month
        monthly_counts = df['month_year'].value_counts().sort_index()

        # Plot the historical data
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_counts.index.astype(str), monthly_counts.values, marker='o')
        plt.title('Monthly "No" Feedback Count')
        plt.xlabel('Month-Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

        # Prepare the data for linear regression
        X = np.array([i.month for i in monthly_counts.index]).reshape(-1, 1)
        y = monthly_counts.values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions for the next month
        current_month = datetime.now().to_period('M')
        next_month = current_month + 1
        prediction = model.predict([[next_month.month]])

        # Print the prediction
        print(f"Prediction for {next_month.strftime('%Y-%m')}: {int(prediction[0])} 'No' Feedbacks")

        # Evaluate the model performance
        y_pred = model.predict(X_test)
        print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}')

    else:
        print("Not enough data to perform train-test split.")

else:
    print("Sorry, I couldn't identify a specific issue. Please provide more details.")

# Close the database connection
conn.close()

