# Intelligent Customer Support

A sophisticated customer support system powered by machine learning and natural language processing. This repository includes a recommender system that provides tailored troubleshooting solutions based on user queries. It adapts and improves recommendations over time by analyzing user feedback and predicting trends using linear regression.

## Overview

Makes customer support system intelligent and user-friendly.

## Key Features

### Recommender System

- **Tailored Solutions**: The system recommends personalized troubleshooting solutions based on user queries.
- **Adaptive Learning**: Improves recommendations over time by analyzing user feedback.
- **Linear Regression Analysis**: Utilizes linear regression to predict future trends and adapt its responses.

## Screenshots

Include screenshots of the system's output, emphasizing key features.

### Recommendation Output
![Recommendation Output](/images/recommendation_output.png)

### Monthly Feedback Count Graph
![Monthly Feedback Count Graph](/images/monthly_feedback_graph_sample.png)

## Code Highlights

### Incorporating User Feedback

```python
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

# Print the recommended solution and help page
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
```

### Getting Started
Follow these steps to set up and run the Intelligent Customer Support system on your local machine.

Prerequisites
Python (version >= 3.6)
Pip (Python package installer)
SQLite (for data storage)


### Installation
Clone the repository to your local machine.
```
git clone https://github.com/your-username/Intelligent-Customer-Support.git
```
Navigate to the project directory.
```
cd Intelligent-Customer-Support
```
### Install the required Python packages.
```
pip install -r requirements.txt
```
### Usage
Run the main script to start the Intelligent Customer Support system.



