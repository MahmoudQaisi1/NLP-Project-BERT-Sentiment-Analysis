from transformers import pipeline
import csv
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix,ndcg_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np

pipe = pipeline("text-classification", model="wesleyacheng/movie-review-sentiment-classifier-with-bert")


def read_random_rows(filename, column_numbers, num_rows):
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
        header = rows[0]  # Assuming the first row is the header

        # Remove the header from the list of rows
        rows = rows[1:]

        # Check if the number of rows requested is more than the available rows
        if num_rows > len(rows):
            print(f"Warning: Requested {num_rows} rows, but only {len(rows)} available.")
            num_rows = len(rows)

        random_rows = []

        while len(random_rows) < num_rows:
            row = random.choice(rows)
            if len(row[0]) < 512 and len(row[1]) < 512:
                random_rows.append(row)

        # Prepend the header back to the list of random rows
        random_rows.insert(0, header)
        data = []
        for row in random_rows:
            columns = [row[i] for i in column_numbers]
            data.append(columns)
        return data


# Reading The files and preparing the tesiting data.
filename1 = 'rotten_tomatoes_critic_reviews.csv'
filename2 = 'Test.csv'
random_rows1 = read_random_rows(filename1, [4, 7], 200)
random_rows1 = [list(row[::-1]) for row in random_rows1]
replacements = {'Rotten': 0, 'Fresh': 1}
random_rows1 = list(zip(*random_rows1))
random_rows1[1] = ['0' if value == 'Rotten' else '1' for value in random_rows1[1]]
random_rows1 = list(zip(*random_rows1))
random_rows1 = [list(column) for column in random_rows1]
random_rows2 = read_random_rows(filename2, [0, 1], 200)

random_rows = random_rows1 + random_rows2[1:]

random.shuffle(random_rows[1:])


for i, row in enumerate(random_rows):
    row.append('0') if pipe(row[0])[0]['label'] == 'NEGATIVE' else row.append('1')

random_rows = list(zip(*random_rows))
reviews = np.array(random_rows[0])
true_labels = np.array(random_rows[1]).astype(int)
predicted_labels = np.array(random_rows[2]).astype(int)


accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# ROC curve
fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(true_labels, predicted_labels)
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

count_zeros = np.count_nonzero(predicted_labels == 0)
count_ones = np.count_nonzero(predicted_labels == 1)
labels = ['Negative', 'Positive']
counts = [count_zeros, count_ones]
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['gray', 'blue'])
plt.title('Sentiment Distribution')
plt.show()


# Positive reviews
positive_reviews = [reviews[i] for i in range(len(reviews)) if predicted_labels[i] == 1]
positive_reviews_text = ' '.join(positive_reviews)

# Negative reviews
negative_reviews = [reviews[i] for i in range(len(reviews)) if predicted_labels[i] == 0]
negative_reviews_text = ' '.join(negative_reviews)

# Generate word cloud for positive reviews
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews_text)

# Generate word cloud for negative reviews
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews_text)

# Plot word clouds
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Reviews')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Reviews')
plt.axis('off')

plt.tight_layout()
plt.show()

'''
# Calculate recall at various positions
def calculate_recall_at_positions(true_labels, predicted_labels, positions):
    recalls = []
    for pos in positions:
        true_positives = sum((true == 1 and pred == 1) for true, pred in zip(true_labels[:pos], predicted_labels[:pos]))
        false_negatives = sum((true == 1 and pred == 0) for true, pred in zip(true_labels[:pos], predicted_labels[:pos]))
        recall = true_positives / (true_positives + false_negatives)
        recalls.append(recall)
    return recalls

positions = [1, 10, 30, 50, 80, 100]
recall_at_positions = calculate_recall_at_positions(true_labels, predicted_labels, positions)

# Calculate NDCG at various positions
def calculate_ndcg(true_labels, predicted_labels, positions):
    relevance_scores = [1 if label == 1 else 0 for label in true_labels]
    predicted_scores = [1 if label == 1 else 0 for label in predicted_labels]
    ndcg_scores = []

    for k in positions:
        ndcg = ndcg_score([relevance_scores], [predicted_scores], k=k)
        ndcg_scores.append(ndcg)

    return ndcg_scores

ndcg = calculate_ndcg(true_labels, predicted_labels, positions)

# Print the recall and NDCG values at various positions
print("Recall at Positions:", recall_at_positions)
print("NDCG:", ndcg)
import matplotlib.pyplot as plt

# Create a table plot
fig, ax = plt.subplots()

# Plot the recall values
ax.plot(positions, recall, label='Recall')

# Plot the NDCG values
ax.plot(positions, ndcg, label='NDCG')

# Set the axis labels and title
ax.set_xlabel('Positions')
ax.set_ylabel('Value')
ax.set_title('Recall and NDCG at Various Positions')

# Display a legend
ax.legend()

# Display the table plot
plt.show()
'''