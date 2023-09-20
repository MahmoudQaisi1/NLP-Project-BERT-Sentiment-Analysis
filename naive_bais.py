import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import csv
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


tokenizer = RegexpTokenizer(r'\w+')
nltk.download('stopwords')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


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


def getStemmedReview(review):
    review = review.lower()
    review = review.replace("<br /><br />", " ")

    tokens = tokenizer.tokenize(review)
    stopwords = [token for token in tokens if token not in en_stopwords]
    new_token = [ps.stem(tokens) for tokens in stopwords]

    cleaned_review = " ".join(new_token)

    return cleaned_review


# Reading The files and preparing the tesiting data.
filename1 = 'rotten_tomatoes_critic_reviews.csv'
filename2 = 'Test.csv'
random_rows1 = read_random_rows(filename1, [4, 7],5000)
random_rows1 = [list(row[::-1]) for row in random_rows1]
replacements = {'Rotten': 0, 'Fresh': 1}
random_rows1 = list(zip(*random_rows1))
random_rows1[1] = ['0' if value == 'Rotten' else '1' for value in random_rows1[1]]
random_rows1 = list(zip(*random_rows1))
random_rows1 = [list(column) for column in random_rows1]
random_rows2 = read_random_rows(filename2, [0, 1], 5000)

random_rows = random_rows1 + random_rows2[1:]

random.shuffle(random_rows[1:])

random_rows = list(zip(*random_rows))
reviews = np.array(random_rows[0])
true_labels = np.array(random_rows[1]).astype(int)

cleaned_reviews = []
for i in reviews:
    cleaned_reviews.append(getStemmedReview(i))

cv= CountVectorizer()
vectorized_reviews = (cv.fit_transform(cleaned_reviews)).toarray()

train_rev = vectorized_reviews[:8000]
train_labels = true_labels[:8000]
test_rev = vectorized_reviews[8000:]
test_labels = true_labels[8000:]

mnb= MultinomialNB()
#Training
mnb.fit(train_rev,train_labels)
#prediction

predicted = mnb.predict(test_rev)

accuracy = accuracy_score(test_labels, predicted)
precision = precision_score(test_labels, predicted)
recall = recall_score(test_labels, predicted)
f1 = f1_score(test_labels, predicted)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

fpr, tpr, _ = roc_curve(test_labels, predicted)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(test_labels, predicted)
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

cm = confusion_matrix(test_labels, predicted)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

count_zeros = np.count_nonzero(predicted == 0)
count_ones = np.count_nonzero(predicted == 1)
labels = ['Negative', 'Positive']
counts = [count_zeros, count_ones]
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['gray', 'blue'])
plt.title('Sentiment Distribution')
plt.show()

reviews = reviews[8000:]
# Positive reviews
positive_reviews = [reviews[i] for i in range(len(reviews)) if predicted[i] == 1]
positive_reviews_text = ' '.join(positive_reviews)

# Negative reviews
negative_reviews = [reviews[i] for i in range(len(reviews)) if predicted[i] == 0]
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
