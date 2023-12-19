import pandas as pd
from apyori import apriori
from tqdm import tqdm

# Load data
data_learning = pd.read_csv("./dataset/lesson_learning_records.transformed.csv")
data_lesson = pd.read_csv("./dataset/lessons.preprocessed.csv")

# Get lesson IDs
lesson_ids = data_lesson["Lesson ID"].tolist()

# Create transaction list
transactions = []
for i in tqdm(range(len(data_learning))):
    transaction = []
    for j in range(len(lesson_ids)):
        if data_learning.loc[i, "Lesson ID"] == lesson_ids[j]:
            transaction.append(lesson_ids[j])
    transactions.append(transaction)

# Total transactions
total_transactions = len(transactions)

# Progress bar setup
from progressbar import ProgressBar, Percentage

# Create progress bar
pbar = ProgressBar(maxval=total_transactions, progress_char="▓", empty_char="░")

# Run Apriori algorithm (Parameterization with lift> 1.5, confidence> 0.2 and support> 0.01)
rules = apriori(transactions, min_support=0.01, min_confidence=0.2, min_lift=1.5)

# Get results
results = []
for rule in rules:
    lesson_a = rule.items[0][0]
    lesson_b = rule.items[1][0]
    support = rule.support
    confidence = rule.confidence
    lift = rule.lift
    results.append((lesson_a, lesson_b, support, confidence, lift))

    # Update progress bar
    pbar.update()

# Write results to CSV file
with open("./output/association_rules.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Lesson A", "Lesson B", "Support", "Confidence", "Lift"])
    for result in results:
        writer.writerow(result)

# Close progress bar
pbar.close()