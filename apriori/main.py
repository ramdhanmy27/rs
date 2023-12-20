import pandas as pd
from apyori import apriori
from tqdm import tqdm
import csv

# Baca data lesson_learning
data_learning = pd.read_csv("./dataset/lesson_learning_records.transformed.csv")

# Baca data lesson
data_lesson = pd.read_csv("./dataset/lessons.preprocessed.csv")

# Gabungkan data lesson_learning dan data_lesson berdasarkan Lesson ID ; note pada file csv lesson.preprocessed.csv pada row pertama ('ID') dirubah menjadi ('Lesson ID)
merged_data = pd.merge(data_learning, data_lesson, on='Lesson ID')

# Filter data yang telah selesai (Completed=True)
completed_data = merged_data[merged_data['Completed'] == True]

# Ambil kolom Lesson ID sebagai string
transactions = completed_data.groupby('User ID')['Title'].apply(lambda x: list(map(str, x))).tolist()

# Apriori Algorithm
min_support = 0.01
min_confidence = 0.2
min_lift = 1.5

association_rules = apriori(
    transactions,
    min_support=min_support,
    min_confidence=min_confidence,
    min_lift=min_lift,
    min_length=2,  # Set to 2 for rules with exactly two items
    max_length=2,
)

# Simpan hasil ke CSV
output_file = "./output/association_rules_output.csv"
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Lesson A', 'Lesson B', 'Support', 'Confidence', 'Lift'])

    for rule in tqdm(association_rules):
        items = [item for item in rule.items]
        if len(items) == 2:
            support = rule.support
            confidence = rule.ordered_statistics[0].confidence if rule.ordered_statistics else None
            lift = rule.ordered_statistics[0].lift if rule.ordered_statistics else None

            csv_writer.writerow([items[0], items[1], support, confidence, lift])