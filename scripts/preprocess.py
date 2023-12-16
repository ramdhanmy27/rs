import os
import re
import pandas as pd

dataset_path = './dataset'
entities_csv = dataset_path + '/entities.csv'
lessons_csv = dataset_path + '/lessons.csv'
shelf_items_csv = dataset_path + '/shelf_items.csv'
llrs_csv = dataset_path + '/lesson_learning_records.csv'
views_csv = dataset_path + '/views.csv'

# check file existence
for csv in [entities_csv, llrs_csv, lessons_csv, shelf_items_csv, views_csv]:
  if not os.path.exists(csv):
    print('%s is not found' % csv)
    exit()

# DATA CLEANING
trim_whitespaces = lambda val: val.strip() if type(val) == str else None

# clean up entities
df = pd.read_csv(entities_csv)
removed_rows = df['Name'].str.contains('test', case=False, regex=False)
df.loc[~removed_rows].to_csv(dataset_path + '/entities.preprocessed.csv', index=False, encoding='utf-8')

df = df.loc[removed_rows]
df.to_csv(dataset_path + '/entities.removed.csv', index=False, encoding='utf-8')
removed_brands_ids = df.loc[df['Type'] == 'Brand']['ID'].array
removed_outlet_ids = df.loc[df['Type'] == 'Outlet']['ID'].array

# clean up lessons
df = pd.read_csv(lessons_csv)
df['Owner ID'] = df['Owner ID'].fillna(0.0).astype(int)
removed_rows = (
  df['Title'].isna() |
  df['ID'].isin([295, 3054, 5307, 5308, 5309, 5310, 5312, 5314, 5315, 5318, 5319, 5320, 5321, 5322, 5323, 5324, 5325, 268, 1284, 2459, 2675, 3531, 3716, 3889, 3890, 3891, 4572]) |
  (
    df['Owner Type'].isna() &
    df['Owner ID'].isna() &
    df['Title'].str.contains('test', case=False, regex=False)
  ) |
  ((df['Owner Type'] == 'Outlet') & df['Owner ID'].isin(removed_brands_ids)) |
  ((df['Owner Type'] == 'Brand') & df['Owner ID'].isin(removed_outlet_ids))
)
removed_df = df.loc[removed_rows]
removed_df.to_csv(dataset_path + '/lessons.removed.csv', index=False, encoding='utf-8')
removed_lesson_ids = removed_df['ID'].array

df = df.loc[~removed_rows]
df['Title'] = df['Title'].apply(trim_whitespaces)
df['Description'] = df['Description'].apply(trim_whitespaces)
df.to_csv(dataset_path + '/lessons.preprocessed.csv', index=False, encoding='utf-8')

# clean up shelf items
df = pd.read_csv(shelf_items_csv)
df = df.loc[df['Item Type'] == 'Lesson']
df = df.drop('Created At', axis=1)
df = df.drop('Item Type', axis=1)
df = df.loc[~df['Item ID'].isin(removed_lesson_ids)]
df = df.sort_values(['User ID', 'Item ID'])
df.to_csv(dataset_path + '/shelf_items.preprocessed.csv', index=False, encoding='utf-8')

# clean up lesson learning records
df = pd.read_csv(llrs_csv)
df = df.drop('Quiz Questions Answered', axis=1)
df = df.drop('Quiz Questions Correct', axis=1)
df = df.drop('Quiz Passed', axis=1)
df = df.loc[~df['Lesson ID'].isin(removed_lesson_ids)]
df.to_csv(dataset_path + '/lesson_learning_records.preprocessed.csv', index=False, encoding='utf-8')

# transform lesson learning records
df = df.drop('Created At', axis=1)
df = df.groupby(['User ID', 'Lesson ID']).apply(
  lambda rows: pd.Series([
    True in rows['Completed'].values,
    rows[rows['Completed'] == False].count()['Completed']
  ], index=['Completed', 'Failed Counts'])
).reset_index()
df = df.sort_values(['User ID', 'Lesson ID'])
df.to_csv(dataset_path + '/lesson_learning_records.transformed.csv', index=False, encoding='utf-8')

# clean up views
df = pd.read_csv(views_csv)
df = df.drop('Location', axis=1)
df = df.loc[~df['Lesson ID'].isin(removed_lesson_ids)]
df.to_csv(dataset_path + '/views.preprocessed.csv', index=False, encoding='utf-8')

# transform views
df = df.drop('Created At', axis=1)
df = df.groupby(['User ID', 'Lesson ID']).size().reset_index(name='Counts')
df = df.sort_values(['User ID', 'Lesson ID'])
df.to_csv(dataset_path + '/views.transformed.csv', index=False, encoding='utf-8')
