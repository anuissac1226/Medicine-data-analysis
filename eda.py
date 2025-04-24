import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

data = pd.read_csv('data/Medicine_Details.csv')
print(data.head())

# Analysing basic details of dataset
print('\n*****Dataset Overview*****\n')
print(data.info())
print('\n*****Statistical Description*****\n')
print(data.describe(include='all'))

# Checking the shape of the dataset
print('\n***** shape *****\n')
print(data.shape)

# Checking for missing values
print('\n***** missing value info *****\n')
print(data.isnull().sum())

#Checking duplicate entries
duplicate = data[data.duplicated()]
print(duplicate)
data_cleaned = data.drop_duplicates().copy()
print('shape of cleaned dataset: ',data_cleaned.shape)
print('Duplicates in cleaned dataset: ',data_cleaned.duplicated().sum())

#Create a new column 'Rating' based on review category
data_cleaned['Rating'] = ((5*data_cleaned['Excellent Review %'] + 3*data_cleaned['Average Review %'] + 1*data_cleaned['Poor Review %'])/100)
#Saving updated dataset
data_cleaned.to_csv('data/Medicine_Details_Updated.csv', index=False)
#Loading updated data set
df = pd.read_csv('data/Medicine_Details_Updated.csv')

#Top 10 Manufactures by number of products
top10_manufacture = df['Manufacturer'].value_counts().head(10)
plt.figure(figsize=(8,6))
sns.barplot(x=top10_manufacture.values,y=top10_manufacture.index,palette='viridis')
plt.title('Top 10 Manufactures')
plt.xlabel('Medicine Count')
plt.ylabel('Manufacture')
plt.show()

#Top 10 compositions
top10_compositions = df['Composition'].value_counts().head(10)
plt.figure(figsize=(8,6))
sns.barplot(x=top10_compositions.values,y=top10_compositions.index,palette='viridis')
plt.title('Top 10 Compositions')
plt.xlabel('Medicine Count')
plt.ylabel('Composition')
plt.show()

#Word cloud for Composition
wordcloud_composition = ' '.join(df['Composition'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_composition)
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Composition')
plt.show()

#Most commonly used(top 10) ingredients in medicine composition
ingredients = []
for i in df['Composition']:
    ingredient = i.split(' + ')
    clean_ingredient = [re.sub(r'\s*\([^)]*\)', '', x).strip() for x in ingredient]
    ingredients.extend(clean_ingredient)
top10_ingredient = pd.Series(ingredients).value_counts().head(10)

plt.figure(figsize=(8,6))
sns.barplot(x=top10_ingredient.values,y=top10_ingredient.index,palette='viridis')
plt.title('Top 10 Ingredients in Medicine Composition')
plt.xlabel('Medicine Count')
plt.ylabel('Ingredients')
plt.show()

#Top 10(Most common) side effects
df['Side_effects_splitted'] = df['Side_effects'].apply(lambda x: re.findall(r'[A-Z][^A-Z]*', x))
Side_effects_flattened = df['Side_effects_splitted'].explode().str.strip()
top10_side_effects = Side_effects_flattened.value_counts().head(10)

plt.figure(figsize=(8,6))
sns.barplot(x=top10_side_effects.values,y=top10_side_effects.index,palette='viridis')
plt.title('Top 10(Most Common) Side Effects')
plt.xlabel('Medicine Count')
plt.ylabel('Side effects')
plt.show()

#Word cloud for Side effects
wordcloud_uses = ' '.join(df['Side_effects'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_uses)
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Side effects')
plt.show()

#Top 10 Uses
common_phrases = ['Treatment of', 'Prevention of']
def clean_uses(uses):
    for i in common_phrases:
        uses = uses.replace(i, '')
    return uses

df['Uses_cleaned'] = df['Uses'].apply(clean_uses)
df['Uses_splitted'] = df['Uses_cleaned'].apply(lambda x: re.findall(r'[A-Za-z0-9\s]+(?:[A-Z][a-z]+|\([^\)]*\)|[a-z]+|[A-Z]+[a-z]+)', x))
Uses_flattened = df['Uses_splitted'].explode().str.strip()
top10_uses = Uses_flattened.value_counts().head(10)

plt.figure(figsize=(8,6))
sns.barplot(x=top10_uses.values,y=top10_uses.index,palette='viridis')
plt.title('Top 10 Uses')
plt.xlabel('Medcine Count')
plt.ylabel('Uses')
plt.show()

#Word cloud for Uses
wordcloud_uses = ' '.join(df['Uses'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_uses)
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Uses')
plt.show()

#Top 10 medicines based on Rating
sorted_df = df.sort_values(by='Rating', ascending=False)
top10_medicine = sorted_df[['Medicine Name', 'Rating']].head(10)

plt.figure(figsize=(8, 6))
sns.barplot(data=top10_medicine,x='Rating',y='Medicine Name',palette='viridis')
plt.title('Top 10 Medicines Based on Rating')
plt.xlabel('Rating')
plt.ylabel('Medicine Name')
plt.tight_layout()
plt.show()

#Number of side effects vs Rating for a medicine
df['Side_effects_splitted'] = df['Side_effects'].apply(lambda x: re.findall(r'[A-Z][^A-Z]*', x))
df['Num_side_effects'] = df['Side_effects_splitted'].apply(len)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x='Rating',y='Num_side_effects',palette='viridis')
plt.title('Number of side effects vs Rating')
plt.xlabel('Rating')
plt.ylabel('Number of side effects')
plt.show()

#Top 10 Medicine with Rating 5 and Minimal Side Effects
filtered_df = df[(df['Num_side_effects']<2) & (df['Rating']==5)].head(10)
sns.barplot(data=filtered_df,x='Num_side_effects',y='Medicine Name',palette='viridis')
plt.title('Top Medicine with Rating 5 and Minimal Side Effects')
plt.xlabel('Number of Side Effects')
plt.ylabel('Medicine Name')
plt.show()

#Top 10 compositions and their average rating
top10_compositions = df['Composition'].value_counts().head(10)
composition_rating_data = df[df['Composition'].isin(top10_compositions.index)]
composition_rating = composition_rating_data.groupby('Composition')['Rating'].mean()
sorted_composition_rating = composition_rating.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_composition_rating.values, y=sorted_composition_rating.index, palette='viridis')
plt.title('Top 10 Compositions and Their Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Composition')
plt.tight_layout()
plt.show()

#Top 10 manufactures(based on number of products) and their average ratings
top10_manufacture = df['Manufacturer'].value_counts().head(10).index
top10_manufacture_data = df[df['Manufacturer'].isin(top10_manufacture)]

manufacturer_ratings = top10_manufacture_data.groupby('Manufacturer')['Rating'].mean()
sorted_manufacturer_ratings = manufacturer_ratings.sort_values(ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=sorted_manufacturer_ratings.values, y=sorted_manufacturer_ratings.index, palette='viridis')
plt.title('Top 10 Manufacturers(by Number of Products) and Their Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Manufacturer')
plt.show()

#Review category of top 10 manufactures(manufactures with highest number of products)
top10_manufacturers = df['Manufacturer'].value_counts().head(10).index
top_df = df[df['Manufacturer'].isin(top10_manufacturers)]
plot_data = top_df[['Manufacturer', 'Excellent Review %', 'Average Review %', 'Poor Review %']]
melted_data = plot_data.melt(id_vars='Manufacturer',var_name='Review Category',value_name='Percentage')

plt.figure(figsize=(10, 8))
sns.barplot(data=melted_data,x='Manufacturer',y='Percentage',hue='Review Category',palette='viridis')
plt.title('Review Categories Across Top 10 Manufacturers')
plt.xlabel('Manufacturer')
plt.ylabel('Review Percentage')
plt.xticks(rotation=90)
plt.legend(title='Review Category', bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.show()

#Heatmap to visualize correlation matrix
corr_matrix = df[['Excellent Review %','Average Review %','Poor Review %','Rating']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix,annot=True,cmap='viridis',cbar_kws={'label':'correlation coefficient'})
plt.title('Correlation Heatmap')
plt.yticks(rotation=0)
plt.show()


