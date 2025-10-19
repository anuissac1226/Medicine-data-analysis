from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import pandas as pd

#Load updated Dataset
df = pd.read_csv('data/Medicine_Details_Updated.csv')

#Preprocessing
label_encoder = {}
for i in ['Medicine Name','Manufacturer']:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])
    label_encoder[i] = le

#Save the label_encoder to a file
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
print("Label encoder saved successfully!!!")

tfidf = {}
tfidf_matrices = []
df_encoded = df.copy()
for column in ['Composition']:
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(df[column])
    df_tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_matrices.append(df_tfidf_matrix)
    tfidf[column] = vectorizer

# Save the TF-IDF vectorizer to a file
with open('tfidf.pkl', 'wb') as file:
    pickle.dump(tfidf, file)
print("TF-IDF vectorizer saved successfully!!!")

# Concatenate the TF-IDF matrices to the encoded DataFrame
df_encoded = pd.concat([df_encoded, tfidf_matrices[0]], axis=1)

#Feature and target set
X_rating = df_encoded.drop(['Rating', 'Image URL','Composition', 'Uses', 'Side_effects'], axis=1)
Y_rating = df_encoded['Rating']

#Train-Test Split
X_train_rating, X_test_rating, Y_train_rating, Y_test_rating = train_test_split(X_rating, Y_rating, test_size=0.2, random_state=100)

# Initialize scaler
scaler = StandardScaler()
X_train_scaled_rating = scaler.fit_transform(X_train_rating)
X_test_scaled_rating = scaler.transform(X_test_rating)
# Save the scaler to a file
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Scaler saved successfully!!!")

# Define models
models_rating = {
    'knn' : KNeighborsRegressor(n_neighbors=5),
    'decision_tree': DecisionTreeRegressor(max_depth=3,min_samples_split=10,min_samples_leaf=2,random_state=42),
    'random_forest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
}

for name,model in models_rating.items():
    model.fit(X_train_scaled_rating,Y_train_rating)
    Y_pred_rating=model.predict(X_test_scaled_rating)
    mse=mean_squared_error(Y_test_rating,Y_pred_rating)
    r2=r2_score(Y_test_rating,Y_pred_rating)
    print(f'{name}-mean squared error: {mse}')
    print(f'{name}-R2: {r2}')

# Save trained model
for name, model in models_rating.items():
    filename = f'{name}_model_for_rating_prediction.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"{name} saved successfully as {filename}")
