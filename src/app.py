import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_excel("C:\\Users\\cdrap\\Documents\\Combined Social Media Data.xlsx")
#print(df)

# Identify columns with missing values
#print(df.isnull().sum())

# Convert specified columns to numeric, coercing errors to NaN
numerical_cols_to_convert = ['Duration_(sec)', 'Impressions', 'Reach', 'Plays', 'Saves', 'Likes', 'Shares', 'Comments']
for col in numerical_cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Set 'Duration (sec)' to 0 for rows where 'Platform' is 'IG image'
df.loc[df["Platform"] == "IG image", "Duration_(sec)"] = 0
df.loc[df["Platform"] == "IG image", "Plays"] = 0

# Fill missing numerical values with the mean
numerical_cols_with_missing = ['Duration_(sec)', 'Impressions', 'Reach', 'Plays', 'Saves', 'Likes', 'Shares', 'Comments']
for col in numerical_cols_with_missing:
    df[col] = df[col].fillna(df[col].mean())


# Fill missing 'Description' values with an empty string
df['Description'] = df['Description'].fillna("")

# Fill missing 'Genres' and 'Content_type' with a placeholder
df['Genres'] = df['Genres'].fillna('Unknown')
df['Content_type'] = df['Content_type'].fillna('Unknown')


# Convert 'Publish_Date' and 'Time' to datetime objects
df["Publish_Date"] = pd.to_datetime(df["Publish_Date"].astype(str), errors='coerce')
df["Time"] = pd.to_datetime(df["Time"].astype(str), errors='coerce')
df["Time"] = df["Time"].fillna(df["Time"].mean())
df['Year'] = df['Publish_Date'].dt.year
df['Month'] = df['Publish_Date'].dt.month
df['Day_of_Week'] = df['Publish_Date'].dt.dayofweek # Monday=0, Sunday=6
df['Hour'] = pd.to_datetime(df['Time'].astype(str)).dt.hour
#df['Publish_Date'] = datetime.datetime.strptime(df['Publish_Date'], "%Y-%m-%d %H:%M:%S")
#print(df['Publish_Date'].hour)

#print(df['Hour'])
#print(df["Day_of_Week"])

# Print the dtypes of the numerical columns after conversion
#print(df[numerical_cols_to_convert].dtypes)

#print(df.isnull().sum())

from sklearn import preprocessing
#categorical_features = ["Platform", "Post_type", "Genres", "Content_type", "Hour", "Month", "Day_of_Week"] # with genre, content
#categorical_features = ["Platform", "Post_type", "Hour","Month", "Day_of_Week"] # without genre, content
categorical_features = ["Post_type", "Publish_Date", "Time"]
encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
one_hot_features = encoder.fit_transform(df[categorical_features])
one_hot_names = encoder.get_feature_names_out()
#print("Type of one_hot_columns is:",type(one_hot_features))
one_hot_df = pd.DataFrame.sparse.from_spmatrix(one_hot_features)
one_hot_df.columns = one_hot_names # Now we can see the actual meaning of the one-hot feature in the DataFrame
#one_hot_df.head()
#print(df.isnull().sum())

import re
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import numpy as np
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

#df['clean_caption'] = df['Description'].apply(clean_text)

# Apply TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(df['Description'])

#numerical_feature_names = ["Reach", "Plays", "Shares", "Saves", "Comments", "Duration_(sec)", "Hour", "Month", "Day_of_Week"] # Make Year, Month, and day categorial
#numerical_feature_names = ['Duration_(sec)', 'Plays', 'Comments', 'Saves', 'Day_of_Week']
#numerical_feature_names = ['Duration_(sec)', 'Plays', 'Comments', 'Saves']
numerical_feature_names = ['Plays', 'Comments', 'Saves']
#numerical_feature_names = ['Duration_(sec)', 'Shares', 'Plays', 'Comments', 'Saves', 'Hour', 'Month', 'Day_of_Week']
map_numerical_feature_names = ["Likes", "Reach", "Plays", "Shares", "Saves", "Comments", "Duration_(sec)", "Hour", "Month", "Day_of_Week"] # for correlation map
#numerical_feature_names = ["Duration (sec)", "Reach","Plays"] # Make Year, Month, and day categorial
#numerical_feature_names = ["Duration (sec)"]
#numerical_feature_names = ["Duration (sec)", "Reach", "Plays", "Saves"] # Comments has the best accuracy so far
#numerical_feature_names = ["Reach", "Plays", "Comments", "Duration (sec)"]
numerical_features = df[numerical_feature_names]
map_numerical_features = df[map_numerical_feature_names]
# Print dtypes to identify the problematic column
#print(numerical_features.dtypes)

features = scipy.sparse.hstack((numerical_features, one_hot_features),format='csr')
#features_without_numerical = scipy.sparse.csr_matrix(one_hot_features)
features_with_text = scipy.sparse.hstack((features, tfidf_matrix),format='csr')
#features_with_text_without_numerical = scipy.sparse.hstack((features_without_numerical, tfidf_matrix),format='csr')
text_without_categoraial = scipy.sparse.hstack((numerical_features, tfidf_matrix),format='csr')
#print(feature_with_text)
#all_feature_names = np.hstack((numerical_feature_names,one_hot_names))
#all_feature_names = np.hstack((numerical_feature_names, tfidf_matrix))
#print(all_feature_names
#target_column = ['Likes']
#target = df[target_column].values
target = df['Likes']
#print(target)

# Perform train and test split of data
rand_seed = 52 # For other models we will use the same random seed, so that we're always using the same train-test split
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=rand_seed) # 80 / 20 split

features_train_with_text, features_test_with_text, target_train, target_test = train_test_split(
    features_with_text, target, test_size=0.2, random_state=rand_seed)

#features_train_with_text_without_numerical, features_test_with_text_without_numerical, target_train, target_test = train_test_split(
    #features_with_text_without_numerical, target, test_size=0.2, random_state=rand_seed)

text_train, text_test, target_train, target_test = train_test_split(
    text_without_categoraial, target, test_size=0.2, random_state=rand_seed)

# Initializing the Random Forest Regression model with 100 decision trees
model_no_text = RandomForestRegressor(n_estimators = 100, random_state = 12)
model_with_text = RandomForestRegressor(n_estimators = 100, random_state = 12)
#model_with_text_without_categorial = RandomForestRegressor(n_estimators = 100, random_state = 12)
#model_with_text_without_numerical = RandomForestRegressor(n_estimators = 100, random_state = 12)

#target_train = target_train.ravel()
# Fitting the Random Forest Regression model to the data
model_no_text.fit(features_train, target_train)
model_with_text.fit(features_train_with_text, target_train)
#model_with_text_without_categorial.fit(text_train, target_train)
#model_with_text_without_numerical.fit(features_train_with_text_without_numerical, target_train)

test_score_no_text = model_no_text.score(features_test,target_test)
test_score_with_text = model_with_text.score(features_test_with_text,target_test)
#test_score_with_text_without_categorial = model_with_text_without_categorial.score(text_test,target_test)
#test_score_with_text_without_numerical = model_with_text_without_numerical.score(features_test_with_text_without_numerical,target_test)
#print("Test score for Regression WITHOUT text features:", test_score_no_text)
#print("Test score for Regression WITH text features:", test_score_with_text)

#numerical_feature_names = ["Impressions", "Reach", "Shares", "Plays", "Comments", "Saves", "Year", "Month", "Day_of_Week", "Hour"] # Make Year, Month, and day categorial
#Plays_numerical_feature_names = ['Duration_(sec)', 'Likes', 'Comments', 'Saves', 'Day_of_Week'] # Make Year, Month, and day categorial
#Plays_numerical_feature_names = ['Duration_(sec)', 'Likes', 'Comments', 'Saves']
Plays_numerical_feature_names = ['Likes', 'Comments', 'Saves', 'Shares']
Plays_numerical_features = df[Plays_numerical_feature_names]

# Print dtypes to identify the problematic column
#print(Plays_numerical_features.dtypes)

Plays_features = scipy.sparse.hstack((Plays_numerical_features, one_hot_features),format='csr')
Plays_features_without_numerical = scipy.sparse.csr_matrix(one_hot_features)
Plays_features_with_text = scipy.sparse.hstack((Plays_features, tfidf_matrix),format='csr')
#Plays_features_with_text_without_numerical = scipy.sparse.hstack((features_without_numerical, tfidf_matrix),format='csr')
#print(feature_with_text)
Plays_all_feature_names = np.hstack((Plays_numerical_feature_names,one_hot_names))
#target_column = ['Likes']
#target = df[target_column].values
target_Plays = df['Plays']
#print(target_Plays)

# Perform train and test split of data
rand_seed = 52 # For other models we will use the same random seed, so that we're always using the same train-test split
Plays_features_train, Plays_features_test, Plays_target_train, Plays_target_test = train_test_split(
    Plays_features, target_Plays, test_size=0.2, random_state=rand_seed) # 80 / 20 split

Plays_features_train_with_text, Plays_features_test_with_text, Plays_target_train, Plays_target_test = train_test_split(
    Plays_features_with_text, target_Plays, test_size=0.2, random_state=rand_seed)

#features_train_with_text_without_numerical, features_test_with_text_without_numerical, target_train, target_test = train_test_split(
    #features_with_text_without_numerical, target, test_size=0.2, random_state=rand_seed)

from sklearn import linear_model
Plays_ridge_fit_without_text = linear_model.RidgeCV(cv= 5)
Plays_ridge_fit_without_text.fit(Plays_features_train, Plays_target_train)
Plays_ridge_fit_with_text = linear_model.RidgeCV(cv= 5)
Plays_ridge_fit_with_text.fit(Plays_features_train_with_text, Plays_target_train)
#ridge_fit_with_text_without_numerical = linear_model.RidgeCV(cv=5)
#ridge_fit_with_text_without_numerical.fit(features_train_with_text_without_numerical, target_train)

Plays_ridge_test_score_no_text = Plays_ridge_fit_without_text.score(Plays_features_test, Plays_target_test)
Plays_ridge_test_score_with_text = Plays_ridge_fit_with_text.score(Plays_features_test_with_text, Plays_target_test)

import scipy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def predict_metrics(model1, model2, encoder, tfidf, df, post_type, publish_date, time, description):
    # Define the lists of feature names used during training and the ones to collect from the user
    likes_all_numerical_feature_names = ['Plays','Comments', 'Saves']
    plays_all_numerical_feature_names = ['Likes','Comments', 'Saves', 'Shares']
    #numerical_feature_names_to_collect = ["Duration_(sec)" ] # Only collect Duration (sec) from user
    categorical_feature_names_for_prediction = ["Post_type", "Publish_Date", "Time"] # without genre and content type

    user_input = {
        'Post_type': post_type,
        'Publish_Date': publish_date,
        'Time': time,
        'Description': description
    }


    # Create a DataFrame from user input, including all numerical features
    user_df = pd.DataFrame([user_input])

    # Calculate conditional means for numerical features based on 'Post_type'
    post_type = user_input.get('Post_type', 'Unknown')
    df_filtered_by_post_type = df[df['Post_type'] == post_type]

    likes_numerical_means = df_filtered_by_post_type[likes_all_numerical_feature_names].mean().to_dict()
    plays_numerical_means = df_filtered_by_post_type[plays_all_numerical_feature_names].mean().to_dict()


    # Fill in means for numerical features not collected from the user for Likes prediction
    for feature in likes_all_numerical_feature_names:
        if feature not in user_df.columns:
            user_df[feature] = likes_numerical_means.get(feature, df[feature].mean()) # Use overall mean as fallback

    # Process categorical features using the fitted encoder
    user_categorical_features = encoder.transform(user_df[categorical_feature_names_for_prediction])

    # Process text feature using the fitted TF-IDF vectorizer
    user_tfidf_matrix = tfidf.transform(user_df['Description'])

    # Combine all features for Likes prediction
    # Ensure the order of numerical features matches the training data
    user_numerical_features_likes_sparse = scipy.sparse.csr_matrix(user_df[likes_all_numerical_feature_names].values)
    user_features_likes = scipy.sparse.hstack((user_numerical_features_likes_sparse, user_categorical_features), format='csr')
    user_features_with_text_likes = scipy.sparse.hstack((user_features_likes, user_tfidf_matrix), format='csr')

    # Predict Likes
    predicted_likes = model1.predict(user_features_with_text_likes)


    # Predict Plays
    if user_input["Post_type"] == "IG_image":
        predicted_plays = None
    elif user_input["Post_type"] == "IG_carousel":
        predicted_plays = None
    else:
        # For Plays prediction, use Plays specific numerical features
        # Ensure all required numerical columns for Plays prediction are in user_df, fill with means if not provided by user
        for feature in plays_all_numerical_feature_names:
            if feature not in user_df.columns:
                 user_df[feature] = plays_numerical_means.get(feature, df[feature].mean()) # Use overall mean as fallback

        user_numerical_features_plays_sparse = scipy.sparse.csr_matrix(user_df[plays_all_numerical_feature_names].values)
        user_features_plays = scipy.sparse.hstack((user_numerical_features_plays_sparse, user_categorical_features), format='csr')
        user_features_with_text_plays = scipy.sparse.hstack((user_features_plays, user_tfidf_matrix), format='csr')
        predicted_plays = model2.predict(user_features_with_text_plays)

    # Find similar descriptions
    user_description_tfidf = tfidf.transform([user_input['Description']])
    similarity_scores = cosine_similarity(user_description_tfidf, tfidf_matrix)
    similar_indices = similarity_scores.argsort()[0][::-1][1:6] # Get top 5 most similar (excluding the input itself)
    similar_posts = df.iloc[similar_indices][['Description', 'Permalink', 'Post_type', 'Likes', 'Plays']] # Display relevant columns

    return predicted_likes[0], predicted_plays[0] if predicted_plays is not None else None, similar_posts # Return both predictions and similar posts


# estimated_likes, estimated_plays, similar_posts = predict_metrics(
#     #xgb_model_with_text,
#     model_with_text,
#     #Plays_xgb_model_with_text,
#     Plays_ridge_fit_with_text,
#     encoder,
#     tfidf,
#     df
# )
# print(f"\nEstimated Likes: {estimated_likes:.2f}")
# print(f"\nRounded Likes:  {round(estimated_likes):.2f}")
# if estimated_plays is not None:
#     print(f"\nEstimated Plays: {estimated_plays:.2f}")
#     print(f"\nRounded Plays:  {round(estimated_plays):.2f}")
# else:
#     print("\nEstimated Plays: None")

# print("\n--- Posts with Similar Descriptions ---")
# display(similar_posts)
# print("-" * 30)

#pip install dash
#pip install dash-bootstrap-components

#!pip freeze > requirements.txt
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc # Using Bootstrap components for better styling

# Initialize the Dash app

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the app layout
app.layout = dbc.Container([
    html.H1("Social Media Post Performance Predictor", className="text-center my-4"),

    dbc.Row([
        dbc.Col(dbc.Label("Post Type:"), width=3),
        dbc.Col(dcc.Dropdown(
            id='post-type-dropdown',
            options=[
                {'label': 'IG Reel', 'value': 'IG_reel'},
                {'label': 'IG Image', 'value': 'IG_image'},
                {'label': 'IG Carousel', 'value': 'IG_carousel'},
                {'label': 'TikTok Video', 'value': 'TikTok_video'},
                {'label': 'YT Video', 'value': 'YT_video'}
            ],
            value='IG_reel', # default value
            clearable=False
        ), width=9),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Label("Publish Date (MM/DD/YY):"), width=3),
        dbc.Col(dbc.Input(id='publish-date-input', type='text', placeholder='MM/DD/YY'), width=9),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Label("Time (HH:MM:SS):"), width=3),
        dbc.Col(dbc.Input(id='time-input', type='text', placeholder='HH:MM:SS'), width=9),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Label("Caption / Description:"), width=3),
        dbc.Col(dbc.Textarea(id='description-input', placeholder='Enter post caption/description', rows=4), width=9),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Button("Predict", id="predict-button", className="me-2", n_clicks=0), width={"size": 6, "offset": 3}),
    ], className="mb-3 text-center"),

    dbc.Row([
        dbc.Col(dbc.Card(
            dbc.CardBody([
                html.H4("Prediction Results", className="card-title"),
                html.Div(id='prediction-output')
            ])
        ), width=12)
    ])
])

# Define the callback function
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('post-type-dropdown', 'value'),
     State('publish-date-input', 'value'),
     State('time-input', 'value'),
     State('description-input', 'value')]
)
def update_output(n_clicks, post_type, publish_date, time, description):
    if n_clicks > 0:
        try:
            estimated_likes, estimated_plays, similar_posts = predict_metrics(
                model_with_text,
                Plays_ridge_fit_with_text,
                encoder,
                tfidf,
                df,  # Pass the 'df' DataFrame
                post_type,
                publish_date,
                time,
                description
            )

            output_text = [
                html.P(f"Estimated Likes: {estimated_likes:.2f}"),
                html.P(f"Rounded Likes: {round(estimated_likes):.2f}"),
                html.Br()
            ]

            if estimated_plays is not None:
                 output_text.extend([
                     html.P(f"Estimated Plays: {estimated_plays:.2f}"),
                     html.P(f"Rounded Plays: {round(estimated_plays):.2f}"),
                     html.Br()
                 ])
            else:
                 output_text.append(html.P("Estimated Plays: N/A for this post type"))

            output_text.append(html.P("--- Posts with Similar Descriptions ---"))
            similar_posts_table = dbc.Table.from_dataframe(similar_posts, striped=True, bordered=True, hover=True)
            output_text.append(similar_posts_table)

            return output_text

        except Exception as e:
            return html.P(f"Error: {e}", style={'color': 'red'})
    return "Enter post details and click Predict" # Initial message

#server = app.server

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

#app.run_server(port = 8050)       
