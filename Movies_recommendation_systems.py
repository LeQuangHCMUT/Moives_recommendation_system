import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data_movies = pd.read_csv(r"C:\Users\Admin\PycharmProjects\pythonProject\Mechine Learning\ML_Learn\movie_data\movies.csv", sep='\t', encoding='latin1')
data_rates = pd.read_csv(r"C:\Users\Admin\PycharmProjects\pythonProject\Mechine Learning\ML_Learn\movie_data\ratings.csv", sep='\t', encoding='latin1')
data_users = pd.read_csv(r"C:\Users\Admin\PycharmProjects\pythonProject\Mechine Learning\ML_Learn\movie_data\users.csv", sep='\t', encoding='latin1')

def enter_gender():
    print("Choose your gender.")
    print("Male : Enter number 0.")
    print("Female : Enter number 1.")
    gender = int(input("Enter your gender: "))
    while gender != 0 and gender != 1:
        print("Invalid input!")
        print("Please enter 0 for Male or 1 for Female")
        gender = int(input("Enter your gender: "))
    if gender == 0:
        return "M"
    elif gender == 1:
        return "F"

def enter_age_desc():
    data_user_age = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    print("Choose your age.")
    print("Please enter the number.")
    for index, age in enumerate(data_user_age):
        print(f"{index} : {age}")
    age = int(input("Enter your age: "))
    while age >= len(data_user_age) or age < 0:
        print("Invalid input!")
        age = int(input("Enter your age: "))
    return data_user_age[age]

def enter_occupation():
    occ_list = ['K-12 student', 'self-employed', 'scientist', 'executive/managerial', 'writer', 'homemaker', 'academic/educator',
     'programmer', 'technician/engineer', 'other or not specified', 'clerical/admin', 'sales/marketing',
     'college/grad student', 'lawyer', 'farmer', 'unemployed', 'artist', 'tradesman/craftsman', 'customer service',
     'retired', 'doctor/health care']
    print("Choose your occupation.")
    print("Please enter the number.")
    for index, occ in enumerate(occ_list):
        print(f"{index} : {occ}")
    occ = int(input("Enter your occupation: "))
    while occ >= len(occ_list) or occ < 0:
        print("Invalid input!")
        occ = int(input("Enter your occupation: "))
    return occ_list[occ]

# Filter users based on gender, age description, and occupation
filtered_users = data_users[(data_users['gender'] == enter_gender()) & (data_users['age_desc'] == enter_age_desc()) & (data_users['occ_desc'] == enter_occupation())]

# Merge filtered users with ratings data
merged_data = pd.merge(filtered_users, data_rates, on='user_id')

# Merge the result with movies data
final_data = pd.merge(merged_data, data_movies, on='movie_id')
final_data = final_data[['user_id', 'gender', 'age_desc', 'occupation', 'movie_id', 'title', 'genres', 'rating']]
final_data['average_rating'] = final_data.groupby('title')['rating'].transform('mean').round(3)

def clean_genres(genres):
    return genres.replace("|", " ").replace("-", "")
final_data["genres"] = final_data["genres"].apply(clean_genres)

df = final_data.drop(["user_id","gender","age_desc","occupation","movie_id","rating"], axis = 1)
data_use_movie = df.drop("average_rating", axis =1)
data_use_movie = data_use_movie.drop_duplicates(subset =["title"])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data_use_movie["genres"])
tfidf_dense = pd.DataFrame(tfidf_matrix.todense(), columns=vectorizer.get_feature_names_out(), index=data_use_movie.title)

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, columns=data_use_movie.title, index=data_use_movie.title)

def enter_movie():
    list_movie = []
    for movie in data_use_movie["title"]:
        list_movie.append(movie)
    print("Choose your movie.")
    for index, movie in enumerate(list_movie):
        print(f"{index} : {movie}")
    movie = int(input("Enter the number corresponding to your movie: "))
    while movie >= len(list_movie) or movie < 0:
        print("Invalid input!")
        movie = int(input("Enter the number corresponding to your movie: "))
    return list_movie[movie]

input_movie = enter_movie()

result = cosine_sim_df.loc[input_movie,:].sort_values(ascending=False).drop(input_movie)
result = result[:20]
movies_list = result.index.tolist()


filtered_df = df[df['title'].isin(movies_list)]

# Sort movies by average_rating in descending order and get top 5 movies with the highest average_rating
sorted_movies = filtered_df[['title', 'average_rating']].drop_duplicates().sort_values(by='average_rating', ascending=False)
top_movies = sorted_movies.head(5)

print(top_movies)

