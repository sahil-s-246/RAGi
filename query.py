import json
import streamlit as st
import weaviate
import google.generativeai as genai
import random
import ast

from weaviate.exceptions import WeaviateQueryError

genai.configure(api_key=st.secrets["API_KEY"])
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

with open("plan.json", 'r') as f:
    plans = json.load(f)

st.title("RAGi - The Meal Plan Recommender")


def fill_form():
    # Define options for select boxes
    status = False
    genders = ["Male", "Female", "Other"]
    activity_levels = ["Low", "Moderate", "High", "Active"]
    dietary_preferences = ["Vegetarian", "Non-Vegetarian", "Vegan", "Pescatarian", "Keto"]
    health_goals = ["Weight Loss", "Muscle Gain", "Weight Maintenance", "Heart Health", "Fat Loss"]

    # Create the form
    with st.form(key='user_form', border=False,):
        age = st.number_input("age", min_value=0, max_value=120)
        gender = st.selectbox("gender", genders, index=None)
        weight = st.number_input("Weight (kg)", min_value=0.0,value=50.0)
        height = st.number_input("Height (cm)", min_value=100)
        activity_level = st.selectbox("Activity Level", activity_levels,index=None)
        dietary_preference = st.selectbox("Dietary Preference", dietary_preferences,index=None)
        health_goal = st.selectbox("Health Goal", health_goals,index=None)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Concatenate values into a space-separated string
        status = True
        user_info = f"{age} {gender} {weight} {height} {activity_level} {dietary_preference} {health_goal} "
        return user_info, status
    else:
        return "", status


def extract_features(query_result):
    """Weaviate's Near Text returns a weaviate object.Extracts this to a dictionary"""
    extracted_data = {}
    for i, item in enumerate(query_result.objects):
        properties = item.properties
        extracted_data[item.properties["mealPlan"]] = {
            'UserID': properties.get('userID', ''),
            'Age': properties.get('age', ''),
            'Gender': properties.get('gender', ''),
            'Weight': properties.get('weight', ''),
            'Height': properties.get('height', ''),
            'ActivtyLevel': properties.get('activityLevel', ''),
            'DietaryPreference': properties.get('dietaryPreference', ''),

        }
    return extracted_data


def ask_ai_for_recommendations():
    """Prompt AI and gemini will retrieve and rank dishes"""
    ask_ai = st.text_input("Ask AI🤖")
    if ask_ai:
        res = model.generate_content([f"Give a meal plan from {plans}"
                                      f"according to the prompt: "
                                      f" {ask_ai} In the format breakfast lunch dinner and optional random juice/snacks"])
        st.write(res.text)


def recommend():

    """Connect to a weaviate collection in your weaviate cloud's sandbox, In this case FoodRecommend
    There are two steps in this recommendation \n
    Retrieval: Semantically search through the data wrt the query \n
    Ranking: Use Gemini Flash API to rank the data"""

    res = ""
    client = weaviate.connect_to_wcs(
        cluster_url=st.secrets["CLUSTER_URL"],
        auth_credentials=weaviate.auth.AuthApiKey(st.secrets["Weav_API_KEY"]),
        headers={

            "X-HuggingFace-Api-Key": st.secrets["hf_key"]
        }
    )

    try:
        questions = client.collections.get("MealRAGi")
        # Retrieval
        query, status = fill_form()
        if not status:
            return "Submit your data please📚"
        response = questions.query.near_text(
            query=query,
            limit=10
        )
        data = extract_features(response)
        with open("resp.json", 'w') as file:
            json.dump(data, file, indent=4)
        # Ranking

        res = model.generate_content([f"We have to suggest a meal plan for the user."
                                      f"Rerank the data\n\n{data} \n\n according to the query: {query}\n\n"
                                      f"Accordingly provide the most suitable meal plan in the following format:"
                                      f"\n {plans[list(data.keys())[0]]}"],
                                     generation_config={"response_mime_type": "application/json"})

    except WeaviateQueryError:
        st.warning("An error occurred. Please try again. Sorry, your tastes to complex to be catered for by us😝")
    finally:
        client.close()  # Close client gracefully
    return res.text


st.write("Enter your choice:")

choice = st.selectbox("Choice", ["Custom", "ask AI"], index=None, placeholder="Custom or Ask AI...")
if choice is None:
    st.write({
        "Custom": "Dishes will be retrieved according to your query from weaviate's db",
        "ask AI": "Dishes will be recommended by gemini according to your prompt"
    })


elif choice == "Custom":

    order = recommend()

    st.success(f"{order}")


elif choice == "ask AI":
    ask_ai_for_recommendations()

rand = st.button("I'm Feeling Lucky✨")
if rand:
    st.write(plans[random.choice(list(plans.keys()))])
