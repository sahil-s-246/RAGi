import json
import streamlit as st
import weaviate
import google.generativeai as genai
import random
from PIL import Image
import requests
import io

from weaviate.exceptions import WeaviateQueryError

# Example Query
# query= "Veg Indian Dishes"
genai.configure(api_key=st.secrets["API_KEY"])
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

st.title("Restaurant from Novigrad")
with open("data.json", "r") as f:
    menu = json.load(f)
st.dataframe(menu)


def extract_features(query_result):
    """Weaviate's Near Text returns a weaviate object.Extracts this to a dictionary"""
    extracted_data = {}
    for i, item in enumerate(query_result.objects):
        properties = item.properties
        extracted_data[item.properties["dish"]] = {
            'cuisine': properties.get('cuisine', ''),
            'category': properties.get('category', ''),
            'description': properties.get('description', ''),
            'allergy': properties.get('allergy', '')
        }
    return extracted_data


def ask_ai_for_recommendations():
    """Prompt AI and gemini will retrieve and rank dishes"""
    ask_ai = st.text_input("Ask AI🤖")
    if ask_ai:
        res = model.generate_content([f"Recommend some dishes along with description, allegies etc. from the menu: "
                                      f"{menu}, \n\n ranked by relevance, according to the prompt: "
                                      f" {ask_ai} Eg. If the prompt is something like a mildly spicy gravy dish with"
                                      f" paneer then recommend paneer butter masala"])
        st.write(res.text)


def recommend(query):
    """Connect to a weaviate collection in your weaviate cloud's sandbox, In this case FoodRecommend
    There are two steps in this recommendation \n
    Retrieval: Semantically search through the data wrt the query \n
    Ranking: Use Gemini Flash API to rank the data"""

    result = first = {}
    client = weaviate.connect_to_wcs(
        cluster_url=st.secrets["CLUSTER_URL"],
        auth_credentials=weaviate.auth.AuthApiKey(st.secrets["Weav_API_KEY"]),
        headers={

            "X-HuggingFace-Api-Key": st.secrets["hf_key"]
        }
    )

    try:
        questions = client.collections.get("FoodRecommend")
        # Retrieval
        response = questions.query.near_text(
            query=query,
            limit=10
        )
        data = extract_features(response)
        with open("resp.json", 'w') as file:
            json.dump(data, file, indent=4)
        # Ranking

        res = model.generate_content([f"Rerank json objects in this data {data} \n\n according to the query: {query}\n\n"
                                      f"Remove the most irrelevant ones but dont remove many"],
                                     generation_config={"response_mime_type": "application/json"})
        # print(res.text)
        result = json.loads(res.text)
        first = result[list(result.keys())[0]]

    except WeaviateQueryError:
        st.write("An error occurred. Please try again. Sorry, your tastes to complex to be catered for by us😝")
    finally:
        client.close()  # Close client gracefully
    return result, first


def give_random():
    """Feeling Lucky? Get a Random dish!"""
    dish = random.choice(list(menu))
    return dish


st.write("Enter your choice:")

choice = st.selectbox("Choice", ["Custom", "ask AI"], index=None, placeholder="Custom or Ask AI...")
if choice is None:
    st.write({
        "Custom": "Dishes will be retrieved according to your query from weaviate's db",
        "ask AI": "Dishes will be recommended by gemini according to your prompt"
    })
    st.info("Please describe your preference eg: Indian or Japanese, Veg or Non-Veg, Sweet or Spicy etc. "
            "Or Click 'I'm Feeling Lucky' for a random suggestion")
    st.warning("Rn there are only 2 cuisines : Indian and Japanese😅")

elif choice == "Custom":
    q = st.text_input("What would you like to have🍉", placeholder="Indian sweet dish")
    if q:
        order, first = recommend(q)
        # Generate Image of first recommendation
        API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
        headers = {"Authorization": st.secrets["hftoken"]}
        payload = {"inputs": first["description"]}
        response = requests.post(API_URL, headers=headers, json=payload)
        image_bytes = response.content
        name_of_first = list(order.keys())[0]
        order.pop(name_of_first)
        with st.spinner("Loading..."):
            image = Image.open(io.BytesIO(image_bytes))
            st.write("Recommended for You: ")
            st.markdown(f"#### :orange[{name_of_first}]")
            st.write(first)
            st.image(image)
        "You might also like:"

        st.write(order)
        st.warning("Some Recommendations may be inaccurate or irrelevant")

elif choice == "ask AI":
    ask_ai_for_recommendations()

click = st.button("I'm Feeling Lucky✨")
if click:
    menu_item = give_random()
    st.write(menu_item)
