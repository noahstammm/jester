import streamlit as st
from main import createtemplatemessage


# Eine einfache Funktion, die die Eingabe des Benutzers verarbeitet und eine Antwort generiert.
def get_bot_response(user_input):
    responses = {
        "hallo": "Hallo! Wie kann ich dir helfen?",
        "wie geht es dir?": "Mir geht es gut, danke! Und dir?",
        "danke": "Gern geschehen! Kann ich noch etwas für dich tun?",
    }
    # Standardantwort, falls keine Übereinstimmung gefunden wird
    return responses.get(user_input.lower(), "Entschuldigung, das habe ich nicht verstanden.")


# Streamlit-App
st.title('Jester')

# Benutzereingabe
user_input = st.text_input("Wie kann ich behilflich sein:")

# Button, um die Antwort zu erhalten
if st.button('Antworten'):
    bot_response = get_bot_response(user_input)
    st.text_area("Chatbot-Antwort:", value=bot_response, height=100, max_chars=None, key=None)
