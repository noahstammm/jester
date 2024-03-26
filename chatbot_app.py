import streamlit as st
# Annahme: createtemplatemessage, initializemodel, invokechain sind in 'main.py' definiert.
from main import createtemplatemessage, initializemodel, invokechain

# Funktion zur Schätzung der Höhe des Textfeldes
def estimate_text_area_height(text):
    lines = text.count("\n") + 1
    min_lines = max(lines, 4)
    height_per_line = 20
    return min_lines * height_per_line

# Initialisieren des Chat-Modells und der Kette
chat_model = initializemodel()
chain = createtemplatemessage(chat_model)

# Streamlit-App Titel
st.title('Jester')

# Benutzereingabe mit dynamischem Schlüssel
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

user_input = st.text_input("Wie kann ich behilflich sein?", key=f"user_input_{st.session_state.input_key}")

# Antwort-Button und Aktualisierung der Konversationshistorie
def on_click():
    # Bot-Antwort generieren
    bot_response = invokechain(chain, st.session_state[f"user_input_{st.session_state.input_key}"])
    # Konversationshistorie aktualisieren
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    st.session_state.conversation_history.append(("Du:", st.session_state[f"user_input_{st.session_state.input_key}"]))
    st.session_state.conversation_history.append(("Bot:", bot_response))
    # Eingabefeld zurücksetzen durch Inkrementieren des Schlüssels
    st.session_state.input_key += 1

if st.button('Antworten', on_click=on_click):
    pass  # Logik wird durch das on_click-Event des Buttons getriggert

# Konversationshistorie anzeigen
if 'conversation_history' in st.session_state:
    for speaker, message in st.session_state.conversation_history:
        height = estimate_text_area_height(message)
        # Eindeutiger Schlüssel für jedes Textfeld, um Duplikate zu vermeiden
        unique_key = f"{speaker}_{hash(message)}"
        st.text_area(label=speaker, value=message, height=height, key=unique_key)
