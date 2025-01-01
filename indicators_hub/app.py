import streamlit as st

def main():
    # App Title
    st.title("Indicators Hub - Language Selection")

    # Dropdown or radio for language choice
    language_choice = st.radio("Select Language:", ["English", "Arabic"])

    # Retrieve text from secrets
    eng_text = st.secrets["overview_english"]
    arb_text = st.secrets["overview_arabic"]

    # Display the relevant text depending on the selected language
    if language_choice == "English":
        st.markdown(eng_text, unsafe_allow_html=True)
    else:
        st.markdown(arb_text, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
