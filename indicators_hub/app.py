import streamlit as st

def main():
    st.title("Indicators Hub - Language Selection")

    language_choice = st.radio("Select Language:", ["English", "Arabic"], index=0)

    # Retrieve text from secrets
    eng_text = st.secrets["overview_english"]
    arb_text = st.secrets["overview_arabic"]

    if language_choice == "English":
        # Render English LTR
        st.markdown(eng_text, unsafe_allow_html=True)
    else:
        
        st.markdown("""
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div dir="rtl" style="text-align: right;">
                {arb_text}
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
