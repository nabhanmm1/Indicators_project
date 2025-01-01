import streamlit as st

def main():
    st.title("Indicators Hub - Language Selection")

    # Radio or selectbox for language
    language_choice = st.radio("Select Language:", ["English", "Arabic"], index=0)

    # Retrieve text from secrets (already stored in TOML format)
    eng_text = st.secrets["overview_english"]  # normal LTR
    arb_text = st.secrets["overview_arabic"]   # we want RTL

    if language_choice == "English":
        # Render as-is (LTR)
        st.markdown(eng_text, unsafe_allow_html=True)
    else:
        # Wrap Arabic text in a right-to-left div
        st.markdown(
            f"""
            <div dir="rtl" style="text-align: right;">
                {arb_text}
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
