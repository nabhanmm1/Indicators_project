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
        # Render Arabic, using an HTML <h2> for the heading
        # and wrapping the rest of the text in a right-aligned <div>.
        # You can remove <h2> if your secrets text already includes
        # its own headings, but this shows how to fix the "heading not recognized" issue.
        
        st.markdown("""
        <h2 style="text-align: right;">القسم باللغة العربية</h2>
        """, unsafe_allow_html=True)

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
