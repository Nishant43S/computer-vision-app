import streamlit as st

## app function
@st.cache_data
def insert_css(css_file:str):
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
def insert_html(html_file):
    with open(html_file) as htm:
        return htm.read()

### page setup        
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(insert_html("html_files/about.html"),unsafe_allow_html=True)

insert_css("cssfiles/settings.css")