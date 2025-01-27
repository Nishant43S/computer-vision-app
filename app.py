import streamlit as st

### page setup

face_detection = st.Page(
    page="Face_detection.py",
    title="Face & Hand Detection",
    icon=":material/person:",
    default=True
)

object_detection = st.Page(
    page="Object_detection.py",
    title="Object Detection",
    icon=":material/emoji_objects:"
)

about_app = st.Page(
    page="about_app.py",
    title="About App",
    icon=":material/person:"
)

pg = st.navigation(
    pages=[face_detection,object_detection,about_app],
    expanded=False,position="sidebar"
)
pg.run()

app_sidebar = st.sidebar

with app_sidebar:
    st.link_button(
        label="Project Link",
        url="https://github.com/Nishant43S/computer-vision-app.git",
        icon=":material/code_off:",
        use_container_width=True

    )
