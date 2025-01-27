import streamlit as st
from streamlit_extras.colored_header import colored_header
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

### app functions
@st.cache_data
def insert_css(css_file:str):
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

#### Load the model and processor5
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)


#### Function to perform object detection
def detect_objects(image):
    """
    detection of objects from image and
    draw rectangular boxon the object
    """
    # Perform object detection
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Process results
    target_sizes = [image.size[::-1]]  # (height, width)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    ### Draw bounding boxes and labels
    draw = ImageDraw.Draw(image)
    detected_objects = []  # List to store detected object names
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        detected_objects.append(f"{label_name} ({score:.2f})")
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{label_name} ({score:.2f})", fill="red")

    return image, detected_objects

#### page setup

st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
)

Main_app_section = st.columns([2,8,2],gap="small")

with Main_app_section[0]:
    pass

with Main_app_section[2]:
    pass

with Main_app_section[1]:

    ### app heading
    colored_header(
        label="Object Detection With DERT",
        color_name="violet-70",
        description="Use generative AI to detect objects"
    )
    st.write(
        "DETR model (`facebook/detr-resnet-50`) by Hugging Face. "
    )


    ## radio button to select file upload and url
    input_method = st.radio("Select input method", ["Upload an Image", "Image URL"],horizontal=True)


    if input_method == "Upload an Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:

            image_upload_file_col = st.columns([4,4],gap="small")

            ### uploaded image column
            with image_upload_file_col[0]:
                ## displaying image
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with image_upload_file_col[1]:
                ## Detect objects
                with st.spinner("Detecting objects..."):
                    result_image, detected_objects = detect_objects(image)
                st.image(result_image, caption="Detected Objects", use_container_width=True)

            # ## Display detected objects name
            if detected_objects:
                st.subheader("Detected Objects:")
               
                detected_objects_cols = st.columns([4,4],gap="small")

                with detected_objects_cols[0]:
                    for i in detected_objects[0::2]:
                        st.warning(f"- {i}")

                with detected_objects_cols[1]:
                    for i in detected_objects[1::2]:
                        st.warning(f"- {i}")
            else:
                st.write("No objects detected with the given threshold.")


    elif input_method == "Image URL":
        image_url = st.text_input("Enter Image URL",placeholder="enter image url",key="image url input")
        detect_object_btn = st.button(label="Detect Object",key='btn detect object')
        if detect_object_btn:
            try:

                image_upload_url_col = st.columns([4,4],gap="small")

                with image_upload_url_col[0]:
                    ###  display the image from the URL
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    st.image(image, caption="Image from URL", use_container_width=True)

                with image_upload_url_col[1]:
                    ## Detect objects
                    with st.spinner("Detecting objects..."):
                        result_image, detected_objects = detect_objects(image)
                    st.image(result_image, caption="Detected Objects", use_container_width=True)

                ## Display detected objects name
                if detected_objects:
                    st.subheader("Detected Objects:")
                    detected_url_objects_cols = st.columns([4,4],gap="small")

                    with detected_url_objects_cols[0]:
                        for i in detected_objects[0::2]:
                            st.warning(f"- {i}")

                    with detected_url_objects_cols[1]:
                        for i in detected_objects[0::2]:
                            st.warning(f"- {i}")
                    

                else:
                    st.write("No objects detected with the given threshold.")

            except Exception as e:
                st.error(f"Failed to load image from URL. Error: {e}")

### inserting css settings
insert_css("cssfiles/settings.css")