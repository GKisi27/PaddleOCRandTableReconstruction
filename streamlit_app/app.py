#######################
# Import libraries
import os
import io
import easygui
import pandas as pd
import altair as alt
from PIL import Image
import streamlit as st
from data_extraction import *
import base64
from pathlib import Path

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


#######################
# Load data
sample_image = Image.open('data/SampleImage.jpg')

#######################
# Global Variable
enhance_image = True
upscale_factor = 2            
split_image = True
width_tolerance = 10
remove_duplicate_columns=True
remove_similar_columns=True
similarity_threshold = 0.5
dataframe_header = True
image_name = ""

df = pd.DataFrame()

def find_file(filename, search_path):
  for root, dirs, files in os.walk(search_path):
    if filename in files:
      return os.path.join(root, filename)
  return None

def extract_table_preprocess(ocr, img, remove_duplicate_columns=True, remove_similar_columns=True, similarity_threshold = 0.5, dataframe_header=True):
    global df, image_name
    image_name = img
    #Get current working directory to create full path for input and output
    current_directory = os.getcwd()
    input_file_path = find_file(img, current_directory)
    output_file_path = "\\".join(input_file_path.split("\\")[:-1])
    start_time_to_extract_data = time.time()
    
    df = extract_data(ocr, input_file_path, upscale_factor)
    
    df.reset_index(drop=True)
    if remove_duplicate_columns:
        ## Check and remove duplicate columns if exists
        df = remove_repeated_columns(df.copy())
    if remove_similar_columns:
        ## Check and remove for similar columns if threshold match.
        df = remove_similar_columns_func(df.copy(), similarity_threshold)
        df = df.iloc[1:].rename(columns=df.iloc[0]) 
#######################
# Page configuration
st.set_page_config(
    page_title="Table Extractor OCR",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# Sidebar
with st.sidebar:
    st.title('📈 Table Extractor OCR')
    file_selected = st.file_uploader('File Uploader',type=['.jpg', '.png', '.jpeg'])
    ## Data Cleaning
    st.sidebar.title('Data Cleaning')
    duplicate_columns = st.radio("Remove Duplicate Columns", ('On', 'Off'),horizontal=True, index=0)
    similar_columns = st.radio("Remove Similar Columns", ('On', 'Off'),horizontal=True, index=0)
    if similar_columns == 'On':
        threshold = st.number_input("Similarity Threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    header = st.checkbox("Header", value = True)
    
#######################
# Dashboard Main Panel
row_one = st.columns((5,1,1), gap='small')
with st.container():
    with row_one[0]:
        message = st.empty()
    with row_one[1]:
        error_displayed = False

        start_btn = st.button("Start Extraction", use_container_width=True)
        if start_btn:        
            if duplicate_columns == "On":
                remove_duplicate_columns=True
            else:
                remove_duplicate_columns = False
            if similar_columns == "On":
                remove_similar_columns = True
                similarity_threshold = threshold
            else:
                remove_similar_columns = False
                similarity_threshold = 0.5
            if header == True:
                dataframe_header = True
            else:
                dataframe_header = False
                
            start_time_to_load_model = time.time()    
            ocr = PaddleOCR(use_angle_cls=True, show_log = False, table_max_len=1000) # need to run only once to download and load model into memory
            end_time_to_load_model = time.time()
 
            if file_selected:
                try:
                    extract_table_preprocess(ocr, file_selected.name, remove_duplicate_columns, remove_similar_columns, similarity_threshold, dataframe_header)
                except Exception as e:
                    message.error(f"Error: {e}")
            else:
                try:
                    extract_table_preprocess(ocr, "SampleImage.jpg", remove_duplicate_columns, remove_similar_columns, similarity_threshold, dataframe_header)
                except Exception as e:
                    message.error(f"Error: {e}")
                

        else:
            dataframe_header = True
    with row_one[2]:
        if not df.empty:
            save_file_name = image_name.split(".")[0]
            # Create an Excel buffer
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')
            if dataframe_header:
                df.to_excel(writer, sheet_name='Sheet1', index=False)
            else:
                df.to_excel(writer, sheet_name='Sheet1',  index=False, header=None)
            writer.save()

            # Download button
            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=f'{save_file_name}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True
            )
            

## Image and Status
row_two =  st.columns([1,6,1])
with st.container():
    with row_two[1]:
        if file_selected:
            st.markdown(f'#### Image: {file_selected.name}')
        else:
            st.markdown(f'#### Image: Sample Image')
        if file_selected:
            # Load your image
            cwd = os.getcwd()
            image_path = find_file(file_selected.name, cwd)
            img_base64 = image_to_base64(image_path)
            
            # Use a unique class for the image container
            st.markdown(
                f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{img_base64}" style="width:100%;">
                </div>
                <style>
                .image-container {{
                    max-height: 500px;  /* Set the fixed height */
                    overflow-y: scroll; /* Add vertical scrollbar */
                    overflow-x: hidden; /* Hide horizontal scrollbar */
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.image(sample_image, use_column_width=True)

## Result
row_three =  st.columns(1)
with st.container():
    with row_three[0]:
        st.markdown ('#### Result')
        if dataframe_header == True:
            # Identify empty string columns
            empty_cols = [col for col in df.columns if col == '']
            # Create unique names
            new_names = [f'unnamed_{i}' for i in range(len(empty_cols))]
            # Get indices of empty columns
            column_index = [i for i, col in enumerate(df.columns) if col == '']
            # Rename columns
            for col_name, col_index in zip(new_names, column_index):
                df.columns.values[col_index] = col_name
            # Get renamed columns
            renamed_columns = rename_duplicate_columns(df.columns.tolist())  # Pass list of column names
            # Assign new column names to the DataFrame
            df.columns = renamed_columns
            # Display DataFrame
            st.dataframe(df, use_container_width=True)
        else:
            df = reset_column(df)
            #Rename columns
            st.dataframe(df, use_container_width=True)
    
    
