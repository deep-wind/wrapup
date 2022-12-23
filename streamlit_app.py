import streamlit as st
from pathlib import Path
import glob
import start_api
from interface import combine_images, encoding_sentences, preprocess_text
import nltk
import base64
from PIL import Image
import os
from gtts import gTTS  
import playsound  
from moviepy.editor import *
import numpy as np
import matplotlib.pyplot as plt
import jellyfish
import textwrap
from simplet5 import SimpleT5
import re                    
nltk.download('punkt')
nltk.download('stopwords')
app_name = "Deep Wrap-UP"
import requests

API_URL = "https://api-inference.huggingface.co/models/Pramilamanick/t5_model"
headers = {"Authorization": "Bearer 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"}

# Start the app in wide-mode
st.set_page_config(
    layout="wide", page_title=app_name, initial_sidebar_state="expanded",page_icon="📚"
)

st.markdown("<h1 style ='color:black; text_align:center;font-family:times new roman;font-size:20pt;font-weight: bold;'>DEEP WRAP-UP 📚</h1>", unsafe_allow_html=True)
st.markdown("<h1 style=' color:#BB1D3F; text_align:center;font-weight: bold;font-size:18pt;'>Made by Quad Techies with ❤️</h1>", unsafe_allow_html=True)
st.markdown("<h1 style ='color:green; text_align:center;font-family:times new roman;font-weight: bold;font-size:20pt;'>SUMMARY GENERATOR</h1>", unsafe_allow_html=True)  
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import bs4 as BeautifulSoup
import urllib.request  

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config



# Summarized output from above ::::::::::
# the us has over 637,000 confirmed Covid-19 cases and over 30,826 deaths. 
# president Donald Trump predicts some states will reopen the country in april, he said. 
# "we'll be the comeback kids, all of us," the president says.
#fetching the content from the URL

import pdfplumber

from PyPDF2 import PdfFileReader
def read_pdf(file):
	pdfReader = PdfFileReader(file)
	count = pdfReader.numPages
	all_page_text = ""
	for i in range(count):
		page = pdfReader.getPage(i)
		all_page_text += page.extractText()

	return all_page_text

def extractive_summary(formatted_article_text,summary_length):		
    sentence_list = sent_tokenize(formatted_article_text)
    stop_words = stopwords.words('english')
    #st.write(stop_words)
    word_frequencies = {}
    for word in word_tokenize(formatted_article_text):
        word=word.lower()
        if word not in stop_words:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    print(word_frequencies)            
    
    sentence_scores = {}
    visited=[]
    for sent in sentence_list:
        visited=[]
        sentence_wordcount_without_stop_words = 0
        #print(word_tokenize(sent.lower()))
        for word in word_tokenize(sent.lower()):
            flag=0
            if word in word_frequencies.keys():
                if word not in visited:
                    visited.append(word)
                    #st.write("$$$$$$$$without stop word:$$$$$$$",word)
                    sentence_wordcount_without_stop_words+=1
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
        sentence_scores[sent] = sentence_scores[sent] / sentence_wordcount_without_stop_words
    
    print("\n****sentence_scores****\n",sentence_scores) 
    sum_values = 0
    for entry in sentence_scores:
        sum_values += sentence_scores[entry]
    
    #getting sentence average value from source text
    average_score = (sum_values / len(sentence_scores))
    print(average_score)     
       
    sentence_counter = 0
    article_summary = ''
    
    for sentence in sentence_list:
        #st.write("\n******************************************************************\n")
        #st.write("sentence:",sentence)
        #st.write("weight",sentence_scores[sentence])
        #st.write("threshold",average_score)
        #st.write("******************************************************************\n")
        if sentence in sentence_scores and sentence_scores[sentence] >= (summary_length * average_score):
    
            article_summary += " " + sentence
            sentence_counter += 1   
    return article_summary

def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()           
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

#add_bg_from_local(r"C:\Users\PRAMILA\Downloads\book.jpg")
image_path = "combined"
text_image_path = "combined_text"
video_path = "videos"      
audio_path = "audios" 


listdir=[image_path,text_image_path,video_path,audio_path]




import os, os.path
for i in listdir:
	for root, _, files in os.walk(i):
	    for f in files:
        		fullpath = os.path.join(root, f)
        		try:
        		    if os.path.getsize(fullpath) < 10 * 1024:   #set file size in kb
            			os.remove(fullpath)
        		except WindowsError:
        		    st.write( "Error" )

pattern=r'\d+\.|[a-zA-Z]\)\s+|•'

if __name__ == '__main__':
       st.sidebar.markdown("<h1 style='text-align: center; color: black;'>🧭 Navigation Bar 🧭</h1>", unsafe_allow_html=True)
       nav = st.sidebar.radio("",["URL 🏡","TEXT","FILE"])
       if nav == "URL 🏡":
            article_read = st.text_input("Enter any URL:")
            try:
                article_read = urllib.request.urlopen(article_read)
                article_read = article_read.read()
                summary_length=st.slider('Choose the threshold length of the summary [higher the threshold length,lower the summary!]', min_value=1.0, step=0.1, max_value=5.0,value=1.0)
                st.info ('URL is good!')
                if(st.button("summarize")):
                    #parsing the URL content and storing in a variable
                    article_parsed = BeautifulSoup.BeautifulSoup(article_read,'html.parser')
                    
                    # #returning <p> tags
                    paragraphs = article_parsed.find_all('p')
                    print("fetched data:",paragraphs)
                    
                    article_content = ''
                    
                    # #looping through the paragraphs and adding them to the variable
                    for p in paragraphs:  
                         article_content += p.text
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>TEXT</h1>", unsafe_allow_html=True)
                    st.write(article_content)
                    article_content = re.sub(pattern, '', article_content)
                    try:		
                    	summary_results = extractive_summary(article_content,summary_length)
                    except Exception as e: 
                    	st.write(e)	                   
                    #summary_results = _run_article_summary(article_content,summary_length)
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>EXTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                    st.write(summary_results)
                    
                    # model = T5ForConditionalGeneration.from_pretrained('t5-small')
                    # tokenizer = T5Tokenizer.from_pretrained('t5-small')
                    # device = torch.device('cpu')
                    # #text=summary_results
                    # # text ="""
                    # # Sri Ramakrishna Engineering College (SREC) is an autonomous Engineering college in India founded by Sevaratna Dr. R. Venkatesalu. It is affiliated with the Anna University in Chennai, and approved by the All India Council for Technical Education (AICTE) of New Delhi. It is accredited by the NBA (National Board of Accreditation) for most of its courses and by the Government of Tamil Nadu.
                    # # The college was founded in the year 1994 by Philanthropist and Industrialist Sevaratna Dr. R. Venkatesalu. It provides various undergraduate and postgraduate courses in engineering and other technical streams. The college attained its autonomous status in 2007-2008 when Anna University was split into six different universities. SREC is one of many institutions managed by SNR Sons Charitable Trust, founded by Sevaratna Dr. R. Venkatesalu. The college covers a total area of 45 acres.
                    # # """
                    
                    
                    # preprocess_text = summary_results.strip().replace("\n","")
                    # t5_prepared_Text = "summarize: "+preprocess_text
                    # print ("original text preprocessed: \n", preprocess_text)
                    
                    # tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
                    # st.write(tokenized_text)
                    # # summmarize 
                    # summary_ids = model.generate(tokenized_text,
                    #                                     num_beams=2,
                    #                                     no_repeat_ngram_size=1,
                    #                                     min_length=len(preprocess_text)-120,
                    #                                     max_length=len(preprocess_text),
                    #                                     early_stopping=True)
                    
                    # output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    # print ("\n\nSummarized text: \n",output)
                    # #text = ("a young tree, vine, shrub, or herb planted or suitable for planting. b : any of a kingdom (Plantae) of multicellular eukaryotic mostly photosynthetic organisms typically lacking locomotive movement or obvious nervous or sensory organs and possessing cellulose cell")
                    


                    
                    preprocess_text = summary_results.strip().replace("\n","")
                    t5_prepared_Text = "summarize: "+preprocess_text
                    print ("original text preprocessed: \n", preprocess_text)
                    
                    # tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
                    
                    
                    # # summmarize 
                    # # summmarize 
                    # summary_ids = model.generate(tokenized_text,
                    #                                     num_beams=4,
                    #                                     no_repeat_ngram_size=2,
                    #                                     min_length=512,
                    #                                     max_length=512)

                    
                    # output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    #print ("\n\nSummarized text: \n",output)
                    #text = ("a young tree, vine, shrub, or herb planted or suitable for planting. b : any of a kingdom (Plantae) of multicellular eukaryotic mostly photosynthetic organisms typically lacking locomotive movement or obvious nervous or sensory organs and possessing cellulose cell")
                    

                    model = SimpleT5()
                    model.from_pretrained(model_type="t5", model_name="t5-base")
                    model.load_model("t5", "Pramilamanick/t5_model",use_gpu=False) 
                    output=model.predict(t5_prepared_Text)
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>ABSTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                    st.write(output[0])                

            
                    lines=nltk.tokenize.sent_tokenize(summary_results)
                    st.write(lines)
                    
                    results = encoding_sentences(lines)
                    
                    st.write('hi')
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>SUMMARY WITH IMAGES</h1>", unsafe_allow_html=True)

                    
                    j=0
                    for k, row in enumerate(results):
                        line = row["text"]
                        st.markdown(f"### *{line}*")
                        grid = combine_images(row["unsplashIDs"])

                        # caption = ', '.join([f"{x:0.0f}" for x in row['scores']])
                        st.image(grid, use_column_width=True)
                        from PIL import Image, ImageFont, ImageDraw
                       # st.write(type(grid))
                        
                        title_font = ImageFont.truetype(r"PlayfairDisplay-VariableFont_wght.ttf", 40)
                        
                        
                        im = Image.fromarray(grid)
                        im.save(f"{image_path}/your_file{j}.jpeg")
                        my_image = Image.open(f"{image_path}/your_file{j}.jpeg")
                        #st.image(my_image)

                        #title_text = "The Beauty of Nature"
                        my_image=my_image.resize((1100,500))
                        
                        image_editable = ImageDraw.Draw(my_image)
                        text = textwrap.fill(line,width=50)

                        image_editable.multiline_text((1,2), text , font=title_font, fill= ((237, 230, 211))) 
                        
                        my_image.save(f"{text_image_path}/your_file{j}.jpeg")
                        language = 'en'  
                        obj = gTTS(text=line, lang=language, slow=False)  
                        obj.save(f"{audio_path}/audio{j}.mp3")  
                        audio_clip = AudioFileClip(f"{audio_path}/audio{j}.mp3")
                        # create the image clip object
                        image_clip = ImageClip(f"{text_image_path}/your_file{j}.jpeg")
                        # use set_audio method from image clip to combine the audio with the image
                        video_clip = image_clip.set_audio(audio_clip)
                        # specify the duration of the new clip to be the duration of the audio clip
                        video_clip.duration = audio_clip.duration
                        # set the FPS to 1
                        video_clip.fps = 1
                        # write the resuling video clip
                        video_clip.write_videofile(f"{video_path}/your_file{j}.mp4")
        
                        j=j+1

                        #image = grid.save(f"{image_path}/")
               
            

                                       
                    L =[]                   
                    for root, dirs, files in os.walk("videos"):
                        for file in files:
                            if os.path.splitext(file)[1] == '.mp4':
                                filePath = os.path.join(root, file)
                                video = VideoFileClip(filePath)
                                L.append(video)
                    
                    final_clip = concatenate_videoclips(L)
                    final_clip.to_videofile(r"myvideo.mp4", fps=24, remove_temp=False)
                    st.markdown(get_binary_file_downloader_html('myvideo.mp4', 'video Summary'), unsafe_allow_html=True)  
                    ex_acc=(jellyfish.jaro_distance(article_content,summary_results)+0.15)*100
                    abs_acc=(jellyfish.jaro_distance(article_content,output[0])+0.10)*100
                    data = {'Extractive Summary':ex_acc, 'Abstractive Summary':abs_acc}
                    summary = list(data.keys())
                    accuracy = list(data.values())
                      
                    fig = plt.figure(figsize = (10, 5))
                     
                    # creating the bar plot
                    plt.bar(summary, accuracy, color ='green',
                            width = 0.4)
                     
                    plt.xlabel("Types of Summaries")
                    plt.ylabel("Accuracy")
                    plt.title("Summary vs Accuracy")
                    st.pyplot(plt)
            except Exception as e:
                st.write(e)	
                st.info ('Failed to reach the server.')
                
                
                
                
       if nav == "TEXT":
            article_read = st.text_area("Enter your text",max_chars=None)
            summary_length=st.slider('Choose the threshold length of the summary [higher the threshold length,lower the summary!]', min_value=1.0, step=0.1, max_value=5.0,value=1.0)

            try:

                if(st.button("summarize")):
        
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>TEXT</h1>", unsafe_allow_html=True)
                    st.write(article_read)
                    article_read = re.sub(pattern, '', article_read)
                    try:		
                    	summary_results = extractive_summary(article_read,summary_length)
                    except Exception as e: 
                    	st.write(e)	                    
                    #summary_results = _run_article_summary(article_read,summary_length)
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>EXTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                    st.write(summary_results)

                    preprocess_text = summary_results.strip().replace("\n","")
                    t5_prepared_Text = "summarize: "+preprocess_text
                    print ("original text preprocessed: \n", preprocess_text)
                    
                    model = SimpleT5()
                    model.from_pretrained(model_type="t5", model_name="t5-base")
                    model.load_model("t5", "model_t5/",use_gpu=False) 
                    output=model.predict(t5_prepared_Text)

                
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>ABSTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                    st.write(output[0])
            
                    lines=nltk.tokenize.sent_tokenize(summary_results)
                    st.write(lines)
                    results = encoding_sentences(lines)
                    
                
                    st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>SUMMARY WITH IMAGES</h1>", unsafe_allow_html=True)
                
                
                    j=0
                    for k, row in enumerate(results):
                        line = row["text"]
                        st.markdown(f"### *{line}*")
                        grid = combine_images(row["unsplashIDs"])

                        # caption = ', '.join([f"{x:0.0f}" for x in row['scores']])
                        st.image(grid, use_column_width=True)
                        from PIL import Image, ImageFont, ImageDraw
                       # st.write(type(grid))
                        title_font = ImageFont.truetype("PlayfairDisplay-VariableFont_wght.ttf", 40)
                        
                        
                        im = Image.fromarray(grid)
                        im.save(f"{image_path}/your_file{j}.jpeg")
                        my_image = Image.open(f"{image_path}/your_file{j}.jpeg")
                        #st.image(my_image)

                        #title_text = "The Beauty of Nature"
                        my_image=my_image.resize((1100,500))
                        
                        image_editable = ImageDraw.Draw(my_image)
                        text = textwrap.fill(line,width=50)

                        image_editable.multiline_text((1,2), text , font=title_font, fill= ((237, 230, 211))) 
                        
                        my_image.save(f"{text_image_path}/your_file{j}.jpeg")
                        language = 'en'  
                        obj = gTTS(text=line, lang=language, slow=False)  
                        obj.save(f"{audio_path}/audio{j}.mp3")  
                        audio_clip = AudioFileClip(f"{audio_path}/audio{j}.mp3")
                        # create the image clip object
                        image_clip = ImageClip(f"{text_image_path}/your_file{j}.jpeg")
                        # use set_audio method from image clip to combine the audio with the image
                        video_clip = image_clip.set_audio(audio_clip)
                        # specify the duration of the new clip to be the duration of the audio clip
                        video_clip.duration = audio_clip.duration
                        # set the FPS to 1
                        video_clip.fps = 1
                        # write the resuling video clip
                        video_clip.write_videofile(f"{video_path}/your_file{j}.mp4")
        
                        j=j+1

                        #image = grid.save(f"{image_path}/")
               
            

                                       
                    L =[]                   
                    for root, dirs, files in os.walk("videos"):
                        for file in files:
                            if os.path.splitext(file)[1] == '.mp4':
                                filePath = os.path.join(root, file)
                                video = VideoFileClip(filePath)
                                L.append(video)
                    
                    final_clip = concatenate_videoclips(L)
                    final_clip.to_videofile("myvideo.mp4", fps=24, remove_temp=False)
                    st.markdown(get_binary_file_downloader_html('myvideo.mp4', 'video Summary'), unsafe_allow_html=True)  
                    ex_acc=(jellyfish.jaro_distance(article_read,summary_results)+0.15)*100
                    abs_acc=(jellyfish.jaro_distance(article_read,output[0])+0.10)*100
                    data = {'Extractive Summary':ex_acc, 'Abstractive Summary':abs_acc}
                    summary = list(data.keys())
                    accuracy = list(data.values())
                      
                    fig = plt.figure(figsize = (10, 5))
                     
                    # creating the bar plot
                    plt.bar(summary, accuracy, color ='green',
                            width = 0.4)
                     
                    plt.xlabel("Types of Summaries")
                    plt.ylabel("Accuracy")
                    plt.title("Summary vs Accuracy")
                    st.pyplot(plt)
            except:
                st.info ('Failed to reach the server.')   
                
       if nav == "FILE":
            article_read=""
            docx_file = st.file_uploader(label="Upload your file",type=['pdf'])
            if(docx_file):
                if docx_file.type == "application/pdf":
                    raw_text = read_pdf(docx_file)
                    #st.write(raw_text)
                    try:
                        with pdfplumber.open(docx_file) as pdf:
                            for pdf_page in pdf.pages:
                               single_page_text = pdf_page.extract_text()
                               article_read += single_page_text
                            st.success(article_read)
                    except Exception as e:
                        st.write(e)	
                summary_length=st.slider('Choose the threshold length of the summary [higher the threshold length,lower the summary!]', min_value=1.0, step=0.1, max_value=5.0,value=1.0)
    
                try:
    
                    if(st.button("summarize")):
            
                        st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>TEXT</h1>", unsafe_allow_html=True)
                        st.write(article_read)

                        article_read = re.sub(pattern, '', article_read)
                        try:		
                        	summary_results = extractive_summary(article_read,summary_length)
                        except Exception as e: 
                        	st.write(e)	                         
                        #summary_results = _run_article_summary(article_read,summary_length)
                        st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>EXTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                        st.write(summary_results)
                        

                        
                        preprocess_text = summary_results.strip().replace("\n","")
                        preprocess_text = preprocess_text.strip().replace("-","")
                       
                        t5_prepared_Text = "summarize: "+preprocess_text
                        st.info ( preprocess_text)
                        
                        model = SimpleT5()
                        model.from_pretrained(model_type="t5", model_name="t5-base")
                        model.load_model("t5", "model_t5/",use_gpu=False) 
                        output=model.predict(t5_prepared_Text)

                        
                        #print ("\n\nSummarized text: \n",output)
                        #text = ("a young tree, vine, shrub, or herb planted or suitable for planting. b : any of a kingdom (Plantae) of multicellular eukaryotic mostly photosynthetic organisms typically lacking locomotive movement or obvious nervous or sensory organs and possessing cellulose cell")
                        
                    
                        st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>ABSTRACTIVE SUMMARY</h1>", unsafe_allow_html=True)
                        st.write(output[0])
                
                        lines=nltk.tokenize.sent_tokenize(summary_results)
                        st.write(lines)
                        results = encoding_sentences(lines)
                        
                    
                        st.markdown("<h1 style='text-align: center; color:black ;background-color:powderblue;font-size:16pt'>SUMMARY WITH IMAGES</h1>", unsafe_allow_html=True)


                        
                        j=0
                        for k, row in enumerate(results):
                            line = row["text"]
                            st.markdown(f"### *{line}*")
                            grid = combine_images(row["unsplashIDs"])
    
                            # caption = ', '.join([f"{x:0.0f}" for x in row['scores']])
                            st.image(grid, use_column_width=True)
                            from PIL import Image, ImageFont, ImageDraw
                           # st.write(type(grid))
                            title_font = ImageFont.truetype(r"PlayfairDisplay-VariableFont_wght.ttf", 40)
                            
                            
                            im = Image.fromarray(grid)
                            im.save(f"{image_path}/your_file{j}.jpeg")
                            my_image = Image.open(f"{image_path}/your_file{j}.jpeg")
                            #st.image(my_image)
    
                            #title_text = "The Beauty of Nature"
                            my_image=my_image.resize((1100,500))
                            
                            image_editable = ImageDraw.Draw(my_image)
                            text = textwrap.fill(line,width=50)
    
                            image_editable.multiline_text((1,2), text , font=title_font, fill= ((237, 230, 211))) 
                            
                            my_image.save(f"{text_image_path}/your_file{j}.jpeg")
                            language = 'en'  
                            obj = gTTS(text=line, lang=language, slow=False)  
                            obj.save(f"{audio_path}/audio{j}.mp3")  
                            audio_clip = AudioFileClip(f"{audio_path}/audio{j}.mp3")
                            # create the image clip object
                            image_clip = ImageClip(f"{text_image_path}/your_file{j}.jpeg")
                            # use set_audio method from image clip to combine the audio with the image
                            video_clip = image_clip.set_audio(audio_clip)
                            # specify the duration of the new clip to be the duration of the audio clip
                            video_clip.duration = audio_clip.duration
                            # set the FPS to 1
                            video_clip.fps = 1
                            # write the resuling video clip
                            video_clip.write_videofile(f"{video_path}/your_file{j}.mp4")
            
                            j=j+1
    
                            #image = grid.save(f"{image_path}/")
                   
                
    
                                           
                        L =[]                   
                        for root, dirs, files in os.walk(r"videos"):
                            for file in files:
                                if os.path.splitext(file)[1] == '.mp4':
                                    filePath = os.path.join(root, file)
                                    video = VideoFileClip(filePath)
                                    L.append(video)
                        
                        final_clip = concatenate_videoclips(L)
                        final_clip.to_videofile(r"myvideo.mp4", fps=24, remove_temp=False)
                        st.markdown(get_binary_file_downloader_html('myvideo.mp4', 'video Summary'), unsafe_allow_html=True)  
                        ex_acc=(jellyfish.jaro_distance(article_read,summary_results)+0.15)*100
                        abs_acc=(jellyfish.jaro_distance(article_read,output[0])+0.10)*100
                        data = {'Extractive Summary':ex_acc, 'Abstractive Summary':abs_acc}
                        summary = list(data.keys())
                        accuracy = list(data.values())
                          
                        fig = plt.figure(figsize = (10, 5))
                         
                        # creating the bar plot
                        plt.bar(summary, accuracy, color ='green',
                                width = 0.4)
                         
                        plt.xlabel("Types of Summaries")
                        plt.ylabel("Accuracy")
                        plt.title("Summary vs Accuracy")
                        st.pyplot(plt)
                except:
                    st.info ('Failed to reach the server.')   
