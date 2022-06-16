#!/usr/bin/env python
# coding: utf-8

# # LSA

# LSA (Latent Semantic Analysis) juga dikenal sebagai LSI (Latent Semantic Index). LSA menggunakan Bag of Word(BoW) model yang nantinya akan menghasilkan term-document matrix. Baris mewakili istilah dan kolom mewakili dokumen. LSA mempelajari topik laten dengan melakukan dekomposisi matriks pada matriks istilah dokumen menggunakan dekomposisi nilai Singular. Beberapa langkah yang harus dilakukan yaitu :

# ### 1. Crawling Data

# Untuk dapat melakukan crawling data, gunakan library scrapy. Install terlebih dahulu scrapynya dengan mengetikkan perintah dibawah ini pada command prompt atau Anaconda Prompt.

# In[1]:


pip install scrapy


# Lalu setelah proses selesai, jalankan perintah untuk melakukan setup pada Spider.

# In[2]:


scrapy startproject WebScrapingBerita


# Selanjutnya untuk membuat scraper atau crawler, kita harus menambahkan file .py baru di dalam directory, buat file baru bernama berita.py lalu tuliskan script sebagai berikut 

# In[ ]:


import scrapy


class BeritaSpider(scrapy.Spider):
    name = 'berita'
    allowed_domains = ['www.liputan6.com']
    start_urls = ['http://www.liputan6.com/']

    def parse(self, response):
            cont = response.css('.articles--iridescent-list--text-item__details')

            #collecting data
            kat = cont.css('.articles--iridescent-list--text-item__category')
            judul = cont.css('.articles--iridescent-list--text-item__title-link-text')
            c=0
            
            #combining the results
            for review in kat:
                yield{'Kategori': ''.join(review.xpath('.//text()').extract()),
                      'Judul': ''.join(judul[c].xpath('.//text()').extract()),
                
                    }
                c=c+1


# Selanjutnya melalui terminal, masuk terlebih dahulu ke dalam directori WebScrapingBerita/WebScrapingBerita/spiders, lalu jalankan perintah:

# In[ ]:


scrapy runspider berita.py -o Berita.csv


# Maka akan tampil hasil dari crawling dalam bentuk file csv. Langkah selanjutnya yang perlu dilakukan adalah preprocessing.

# ### 2. Preprocessing

# Data preprocessing adalah proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur. Pertama,lakukan import libary yang diperlukan dalam langkah preprocessing dan LSA.

# In[2]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('indonesian'))


# load dataset yang akan di preprocessing dengan menggunakan library pandas untuk membaca file csv.

# In[3]:


df = pd.read_csv('berita.csv')
df.head() #menampilkan 5 data teratas


# Proses selanjutnya hapus kolom 'Kategori' karena tidak terlalu berguna dalam proses LSA nantinya.

# In[4]:


# drop the kategori data.
df.drop(['Kategori'],axis=1,inplace=True)
df.head(10)


# Dilanjutkan dengan proses tokenizing untuk memotong kalimat menjadi kata dengan cara memotong kata pada white space atau spasi dan membuang karakter tanda baca. Di sini lakukan preprocessing data. 

# In[5]:


#fungsi clean_text
def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words  and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[6]:


# melakukan preprocessing pada data judul
df['judul_cleaned']=df['Judul'].apply(clean_text)
df.head(10)


# Lalu hilangkan kolom 'Judul'

# In[7]:


df.drop(['Judul'],axis=1,inplace=True)


# In[8]:


df.head(10)


# ### 3. Ekstraksi Fitur dan Membuat Document-Term-Matrix(DTM)

# Dalam DTM nilainya adalah nilai TFidf.
# 
# Saya juga telah menentukan beberapa parameter dari vectorizer Tfidf.
# 
# Beberapa poin penting:-
# 
# **1) LSA umumnya diimplementasikan dengan nilai Tfidf di mana-mana dan bukan dengan Count Vectorizer.**
# 
# **2) max_features bergantung pada daya komputasi Anda dan juga pada eval. metrik (skor koherensi adalah metrik untuk model topik). Coba nilai yang memberikan evaluasi terbaik. metrik dan tidak membatasi kekuatan pemrosesan.**
# 
# **3) Nilai default untuk min_df & max_df bekerja dengan baik.**
# 
# **4) Dapat mencoba nilai yang berbeda untuk ngram_range.**

# In[12]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[13]:


vect_text=vect.fit_transform(df['judul_cleaned'])


# Sekarang kita dapat melihat kata-kata yang paling sering dan langka di headline berita berdasarkan skor idf. Semakin kecil nilainya.

# In[14]:


print(vect_text.shape)
print(vect_text)


# In[15]:


idf=vect.idf_


# In[16]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(dd['covid'])
print(dd['sinopsis'])


# Dapat terlihat bahwa berdasarkan **nilai idf** , **'covid'** adalah kata **paling sering** sedangkan **'sinopsis'** **paling jarang** muncul di antara berita.

# ### LSA (Latent Semantic Analysis)

# **LSA pada dasarnya adalah dekomposisi nilai tunggal.**
# 
# **SVD menguraikan DTM asli menjadi tiga matriks S=U.(sigma).(V.T). Di sini matriks U menunjukkan matriks topik-dokumen sedangkan (V) adalah matriks istilah-topik.**
# 
# **Setiap baris dari matriks U(matriks term dokumen) adalah representasi vektor dari dokumen yang bersangkutan. Panjang vektor ini adalah jumlah topik yang diinginkan. Representasi vektor untuk suku-suku dalam data kami dapat ditemukan dalam matriks V (matriks istilah-topik).**
# 
# Jadi, SVD memberi kita vektor untuk setiap dokumen dan istilah dalam data kita. Panjang setiap vektor adalah k. **Kemudian kita dapat menggunakan vektor-vektor ini untuk menemukan kata dan dokumen serupa menggunakan metode kesamaan kosinus.**
# 
# Kita dapat menggunakan fungsi truncatedSVD untuk mengimplementasikan LSA. Parameter n_components adalah jumlah topik yang ingin kita ekstrak.
# Model tersebut kemudian di fit dan ditransformasikan pada hasil yang diberikan oleh vectorizer.
# 
# **Terakhir perhatikan bahwa LSA dan LSI (I untuk pengindeksan) adalah sama dan yang terakhir hanya terkadang digunakan dalam konteks pencarian informasi.**

# In[17]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[18]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[19]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)
  


# Mirip dengan dokumen lain kita bisa melakukan ini. Namun perhatikan bahwa nilai tidak menambah 1 seperti di LSA itu bukan kemungkinan topik dalam dokumen.

# In[20]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Sekarang bisa mendapatkan daftar kata-kata penting untuk masing-masing dari 10 topik seperti yang ditunjukkan.

# In[22]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")
         


# In[ ]:




