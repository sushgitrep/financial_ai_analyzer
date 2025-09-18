#!/usr/bin/env python3
"""
Topic Modelling Script - Converted from Jupyter Notebook
This script performs financial text topic modeling using various embedding models.

Usage: python topic_modeling_standalone.py

Make sure to:
1. Install requirements: pip install -r requirements.txt
2. Download spacy model: python -m spacy download en_core_web_sm  
3. Place your PDF file in the same directory or update the file path
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Basic Libraries
import pandas as pd
import numpy as np
import re
from collections import Counter

# Data Visualization  
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# PDF Reader
from pypdf import PdfReader
import string

# NLP Libraries
import nltk
import spacy
from nltk.corpus import words, stopwords, names
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# Topic Modeling
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ML Libraries
import torch
from transformers import AutoTokenizer, AutoModel

# Download NLTK data if not already present
def setup_nltk():
    """Download required NLTK data"""
    try:
        nltk.data.find('corpora/words')
        nltk.data.find('corpora/stopwords') 
        nltk.data.find('corpora/names')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('words', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('names', quiet=True)
        print("NLTK data downloaded successfully!")

# Initialize NLTK
setup_nltk()

def clear_memory():
    """Clear PyTorch GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("GPU memory cleared")

def fix_question_answer_words(transcript):
    """Fix PDF read issues with split words"""
    # Concatenate letters in question/questions when the letters are split
    transcript = re.sub(r'\bq\s*u\s*e\s*s\s*t\s*i\s*o\s*n\s*s?\b', 'questions', transcript, flags=re.IGNORECASE)
    transcript = re.sub(r'\ba\s*n\s*s\s*w\s*e\s*r\s*s?\b', 'answers', transcript, flags=re.IGNORECASE)

    # Fix other common PDF parsing issues
    transcript = re.sub(r'\s+', ' ', transcript)  # Multiple spaces to single
    transcript = re.sub(r'\n+', '\n', transcript)  # Multiple newlines to single

    return transcript

def read_transcript(file_path):
    
    
    import requests
    url = "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/quarterly-earnings/2025/2nd-quarter/jpm-2q25-earnings-call-transcript.pdf"
    response = requests.get(url)
    with open("jpm_q2_2025.pdf", "wb") as f:
        f.write(response.content)
    print("PDF downloaded!")    
    file_path = "jpm_q2_2025.pdf"
    
    """Read PDF transcript and return as string"""
    try:
        reader = PdfReader(file_path)
        transcript = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                transcript += text + "\n"

        # Fix common PDF reading issues
        transcript = fix_question_answer_words(transcript)

        print(f"Successfully read transcript from {file_path}")
        print(f"Transcript length: {len(transcript)} characters")

        return transcript

    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

#Define a function to extract only the Q&A secition of the transcripts
def extract_qa(transcript):

    #Lowercase the transcript and skip the first 2000 words (attempt to avoid situation where early remarks are made about the Q&A session)
    transcript_lower = transcript.lower()[2000:]

    #Define lowercase Q&A markers
    qa_markers = ["questions & answers",
                  "questions and answers",
                  "question & answer",
                  "question and answer",
                  "question & answers",
                  "question and answers",
                  "questions & answer",
                  "questions and answer",
                  "question-and-answer",
                  "question-and-answers",
                  "questions-and-answer",
                  "questions-and-answers",
                  "question-&-answer",
                  "question-&-answers",
                  "questions-&-answer",
                  "questions-&-answers",
                  "q&a",
                  "q & a",
                  "q and a"]

    #Combine the Q&A markers into a regex
    pattern = re.compile('|'.join(qa_markers))

    #Find the first occurrence of the pattern
    match = pattern.search(transcript_lower)

    if match:

        #Return original transcript from the match index onward
        return transcript_lower[match.start():]

    else:
        #Let the user know that the Q&A section was not found
        print("No match found!")
        return ""

#Define a function to preprocess the transcript
def preprocess_transcript(transcript, lemma_model="en_core_web_sm", group_size=1):

    #Break down the transcript in to individual sentences
    sentences = sent_tokenize(transcript)

    #Define a function to clean the sentences
    def clean_sentence(sentence): 
        
        #Define a set of manually selected words to remove
        additional_stop_words = ["thank", "thanks", "you", "operator", "asked", "answer",
        "answered", "next", "please", "line", "call", "open", "close", "jpmorgan", "jpmorganchase",
            "jp morgan", "jpmorgan chase", "jp morgan chase", "chief", "dimon", "jamie", "jpm", "jp", "morgan",
            "executive", "chairman", "question", "questions", "answers", "fool", "motley",
            "so", "see", "think", "ok", "okay", "that", "yeah", "officer", "analyst", "jeremy", "llc", "thats",
            "morning", "afternoon", "hello", "hi", "said", "get", "they", "theyve", "would", "they", "theyre",
            "stephen", "steve", "james", "jim", "alex", "alex", "charlotte", "charlie", "olivia", "liv", "emma", "em",
            "liam", "noah", "isabella", "bella", "mia", "sophia", "sophie", "jack", "john", "henry", "harry", "amelia", "amy",
            "ethan", "lucas", "harper", "elijah", "eli", "ava", "william", "will", "oliver", "ollie", "emily", "emma",
            "logan", "chloe", "daniel", "dan", "sarah", "matthew", "matt", "lily", "nathan", "nate", "grace", "samuel", "sam",
            "zoe", "sebastian", "bastian", "ella", "jacob", "jake", "scarlett", "michael", "mike", "hannah", "alexander", "alex",
            "layla", "ryan", "victoria", "vicky", "joseph", "joe", "nora", "david", "leah", "anthony", "tony", "audrey",
            "andrew", "andy", "caroline", "tyler", "samantha", "sam", "christopher", "chris", "madison", "nicholas", "nick",
            "morgan", "jonathan", "jon", "abigail", "abby", "aaron", "lillian", "brandon", "elyse", "kate", "patrick", "savannah",
            "mark", "zoey", "joshua", "josh", "hailey", "dylan", "addison", "caleb", "aubrey", "adam", "brooklyn", "zachary", "zac",
            "peyton", "cameron", "riley", "blake", "aubree", "tyson", "carter", "jason", "claire", "alexis", "madeline", "julian", "stella",
            "ian", "kevin", "isla", "evan", "lila", "lucy", "mason", "violet", "owen", "natalie", "mila", "hunter", "elena",
            "colin", "luke", "penelope", "ryder", "damian", "micah", "adrian", "madelyn", "jonah", "gabriel", "noel", "sara",
            "leo", "elise", "nina", "lola", "julia", "brian", "carla", "cody", "diana", "eddie", "fiona", "frank", "gwen",
            "harold", "iris", "jenny", "jackson", "karen", "keith", "linda", "naomi", "pamela", "quinn", "paul", "rachael",
            "susan", "taylor", "uma", "vanessa", "steven", "wendy", "thomas", "yara", "timothy", "victor", "vincent", "wade",
            "daisy", "xander", "elizabeth", "liz", "faith", "aarav", "arya", "aditya", "anaya", "aiden", "aya", "benjamin", "charles",
            "clara", "danica", "eva", "felix", "gemma", "harrison", "isaac", "jasmine", "kyle", "katherine", "luna", "ophelia",
            "quincy", "theodore", "xena", "yusuf", "yasmin", "zane", "lizzie", "joey", "jimmy", "stevie", "ellie", "meg", "annie", "danny", "cathy", "beth", "eric", "howell",
            "could", "casey", "going", "like", "well", "maybe", "us", "know", "don’t", "we", "weve", "already", "casey", "fitzgibbon", "mcgratty", "take", "hey", "dave", "wyremski", "good", "well",
            "jared","erin", "lehman", "cfo", "inc", "callan", "thomson", "not", "really", "may", "without", "brother", "brothers", "leh", "officer", "financial", "bank", "thing",
            "talk", "happen", "come", "company", "look", "bit", "true", "say", "chase", "want", "cos", "moment", "make", "go", "one", "lynch", "anything", "always", "erika", "people",
            "tell", "feed", "much", "also", "lot", "kind", "need", "hear", "datum", "obviously", "market", "change", "client", "ken", "point", "manage", "probably", "give", "bind",
            "advertisement", "transcript", "earnings", "earning", "continue", "different", "follow", "marc", "provide", "becker", "beck",
            "quarter", "level", "time", "expect","start", "half", "yep", "great", "haire", "today", "everybody", "perspective", "wood", "year", "president", "chief", "executive", "officer", "ceo", "back",
            "side", "billion", "million", "new", "first", "across", "evening", "everyone","join", "still", "right", "add", "wonder", "around", "way", "part", "high", "bottom",
            "important", "month", "little", "ear", "ex", "proceed", "page", "seaport", "banking", "bank", "actually", "guess", "reason", "yes", "mean"]

        #Define full list of names to add to stop words
        male_names = set(names.words('male.txt'))
        female_names = set(names.words('female.txt'))
        all_names = male_names.union(female_names)

        #Add the manually defined stopwords and names to the standard list
        stop_words = stopwords.words("english") + additional_stop_words + list(all_names)

        #Define all English words
        english_words = set(words.words())

        #Define the set of punctuation to remove
        exclude = set(symb for symb in string.punctuation if symb not in "$%") | set(["’", "–"])  
      
      #Convert to lowercase
        cap_free = sentence.lower()

        #Remove stop words
        stop_free = " ".join([word for word in cap_free.split() if word not in stop_words])

        #Remove URLs
        url_free = re.sub(r'\b(?:https?|www|httpswww)\S*\b', '', stop_free)

        #Remove punctuation
        punc_free = ''.join(char for char in url_free if char not in exclude)

        #Remove numbers
        num_free = re.sub(r'\d+', '', punc_free)

        #Lemmatize the tokens
        nlp = spacy.load(lemma_model, disable=["ner", "parser"])
        doc = nlp(num_free)
        lemma_free = " ".join([token.lemma_ for token in doc])

        #Remove stop words again
        stop_free_2 = " ".join([word for word in lemma_free.split() if word not in stop_words])

        #Remove fake words
        sentence_clean = " ".join([word for word in stop_free_2.split() if word in english_words])

        return sentence_clean

    clean_transcript = [clean_sentence(sentence) for sentence in sentences]

    #Group sentences together
    if group_size > 1:
      grouped_sentences = []
      for i in range(0, len(clean_transcript), group_size):
          group = " ".join(clean_transcript[i:i+group_size])
          grouped_sentences.append(group)

      #Remove empty groups
      grouped_sentences = [group for group in grouped_sentences if group.strip() != ""]

      clean_transcript = grouped_sentences

    return clean_transcript

def topic_model(transcript, model_name, max_topics=None, max_words_per_topic=10, seed_topic_list=None):
    """Create topic model using specified embedding model"""

    try:
        print(f"Creating topic model with {model_name}...")

        # Create sentence transformer
        embedding_model = SentenceTransformer(model_name)

        # Create BERTopic model
        topic_model_obj = BERTopic(
            embedding_model=embedding_model,
            nr_topics=max_topics,
            seed_topic_list=seed_topic_list,
            verbose=True
        )

        # Fit the model
        topics, probs = topic_model_obj.fit_transform(transcript)

        # Get topic words
        topic_words = topic_model_obj.get_topics()

        # Extract top words for each topic
        top_words_per_topic = []
        for topic_id in sorted(topic_words.keys()):
            if topic_id != -1:  # Skip outlier topic
                words = [word for word, score in topic_words[topic_id][:max_words_per_topic]]
                top_words_per_topic.append(words)

        print(f"Model completed. Found {len(top_words_per_topic)} topics.")
        return top_words_per_topic

    except Exception as e:
        print(f"Error creating topic model with {model_name}: {str(e)}")
        return []

def main():
    """Main execution function"""

    print("=== Financial Transcript Topic Modeling ===\n")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
    else:
        print("GPU not available, using CPU")

    # File path - UPDATE THIS TO YOUR PDF FILE PATH
    file_path = "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/quarterly-earnings/2025/2nd-quarter/jpm-2q25-earnings-call-transcript.pdf"  # Change this to your actual PDF file path

    # if not os.path.exists(file_path):
    #     print(f"Error: File {file_path} not found!")
    #     print("Please update the file_path variable with the correct path to your PDF file.")
    #     return

    # Read transcript
    print("\n=== Reading Transcript ===")
    transcript = read_transcript(file_path)   
    # print(transcript)
    if not transcript:
        return

    # Extract Q&A section
    print("\n=== Extracting Q&A Section ===")
    qa_section = extract_qa(transcript)
    # print(qa_section)
    
    # Preprocess transcript
    print("\n=== Preprocessing Transcript ===")
    clean_transcript = preprocess_transcript(qa_section, lemma_model="en_core_web_sm", group_size=3)
    print(clean_transcript)
    if not clean_transcript:
        return

    # Create seed topic list for financial transcripts
    seed_topic_list = [
        # Credit risk & asset quality
        ["credit risk", "loan losses", "provisioning", "non performing loans", "defaults", "counterparty credit", "credit quality", "impairments", "charge offs", "allowances"],

        # Interest rates & monetary policy  
        ["interest rates", "fed funds", "monetary policy", "rate cuts", "rate hikes", "yield curve", "duration risk", "rate environment", "fed policy", "central bank"],

        # Capital & liquidity
        ["capital ratios", "tier capital", "regulatory capital", "liquidity", "deposits", "funding", "capital adequacy", "stress tests", "buffer", "requirements"],

        # Economic outlook & conditions
        ["economic outlook", "recession", "inflation", "gdp growth", "unemployment", "consumer spending", "business conditions", "economic indicators", "market conditions", "outlook"],

        # Banking operations & performance
        ["net interest margin", "efficiency ratio", "operating leverage", "expenses", "revenue", "profitability", "return on assets", "return on equity", "cost of funds", "fee income"]
    ]

    # Model list - financial domain models
    model_list = [
        "tyuan73/finetuned-modernbert-finance-large",
        "BAAI/bge-large-en-v1.5", 
        "ohsuz/k-finance-sentence-transformer",
        "nickmuchi/setfit-finetuned-financial-text-classification",
        "sentence-transformers/all-MiniLM-L6-v2",
        "shail-2512/nomic-embed-financial-matryoshka",
        "mukaj/fin-mpnet-base"
    ]

    # Run topic modeling for each model
    print("\n=== Running Topic Modeling ===")
    topic_results = {}

    for i, model_name in enumerate(model_list, 1):
        print(f"\n[{i}/{len(model_list)}] Processing {model_name}...")
        clear_memory()  # Clear GPU memory between models

        try:
            topics = topic_model(
                clean_transcript, 
                model_name, 
                max_topics=6,  # Limit to 6 topics
                max_words_per_topic=5,
                seed_topic_list=seed_topic_list
            )

            topic_results[model_name] = topics

        except Exception as e:
            print(f"Failed to process {model_name}: {str(e)}")
            topic_results[model_name] = []

    # Create results DataFrame
    print("\n=== Creating Results Summary ===")
    print(topics)
    max_topics = max(len(topics) for topics in topic_results.values() if topics)

    # Create DataFrame
    df_data = {}
    for model_name, topics in topic_results.items():
        model_short = model_name.split('/')[-1] + " Top Words"
        df_data[model_short] = topics + [None] * (max_topics - len(topics))

    topic_output = pd.DataFrame(df_data)

    # Display results
    pd.set_option("display.max_colwidth", None)
    print("\n=== TOPIC MODELING RESULTS ===")
    print(topic_output.to_string())

    # Save results to CSV
    output_file = "topic_modeling_results.csv"
    topic_output.to_csv(output_file, index=True)
    print(f"\nResults saved to {output_file}")

    # Word frequency analysis
    print("\n=== Word Frequency Analysis ===")
    word_counter = Counter()

    # Count words across all models (only non-empty cells)
    min_cols = topic_output.shape[1] * 0.5

    for value in topic_output.values.flatten():
        if value is not None and isinstance(value, list):
            for word in value:
                if isinstance(word, str):
                    word_counter[word] += 1

    # Get top words
    top_words = word_counter.most_common(20)
    word_stats = pd.DataFrame(top_words, columns=["word", "total_count"])

    print("\nTop 20 most frequent words across all models:")
    print(word_stats.to_string(index=False))

    # Create word frequency plot
    plt.figure(figsize=(12, 6))
    plt.bar(word_stats["word"], word_stats["total_count"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Word")
    plt.ylabel("Count")
    plt.title("Top Words Frequency Across All Models")
    plt.tight_layout()
    plt.savefig("word_frequency.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Create word cloud
    if not word_stats.empty:
        freq_dict = dict(zip(word_stats["word"], word_stats["total_count"]))

        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color="white",
            max_words=100,
            colormap='viridis'
        ).generate_from_frequencies(freq_dict)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud - Top Topic Words")
        plt.tight_layout()
        plt.savefig("wordcloud.png", dpi=300, bbox_inches='tight')
        plt.show()

    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- topic_modeling_results.csv")
    print("- word_frequency.png") 
    print("- wordcloud.png")

if __name__ == "__main__":
    main()
