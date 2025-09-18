"""
Topic Modelling
"""

import streamlit as st
from pathlib import Path
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from collections import Counter
import pandas as pd

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

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_manager import DataManager

logger = logging.getLogger(__name__)

class TopicModelingAgent:
    """Final polished topic modeling agent"""

    def __init__(self):
        self.data_manager = DataManager()
        self.current_bank = st.session_state.get('current_bank')
        self.base_path = Path(".")
        
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
            
    def clear_memory(self):
        """Clear PyTorch GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("GPU memory cleared")
            
    def fix_question_answer_words(self,transcript):
        """Fix PDF read issues with split words"""
        # Concatenate letters in question/questions when the letters are split
        transcript = re.sub(r'\bq\s*u\s*e\s*s\s*t\s*i\s*o\s*n\s*s?\b', 'questions', transcript, flags=re.IGNORECASE)
        transcript = re.sub(r'\ba\s*n\s*s\s*w\s*e\s*r\s*s?\b', 'answers', transcript, flags=re.IGNORECASE)

        # Fix other common PDF parsing issues
        transcript = re.sub(r'\s+', ' ', transcript)  # Multiple spaces to single
        transcript = re.sub(r'\n+', '\n', transcript)  # Multiple newlines to single

        return transcript
    
    def read_transcript(self):    
    
        # import requests
        # url = "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/quarterly-earnings/2025/2nd-quarter/jpm-2q25-earnings-call-transcript.pdf"
        # response = requests.get(url)
        # with open("jpm_q2_2025.pdf", "wb") as f:
        #     f.write(response.content)
        # print("PDF downloaded!")    
        # file_path = "jpm_q2_2025.pdf"
        
        # """Read PDF transcript and return as string"""
        try:
        #     reader = PdfReader(file_path)
        #     transcript = ""

        #     for page in reader.pages:
        #         text = page.extract_text()
        #         if text:
        #             transcript += text + "\n"

            # Fix common PDF reading issues
            print("before read fix_question_answer_words")
            transcript = self.fix_question_answer_words(st.session_state.document_data)

            print(f"Successfully read transcript from ")
            print(f"Transcript length: {len(transcript)} characters")

            return transcript

        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return None

    #Define a function to extract only the Q&A secition of the transcripts
    def extract_qa(self,transcript):

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
    def preprocess_transcript(self,transcript, lemma_model="en_core_web_sm", group_size=1):

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

    def topic_model(self,transcript, model_name, max_topics=None, max_words_per_topic=10, seed_topic_list=None):
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
        
    def main_run(self):
        """Main execution function"""

        print("=== Financial Transcript Topic Modeling ===\n")

        # Check GPU availability
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name()}")
        else:
            print("GPU not available, using CPU")

        # File path - UPDATE THIS TO YOUR PDF FILE PATH
        # file_path = "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/quarterly-earnings/2025/2nd-quarter/jpm-2q25-earnings-call-transcript.pdf"  # Change this to your actual PDF file path

        # if not os.path.exists(file_path):
        #     print(f"Error: File {file_path} not found!")
        #     print("Please update the file_path variable with the correct path to your PDF file.")
        #     return

        # Read transcript
        print("\n=== Reading Transcript ===")
        # transcript = self.read_transcript(file_path)   
        transcript = self.read_transcript()
        # print(transcript)
        if not transcript:
            return

        # Extract Q&A section
        print("\n=== Extracting Q&A Section ===")
        qa_section = self.extract_qa(transcript)
        # print(qa_section)
        
        # Preprocess transcript
        
        
        text = self.get_available_text(self.current_bank)
        print(text)
        if text == "" :
            print("\n=== Preprocessing Transcript ===")
            clean_transcript = self.preprocess_transcript(qa_section, lemma_model="en_core_web_sm", group_size=3)
            self.save_preporcess_text(self.current_bank, clean_transcript)
        else:
            clean_transcript = text
            
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
            self.clear_memory()  # Clear GPU memory between models

            try:
                topics = self.topic_model(
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
                
        st.session_state.topic_results = topic_results



    def run(self):
        
        
        """Run polished topic modeling with correct order"""
        st.subheader("🎯 Topic Modeling with Visualizations")

        if not self.current_bank:
            st.error("❌ No bank selected. Please go to Bank Selection tab first.")
            return

        # Check prerequisites
        if 'document_data' not in st.session_state:
            st.warning("⚠️ Please process a document first in the Preprocessing tab")
            return

        bank_info = self.data_manager.get_bank_info(self.current_bank)
        st.info(f"**Analyzing:** {bank_info['name'] if bank_info else self.current_bank}")

        # 1. Analysis section FIRST
        st.markdown("### 🎯 Run Topic Analysis")

        available_plots = self.data_manager.get_available_plots(self.current_bank)
        if available_plots:
            st.warning("⚠️ Running new analysis will overwrite previous topic plots")

        if st.button("🎯 Run Topic Analysis", type="primary", use_container_width=True, key="run_topic_analysis_polished"):
            self._run_topic_analysis()

        # 2. Current Plots section SECOND (after running analysis)
        if 'topic_results' in st.session_state:
            st.markdown("---")
            self._display_current_plots()

        # 3. FIXED: Previous Analysis section THIRD (only one instance, no duplicates)
        if available_plots:
            st.markdown("---")
            self._show_previous_analysis_section(available_plots)

        # 4. Analysis results LAST
        if 'topic_results' in st.session_state:
            st.markdown("---")
            self._display_current_results()

    def _show_previous_analysis_section(self, available_plots: Dict):
        """Show previous analysis plots section - SINGLE INSTANCE ONLY"""
        st.markdown("### 📂 Previous Analysis Plots")

        if st.button("📊 View Previous Plots", type="secondary", use_container_width=True, key="view_topic_plots_polished"):
            self._show_previous_plots(available_plots)

    def _show_previous_plots(self, available_plots: Dict):
        """Display previous topic modeling plots"""
        topic_plots = {}
        for plot_name, plot_files in available_plots.items():
            if any(keyword in plot_name.lower() for keyword in ['topic', 'word', 'frequency', 'analysis', 'cloud']):
                topic_plots[plot_name] = plot_files

        if topic_plots:
            cols = st.columns(2)
            for i, (plot_name, plot_files) in enumerate(topic_plots.items()):
                with cols[i % 2]:
                    st.markdown(f"**{plot_name.replace('_', ' ').title()}**")
                    if plot_files:
                        try:
                            latest_plot = plot_files[0]
                            st.image(latest_plot, caption=f"Previous: {plot_name}", use_container_width=True)
                        except Exception as e:
                            st.write(f"Plot file: {Path(latest_plot).name}")
        else:
            st.info("No previous topic analysis plots found.")

    def _run_topic_analysis(self):
        """Run actual topic analysis"""
        try:
            with st.spinner("🎯 Running topic analysis..."):
                print("RUNNN")      

                self.main_run()

                print("after main run")
                # Save results to session
                results = st.session_state.topic_results

                self.save_csv_results(self.current_bank, 'topic_results', results)

                # Generate and save ONLY 2 plots
                self._generate_and_save_plots(results)

                st.success("🎯 Topic analysis completed and saved!")
                st.rerun()

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Topic analysis error: {e}")

    def get_available_text(self, bank_key: str) -> str:
        print("get available txt")
        try:
            data_path = self.base_path / "data" / "banks" / bank_key
            
            if data_path.exists():
                
                filename = f"{bank_key}_preprocessed.txt"
                filepath = data_path / filename
                print(filepath)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read().splitlines()
                
                return text
        except Exception as e:
            print(f"Error getting available texts: {e}")
        return ""
    
    def save_preporcess_text(self, bank_key: str, text: str) -> bool:
        try:
            print("texts stored")
            data_path = self.base_path / "data" / "banks" / bank_key
            data_path.mkdir(parents=True, exist_ok=True)

            filename = f"{bank_key}_preprocessed.txt"
            filepath = data_path / filename
            
            print("filepath")
          
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f.write("\n".join(text)))
            print(f"\nResults saved to {filepath}")

            return True
        except Exception as e:
            print(f"Error saving analysis results: {e}")
            return False

    def save_csv_results(self, bank_key: str, data_type: str, topic_results: Dict[str, Any]) -> bool:
        try:
            print("CSV Results")
            data_path = self.base_path / "data" / "banks" / bank_key
            data_path.mkdir(parents=True, exist_ok=True)

            filename = f"{data_type}_latest.csv"
            filepath = data_path / filename
            
            print("filepath")

            # Also save timestamped version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_filename = f"{data_type}_{timestamp}.json"
            timestamped_filepath = data_path / timestamped_filename
            
                # Create results DataFrame
            print("\n=== Creating Results Summary ===")

            max_topics = max(len(topics) for topics in topic_results.values() if topics)

            # Create DataFrame
            df_data = {}
            for model_name, topics in topic_results.items():
                model_short = model_name.split('/')[-1] + " Top Words"
                df_data[model_short] = topics + [None] * (max_topics - len(topics))

            topic_output = pd.DataFrame(df_data)
            
            st.session_state.topic_output = topic_output
    
                # Display results
            pd.set_option("display.max_colwidth", None)
            print("\n=== TOPIC MODELING RESULTS ===")
            print(topic_output.to_string())

            topic_output.to_csv(filepath, index=True)
            print(f"\nResults saved to {filepath}")
            
            st.markdown("### Analysis results")
            st.markdown(topic_output)   # renders as a nice table
            return True
        except Exception as e:
            print(f"Error saving analysis results: {e}")
            return False
        
    def _perform_real_topic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform real topic analysis"""
        return
   
        # return {
        #     'total_topics': len(topics_data),
        #     'topics_data': topics_data,
        #     'word_frequency': top_words,
        #     'wordcloud_data': wordcloud_data,
        #     'analysis_timestamp': datetime.now().isoformat(),
        #     'bank_name': self.current_bank,
        #     'text_length': len(text),
        #     'total_keyword_mentions': total_keywords
        # }

    def _generate_and_save_plots(self, results: Dict[str, Any]):
        """Generate and save ONLY 2 plots"""
        try:
            # 1. Word frequency bar chart

            print("before save_plot Word Frequncy")
            fig1 = self._create_word_frequency_chart(results)
            print("after save_plot Word Frequncy")
            self.data_manager.save_plot(self.current_bank, 'word_frequency_analysis', fig1, 'matplotlib')
            plt.close(fig1)

        # 2. Word cloud plot

            print("before save_plot Word ")
            fig2 = self._create_word_cloud_chart(results)
            print("after save_plot Word ")
            self.data_manager.save_plot(self.current_bank, 'word_cloud_visualization', fig2, 'matplotlib')
            plt.close(fig2)

            logger.info(f"Generated and saved 2 topic analysis plots for {self.current_bank}")

        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            st.warning(f"Analysis completed but plot generation had issues: {str(e)}")

    def _create_word_frequency_chart(self, topic_results: Dict[str, int]):
        """Create word frequency bar chart"""
        print("Create word Frequency")
        fig, ax = plt.subplots(figsize=(12, 8))

        max_topics = max(len(topics) for topics in topic_results.values() if topics)

        # Create DataFrame
        df_data = {}
        for model_name, topics in topic_results.items():
            model_short = model_name.split('/')[-1] + " Top Words"
            df_data[model_short] = topics + [None] * (max_topics - len(topics))

        topic_output = pd.DataFrame(df_data)
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
        # plt.figure(figsize=(12, 6))
        plt.bar(word_stats["word"], word_stats["total_count"])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Word")
        plt.ylabel("Count")
        plt.title("Top Words Frequency Across All Models")

        plt.tight_layout()
        return fig

    def _create_word_cloud_chart(self, topic_results: Dict[str, int]):
        """Create word cloud visualization"""
        print("Create word cloud visualization")
        fig, ax = plt.subplots(figsize=(12, 8))   
        """Create word frequency bar chart"""
        max_topics = max(len(topics) for topics in topic_results.values() if topics)

        # Create DataFrame
        df_data = {}
        for model_name, topics in topic_results.items():
            model_short = model_name.split('/')[-1] + " Top Words"
            df_data[model_short] = topics + [None] * (max_topics - len(topics))

        topic_output = pd.DataFrame(df_data)
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
        
         # Create word cloud
        if not word_stats.empty:
            freq_dict = dict(zip(word_stats["word"], word_stats["total_count"]))
            print("wewewwe")
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color="white",
                max_words=100,
                colormap='viridis'
            ).generate_from_frequencies(freq_dict)


            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Word Cloud - Top Topic Words")
            plt.tight_layout()

        print("\n=== Analysis Complete ===")
        print("Generated files:")
        print("- topic_modeling_results.csv")
        print("- word_frequency.png") 
        print("- wordcloud.png")
        return fig
    
    def _display_current_plots(self):
        """Display current analysis plots section"""
        st.markdown("### 📊 Current Plots")
        st.success("✅ **New analysis plots generated!**")

        # Get most recent plots
        available_plots = self.data_manager.get_available_plots(self.current_bank)

        if available_plots:
            topic_plots = {}
            for plot_name, plot_files in available_plots.items():
                if any(keyword in plot_name.lower() for keyword in ['word_frequency', 'word_cloud']):
                    topic_plots[plot_name] = plot_files

            if topic_plots:
                cols = st.columns(2)
                for i, (plot_name, plot_files) in enumerate(topic_plots.items()):
                    with cols[i % 2]:
                        st.markdown(f"**{plot_name.replace('_', ' ').title()}**")
                        if plot_files:
                            try:
                                latest_plot = plot_files[0]  # Most recent
                                st.image(latest_plot, caption=f"Current: {plot_name}", use_container_width=True)
                            except Exception as e:
                                st.write(f"Plot file: {Path(latest_plot).name}")
        else:
            st.info("Run analysis to generate current plots.")

    def _display_current_results(self):
        """Display current analysis results - streamlined"""
        results = st.session_state.topic_results

        st.markdown("### 📊 Topic Analysis Results")

        # Key metrics (streamlined)
        col1,  = st.columns(1)

        with col1:
            topic_df = pd.DataFrame(st.session_state.topic_output)
            st.dataframe(topic_df, use_container_width=True)

        # # Topics overview
        # topics_data = results.get('topics_data', [])
        # if topics_data:
        #     st.markdown("#### 🏷️ Financial Topics Identified")

        #     # Create DataFrame for better display
        #     topics_df = pd.DataFrame([{
        #         'Topic': topic['name'],
        #         'Mentions': topic['keyword_count'],
        #         'Percentage': f"{topic['percentage']:.1f}%"
        #     } for topic in topics_data[:8]])  # Show top 8

        #     st.dataframe(topics_df, use_container_width=True)

        #     # Condensed topic details
        #     with st.expander("🔍 Topic Details", expanded=False):
        #         for topic in topics_data[:3]:  # Show top 3
        #             st.markdown(f"**📊 {topic['name']}:** {', '.join(topic['keywords'])} ({topic['keyword_count']} mentions)")

def run_topic_modeling_agent():
    """Entry point for topic modeling agent"""
    agent = TopicModelingAgent()
    agent.run()