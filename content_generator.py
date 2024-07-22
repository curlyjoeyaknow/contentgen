import yaml
import spacy
import sqlite3
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModel
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import List, Dict, Any
import requests
import json
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from langdetect import detect
import random
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keybert import KeyBERT  # Replacing gensim.summarization
from sentence_transformers import SentenceTransformer  # For better text embeddings
from rank_bm25 import BM25Okapi  # For better keyword extraction


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class AIContentGenerator:
    def __init__(self, config_file: str):
        try:
            with open(config_file, 'r') as file:
                self.config = yaml.safe_load(file)
                print("Loaded configuration:", self.config)  # Add this line
                self.perplexity_api_key = self.config['perplexity']['api_key']
                self.perplexity_base_url = "https://api.perplexity.ai"
                self.nlp = spacy.load("en_core_web_sm")
                self.word2vec_model = KeyedVectors.load_word2vec_format(
                    self.config['word2vec']['model_path'], 
                    binary=True
                )
                self.setup_models()
                self.setup_data_source()
                
                self.db_conn = sqlite3.connect(self.config['database']['path'])
                self.cursor = self.db_conn.cursor()
                self.create_feedback_table()
                
                self.personas = self._load_personas()
                self.stop_words = set(stopwords.words('english'))
                
                # New models for improved performance
                self.keybert_model = KeyBERT()
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_file}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required configuration: {e}")

    def _load_config(self, config_file: str) -> dict:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def setup_models(self):
        print("Setting up models...")
        try:
            model_name = self.config['models']['content_generation']['model']
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print(f"Models set up successfully: {model_name}")
        except Exception as e:
            print(f"Error setting up models: {e}")
            raise
    def setup_data_source(self):
        data_source = self.config['data_source']
        if data_source['type'] == 'csv':
            with open(data_source['path'], 'r') as file:
                reader = csv.reader(file)
                self.data = [row[0] for row in reader]
        elif data_source['type'] == 'api':
            response = requests.get(data_source['url'])
            self.data = response.json()
        else:
            raise ValueError(f"Unsupported data source type: {data_source['type']}")

    def get_model_name(self, language):
        return self.config['models'][language]['model']

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message['content'].strip()
        except openai.Error as e:
            raise openai.Error(f"Error generating text: {e}")

    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
    # Using KeyBERT for keyword extraction
        keywords = self.keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', 
        use_maxsum=True, nr_candidates=20, top_n=num_keywords)
        return [keyword for keyword, _ in keywords]

    def _get_word_embeddings(self, tokens: List[str]) -> Dict[str, np.array]:
        # Using SentenceTransformer for better word embeddings
        embeddings = self.sentence_transformer.encode(tokens)
        return {token: embedding for token, embedding in zip(tokens, embeddings)}

    def perplexity_request(self, endpoint: str, payload: Dict) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(f"{self.perplexity_base_url}/{endpoint}", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def web_search(self, query: str, keywords: List[str]) -> List[Dict]:
        optimized_query = f"{query} {' '.join(keywords)}"
        payload = {
            "query": optimized_query,
            "max_results": self.config['research']['max_results']
        }
        response = self.perplexity_request("search", payload)
        results = response.get('results', [])
        
        # Use BM25 to rank results based on relevance to keywords
        corpus = [result['summary'] for result in results]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(' '.join(keywords))
        
        ranked_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [result for result, _ in ranked_results]

    def fact_check(self, claim: str) -> Dict:
        payload = {
            "query": f"Fact check: {claim}"
        }
        response = self.perplexity_request("factcheck", payload)
        return response

    def generate_content(self, prompt: str) -> str:
        keywords = self.extract_keywords(prompt)
        persona = random.choice(self.personas)
        search_results = self.web_search(prompt, keywords)
        fact_check_results = [self.fact_check(result['summary']) for result in search_results[:2]]
        comprehensive_prompt = self.create_structured_prompt(prompt, persona, search_results, fact_check_results, keywords)
        
        if self.config['use_gpt4']:
            generated_content = self.generate_with_gpt4(comprehensive_prompt, persona)
        else:
            model_info = self.models[self.config['multilingual']['default_language']]
            tokenizer = model_info['tokenizer']
            model = model_info['model']
            
            input_ids = tokenizer.encode(comprehensive_prompt, return_tensors='pt')
            output = model.generate(input_ids, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2)
            generated_content = tokenizer.decode(output[0], skip_special_tokens=True)
        
        optimized_content = self.optimize_for_seo(generated_content, keywords)
        return optimized_content

    def generate_with_gpt4(self, comprehensive_prompt, persona):
        response = openai.Completion.create(
            engine="text-davinci-002",  # or the appropriate GPT-4 engine
            prompt=comprehensive_prompt,
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()

    def create_structured_prompt(self, base_prompt: str, persona: Dict, search_results: List[Dict], fact_check_results: List[Dict], keywords: List[str]) -> str:
        structure = """
        Create a well-structured article based on the following topic, written in the style of {author}. Include:
        1. An attention-grabbing headline that {author} might use
        2. An engaging introduction that hooks the reader in the voice of {author}
        3. A clear statement of the main topic and why it's relevant or important, presented as {author} would
        4. Main body with descriptive subheadings (at least 3), using {author}'s writing style and {tone}
        5. Use of examples, anecdotes, or case studies that {author} might include to illustrate key points
        6. Inclusion of relevant data or statistics, as {author} would present them
        7. A strong conclusion with key takeaways, written as {author} would summarize
        8. A call-to-action or next steps for the reader, as {author} might suggest

        Topic: {prompt}
        
        Subject Matter: {subject_matter}

        Incorporate the following research findings:
        {search_results}

        Consider these fact-check results:
        {fact_checks}

        Optimize the content for the following keywords:
        {keywords}

        Ensure the content is engaging, uses appropriate transitions between ideas, and maintains consistency with {author}'s style and {tone}. 
        Use formatting and structures that {author} typically employs in their writing.
        Develop the content as {author} would, focusing on the {subject_matter}.
        Suggest places where images, infographics, or videos could enhance the content by adding [IMAGE SUGGESTION] tags, considering what media {author} might use.
        Naturally incorporate the provided keywords throughout the content for SEO optimization.
        """
        search_results_text = "\n".join([f"- {result['title']}: {result['summary']}" for result in search_results])
        fact_checks_text = "\n".join([f"- Claim: {check['claim']}, Verdict: {check['verdict']}" for check in fact_check_results])
        keywords_text = ", ".join(keywords)
        
        return structure.format(
            prompt=base_prompt,
            author=persona['author'],
            tone=persona['tone'],
            subject_matter=persona['subject_matter'],
            search_results=search_results_text,
            fact_checks=fact_checks_text,
            keywords=keywords_text
        )

    def optimize_for_seo(self, content: str, keywords: List[str]) -> str:
        for keyword in keywords:
            if keyword.lower() not in content.lower():
                content += f"\n\nRelated topic: {keyword}"

        lines = content.split('\n')
        title = lines[0]
        first_para = lines[1] if len(lines) > 1 else ""

        if not any(keyword.lower() in title.lower() for keyword in keywords):
            title = f"{keywords[0].capitalize()}: {title}"

        if first_para and not any(keyword.lower() in first_para.lower() for keyword in keywords):
            first_para = f"{first_para} This article discusses {', '.join(keywords)}."

        lines[0] = title
        if len(lines) > 1:
            lines[1] = first_para

        return '\n'.join(lines)
    
    def assess_readability(self, text):
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        smog_index = textstat.smog_index(text)
    
        return {
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'flesch_reading_ease': flesch_reading_ease,
        'smog_index': smog_index
        }
    
    def post_process(self, text, language, keywords):
        if not self.config['post_processing']['enable']:
            return text

        text = re.sub(r'([.!?]){2,}', r'\1', text)
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        if language == 'en':
            text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda p: p.group(1) + p.group(2).upper(), text)
        elif language == 'de':
            text = re.sub(r'(^|[.!?]\s+)([a-zäöü])', lambda p: p.group(1) + p.group(2).upper(), text)

        readability_scores = self.assess_readability(text)
        if readability_scores['flesch_kincaid_grade'] > self.config['post_processing']['max_grade_level']:
            text = self.simplify_text(text)

        text = self.optimize_for_seo(text, keywords)
        text = self.ensure_consistent_style(text)
        text = self.add_transitions(text)

        return text.strip()

    def simplify_text(self, text):
        doc = self.nlp(text)
        simplified_sentences = []
        for sent in doc.sents:
            if len(sent) > 20:
                simplified_sentences.extend(self.split_sentence(sent))
            else:
                simplified_sentences.append(sent.text)
        return ' '.join(simplified_sentences)

    def split_sentence(self, sentence):
        chunks = []
        current_chunk = []
        for token in sentence:
            current_chunk.append(token.text)
            if token.pos_ in ['VERB', 'NOUN'] and len(current_chunk) > 10:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def ensure_consistent_style(self, text):
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        avg_sent_len = sum(len(sent) for sent in sentences) / len(sentences)
        
        adjusted_sentences = []
        for sent in sentences:
            if len(sent) > 1.5 * avg_sent_len:
                adjusted_sentences.extend(self.split_sentence(sent))
            elif len(sent) < 0.5 * avg_sent_len:
                if len(adjusted_sentences) > 0:
                    adjusted_sentences[-1] = f"{adjusted_sentences[-1]} {sent.text}"
                else:
                    adjusted_sentences.append(sent.text)
            else:
                adjusted_sentences.append(sent.text)
        
        return ' '.join(adjusted_sentences)

    def add_transitions(self, text):
        transitions = [
            "Furthermore,", "Moreover,", "In addition,", "However,", "On the other hand,",
            "Consequently,", "As a result,", "Therefore,", "Thus,", "In contrast,"
        ]
        sentences = sent_tokenize(text)
        for i in range(1, len(sentences)):
            if random.random() < 0.3:  # 30% chance to add a transition
                sentences[i] = f"{random.choice(transitions)} {sentences[i]}"
        return ' '.join(sentences)

    def fact_check(self, text):
        facts = re.findall(r'\d+(?:\.\d+)?%|\d+ out of \d+|\d+th percentile', text)
        claims = []
        for fact in facts:
            claims.append(f"Claim: {fact} - Status: Unverified")
        return "\n".join(claims)
    def extract_title_and_content(self, text):
        lines = text.split('\n')
        title = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()
        return title, content

    def generate_alternative_title(self, original_title):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Generate an alternative headline for: {original_title}",
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()

    def _load_personas(self) -> List[Dict]:
        personas_file = self.config['personas']['file']
        with open(personas_file, 'r') as file:
            return yaml.safe_load(file)

    def create_feedback_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                article_id INTEGER PRIMARY KEY,
                rating INTEGER,
                comments TEXT
            )
        ''')
        self.db_conn.commit()

    def collect_feedback(self, article_id, rating, comments):
        self.cursor.execute('''
            INSERT OR REPLACE INTO feedback (article_id, rating, comments)
            VALUES (?, ?, ?)
        ''', (article_id, rating, comments))
        self.db_conn.commit()

    def get_feedback_data(self):
        self.cursor.execute('SELECT * FROM feedback')
        return self.cursor.fetchall()

    def fine_tune(self):
        feedback_data = self.get_feedback_data()
        if len(feedback_data) < 100:
            print("Not enough feedback data for fine-tuning.")
            return

        X = [[row[1]] for row in feedback_data]  # rating
        y = [1 if row[1] >= 4 else 0 for row in feedback_data]  # binary classification: good (>=4) or bad (<4)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Fine-tuning model accuracy: {accuracy}")

        self.quality_threshold = model.predict_proba([[3.5]])[0][1]  # probability of being "good" for a 3.5 rating

    def generate_and_process(self):
        output_file = self.config['output']['file']
        iterations = self.config['generation']['iterations']
        default_language = self.config['multilingual']['default_language']

        with open(output_file, 'w', encoding='utf-8') as file:
            for i in range(iterations):
                prompt = self.data[i % len(self.data)]
                prompt_language = detect(prompt)
                target_language = prompt_language if prompt_language in self.models else default_language
                persona = random.choice(self.personas)

                keywords = self.extract_keywords(prompt)

                generated_content = self.generate_content(prompt)
                processed_content = self.post_process(generated_content, target_language, keywords)
                fact_check_result = self.fact_check(processed_content)

                title, content = self.extract_title_and_content(processed_content)
                alternative_title = self.generate_alternative_title(title)

                # Simulating quality check using fine-tuned model
                if hasattr(self, 'quality_threshold'):
                    simulated_rating = random.uniform(1, 5)
                    if self.quality_threshold > simulated_rating:
                        print(f"Article {i+1} didn't meet quality threshold. Skipping.")
                        continue

                file.write(f"Article {i+1}:\n")
                file.write(f"Title A: {title}\n")
                file.write(f"Title B: {alternative_title}\n\n")
                file.write(f"{content}\n")
                file.write(f"Fact Check Result:\n{fact_check_result}\n")
                file.write(f"Author: {persona['author']}\n")
                file.write(f"Tone: {persona['tone']}\n")
                file.write(f"Subject Matter: {persona['subject_matter']}\n")
                file.write(f"Keywords: {', '.join(keywords)}\n")
                file.write("\n" + "="*50 + "\n")

                # Simulate feedback collection (in a real scenario, this would come from users)
                simulated_rating = random.randint(1, 5)
                simulated_comment = f"Simulated feedback for article {i+1}"
                self.collect_feedback(i+1, simulated_rating, simulated_comment)

    def __del__(self):
        if hasattr(self, 'db_conn'):
            self.db_conn.close()

def main():
    try:
        generator = AIContentGenerator('config.yaml')
        generator.generate_and_process()
        generator.fine_tune()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()