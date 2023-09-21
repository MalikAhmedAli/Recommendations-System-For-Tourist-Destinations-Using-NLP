import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from tqdm import tqdm

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BERT_BATCH_SIZE = 4
MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

class BertModel:
    def __init__(self, model_name, device=-1, small_memory=True, batch_size=BERT_BATCH_SIZE):
        self.model_name = model_name
        self._set_device(device)
        self.small_device = 'cpu' if small_memory else self.device
        self.batch_size = batch_size
        self.load_pretrained_model()


    def _set_device(self, device):
        if device == -1 or device == 'cpu':
            self.device = 'cpu'
        elif device == 'cuda' or device == 'gpu':
            self.device = 'cuda'
        elif isinstance(device, int) or isinstance(device, float):
            self.device = 'cuda'
        else:  # default
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        device = -1 if self.device == 'cpu' else 0
        self.pipeline = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer, device=device)

    def embed(self, data):
        """ Create the embedded matrix from original sentences """
        nb_batches = 1 if (len(data) < self.batch_size) else len(data) // self.batch_size
        batches = np.array_split(data, nb_batches)
        mean_pooled = []
        for batch in tqdm(batches, total=len(batches), desc='Training...'):
            mean_pooled.append(self.transform(batch))
        mean_pooled_tensor = torch.tensor(len(data), dtype=float).to(self.small_device)
        mean_pooled = torch.cat(mean_pooled, out=mean_pooled_tensor)
        self.embed_mat = mean_pooled

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def transform(self, data):
        if isinstance(data, str):
            data = [data]
        data = list(data)
        token_dict = self.tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors="pt")
        token_dict = self.to(token_dict, self.device)
        with torch.no_grad():
            token_embed = self.model(**token_dict)
        attention_mask = token_dict['attention_mask']
        mean_pooled = self.mean_pooling(token_embed, attention_mask)
        mean_pooled = mean_pooled.to(self.small_device)
        return mean_pooled

    def to(self, data: dict, device: str):
        """Send all values to device by calling v.to(device)"""
        data = {k: v.to(device) for k, v in data.items()}
        return data

    def predict(self, in_sentence, topk=3):
        input_vec = self.transform(in_sentence)
        mat = cosine_similarity(input_vec, self.embed_mat)
        best_index = extract_best_indices(mat, topk=topk)
        return best_index


DATAPATH = '../RecommendationSystemCode/data/Testing.csv'

# df = pd.read_csv(DATAPATH, encoding="utf-8", sep=",")
# df.rename(columns={'description': 'sentence'}, inplace=True)

df = pd.read_csv(DATAPATH, encoding="utf-8", sep=",")
df.rename(columns={'description':'sentence'}, inplace=True)

STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 4
MAX_WORDS = 200

PATTERN_S = re.compile("\'s")
PATTERN_RN = re.compile("\\r\\n")
PATTERN_PUNC = re.compile(r"[^\w\s]")


def clean_text(text):
    """
    Series of cleaning. String to lower case, remove non words characters and numbers.
        text (str): input text
    return (str): modified initial text
    """
    if isinstance(text, float):
        # If text is a float, return an empty string
        return ''
    text = text.lower()  # lowercase text
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    text = re.sub(PATTERN_PUNC, ' ', text)
    
    return text

def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
    """
    Lemmatize, tokenize, crop and remove stop words.
    """
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
    else:
        tokens = [w for w in word_tokenize(sentence)]
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                                        and w not in stopwords)]
    return tokens    


def clean_sentences(df):
    """
    Remove irrelavant characters (in new column clean_sentence).
    Lemmatize, tokenize words into list of words (in new column tok_lem_sentence).
    """
    print('Cleaning sentences...')
    df['clean_sentence'] = df['sentence'].apply(clean_text)
    df['tok_lem_sentence'] = df['clean_sentence'].apply(
        lambda x: tokenizer(x, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True))
    return df

def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]  
    return best_index


df = clean_sentences(df)
token_stop = tokenizer(' '.join(STOPWORDS), lemmatize=False)
df.fillna("", inplace=True)
text_data = df['sentence'].tolist()

# Fit TFIDF
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer) 
tfidf_mat = vectorizer.fit_transform(df['sentence'].values)



class BertModel:
    def __init__(self, model_name, device=-1, small_memory=True, batch_size=BERT_BATCH_SIZE):
        self.model_name = model_name
        self._set_device(device)
        self.small_device = 'cpu' if small_memory else self.device
        self.batch_size = batch_size
        self.load_pretrained_model()


    def _set_device(self, device):
        if device == -1 or device == 'cpu':
            self.device = 'cpu'
        elif device == 'cuda' or device == 'gpu':
            self.device = 'cuda'
        elif isinstance(device, int) or isinstance(device, float):
            self.device = 'cuda'
        else:  # default
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        device = -1 if self.device == 'cpu' else 0
        self.pipeline = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer, device=device)

    def embed(self, data):
        """ Create the embedded matrix from original sentences """
        nb_batches = 1 if (len(data) < self.batch_size) else len(data) // self.batch_size
        batches = np.array_split(data, nb_batches)
        mean_pooled = []
        for batch in tqdm(batches, total=len(batches), desc='Training...'):
            mean_pooled.append(self.transform(batch))
        mean_pooled_tensor = torch.tensor(len(data), dtype=float).to(self.small_device)
        mean_pooled = torch.cat(mean_pooled, out=mean_pooled_tensor)
        self.embed_mat = mean_pooled

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def transform(self, data):
        if isinstance(data, str):
            data = [data]
        data = list(data)
        token_dict = self.tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors="pt")
        token_dict = self.to(token_dict, self.device)
        with torch.no_grad():
            token_embed = self.model(**token_dict)
        attention_mask = token_dict['attention_mask']
        mean_pooled = self.mean_pooling(token_embed, attention_mask)
        mean_pooled = mean_pooled.to(self.small_device)
        return mean_pooled

    def to(self, data: dict, device: str):
        """Send all values to device by calling v.to(device)"""
        data = {k: v.to(device) for k, v in data.items()}
        return data

    def predict(self, in_sentence, topk=3):
        input_vec = self.transform(in_sentence)
        mat = cosine_similarity(input_vec, self.embed_mat)
        best_index = extract_best_indices(mat, topk=topk)
        return best_index

bert_model = BertModel(MODEL_NAME)


def transform_df(df):
    df = df.to_dict()
    output_list = []

    for key, value in df['city'].items():
        description = df['sentence'][key]
        output_list.append({'city': value.lower(), 'description': description})
    
    return output_list

def get_recommendations_tfidf(sentence):
    
    """
    Return the database sentences in order of highest cosine similarity relatively to each 
    token of the target sentence. 
    """
    # Embed the query sentence
    tokens = [str(tok) for tok in tokenizer(sentence)]
    vec = vectorizer.transform(tokens)
    # Create list with similarity between query and dataset
    mat = cosine_similarity(vec, tfidf_mat)
    # Best cosine distance for each token independantly
    print(mat.shape)
    best_index = extract_best_indices(mat, topk=5)
    return best_index

def predict(sentence):
    res = get_recommendations_tfidf(sentence)
    columns = ['city', 'sentence']
    if 'image' in df.columns:
        columns.append('image')
    pred = df[columns].iloc[res]
    pred = transform_df(pred)
    
    pred_df = pd.DataFrame(pred, columns=columns)  # Convert pred list to DataFrame

    formatted_recommendations = []
    for index, row in pred_df.iterrows():
        city = row['city']
        description = row['sentence']
        image_url = None
        if 'image' in row:
            image_name = row['image']
            image_url = f"http://your-domain.com/image?q={image_name}"
        recommendation = {'city': city, 'description': description, 'image_url': image_url}
        formatted_recommendations.append(recommendation)

    return {'recommendations': formatted_recommendations}

if __name__ == "__main__":
    recommendations = predict("often snowfall beautiful mountains")
    print(recommendations)


# def preprocess_sentences(sentences):
#     cleaned_sentences = []
#     for sentence in sentences:
#         cleaned_sentence = sentence.lower()
#         cleaned_sentence = cleaned_sentence.replace("\'s", " ")
#         cleaned_sentence = re.sub("\\r\\n", " ", cleaned_sentence)
#         cleaned_sentence = re.sub(r"[^\w\s]", " ", cleaned_sentence)
#         cleaned_sentences.append(cleaned_sentence)
#     return cleaned_sentences

# def get_bert_embeddings(sentences):
#     preprocessed_sentences = preprocess_sentences(sentences)
#     tokenized_inputs = tokenizer(preprocessed_sentences, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**tokenized_inputs)
#         embeddings = outputs.last_hidden_state.squeeze(0)  # Squeeze the extra dimension
#     return embeddings

# def get_recommendations_bert(embeddings, query_embedding, topk=5):
#     reshaped_query_embedding = query_embedding.unsqueeze(0)  # Reshape the query embedding
#     cosine_similarities = cosine_similarity(reshaped_query_embedding, embeddings)
#     best_indices = cosine_similarities.argsort()[0][-topk:][::-1]
#     return best_indices

# def predict(sentence):
#     embeddings = get_bert_embeddings(df['sentence'])
#     query_embedding = get_bert_embeddings([sentence])
#     recommendations = get_recommendations_bert(embeddings, query_embedding, topk=5)
#     return df.iloc[recommendations]

# if __name__ == "__main__":
#     recommendations = predict("often snowfall beautiful mountains")
#     print(recommendations)
