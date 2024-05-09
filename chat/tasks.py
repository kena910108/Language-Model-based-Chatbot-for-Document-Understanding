from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import re

# LLM required packages // Required confirmation

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
import prompt
import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print ("MPS device not found.")

class Phi2StoppingCriteria(StoppingCriteria):
    def __init__(self):
        stop_list = ["Exercise", "Exercises", "exercises:", "<|endoftext|>"]
        tokenphrases = []
        for token in stop_list:
            tokenphrases.append(
                tokenizer(token, return_tensors="pt").input_ids[0].tolist()
            )
        self.tokenphrases = tokenphrases

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for tokenphrase in self.tokenphrases:
            if tokenphrase == input_ids[0].tolist()[-len(tokenphrase):]:
                return True
            
def init_context():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    lemmatizer = WordNetLemmatizer()

    sentence_corpus = prompt.get_context("example.pdf")
    tokenized_sentence = prompt.tokenize_corpus(tokenizer, lemmatizer, sentence_corpus)
    bm25 = BM25Okapi(tokenized_sentence)
    

    return tokenizer, lemmatizer, bm25, sentence_corpus

def find_relevent_context(tokenizer, lemmatizer, query):
    tokenized_query = prompt.tokenize_query(tokenizer, lemmatizer, query)
    
    top = prompt.get_top_blocks(bm25, tokenized_query, sentence_corpus)
    
    return top[0]

#initialization
tokenizer, lemmatizer, bm25, sentence_corpus = init_context()

channel_layer = get_channel_layer()

response_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float32)
response_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
text_generator = transformers.pipeline('text-generation', model = response_model, tokenizer= response_tokenizer,
                                        torch_dtype=torch.float32,
                                        trust_remote_code=True,
                                        max_length=400,
                                        do_sample=True,
                                        top_k=10,
                                        num_return_sequences=1,
                                        eos_token_id=tokenizer.eos_token_id
                                    )

def get_response(channel_name, input_data):

    user_input = input_data["text"]
    print(user_input)
    context = re.sub(r'\s+', ' ', find_relevent_context(tokenizer, lemmatizer, user_input))

    template = f""" Summarize {user_input} in the following context: {context}.

    Answer: """
    
    print(template)
    #chat_history.append("User: " + user_input)
    #formatted_history = "\n".join(chat_history)
    #print(formatted_history)
    #formatted_prompt = f"The following conversation takes place:\nA User: '{user_input}'\n Chatbot:"

    response = text_generator(
        template,
        max_length=400,
        num_return_sequences=1,
        temperature=0.1,  # Lower for less random responses
        top_p=0.9,
        stop_sequence= "Exercise"
        , eos_token_id= text_generator.tokenizer.eos_token_id)[0]['generated_text']

    print('\n','\n','\n','\n',print(response))
    print("/////////////////////////////////////")
    print('\n','\n','\n','\n',print(repr(response)))
    #response = response.split("User: " + user_input)[-1].strip().split("\n")[0]
    #print(response)

    #chat_history.append(response)

    async_to_sync(channel_layer.send)(
        channel_name,
        {
            "type": "chat.message",
            "text": {"msg": response.split("Answer: ")[1].split("\n\n")[0], "source": "bot"},
        },
    )

def truncate_history(history, tokenizer, max_length=512):
    # Tokenize the chat history and truncate it to the maximum length
    tokens = tokenizer.encode(history)
    if len(tokens) > max_length:
        truncated_tokens = tokens[-max_length:]  # Keep only the last max_length tokens
        return tokenizer.decode(truncated_tokens)
    return history