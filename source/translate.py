import tensorflow as tf
pretrained_model = tf.keras.models.load_model('D:\Acer\hbi\model\machine_v2.keras')
# Show the model architecture
pretrained_model.summary()
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
import numpy as np
import spacy
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pymupdf
import numpy as np
import pandas as pd
# from googletrans import Translator, constants
import io
import re
input()
spacy_en = spacy.load('en_core_web_sm')
eos = '<eos>'
bos = '<bos>'
import pickle
with open('D:\Acer\hbi\data\dictionary_en.pkl', 'rb') as f:
    en_vocabulary = pickle.load(f)
with open('D:\Acer\hbi\data\dictionary_vi.pkl', 'rb') as f:
    vi_vocabulary = pickle.load(f)
with open('D:\Acer\hbi\data\dictionary_vi_reverse.pkl', 'rb') as f:
    vi_vocabulary_reverse = pickle.load(f)
with open('D:\Acer\hbi\data\col_no_translate') as f:
    no_translate_vocabulary = [line.replace('\n','') for line in f]

pretrained_emb_layer = pretrained_model.get_layer('embedding')
encoder_inputs = Input(shape=(None,))
encoder_emb = pretrained_emb_layer(encoder_inputs)  # Use the pre-trained layer
encoder_lstm = pretrained_model.get_layer('lstm')
_, state_h, state_c = encoder_lstm(encoder_emb)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = pretrained_model.get_layer('embedding_1')
dec_emb = dec_emb_layer(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm =pretrained_model.get_layer('lstm_1')
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = pretrained_model.get_layer('dense')
decoder_outputs = decoder_dense(decoder_outputs)
# Encode the input sequence to get the "Context vectors"
encoder_model = Model(encoder_inputs, encoder_states)
latent_dim=128
# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_input = [decoder_state_input_h, decoder_state_input_c]
# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_input)
decoder_states2 = [state_h2, state_c2]

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_state_input,
    [decoder_outputs2] + decoder_states2)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = vi_vocabulary[bos]

    # Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    # greedy search
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # print(output_tokens)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # print(sampled_token_index)
        sampled_word =vi_vocabulary_reverse[sampled_token_index]
        decoded_sentence += ' '+ sampled_word

        # Exit condition: either hit max length or find stop character.
        if (sampled_word == eos or
           len(decoded_sentence) > 300):
            stop_condition = True
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        # Update states
        states_value = [h, c]
    return decoded_sentence
def fix_text(text):
  if isinstance(text, str):
    if text[:3]=='Col' or text[:3]=='Unn':
      text=""
    else:
      text=text.replace('\n', ' ')
  else:
    text=""
  return text
def check_alpha(text):
   if text.isalpha():
     return True
   else:
     return False
def check_numeric(text):
    try:
        float(text)
        return True
    except ValueError:
        return False

def translate(seq_input):
  if type(seq_input)==str:
    seq_input = seq_input.lower()
  else:
    seq_input=str(seq_input)
  seq =[tok.text for tok in spacy_en.tokenizer(seq_input)]
  if len(seq)==1 and (not check_alpha(seq_input) or check_numeric(seq_input) or len(seq_input)==1):
    output = seq_input
  else:
    list_isalnum=[]
    input_arr = [1]
    input_arr_=[1]
    index_next=0
    enmaxlen=48 # max length of english sequence, get from source translate_en_vi.ipynb
    for text in seq:
      if text in en_vocabulary and not (check_numeric(text) or not check_alpha(text) or len(text)==1):
        input_arr.append(en_vocabulary[text])
        input_arr_.append(en_vocabulary[text])
      else:
        input_arr.append(en_vocabulary['<bos>'])
        list_isalnum.append(text)
        try:
          input_arr_.append(en_vocabulary[text])
        except:
          input_arr_.append(en_vocabulary['<bos>'])
    if 'i' in list_isalnum and 'd' in list_isalnum:
      list_isalnum.remove('i')
      list_isalnum.remove('d')
    # Translate
    if np.unique(input_arr).shape[0] == 1:
      output=seq_input
    else:
      input_arr_ = pad_sequences([input_arr_], maxlen = enmaxlen, padding = 'post')
      decoded_sentence = decode_sequence(input_arr_)
      out_str=decoded_sentence[:-5].replace('_', ' ').strip()
      out_list=out_str.split(' ')
      output=""
      arr=[]
      for index_item, item in enumerate(out_list):
        if check_alpha(item):
          output+=item+" "
        else:
          arr.append(index_item)
      for i in list_isalnum:
        spaces_indices = [match.start() for match in re.finditer(' ', output)]
        if spaces_indices:
          try:
            index=arr.pop(0)-1
          except:
            index=0
          output=output[:spaces_indices[index]]+" "+i+output[spaces_indices[index]:]
        else:
          output += " " + i
  return output
# translator = Translator()
col_name_tranl=["Product Type","Piece name"]
def trans_df(df):
  name_col=[]
  list_col=[col for col in df]
  if list_col[0] in col_name_tranl:
    name_col=[translate(fix_text(col)) for col in list_col]
  else:
    name_col=[fix_text(col) for col in list_col]
  for col in list_col:
    for row in range(len(df[col])):
      df.loc[row,col]=fix_text(df.loc[row,col])
      if df.loc[row,col] in no_translate_vocabulary :
        if df.loc[row,col][0]!="(":
          df.loc[row,col]=translate(df.loc[row,col])
        else:
          df.loc[row,col]=df.loc[row,col]
        break
      else:
        df.loc[row,col]=translate(df.loc[row,col])
  df2=df.set_axis(name_col, axis=1)
  return df2
doc = pymupdf.open("D:/Acer/hbi/data/test.pdf")
with pd.ExcelWriter('pdf_output.xlsx') as writer:
    for index_page in range(len(doc)):
        page = doc[index_page]
        tabs = page.find_tables(strategy='lines_strict')
        for table_index, table in enumerate(tabs):
            # Convert pymupdf Table to Pandas DataFrame
            df = table.to_pandas()
            # df.to_excel(writer, sheet_name=f'Sheet{index_page}_{table_index+1}')
            result = trans_df(df)
            result.to_excel(writer, sheet_name=f'Sheet{index_page}_{table_index+1}',index=False)