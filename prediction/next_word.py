import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Read the text file

with open('content/sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1   


input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
# Now convert input_sequences to a NumPy array after all sequences are appended
max_sequence_len = max([len(seq) for seq in input_sequences])  # Find the maximum sequence length
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Now let’s split the sequences into input and output:
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Now let’s convert the output to one-hot encode vectors:
y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))


model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)


# After training your model
model.save('text_generation_model.h5')  # Save the trained model to a .h5 file
print("Model saved as 'text_generation_model.h5'")


seed_text = "I will leave if they"
next_words = 3

# The below code will run predictoin from the generated model in memory on run time
# for _ in range(next_words):
#     token_list = tokenizer.texts_to_sequences([seed_text])[0]
#     token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
#     predicted = np.argmax(model.predict(token_list), axis=-1)
#     output_word = ""
#     for word, index in tokenizer.word_index.items():
#         if index == predicted:
#             output_word = word
#             break
#     seed_text += " " + output_word


# Load the saved model
loaded_model = tf.keras.models.load_model('text_generation_model.h5')
print("Model loaded successfully")

# Define the tokenizer and max_sequence_len again as per your setup
# tokenizor and max_sequence_len should be the same as used during training

# Example seed text for prediction
seed_text = "I will leave if they"
next_words = 3

for _ in range(next_words):
    # Convert seed text to sequence of integers using tokenizer
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    
    # Pad the sequence to match the input length expected by the model
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Predict the next word
    predicted = np.argmax(loaded_model.predict(token_list), axis=-1)
    
    # Convert the predicted index back to a word
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    
    # Append the predicted word to the seed text
    seed_text += " " + output_word

print(seed_text)  # Output the generated text

