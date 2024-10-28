import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem  # Ensure these are correctly defined
from model import NeuralNet  # Ensure this model class is defined correctly

# Load intents file
try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    print("The intents.json file was not found.")
    exit()

all_words = []
tags = []
xy = []

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and lower each word, remove punctuation
ignore_words = ['?', '.', '!']
all_words = [stem(w.lower()) for w in all_words if w not in ignore_words]

# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"{len(xy)} patterns")
print(f"{len(tags)} tags: {tags}")
print(f"{len(all_words)} unique stemmed words: {all_words}")

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # Create bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # Get class label for each tag
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(f"Input size: {input_size}, Output size: {output_size}")


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.FloatTensor(X_train)  # Convert to Tensor
        self.y_data = torch.LongTensor(y_train)  # Convert to Tensor

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# Save the model and other data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')
