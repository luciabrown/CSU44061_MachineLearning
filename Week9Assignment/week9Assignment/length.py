file_path = "input_shakespeare.txt"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

words = content.split()
word_count = len(words)
unique_words = len(set(words))

print("Number of words:", word_count)
print("Number of unique words:", unique_words)