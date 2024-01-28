import random
import re

def replace_words_with_token(input_path, output_path, replacement_token, percentage_to_replace):
    # Read the contents of the input file
    with open(input_path, 'r', encoding='utf-8') as file:
        text_data = file.read()

    # Tokenize the input text
    words = re.findall(r'\b\w+\b', text_data)

    # Calculate the number of words to replace
    total_words = len(words)
    words_to_replace = int(percentage_to_replace / 100 * total_words)

    # Select random words to replace
    words_indices_to_replace = random.sample(range(total_words), words_to_replace)

    # Replace the selected words with the replacement token
    for index in words_indices_to_replace:
        words[index] = replacement_token

    # Join the words back into a string
    modified_text = ' '.join(words)

    # Write the modified text to the output file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(modified_text)

# Example usage
replacement_token = '[MASK]'
percentage_to_replace = 2  # 10% of words to be replaced
replace_words_with_token('wiki.test.raw', 'wiki.test.txt', replacement_token, percentage_to_replace)