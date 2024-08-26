import os

from anthropic import Anthropic


def get_symmetric_examples(pairs, current_index):
    """
    Get examples in a symmetric order, treating the pairs as a ring.

    Args:
    pairs (list of tuples): All (input, output) pairs
    current_index (int): Index of the current pair being rebuilt

    Returns:
    list of tuples: Ordered list of example pairs, excluding the current pair
    """
    return pairs[current_index + 1:] + pairs[:current_index]


def parallel_output_replacement(input_output_pairs):
    """
    Perform a single turn of parallel output replacement with consistent symmetric example ordering.

    Args:
    input_output_pairs (list of tuples): Current set of (input, output) pairs

    Returns:
    list of tuples: Updated set of (input, output) pairs after one turn
    """

    for i, (current_input, _) in enumerate(input_output_pairs):
        # Get examples
        symmetric_examples = get_symmetric_examples(input_output_pairs, i)

        # Generate new output using the LLM function
        new_output = llm_function(symmetric_examples, current_input)
        yield current_input, new_output


anthropic_client = Anthropic()


def llm_function(example_pairs, input_text):
    messages = []
    for example_input, example_output in example_pairs:
        messages.append({"role": "user", "content": example_input})
        messages.append({"role": "assistant", "content": example_output})
    messages.append({"role": "user", "content": input_text})
    response = anthropic_client.messages.create(
        messages=messages,
        temperature=0,
        model='claude-3-haiku-20240307',
        max_tokens=4096,
    )
    return ''.join(
        block.text
        for block in response.content
    )


def iterate_parallel_replacement(directory, max_num_iterations):
    # load the initial pairs
    examples = []
    example_names = []
    for example_name in os.listdir(directory):
        input_text = read_file(os.path.join(directory, example_name, 'input.txt'))
        output_text = read_file(os.path.join(directory, example_name, 'output.txt'))
        examples.append((input_text, output_text))
        example_names.append(example_name)

    # run the algorithm
    for i in range(max_num_iterations):
        new_examples_iter = parallel_output_replacement(examples)
        new_examples = []
        for example_name, (input_text, output_text) in zip(example_names, new_examples_iter):
            write_file(os.path.join(directory, example_name, f'output_{i:03d}.txt'), output_text)
            new_examples.append((input_text, output_text))
        if examples == new_examples:
            break
        examples = new_examples


def read_file(file_path):
    print(f"Reading file: {file_path}")
    with open(file_path, 'rt') as file:
        return file.read()


def write_file(file_path, content):
    print(f"Writing file: {file_path}")
    with open(file_path, 'wt') as file:
        file.write(content)


if __name__ == "__main__":
    iterate_parallel_replacement('examples', 10)
