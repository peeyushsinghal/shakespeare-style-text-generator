# Text Generation with GPT

This project implements a text generation application using a GPT model. The application allows users to input a prompt and generate multiple variations of text based on that prompt.

## Features

- Enter a prompt in a text box.
- Select the number of sequences to generate (1 to 5).
- View each generated sequence clearly labeled.

## Requirements

To run this project, you need to have the following Python packages installed:

- `torch`
- `transformers`
- `tiktoken`
- `gradio`
- `rust`
You can install the required packages using pip:

```bash
pip install torch transformers tiktoken gradio
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/peeyushsinghal/shakespeare-style-text-generation.git
cd shakespeare-style-text-generation
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Gradio application:

```bash
python src/app.py
```

4. Open your web browser and navigate to the URL provided in the terminal (usually `http://127.0.0.1:7860`).


5. Usage:

- Enter your desired prompt in the input box.
- Use the slider to select the number of text sequences you want to generate.
- Click the "Submit" button to see the generated text sequences.


