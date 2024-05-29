



# Lexicon Construction Tool

## Overview

The Lexicon Construction Tool is designed to enhance natural language processing (NLP) applications by integrating human-level attributes (HLAs) such as psychological and personality dimensions. By incorporating these dimensions, the tool aims to improve sentiment analysis, personality detection, and overall text analysis to create more natural and empathetic AI-driven communication tools, like chatbots.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jessedoka/LCT.git
   cd LCT
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Collection:**
   - Please ensure you have access to the required datasets, such as the Goodreads Book Graph Dataset and the Essay Dataset.
   - Anonymize data to comply with ethical standards.

2. **Preprocessing:**
   - Run the preprocessing script to clean and prepare the text data.
   ```bash
   python preprocessing.py
   ```

3. **Lexicon Construction:**
   - Use the provided scripts to generate word embeddings and construct the lexicon.
   ```bash
   python lexicon_construction.py
   ```

4. **Model Training:**
   - Train the models using the prepared data and the constructed lexicon.
   - This is used in notebooks `text_analysis.ipynb` and `lexicon_analysis.ipynb`

5. **Evaluation:**
   - Evaluate the performance of the models and compare them with base models.
   ```bash
   python evaluation.py
   ```

## Key Components

- **Data Sources:**
  - Goodreads Book Graph Dataset.
  - Essay Dataset with Big Five personality traits.
  - LIWC Dictionary and OCEAN Words Dataset.

- **Preprocessing Techniques:**
  - Sentiment term extraction.
  - Tokenization.
  - Stop word removal.
  - Lemmatization.

- **Lexicon Construction:**
  - Word embeddings using Word2Vec.
  - Semantic graph construction using NetworkX.
  - Multi-label propagation for categorization.

- **Models:**
  - Logistic Regression.
  - Support Vector Machine (SVM).
  - Random Forest.


## Future Work

- Expand the lexicon to support multiple languages.
- Continuously update the lexicon with new sources.
- Enhance the integration with large language models (LLMs).
- Conduct manual evaluation by linguistic experts.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. Ensure that your contributions comply with the project's coding standards and include relevant tests.

## Acknowledgements

- Supervisor: Dr. Huizhi Liang
- Special thanks to Dr. Rusnachenko at Bournemouth University for guidance on NLP techniques.
