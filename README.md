# Note to maintainers - Scancode Required Phrase Predictor (Streamlit Demo)

This repository contains the code for Streamlit app for my GSoC 2026 proposal to AboutCode organization (scancode-toolkit)

The application uses my fine-tuned model (`Kaushik-Kumar-CEG/scancode-required-phrases-deberta-large`)

## Files
* **`app.py`**: The main Streamlit web application. It handles the UI, pre-caches the model weights, processes the example inputs and displays the color-coded diff results
* **`add_ml_phrases.py`**: The core ML inference script. This contains all the logic for text preprocessing, ONNX model loading, token extraction and boundary math
    * *Note: The command-line interface (CLI) logic at the bottom of this file has been commented out, as this script is currently being imported as a module by the Streamlit app rather than run directly from the terminal*
* **`requirements.txt`**: The specific Python dependencies required to run the Hugging Face model and Streamlit locally
* **`.devcontainer/devcontainer.json`**: Configuration for running this project seamlessly in cloud environments like GitHub Codespaces

## License
* The code in this repository is licensed under the [Apache License 2.0](LICENSE).
* The fine-tuned model weights hosted on HuggingFace are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
