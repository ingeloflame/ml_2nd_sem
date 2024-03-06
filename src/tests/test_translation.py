import pytest
from transformers import pipeline

def test_translation_pipeline():
    # Initialize the translation pipeline
    translator = pipeline("translation_ru_to_en", "Helsinki-NLP/opus-mt-ru-en")
    
    # Test input
    input_text = "очень много материала надо изучить"
    expected_output = "There is a lot of material to study" # This is a rough translation, adjust as needed
    
    # Translate the input text
    result = translator(input_text)[0]['translation_text']
    
    # Assert that the translation is as expected
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"
