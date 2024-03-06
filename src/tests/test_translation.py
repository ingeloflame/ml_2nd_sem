import pytest
from transformers import pipeline


def test_translation_pipeline():
    translator = pipeline("translation_ru_to_en", "Helsinki-NLP/opus-mt-ru-en")

    input_text = "очень много материала надо изучить"
    expected_output = "There's a lot of material to study."
 
    result = translator(input_text)[0]["translation_text"]
 
    assert (
        result == expected_output
    ), f"Expected '{expected_output}', but got '{result}'"
