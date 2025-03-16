import google.generativeai as genai
import random
import os
import sys
import spacy
from typing import List, Tuple


try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. Please set it before running this script."
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        nlp = spacy.load("zh_core_web_sm")
    except OSError:
        print("Chinese language model not found. Installing zh_core_web_sm...")
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "zh_core_web_sm"], check=True
        )
        nlp = spacy.load("zh_core_web_sm")
except Exception as e:
    print(f"Error initializing clients: {e}")
    exit(1)


def generate_chinese_response(prompt: str) -> str:
    """Generate an informational response in Chinese based on the prompt."""
    try:
        response = model.generate_content(
            f"Please provide an informational response in Chinese about: {prompt}. The response should be detailed and at least 3-4 sentences long."
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""
        return ""


def extract_morphemes(text: str, num_morphemes: int = 10) -> List[str]:
    doc = nlp(text)

    compounds = []
    for token in doc:
        if (
            token.pos_ in ["NOUN", "VERB", "ADJ"]
            and len(token.text) >= 2
            and not token.is_punct
            and not token.is_space
        ):

            # Get the compound and its POS tag
            compound = {
                "text": token.text,
                "pos": token.pos_,
                "dep": token.dep_,  # Dependency relation
                "head": token.head.text,  # Head word in dependency tree
            }

            if compound["text"] not in [c["text"] for c in compounds] and (
                compound["dep"] in ["nsubj", "dobj", "compound"]
                or any(ancestor.pos_ == "NOUN" for ancestor in token.ancestors)
            ):
                compounds.append(compound)

    compounds.sort(
        key=lambda x: (
            x["pos"] == "NOUN",
            len(x["text"]),
            x["dep"] in ["nsubj", "dobj"],
        ),
        reverse=True,
    )

    morphemes = [c["text"] for c in compounds]

    if len(morphemes) < num_morphemes:
        return morphemes
    return random.sample(morphemes[:10], num_morphemes)  # Sample from top 10 candidates


def generate_alternatives(morpheme: str) -> List[str]:
    """Generate alternative incorrect versions of the given semantic morpheme."""
    try:
        response = model.generate_content(
            f"For the Chinese compound '{morpheme}', generate 3 other Chinese compounds of the same length that are semantically related but have different meanings. The alternatives should be plausible in the context but incorrect. Return only the three alternatives, separated by '|'. Do not include any other text."
        )
        alternatives = response.text.strip().split("|")
        return [alt.strip() for alt in alternatives[:3]]
    except Exception as e:
        print(f"Error generating alternatives: {e}")
        return []


def create_quiz(prompt: str) -> Tuple[str, List[dict], str]:
    """Create a quiz with the original text and questions."""
    # Generate the main response
    response = generate_chinese_response(prompt)
    if not response:
        return "", [], ""

    # Extract random morphemes
    target_morphemes = extract_morphemes(response)

    quiz_items = []
    masked_text = response

    for i, morpheme in enumerate(target_morphemes):
        alternatives = generate_alternatives(morpheme)
        if alternatives:
            # Create quiz item
            quiz_items.append(
                {"original_morpheme": morpheme, "alternatives": alternatives}
            )
        # Replace morpheme with underscores of the same length
        masked_text = masked_text.replace(morpheme, f" ({i}) ")
    return response, quiz_items, masked_text


def main():
    # Example usage
    prompt = input("Enter a topic: ")

    response, quiz_items, masked_text = create_quiz(prompt)

    print("\n=== Generated Content ===")
    print(response)

    print("\n=== Masked Text ===")
    print(masked_text)

    print("\n=== Quiz Items ===")
    for i, item in enumerate(quiz_items, 1):
        print(f"\nQuestion {i - 1}:")

        joined_answers = item["alternatives"] + [item["original_morpheme"]]
        random.shuffle(joined_answers)
        for j, alt in enumerate(joined_answers, 1):
            print(f"  {j}. {alt}")


if __name__ == "__main__":
    main()
