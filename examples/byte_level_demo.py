#!/usr/bin/env python3
"""
Demonstration of byte-level Infinigram with UTF-8 text.

This example shows how Infinigram operates on bytes (0-255), making it
compatible with UTF-8 text, multiple documents, and augmentations.
"""

from infinigram import Infinigram, IdentityAdapter
from infinigram.corpus_utils import build_corpus_from_documents, text_to_bytes, bytes_to_text


def demo_basic_utf8():
    """Basic UTF-8 text prediction."""
    print("=" * 60)
    print("DEMO 1: Basic UTF-8 Text Prediction")
    print("=" * 60)

    # Create corpus from text
    text = "the cat sat on the mat. the cat ran on the mat."
    corpus = list(text.encode('utf-8'))

    print(f"Corpus text: '{text}'")
    print(f"Corpus size: {len(corpus)} bytes")
    print(f"First 20 bytes: {corpus[:20]}")
    print()

    # Create model
    model = Infinigram(corpus, max_length=20)
    print(f"Model: {model}")
    print()

    # Predict next byte after "the cat "
    context_text = "the cat "
    context = list(context_text.encode('utf-8'))

    print(f"Context: '{context_text}'")
    print(f"Context bytes: {context}")
    print()

    # Get predictions
    probs = model.predict(context, top_k=5)

    print("Top 5 predictions:")
    for byte_val, prob in sorted(probs.items(), key=lambda x: -x[1])[:5]:
        # Try to decode byte as ASCII for display
        char = chr(byte_val) if 32 <= byte_val < 127 else f"0x{byte_val:02X}"
        print(f"  Byte {byte_val:3d} ('{char}'): {prob:.4f}")
    print()


def demo_unicode():
    """UTF-8 with Unicode characters."""
    print("=" * 60)
    print("DEMO 2: Unicode (Multi-byte) Characters")
    print("=" * 60)

    # Text with Unicode
    text = "café café café"
    corpus = list(text.encode('utf-8'))

    print(f"Text: '{text}'")
    print(f"Byte count: {len(corpus)} (note: 'é' is 2 bytes in UTF-8)")
    print(f"Bytes: {corpus}")
    print()

    model = Infinigram(corpus)

    # Predict after "caf"
    context = list("caf".encode('utf-8'))
    probs = model.predict(context, top_k=3)

    print("Context: 'caf'")
    print("Top 3 predictions:")
    for byte_val, prob in sorted(probs.items(), key=lambda x: -x[1])[:3]:
        print(f"  Byte {byte_val:3d} (0x{byte_val:02X}): {prob:.4f}")

    # Note: 0xC3 is first byte of 'é' in UTF-8
    if 195 in probs:  # 0xC3 = 195
        print("\n  ✓ Model learned to predict first byte of 'é' (0xC3)")
    print()


def demo_multi_document():
    """Multiple documents with separators."""
    print("=" * 60)
    print("DEMO 3: Multiple Documents with Separators")
    print("=" * 60)

    # Create corpus from multiple documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "The lazy cat sleeps on the warm mat.",
        "The quick cat runs across the yard."
    ]

    print("Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    print()

    # Build corpus with document separators
    corpus = build_corpus_from_documents(documents, separator=b"\n\n")

    print(f"Corpus size: {len(corpus)} bytes")
    print(f"Decoded corpus:\n{bytes_to_text(corpus)}")
    print()

    model = Infinigram(corpus, max_length=30)

    # Predict after "The quick "
    context = list("The quick ".encode('utf-8'))
    probs = model.predict(context, top_k=5)

    print("Context: 'The quick '")
    print("Top predictions (should include 'b' from 'brown' and 'c' from 'cat'):")
    for byte_val, prob in sorted(probs.items(), key=lambda x: -x[1])[:5]:
        char = chr(byte_val) if 32 <= byte_val < 127 else f"0x{byte_val:02X}"
        print(f"  '{char}': {prob:.4f}")
    print()


def demo_augmentation():
    """Data augmentation for improved coverage."""
    print("=" * 60)
    print("DEMO 4: Data Augmentation")
    print("=" * 60)

    from infinigram.corpus_utils import build_corpus_with_augmentation

    documents = ["Hello World"]

    print(f"Original document: '{documents[0]}'")
    print()

    # Augmentation functions
    def lowercase(text):
        return text.lower()

    def uppercase(text):
        return text.upper()

    # Build augmented corpus
    corpus = build_corpus_with_augmentation(
        documents,
        augmentations=[lowercase, uppercase],
        separator=b"\n\n"
    )

    print("Augmented corpus:")
    print(bytes_to_text(corpus))
    print()

    model = Infinigram(corpus)

    # Test with lowercase query
    context = list("hello ".encode('utf-8'))
    pos, length = model.longest_suffix(context)

    print(f"Query: 'hello ' (lowercase)")
    print(f"Match length: {length} bytes")
    if length > 0:
        print("  ✓ Found match from augmentation!")
    print()


def demo_identity_adapter():
    """Using IdentityAdapter for text conversion."""
    print("=" * 60)
    print("DEMO 5: IdentityAdapter for Text Handling")
    print("=" * 60)

    adapter = IdentityAdapter()

    # Text to bytes
    text = "Hello 世界 🌍"
    byte_seq = adapter.text_to_bytes(text)

    print(f"Text: '{text}'")
    print(f"Bytes: {byte_seq}")
    print(f"Byte count: {len(byte_seq)}")
    print()

    # Create model
    corpus_text = "Hello world! Hello 世界! Hello everyone!"
    corpus = adapter.text_to_bytes(corpus_text)
    model = Infinigram(corpus)

    # Query
    query_text = "Hello "
    context = adapter.text_to_bytes(query_text)
    probs = model.predict(context, top_k=5)

    print(f"Corpus: '{corpus_text}'")
    print(f"Query: '{query_text}'")
    print("\nTop 5 next-byte predictions:")
    for byte_val, prob in sorted(probs.items(), key=lambda x: -x[1])[:5]:
        char = chr(byte_val) if 32 <= byte_val < 127 else f"byte {byte_val}"
        print(f"  {char}: {prob:.4f}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  INFINIGRAM BYTE-LEVEL DEMONSTRATIONS")
    print("=" * 60)
    print()

    demo_basic_utf8()
    demo_unicode()
    demo_multi_document()
    demo_augmentation()
    demo_identity_adapter()

    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
