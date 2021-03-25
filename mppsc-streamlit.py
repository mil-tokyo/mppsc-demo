# coding=utf-8

import copy
import torch
import streamlit as st
import numpy as np

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from nltk.tokenize import sent_tokenize

from sbert_context_bert_multitask_storycompletion import MultiTask_StoryCompletionModel


def fit_to_block_size(sequence, block_size, pad_token):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter than the block size we pad it with -1 ids
    which correspond to padding tokens.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token] * (block_size - len(sequence)))
        return sequence


def build_mask(sequence, pad_token):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1. """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token
    mask[idx_pad_tokens] = 0
    return mask


def collate_demo(input_text, tokenizer, block_size, device):
    """ Collate formats the data passed to the data loader.
    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    story_line = input_text

    story_line_token_ids = [
        tokenizer.encode(line, add_special_tokens=True) for line in story_line
    ]

    encoded_stories = torch.tensor(
        [
            [
                fit_to_block_size(line, block_size, tokenizer.pad_token_id)
                for line in story_line_token_ids
            ]
        ]
    )

    encoder_mask = build_mask(encoded_stories, tokenizer.pad_token_id)

    batch = {
        "batch_size": 1,
        "src": [story_line],
        "mask_src": encoder_mask.to(device),
    }

    return batch


@st.cache(suppress_st_warning=True)
def load_pretrained_model(model_path, device):

    sentbertmodel = SentenceTransformer("bert-base-nli-mean-tokens")

    # --- model ---
    model = MultiTask_StoryCompletionModel(
        SentenceEncoder=sentbertmodel, device=device, ContextEncoder="GRUContextEncoder"
    ).to(device)

    model.load_state_dict(torch.load(model_path), strict=False)

    model.eval()

    return model


@st.cache(suppress_st_warning=True)
def load_examples(example_path):
    with open(example_path, "r") as f:
        examples = f.readlines()

    return examples


if __name__ == "__main__":

    st.title("Missing Position Prediction + Story Completion Demo System")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_pretrained_model("./trained_model/best_mpp_acc_model.pt", device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = copy.deepcopy(tokenizer)

    with open("examples.txt", "r") as f:
        examples = f.readlines()

    # Selectors
    # model_name = st.sidebar.selectbox("Model", list(MODEL_CLASSES.keys()))

    st.sidebar.header("Task")

    run_mpp = st.sidebar.checkbox("Missing Position Prediction", value=True)
    run_sc = st.sidebar.checkbox("Story Completion", value=True)

    st.sidebar.header("Parameters")

    max_length = st.sidebar.slider("Max Length", 0, 100, 32)
    # temperature = st.sidebar.slider("Temperature", 0.0, 3.0, 0.8)
    # top_k = st.sidebar.slider("Top K", 0, 10, 0)
    # top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.7)

    block_size = max_length

    if not run_mpp and run_sc:
        st.sidebar.header("Story Completion only mode parameter")
        apriori_missing_id = st.sidebar.number_input(
            "a priori knowledge of missing position (from 0 to 4)",
            min_value=0,
            max_value=4,
        )

    examples = load_examples("examples.txt")

    example = st.selectbox("Select from example stories", examples)
    st.write("Selected example :")
    st.write(example)

    run_example = st.checkbox("Use example text", value=False)

    if run_example:
        raw_text = None
        if st.button("Start"):
            raw_text = example
    else:
        raw_text = st.text_input("Enter start text and press enter")

    if raw_text:
        # context_tokens = tokenizer.encode(raw_text)

        split_text = sent_tokenize(raw_text)

        st.write("Sentence-tokenized text:")
        st.write(split_text)

        if len(split_text) != 4:
            st.write(
                "Please input four-sentence context, where one sentence is dropped from the original five-sentence story."
            )

        else:
            with torch.no_grad():

                batch = collate_demo(split_text, tokenizer, block_size, device)

                print(batch)

                sc_generated, mpp_output = model.generate(
                    batch["src"],
                    batch["mask_src"],
                    tokenizer,
                    max_length=max_length,
                    trg_start_id=101,
                )

                # batch size = 1
                i = 0

                if run_mpp:
                    predicted = mpp_output.argmax(1)
                    predicted = predicted.to("cpu").numpy()

                if run_mpp and run_sc:
                    input_story = raw_text.lower()
                    predicted_story = " ".join(
                        np.insert(split_text, predicted[i], sc_generated[i])
                    ).lower()

                    predicted_story_markdown = " ".join(
                        np.insert(
                            split_text, predicted[i], "**" + sc_generated[i] + "**"
                        )
                    ).lower()

                elif not run_mpp and run_sc:
                    i = 0
                    input_story = raw_text.lower()
                    predicted_story = " ".join(
                        np.insert(split_text, apriori_missing_id, sc_generated[i])
                    ).lower()

                    predicted_story_markdown = " ".join(
                        np.insert(
                            split_text,
                            apriori_missing_id,
                            "**" + sc_generated[i] + "**",
                        )
                    ).lower()

                if run_mpp:
                    st.write(f"Predicted missing position: {predicted[i]}")
                if run_sc:
                    st.write(f"Generated sentence: {sc_generated[i]}")
                    st.markdown(predicted_story_markdown)
