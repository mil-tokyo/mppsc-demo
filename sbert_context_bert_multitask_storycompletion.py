import os
import itertools
import argparse
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    BertLMHeadModel,
    BertTokenizer,
    AdamW,
    get_constant_schedule_with_warmup,
)
from sentence_transformers import SentenceTransformer

from torch.utils.tensorboard import SummaryWriter

import mlflow

from tqdm import tqdm, trange
from collections import namedtuple


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# ----------------------
# TensorBoard and mlflow
# ----------------------


def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    tb_writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value, step)


# ------------
# Data loading
# ------------


class ROCStoriesDataset_with_missing(Dataset):
    def __init__(self, data_path=""):
        assert os.path.isfile(data_path)

        self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].values

        story_lines = row[0:4]
        missing_id = row[4]
        missing_sentence = row[5:6]

        return story_lines, missing_sentence, missing_id


class ROCStoriesDataset_random_missing(Dataset):
    def __init__(self, data_path=""):
        assert os.path.isfile(data_path)

        self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].values

        story_lines = row[0:5]

        missing_id = np.random.randint(low=0, high=5)

        missing_sentence = np.array([story_lines[missing_id]], dtype=object)
        remain_sentences = np.delete(story_lines, missing_id)

        return remain_sentences, missing_sentence, missing_id


# --------------------------
# Encoding and preprocessing
# --------------------------


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


def build_lm_labels(sequence, pad_token):
    """ Padding token, encoded as 0, are represented by the value -1 so they
    are not taken into account in the loss computation. """
    padded = sequence.clone()
    padded[padded == pad_token] = -1
    return padded


def build_mask(sequence, pad_token):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1. """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token
    mask[idx_pad_tokens] = 0
    return mask


def encode_for_storycompletion(story_lines, missing_sentence, tokenizer):
    """ Encode the story lines and missing sentence, and join them
    as specified in [1] by using `[SEP] [CLS]` tokens to separate
    sentences.
    """
    story_lines_token_ids = [
        tokenizer.encode(line, add_special_tokens=True) for line in story_lines
    ]
    missing_sentence_token_ids = [
        tokenizer.encode(line, add_special_tokens=True) for line in missing_sentence
    ]

    story_token_ids = [
        token for sentence in story_lines_token_ids for token in sentence
    ]
    missing_sentence_token_ids = [
        token for sentence in missing_sentence_token_ids for token in sentence
    ]

    return story_token_ids, missing_sentence_token_ids, story_lines_token_ids


def compute_token_type_ids(batch, separator_token_id):
    """ Segment embeddings as described in [1]
    The values {0,1} were found in the repository [2].
    Attributes:
        batch: torch.Tensor, size [batch_size, block_size]
            Batch of input.
        separator_token_id: int
            The value of the token that separates the segments.
    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    [2] https://github.com/nlpyang/PreSumm (/src/prepro/data_builder.py, commit fac1217)
    """
    batch_embeddings = []
    for sequence in batch:
        sentence_num = 0
        embeddings = []
        for s in sequence:
            if s == separator_token_id:
                sentence_num += 1
            embeddings.append(sentence_num % 2)
        batch_embeddings.append(embeddings)
    return torch.tensor(batch_embeddings)


# ----------------
# LOAD the dataset
# ----------------

Batch = namedtuple(
    "Batch",
    ["batch_size", "src", "mask_src", "missing_ids", "trg", "mask_trg", "tgt_str"],
)


def collate(data, tokenizer, block_size, device):
    """ Collate formats the data passed to the data loader.
    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    story_lines = [story_lines for story_lines, _, _ in data]
    missing_ids = torch.tensor([ids for _, _, ids in data])
    missing_sentences = [" ".join(missing_sentence) for _, missing_sentence, _ in data]

    encoded_text = [
        encode_for_storycompletion(story_lines, missing_sentence, tokenizer)
        for story_lines, missing_sentence, _ in data
    ]

    encoded_stories = torch.tensor(
        [
            [
                fit_to_block_size(line, block_size, tokenizer.pad_token_id)
                for line in story
            ]
            for _, _, story in encoded_text
        ]
    )

    encoded_missing_sentences = torch.tensor(
        [
            fit_to_block_size(missing_sentence, block_size, tokenizer.pad_token_id)
            for _, missing_sentence, _ in encoded_text
        ]
    )

    # encoder_token_type_ids = compute_token_type_ids(encoded_stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(encoded_stories, tokenizer.pad_token_id)
    decoder_mask = build_mask(encoded_missing_sentences, tokenizer.pad_token_id)

    batch = Batch(
        batch_size=len(encoded_stories),
        src=story_lines,
        # segs=encoder_token_type_ids.to(device),
        mask_src=encoder_mask.to(device),
        missing_ids=missing_ids.to(device),
        trg=encoded_missing_sentences.to(device),
        mask_trg=decoder_mask.to(device),
        tgt_str=missing_sentences,
    )

    return batch


# -----
# Model
# -----


class GRUContextEncoder(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super().__init__()

        self.rnn = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.linear.weight, 0.0, 0.01)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # x = [batch size, seq len (num sentence), hidden size]

        # trans_x = [seq len (num sentence), batch size, hidden size]
        trans_x = x.transpose(0, 1)

        # h = [batch size, hidden size]
        h = self.rnn(trans_x)[1][-1]

        h = self.linear(h)
        h = F.relu(self.bn(h))

        return h


class PoolContextEncoder(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super().__init__()

        self.linear = nn.Linear(input_size, hidden_size)
        nn.init.normal_(self.linear.weight, 0.0, 0.01)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # x = [batch size, seq len (num sentence), hidden size]

        # trans_x = [seq len (num sentence), batch size, hidden size]
        trans_x = x.transpose(0, 1)

        # h = [batch size, hidden size]

        # max pooling
        h = torch.max(trans_x, 0)[0]
        h = self.linear(h)
        h = F.relu(self.bn(h))

        return h


class MultiTask_StoryCompletionModel(nn.Module):
    def __init__(
        self,
        SentenceEncoder,
        device,
        ContextEncoder,
        no_contextencoder_before_languagemodel=False,
    ):
        super().__init__()

        self.sentence_encoder = SentenceEncoder

        # Context Encoder
        if ContextEncoder == "GRUContextEncoder":
            self.context_encoder = GRUContextEncoder(input_size=768, hidden_size=768)
        elif ContextEncoder == "PoolContextEncoder":
            self.context_encoder = PoolContextEncoder(input_size=768, hidden_size=768)

        self.decoder = BertLMHeadModel.from_pretrained(
            "bert-base-uncased",
            is_decoder=True,
            add_cross_attention=True,
            output_hidden_states=True,
        )

        self.mpp_classifier = nn.Linear(768, 5)

        self.device = device
        self.no_contextencoder_before_languagemodel = (
            no_contextencoder_before_languagemodel
        )

    def forward(self, story, story_mask, target=None, max_length=32, trg_start_id=101):

        batch_size = len(story)

        all_sentences_in_batch = list(itertools.chain.from_iterable(story))
        embeddings = self.sentence_encoder.encode(
            all_sentences_in_batch, show_progress_bar=False
        )
        embeddings = np.stack(embeddings, axis=0)
        embeddings = embeddings.reshape(batch_size, 4, -1)

        # embeddings_tensor = [batch size, num sentences, feature]
        embeddings_tensor = torch.tensor(embeddings).to(self.device)

        # context = [batch size, feature]
        context = self.context_encoder(embeddings_tensor)

        #
        # Missing Position Prediction
        #
        mpp_outputs = self.mpp_classifier(context)

        if self.no_contextencoder_before_languagemodel is False:
            # context also used as the initial hidden state of the decoder
            # hidden = [batch size, 1, feature]
            hidden = context.unsqueeze(1)
        else:
            # the output of Sentence-BERT is directly input to the BERT LM.
            # In other words: no-multitask mode.
            hidden = embeddings_tensor

        # Training
        if self.training and target is not None:
            # decoder_outputs = self.decoder(input_ids=target, encoder_hidden_states=hidden, lm_labels=target)
            decoder_outputs = self.decoder(
                input_ids=target, encoder_hidden_states=hidden, labels=target
            )

            sc_loss = decoder_outputs[0]

            return sc_loss, mpp_outputs

        # Inference
        else:
            generated = torch.tensor([[trg_start_id]] * batch_size).to(self.device)

            for t in range(1, max_length):
                decoder_outputs = self.decoder(
                    input_ids=generated, encoder_hidden_states=hidden
                )
                predictions = decoder_outputs[0]

                last_predictions = predictions[:, -1].unsqueeze(1)
                # sc_outputs = [batch size, length+1, trg_vocab_size]

                # When t == 1, make zeros for t == 0 ([CLS])
                if t == 1:
                    sc_outputs = torch.zeros(
                        batch_size, 1, last_predictions.shape[-1]
                    ).to(self.device)

                sc_outputs = torch.cat([sc_outputs, last_predictions], dim=1)

                predicted_index = torch.argmax(predictions[:, -1], dim=1).unsqueeze(1)
                generated = torch.cat([generated, predicted_index], dim=1)

            return generated, sc_outputs, mpp_outputs

    def generate(self, story, story_mask, tokenizer, max_length=32, trg_start_id=101):
        with torch.no_grad():
            generated_sentences = []
            generated, _, mpp_outputs = self.forward(
                story, story_mask, max_length=max_length
            )

            # generated = generated[1:].transpose(0, 1)

            for output_ind in generated:
                try:
                    output_decoded = tokenizer.decode(
                        output_ind.cpu().numpy(), skip_special_tokens=True
                    )
                    generated_sentences.append(output_decoded)
                except:
                    generated_sentences.append("")

        return generated_sentences, mpp_outputs


def train(
    model,
    iterator,
    optimizer,
    sc_criterion,
    mpp_criterion,
    clip,
    scheduler,
    loss_weight,
):

    model.train()

    epoch_loss = 0
    epoch_mpp_acc = 0

    for i, batch in enumerate(tqdm(iterator, desc="Iteration")):
        optimizer.zero_grad()

        batch_size = batch.batch_size
        story = batch.src
        story_mask = batch.mask_src
        target = batch.trg
        cls = batch.missing_ids

        trg = target.transpose(0, 1)
        sc_loss, mpp_outputs = model(story, story_mask, target)

        mpp_loss = mpp_criterion(mpp_outputs, cls)

        total_loss = (1 - loss_weight) * sc_loss + loss_weight * mpp_loss

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        scheduler.step()

        epoch_loss += total_loss.item()
        epoch_mpp_acc += (mpp_outputs.argmax(1) == cls).sum().item() / (
            batch_size + 0.0
        )

    return epoch_loss / len(iterator), epoch_mpp_acc / len(iterator)


def evaluate(
    model, iterator, sc_criterion, mpp_criterion, loss_weight, tb_valid_text=0
):

    model.eval()

    epoch_loss = 0
    epoch_mpp_acc = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Iteration")):

            batch_size = batch.batch_size
            story = batch.src
            story_mask = batch.mask_src
            target = batch.trg
            cls = batch.missing_ids

            trg = target.transpose(0, 1)

            # model inference
            generated, sc_outputs, mpp_outputs = model(story, story_mask)

            # trg = [trg len, batch size]
            # sc_outputs = [batch size, trg len, output dim]

            output_dim = sc_outputs.shape[-1]

            sc_outputs = sc_outputs.transpose(0, 1)
            sc_outputs = sc_outputs[1:].reshape(-1, output_dim)

            trg = trg[1:].reshape(-1)

            # trg = [(trg len - 1) * batch size]
            # sc_outputs = [(trg len - 1) * batch size, output dim]

            sc_loss = sc_criterion(sc_outputs, trg)
            mpp_loss = mpp_criterion(mpp_outputs, cls)

            total_loss = (1 - loss_weight) * sc_loss + loss_weight * mpp_loss

            epoch_loss += total_loss.item()
            epoch_mpp_acc += (mpp_outputs.argmax(1) == cls).sum().item() / (
                batch_size + 0.0
            )

            if tb_valid_text != 0 and i == 0:
                if tb_valid_text > len(generated):
                    tb_valid_text = len(generated)
                for j, output_ind in enumerate(generated[:tb_valid_text]):
                    try:
                        output_decoded = tokenizer.decode(
                            output_ind.cpu().numpy(), skip_special_tokens=True
                        )
                        tb_writer.add_text(
                            f"example {j}", output_decoded, global_step=epoch + 1
                        )
                    except:
                        tb_writer.add_text(f"example {j}", "", global_step=epoch + 1)

    return epoch_loss / len(iterator), epoch_mpp_acc / len(iterator)


def for_heatmap(model, iterator):
    model.eval()

    acc_heatmap = np.zeros((5, 5))
    cls_count = np.zeros(5)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Iteration")):

            batch_size = batch.batch_size
            story = batch.src
            cls = batch.missing_ids

            output = model(story)
            predicted = output.argmax(1)

            cls = cls.to("cpu").numpy()
            predicted = predicted.to("cpu").numpy()

            for e, c in zip(predicted, cls):
                acc_heatmap[e][c] += 1
                cls_count[c] += 1

    for i, cc in enumerate(cls_count):
        acc_heatmap[:][i] /= cc

    return acc_heatmap, cls_count


def show_result(model, iterator, tokenizer):
    model.eval()

    # missing_sentence = "____________________."

    result_to_show = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Iteration")):
            batch_size = batch.batch_size
            story = batch.src
            story_mask = batch.mask_src
            cls = batch.missing_ids
            original_sentence = batch.tgt_str

            sc_generated, mpp_output = model.generate(story, story_mask, tokenizer)
            predicted = mpp_output.argmax(1)

            cls = cls.to("cpu").numpy()
            predicted = predicted.to("cpu").numpy()

            for i in range(batch_size):
                input_story = " ".join(story[i]).lower()
                predicted_story = " ".join(
                    np.insert(story[i], predicted[i], sc_generated[i])
                ).lower()
                # gt_missing_story = " ".join(np.insert(story[i], cls[i], missing_sentence))
                gt_story = " ".join(
                    np.insert(story[i], cls[i], original_sentence[i])
                ).lower()

                result_to_show.append(
                    [
                        input_story,
                        predicted[i],
                        sc_generated[i],
                        predicted_story,
                        cls[i],
                        original_sentence[i],
                        gt_story,
                    ]
                )

    show_result_df = pd.DataFrame(
        result_to_show,
        columns=[
            "input",
            "pred_missing_id (0_indexed)",
            "generated_sentence",
            "pred_story",
            "gt_missing_id (0_indexed)",
            "gt_sentence",
            "gt_story",
        ],
    )

    return show_result_df


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of iterations to train"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="completion",
        help="task to solve: completion, ending",
    )

    parser.add_argument(
        "--context-encoder",
        "-ce",
        type=str,
        default="GRUContextEncoder",
        help="type of context encoder",
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--warmup_epochs", default=0, type=int, help="Linear warmup over warmup_epochs."
    )

    parser.add_argument(
        "--loss_weight",
        default=0.5,
        type=float,
        help="MPP loss weight in total loss. \
                              total_loss = (1 - loss_weight) * sc_loss + loss_weight * mpp_loss",
    )

    parser.add_argument(
        "--no_contextencoder_lm",
        action="store_true",
        help="Not using Context Encoder before BERT decoder LM.",
    )

    parser.add_argument(
        "--save_every_epoch", action="store_true", help="Save the model of every epoch."
    )

    parser.add_argument(
        "--tb_valid_text",
        default=0,
        type=int,
        help="The number of example texts for tensorboard every epoch.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set seed
    set_seed(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    block_size = 32
    N_EPOCHS = args.epochs
    CLIP = args.max_grad_norm

    if args.task == "completion":
        print("Task: Story Completion")
        train_dataset = ROCStoriesDataset_random_missing(
            data_path="../data/rocstories_completion_train.csv"
        )
        val_dataset = ROCStoriesDataset_with_missing(
            data_path="../data/rocstories_completion_dev.csv"
        )
    elif args.task == "ending":
        print("Task: Story Ending Generation")
        train_dataset = ROCStoriesDataset(
            data_path="../data/rocstories_for_storyendinggeneration_train.csv"
        )
        val_dataset = ROCStoriesDataset(
            data_path="../data/rocstories_for_storyendinggeneration_val.csv"
        )

    sentbertmodel = SentenceTransformer("bert-base-nli-mean-tokens")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # --- model ---
    model = MultiTask_StoryCompletionModel(
        SentenceEncoder=sentbertmodel,
        device=device,
        ContextEncoder=args.context_encoder,
        no_contextencoder_before_languagemodel=args.no_contextencoder_lm,
    ).to(device)

    # --- DataLoader ---
    collate_fn = lambda data: collate(
        data, tokenizer, block_size=block_size, device=device
    )
    train_iterator = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    valid_iterator = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    TRG_PAD_IDX = tokenizer.pad_token_id
    START_ID = tokenizer.cls_token_id
    sc_criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    mpp_criterion = nn.CrossEntropyLoss()

    # best_valid_loss = float('inf')
    best_valid_mpp_acc = 0.0

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    # learning rate scheduler
    warmup_steps = args.warmup_epochs * len(train_iterator)
    print(f"warmup steps: {warmup_steps}")
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps
    )

    with mlflow.start_run():
        # Log our parameters into mlflow
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        comment = os.path.basename(args.output_dir)
        comment = "" if comment == "" else "_" + comment

        tb_writer = SummaryWriter(comment=comment)

        for epoch in trange(N_EPOCHS, desc="Epoch"):

            start_time = time.time()

            train_loss, train_mpp_acc = train(
                model,
                train_iterator,
                optimizer,
                sc_criterion,
                mpp_criterion,
                CLIP,
                scheduler=scheduler,
                loss_weight=args.loss_weight,
            )
            valid_loss, valid_mpp_acc = evaluate(
                model,
                valid_iterator,
                sc_criterion,
                mpp_criterion,
                loss_weight=args.loss_weight,
                tb_valid_text=args.tb_valid_text,
            )

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # save best model

            # if valid_loss < best_valid_loss:
            #     best_valid_loss = valid_loss
            #     torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))

            if valid_mpp_acc > best_valid_mpp_acc:
                best_valid_mpp_acc = valid_mpp_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "best_mpp_acc_model.pt"),
                )

            # save each epoch model
            if args.save_every_epoch:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.output_dir, "epoch_{:0=3}_model.pt".format(epoch + 1)
                    ),
                )

            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            # print(f'lr: {scheduler.get_lr()[0]}')
            print(f"\tlr: {scheduler.get_last_lr()[0]}")
            print(
                f"\tTrain Loss: {train_loss:.3f} | Train MPP Accuracy: {train_mpp_acc * 100:.1f}%"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f} |  Val. MPP Accuracy: {valid_mpp_acc * 100:.1f}%"
            )

            # tensorboard and mlflow
            log_scalar("train_loss", train_loss, epoch + 1)
            log_scalar("train_mpp_acc", train_mpp_acc, epoch + 1)
            log_scalar("valid_loss", valid_loss, epoch + 1)
            log_scalar("valid_mpp_acc", valid_mpp_acc, epoch + 1)
            log_scalar("lr", scheduler.get_last_lr()[0], epoch + 1)

            # tensorboard
            # tb_writer.add_scalar('train_loss', train_loss, epoch+1)
            # tb_writer.add_scalar('train_mpp_acc', train_mpp_acc, epoch+1)
            # tb_writer.add_scalar('valid_loss', valid_loss, epoch+1)
            # tb_writer.add_scalar('valid_mpp_acc', valid_mpp_acc, epoch+1)
            # tb_writer.add_scalar('lr', scheduler.get_last_lr(), epoch+1)

        # Upload the TensorBoard event logs as a run artifact
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(tb_writer.log_dir, artifact_path="events")
        # print("\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s" %
        #    os.path.join(mlflow.get_artifact_uri(), "events"))

        tb_writer.close()
