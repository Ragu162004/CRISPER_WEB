import torch
import asyncio
import warnings
from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer
from flair.data import Sentence
from flair.models import SequenceTagger

app = Flask(__name__)

warnings.filterwarnings("ignore")
model_name = "facebook/bart-large-xsum"  # Summarization Model
tokenizer = BartTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
summarization_model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
ner_tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")  # NER Model


async def process_data(input_text):
    # Text Summarization
    tokenized_text = tokenizer.encode("summarize: " + input_text, return_tensors='pt', max_length=512,
                                      truncation=True).to(device)
    summary_ = summarization_model.generate(
        tokenized_text, min_length=80, max_length=2000, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=2
    )
    summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

    # NER
    sentence = Sentence(input_text)
    ner_tagger.predict(sentence)
    ner_spans = [(entity.text, entity.tag) for entity in sentence.get_spans('ner')]

    return summary, ner_spans


@app.route('/')
def home():
    return render_template('input.html')


@app.route('/text-summarization-and-ner', methods=["POST"])
def process_input():
    global ner_spans, summary
    if request.method == "POST":
        input_text = request.form["inputtext_"]

        # Run the processing asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        summary, ner_spans = loop.run_until_complete(process_data(input_text))

    return render_template("output.html", data={"summary": summary, "ner_spans": ner_spans})


if __name__ == '__main__':
    app.run()
