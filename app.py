from flask import Flask, render_template, request
from transformers import pipeline
# pip install flask
# pip install transformers
# pip install tensorflow
# pip install sentencepiece
# pip install huggingface_hub[hf_xet]
# pip install hf_xet



app = Flask(__name__)

# Load NLP pipelines
sentiment_pipeline = pipeline('sentiment-analysis')
generation_pipeline = pipeline('text-generation')
translation_pipeline = pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
summarization_pipeline = pipeline('summarization')
NER_pipeline = pipeline("ner", grouped_entities=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    sent = None
    gen = None
    trans = None
    ner = None
    summ = None
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        task = request.form.get('task')

        if task == 'sentiment':
            sent = sentiment_pipeline(user_input)
            return render_template('index.html', sent=sent)
        elif task == 'generation':
            gen = generation_pipeline(user_input)
            return render_template('index.html', gen=gen)
        elif task == 'translation':
            trans = translation_pipeline(user_input)
            return render_template('index.html', trans=trans)
        elif task == 'summarization':
            summ = summarization_pipeline(user_input)
            return render_template('index.html', summ=summ)
        elif task == 'named_entity_recognition':
            ner = NER_pipeline(user_input)
            return render_template('index.html', ner=ner)



    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)