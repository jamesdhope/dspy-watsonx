# dspy-watsonx

notebooks adapted from Stanford NLP for use with IBM watsonx.ai models:

1. Optimisation of a prompt for a RAG system

This notebook demonstrates 3-shot prompt optimisation for retrieval-augmented generation. The Wikipedia 2017 "abstracts" is used as the source data. The HotPotQA dataset is used for question-answer candidate pairs to optimise and evaluate the prompt. The metric used is `dspy.evaluate.answer_exact_match` and `dspy.evaluate.answer_passage_match`.

2. Optimisation of a prompt for a multi-hop QA RAG system

This notebook demonstrates few-shot prompt optimisation with multiple QA hops (or multi-turn QA). The Wikipedia 2017 "abstracts" is used as the source data. The HotPotQA dataset is used for question-answer candidate pairs to optimise the prompt over several iterations of query generation, retrieval and answer generation. The metric is extended to penalise verbose model responses.

3. Optimisation of a prompt for a multi-hop QA RAG system with model coercion

This notebook is similar to the previous however `dspy.Assert` and `dspy.Suggest` classes are used to coerce the model during forward the pass. See https://dspy-docs.vercel.app/api/assertions#dspyassert-and-dspysuggest-api for further info.

In each case, the DSPy Language Model Abstract Class is implemented for calling IBM watsonx.ai models:

```
class WatsonX(LM):
    def __init__(self,model,api_key):
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        self.model = model
        self.api_key = api_key
        self.provider = "default"
        self.history = []
        
        self.base_url = os.environ['WATSONX_URL']

        self.project_id = os.environ['WATSONX_PROJECTID']

    def basic_request(self, prompt: str, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "content-type": "application/json"
        }

        data = {
            "parameters": {**kwargs},
            "model_id": self.model,
            "input": prompt,
            "project_id": self.project_id
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        response = response.json()

        self.history.append({
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        })
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)

        print(response)

        completions = [result["generated_text"] for result in response["results"]]

        return completions
```

For tutorials see: https://github.com/stanfordnlp/dspy?tab=readme-ov-file#2-documentation


References:

[1] https://github.com/stanfordnlp/dspy/tree/main/examples/qa/hotpot

[2] https://dspy-docs.vercel.app/docs/tutorials/rag

[3] https://github.com/stanfordnlp/dspy?tab=readme-ov-file#2-documentation
