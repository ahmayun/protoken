from typing import Literal
from pydantic import BaseModel
from openai import OpenAI





class JudgeVerdict(BaseModel):
    is_correct: Literal[True, False]
    

# --- Core judge ---
def llm_judge(generated: str, actual: str, client=None, model=None) -> bool:
    if client is None:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        models = client.models.list()
        model = models.data[0].id


    system = (
        "You are an evaluator. Judge how well a generated response matches a reference. "
        "The Generated response may contain synonyms or paraphrasing but it should match the meaning to Reference."
    )



    USER_PROMPT_TEMPLATE = (
        f"Reference: {actual}\n\n"
        f"Generated: {generated}\n\n"
        "Rate as one of:\n"
        "True - if Generated and Reference inputs are same.\n"
        "False – Completely wrong or irrelevant\n\n"
        "Return only the structured result (JSON)."
    )

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE,
            },
        ],
        text_format=JudgeVerdict,
        temperature=0,
    )

    # print("Raw LLM output:", resp)

    verdict = resp.output_parsed
    return verdict.is_correct


def main():
    print(llm_judge(" -110916 ", "-110916"))             # True
    # False per rubric
    print(llm_judge('{"missing":"f8c8"}', '{"missing move is this one: hellolkdfjkjf":  "f8c8"}'))


if __name__ == "__main__":
    main()
