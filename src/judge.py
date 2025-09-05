from typing import Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI


class JudgeVerdict(BaseModel):
    match: Literal[True, False] = Field(
        description="True iff predicted and actual are semantically equivalent under the rubric."
    )
    reason: Optional[str] = Field(
        default=None,
        description="1–2 line justification for the decision."
    )


# --- Core judge ---
def llm_judge(predicted: str, actual: str, client: OpenAI) -> bool:
    system = (
        """You are a strict evaluator. Compare PREDICTED and ACTUAL:
        Rules:
        - If JSON, ignore the keys and only compare the values.
        - Ignore whitespace and letter casing.
        - If numbers, treat string and numeric forms as equal.
        - Return a structured result with:
            - match: true if values are equal, false otherwise
            - reason: one short line why
        """
    )
    # pick first available model exposed by vLLM
    models = client.models.list()

    # print("Available models:", models)

    model = models.data[0].id

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"PREDICTED:\n{predicted}\n\nACTUAL:\n{actual}\n\nReturn only the structured result (JSON)."
            },
        ],
        text_format=JudgeVerdict,
        temperature=0,
    )

    print("Raw LLM output:", resp)

    verdict: JudgeVerdict = resp.output_parsed
    return verdict.match


def main():
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    print(llm_judge(" -110916 ", "-110916", client))             # True
    # False per rubric
    print(llm_judge('{"missing":"f8c8"}', '{"missing move":"f8c8"}', client))


if __name__ == "__main__":
    main()
