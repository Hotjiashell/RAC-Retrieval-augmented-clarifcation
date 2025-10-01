import os
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase
from typing import Optional, Union, Callable


class PromptFormatter:
    template: Optional[str] = None
    response_template: Optional[str] = None

    def format_dataset(self, example):
        raise NotImplementedError(
            "Subclasses must implement the format method.")


# class DefaultPromptFormatter(PromptFormatter):
#     def __init__(self, template: Optional[str] = None):
#         super().__init__()
#         self.template = template
#     def format(self, question: str, answers: List[str]) -> str:
#         return f"Q: {question}\n\n" + "\n".join(f"A{i + 1}: {ans}" for i, ans in enumerate(answers))


class PassageClarificationPromptFormatter(PromptFormatter):
    def __init__(
        self,
        template: Optional[
            str
        ] = """You are given a user question and a subtopic. 
Your task is to ask a clarifying question that helps better understand what the user is asking. 
Focus strictly on content of the given passage. 
Do not introduce any new information or assumptions. 
Your clarifying question should be simple, clear, and directly related to both the user question and the passage.
"""
        "You are given a user question and a passage. Your task is to ask a clarifying question that helps better understand what the user needs. Your clarifying question must be faithful to the passage: do not introduce any information not present in the passage. Focus only on the content of the given passage.Your question should be simple, clear, and directly related to the user's question and the passage.",
        response_template: Optional[
            str
        ] = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    ):
        super().__init__()
        self.template = template
        self.response_template = response_template

#     def format_dataset(self, example):
#         # example['text'] = f"""You are given a user question and a passage. Your task is to ask a clarifying question that helps better understand what the user needs. Your clarifying question must be faithful to the passage: do not introduce any information not present in the passage. Focus only on the content of the given passage. Your question should be simple, clear, and directly related to the user's question and the passage. ### User Question: {example['topic']} ### Passage: {example['Passage']} ### Clarifying Question : {example["question"]}"""
#         # example["text"] = (
#         #     f""" {self.template} ### User Question: {example['topic']} ### Passage: {example['Passage']} ### Clarifying Question : {example["question"]}"""
#         # )

#         example[
#             "text"
#         ] = f"""You are given a user question and a clarifying question.
# Your task is to ask a clarifying question that helps better understand what the user is asking.
# Focus strictly on content of the given passage.
# Do not introduce any new information or assumptions.
# Your clarifying question should be simple, clear, and directly related to both the user question and the passage.

# User Question: {example['topic']}
# Passage: {example['Passage']}
# Clarifying Question: {example["question"]}"""
#         return example

    def format_dataset_no_chat(self, examples):
        """
        Formats the dataset for no_chat template generation.
        """

        # System + user content as plain string for main prompt
        examples["prompt"] = (
            "You are given a user question and a list of passages. \n"
            "Your task is to ask a clarifying question that helps better understand what the user is asking. \n"
            "Focus strictly on content of the given passages. \n"
            "Do not introduce any new information or assumptions. \n"
            "Your clarifying question should be simple, clear, and directly related to both the user question and the passages.\n\n"
            f"User Query: {examples['topic']}\n"
            f"Passages: {examples['Passage']}\n"
            "Clarifying Question:"
        )

        # Completion is a plain string
        examples["completion"] = examples["question"]

        # conv is same as prompt
        examples["conv"] = examples["prompt"]

        # Weak supervision version (no passages)
        examples["weak_conv"] = (
            "You are given a user question.\n"
            "Your task is to ask a clarifying question that helps better understand what the user is asking.\n"
            "Your clarifying question should be simple, clear, and directly related to a particular part of the user’s question.\n\n"
            f"User Query: {examples['topic']}\n"
            "Clarifying Question:"
        )

        return examples

    def format_dataset(self, examples):
        """

        """
        examples["prompt"] = [{
            "role": "system",
            "content": """You are given a user question and a list of passages. 
    Your task is to ask a clarifying question that helps better understand what the user is asking. 
    Focus strictly on content of the given passages. 
    Do not introduce any new information or assumptions. 
    Your clarifying question should be simple, clear, and directly related to both the user question and the passages.""",
        },
            {
                "role": "user",
                "content": f"""User Query: {examples["topic"]}
                \n Passages: {examples["Passage"]} \nClarifying Question:""",
        }]
        examples["completion"] = [{
            "role": "assistant",
            "content": examples["question"]
        }
        ]

        examples["conv"] = [
            {
                "role": "system",
                "content": """You are given a user question and a list of passages. 
    Your task is to ask a clarifying question that helps better understand what the user is asking. 
    Focus strictly on content of the given passages. 
    Do not introduce any new information or assumptions. 
    Your clarifying question should be simple, clear, and directly related to both the user question and the passages.""",
            },
            {
                "role": "user",
                "content": f"""User Query: {examples["topic"]}
                \n Passages: {examples["Passage"]} \nClarifying Question:""",
            }
        ]
    #     examples["weak_conv"] = [
    #         {
    #             "role": "system",
    #             "content": """You are given a user question.
    # Your task is to ask a clarifying question that helps better understand what the user is asking.
    # Avoid asking vague or general questions. Instead, focus on uncovering specific subtopics or aspects of the user's query that need clarification.
    # Your clarifying question should be simple, clear, and directly related to a particular part of the user’s question."""},
    #         {"role": "user",
    #             "content": f"""User Query: {examples["topic"]} \nClarifying Question:"""}
    #     ]

        # examples["weak_conv"] = [
        #     {
        #         "role": "user",
        #         "content": f"""User question: Who invented the telephone?
        # Clarifying question: Are you asking about Alexander Graham Bell’s 1875 prototype in Boston or the experimental device built in Turin in 1873 by Carlo Bianchi?

        # User question: What is the fastest animal on Earth?
        # Clarifying question: Do you mean the 2020 Dubai-trained peregrine falcon "Skyflash" or the Kenyan cheetah that set the 2015 unofficial sprint record?

        # User question: ${examples["topic"]}
        # Clarifying question:"""
        #     }
        # ]
        examples["weak_conv"] = [
            {
                "role": "system",
                "content": """You are given a user question.
    Your task is to ask a clarifying question that helps better understand what the user is asking.
    Your clarifying question should be simple, clear, and directly related to a particular part of the user’s question."""},
            {"role": "user",
                "content": f"""User Query: {examples["topic"]} \nClarifying Question:"""}
        ]
        return examples

    def format_dataset_preference_tuning_no_chat(self, examples):
        """
        Formats the dataset for preference tuning in no_chat template format.
        """

        # Instruction + prompt as a plain string
        examples["prompt"] = (
            "You are given a user question and a list of passages.\n"
            "Your task is to ask a clarifying question that helps better understand what the user is asking.\n"
            "Focus strictly on content of the given passages.\n"
            "Do not introduce any new information or assumptions.\n"
            "Your clarifying question should be simple, clear, and directly related to both the user question and the passages.\n\n"
            f"User Query: {examples['topic']}\n"
            f"Passages: {examples['Passage']}\n"
            "Clarifying Question:"
        )

        # Preferred and rejected outputs as plain strings
        examples["chosen"] = examples["chosen"]
        examples["rejected"] = examples["rejected"]

        return examples

    def format_dataset_preference_tuning(self, examples):
        # example["prompt"] = (
        #     f""" {self.template} ### User Question: {example['topic']} ### Passage: {example['Passage']} ### Clarifying Question : """
        # )
        #         example[
        #             "prompt"
        #         ] = f"""You are given a user question and a clarifying question.
        # Your task is to ask a clarifying question that helps better understand what the user is asking.
        # Focus strictly on content of the given passage.
        # Do not introduce any new information or assumptions.
        # Your clarifying question should be simple, clear, and directly related to both the user question and the passage.

        # User Question: {example['topic']}
        # Passage: {example['Passage']}
        # Clarifying Question:"""

        #         example["chosen"] = f"""{example.chosen}"""
        #         example["rejected"] = f"""{example.rejected}"""
        #         return example

        examples["prompt"] = [


            {
                "role": "system",
                "content": """You are given a user question and a list of passages. 
    Your task is to ask a clarifying question that helps better understand what the user is asking. 
    Focus strictly on content of the given passages. 
    Do not introduce any new information or assumptions. 
    Your clarifying question should be simple, clear, and directly related to both the user question and the passages.""",
            },
            {
                "role": "user",
                "content": f"""User Query: {examples["topic"]}
                \n Passages: {examples["Passage"]} \nClarifying Question:""",
            }]
        examples["chosen"] = [{
            "role": "assistant",
            "content": examples["chosen"]
        }]
        examples["rejected"] = [{
            "role": "assistant",
            "content": examples["rejected"]
        }]

        return examples

    def format_dataset_multi_preference_tuning_no_chat(self, examples):
        """
        Formats the dataset for multi-preference tuning in no_chat template format.
        """

        # Plain string prompt (system + user content)
        examples["prompt"] = (
            "You are given a user question and a list of passages.\n"
            "Your task is to ask a clarifying question that helps better understand what the user is asking.\n"
            "Focus strictly on content of the given passages.\n"
            "Do not introduce any new information or assumptions.\n"
            "Your clarifying question should be simple, clear, and directly related to both the user question and the passages.\n\n"
            f"User Query: {examples['topic']}\n"
            f"Passages: {examples['Passage']}\n"
            "Clarifying Question:"
        )

        # Convert 'chosen' to plain string
        examples["chosen"] = examples["chosen"]

        # Convert all rejected_* keys to plain string
        rejected_columns = [
            col for col in examples if col.startswith("rejected_")]
        for col in rejected_columns:
            examples[col] = examples[col]

        return examples

    def format_dataset_multi_preference_tuning(self, examples):

        examples["prompt"] = [


            {
                "role": "system",
                "content": """You are given a user question and a list of passages. 
    Your task is to ask a clarifying question that helps better understand what the user is asking. 
    Focus strictly on content of the given passages. 
    Do not introduce any new information or assumptions. 
    Your clarifying question should be simple, clear, and directly related to both the user question and the passages.""",
            },
            {
                "role": "user",
                "content": f"""User Query: {examples["topic"]}
                \n Passages: {examples["Passage"]} \nClarifying Question:""",
            }]
        examples["chosen"] = [{
            "role": "assistant",
            "content": examples["chosen"]
        }]

        columns = [rej for rej in examples.keys(
        ) if rej.startswith("rejected_")]
        for col in columns:
            examples[col] = [{
                "role": "assistant",
                "content": examples[col]
            }]
        # examples["rejected"] = [{
        #     "role": "assistant",
        #     "content": examples["rejected"]
        # }]

        return examples


    def format_dataset_zero_shot_compact(self, examples):
        """

        """
        examples["prompt"] = [{
            "role": "system",
            "content": """You are given a user question and a list of passages. 
    Your task is to ask a clarifying question that helps better understand what the user is asking. 
    Focus strictly on content of the given passages. 
    Do not introduce any new information or assumptions. 
    Your clarifying question should be simple, consice, clear, and directly related to both the user question and the passages.
    It should addresse only one facet or subtopic.
    DO NOT:
        - Include any explanation, justification, or commentary.
        - Add phrases like 'This clarifying question is simple, clear' or similar.
        - Anything after the clarification
    DO NOT SAY this question is simple, this question is good ....""",
        },
            {
                "role": "user",
                "content": f"""User Query: {examples["topic"]}
                \n Passages: {examples["Passage"]} \n(only output the clarifying question, no explanation)Clarifying Question:""",
        }]
        examples["completion"] = [{
            "role": "assistant",
            "content": examples["question"]
        }
        ]

        examples["conv"] = [
            {
                "role": "system",
                "content": """You are given a user question and a list of passages. 
    Your task is to ask a clarifying question that helps better understand what the user is asking. 
    Focus strictly on content of the given passages. 
    Do not introduce any new information or assumptions. 
    Your clarifying question should be simple, consice, clear, and directly related to both the user question and the passages.
    It should addresse only one facet or subtopic.
    DO NOT:
        - Include any explanation, justification, or commentary.
        - Add phrases like 'This clarifying question is simple, clear' or similar.
        - Anything after the clarification
    DO NOT SAY this question is simple, this question is good ....""",
            },
            {
                "role": "user",
                "content": f"""User Query: {examples["topic"]}
                \n Passages: {examples["Passage"]} \n(only output the clarifying question, no explanation)Clarifying Question:""",
            }
        ]
        examples["weak_conv"] = [
            {
                "role": "system",
                "content": """You are given a user question.
    Your task is to ask a clarifying question that helps better understand what the user is asking.
    Your clarifying question should be simple, clear, and directly related to a particular part of the user’s question."""},
            {"role": "user",
                "content": f"""User Query: {examples["topic"]} \n(only output the clarifying question, no explanation)Clarifying Question:"""}
        ]
        return examples




class FacetClarificationPromptFormatter(PromptFormatter):
    def __init__(
        self,
        template: Optional[
            str
        ] = "You are given a user question and a subtopic. Your job is to ask a clarifying question that helps better understand what the user needs. Focus only on the subtopic given. Do not add any new information. Your question should be simple, clear, and directly related to the user's question and the subtopic.",
        response_template: Optional[
            str
        ] = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    ):
        super().__init__()
        self.template = template
        self.response_template = response_template
    def format_dataset_no_chat(self, examples):
        """
        Formats the dataset for no_chat template generation.
        """

        # System + user content as plain string for main prompt
        examples["prompt"] = (
            "You are given a user question and a a facet. \n"
            "Your task is to ask a clarifying question that helps better understand what the user is asking. \n"
            "Focus strictly on content of the given passages. \n"
            "Do not introduce any new information or assumptions. \n"
            "Your clarifying question should be simple, clear, and directly related to both the user question and the facet.\n\n"
            f"User Query: {examples['topic']}\n"
            f"Sub-topic: {examples['facet_desc']}\n"
            "Clarifying Question:"
        )

        # Completion is a plain string
        examples["completion"] = examples["question"]

        # conv is same as prompt
        examples["conv"] = examples["prompt"]

        # Weak supervision version (no passages)
        examples["weak_conv"] = (
            "You are given a user question.\n"
            "Your task is to ask a clarifying question that helps better understand what the user is asking.\n"
            "Your clarifying question should be simple, clear, and directly related to a particular part of the user’s question.\n\n"
            f"User Query: {examples['topic']}\n"
            "Clarifying Question:"
        )

        return examples

    def format_dataset(self, example):
        example["text"] = (
            f"""{self.template} ### User Question: {example['topic']} ### Subtopic: {example['facet_desc']} ### Clarifying Question : {example["question"]}"""
        )
        return example


class DPOPassagePrefencePromptFormatter(PromptFormatter):
    def __init__(
        self,
        template: Optional[
            str
        ] = "You are given a user question and a passage. Your task is to ask a clarifying question that helps better understand what the user needs. Your clarifying question must be faithful to the passage: do not introduce any information not present in the passage. Focus only on the content of the given passage.Your question should be simple, clear, and directly related to the user's question and the passage.",
    ):
        super().__init__()
        self.template = template

    def format_dataset(self, example):
        example["prompt"] = (
            f""" {self.template} ### User Question: {example['topic']} ### Passage: {example['Passage']} ### Clarifying Question : {example["question"]}"""
        )
        example["chosen"] = f"""{self.chosen}"""
        example["rejected"] = f"""{self.rejected}"""
        return example


class UnconditionalClarificationPromptFormatter(PromptFormatter):
    def __init__(
        self,
        template: Optional[
            str
        ] = """You are given a user question and a subtopic. 
Your task is to ask a clarifying question that helps better understand what the user is asking. 
Focus strictly on content of the given passage. 
Do not introduce any new information or assumptions. 
Your clarifying question should be simple, clear, and directly related to both the user question and the passage.
"""
        "You are given a user question and a passage. Your task is to ask a clarifying question that helps better understand what the user needs. Your clarifying question must be faithful to the passage: do not introduce any information not present in the passage. Focus only on the content of the given passage.Your question should be simple, clear, and directly related to the user's question and the passage.",
        response_template: Optional[
            str
        ] = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    ):
        super().__init__()
        self.template = template
        self.response_template = response_template

    def format_dataset_no_chat(self, examples):
        """
        Formats the dataset for no_chat template (no role-based chat structure).
        """

        ambiguity_type_definitions = """The ambiguity of a query can be multifaceted, and there are multiple possible ambiguity types: 
        [1] Semantic: the query is semantically ambiguous for several common reasons: it may include homonyms; a word in the query may refer to a specific entity while also functioning as a common word; or an entity mention in the query could refer to multiple distinct entities. 
        [2] Generalize: the query focuses on specific information; however, a broader, closely related query might better capture the user's true information needs.
        [3] Specify: the query has a clear focus but may encompass too broad a research scope. It is possible to further narrow down this scope by providing more specific information related to the query.\n"""

        instruction = "You are a conversational search system, generate a short and concise clarification question that you think is most appropriate to gain a better understanding of the user's intention. Do not give example, make it the simplest and shortest possible"

        instruction += ambiguity_type_definitions + f"Before generating Clarifying question, provide a textual explanation of your reasoning about which types of ambiguity apply to the given query. "\
                                                                        "Based on these ambiguity types, describe how you plan to clarify the original query.\n"
        examples["prompt"] = (
            "You are given a user question.\n"
            "Your task is to ask a clarifying question that helps better understand what the user is asking.\n"
            "Your clarifying question should be simple, clear, and directly related to a particular part of the user’s question.\n\n"
            f"User Query: {examples['topic']}\n"
            "Clarifying Question:"
        )

        # examples["prompt"] = {
        #     f"{instruction}\n"
        #     f"User Query: Find me a book about the civil war.\n"
        #     "The query \"Find me a book about the civil war\" appears to suffer from **Semantic** and **Specify** ambiguities.\n\n- **Semantic**: The phrase \"the civil war\" could refer to different historical events depending on the user's region or interest. For example, it could mean the American Civil War, the English Civil War, the Spanish Civil War, or others.\n- **Specify**: Even if the intended civil war is known (e.g., the American Civil War), the query is broad—it doesn't indicate whether the user wants a factual, historical nonfiction book or a fictional story set during that period.\n\nTo clarify the original query, I would like to ask this Clarifying Question:\n\n\"Are you looking for a nonfiction historical account or a fictional story set during the Civil War?\"\n\nThis clarifying question helps disambiguate the user's intent in terms of genre and narrows down the search scope to better satisfy their information need."
        # }

        examples["completion"] = examples["question"]

        return examples

    def format_dataset(self, examples):
        """

        """
        examples["prompt"] = [
            {
                "role": "system",
                "content": """You are given a user question.
    Your task is to ask a clarifying question that helps better understand what the user is asking.
    Your clarifying question should be simple, clear, and directly related to a particular part of the user’s question."""},
            {"role": "user",
                "content": f"""User Query: {examples["topic"]} \nClarifying Question:"""}
        ]
        examples["completion"] = [{
            "role": "assistant",
            "content": examples["question"]
        }
        ]
        return examples


def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

    For more details, see [`maybe_apply_chat_template`].
    """
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "completion", "messages", "label"]
    base_keys = set(example.keys())
    # Identify rejected keys dynamically
    rejected_keys = [key for key in base_keys if key.startswith("rejected")]
    example_keys = {
        key for key in base_keys if key in supported_keys or key in rejected_keys}

    # Validate
    valid_key_sets = [
        {"messages"},
        {"prompt"},
        {"prompt", "completion"},
        {"prompt", "chosen"} | set(rejected_keys),
        {"chosen"} | set(rejected_keys),
        {"prompt", "completion", "label"},
    ]
    if not any(example_keys == valid_set for valid_set in valid_key_sets):
        raise KeyError(f"Invalid keys in the example: {example_keys}")

    # Apply the chat template to the whole conversation
    if "messages" in example:
        messages = tokenizer.apply_chat_template(
            example["messages"], tools=tools, tokenize=False)

    # Apply chat template to prompt
    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role == "user":
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role in the last message: {last_role}")
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Apply the chat template to completions (chosen, rejected, etc.)
    output = {}

    if "prompt" in example:
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"], tools=tools, tokenize=False
            )
            chosen = prompt_chosen[len(prompt):]
            output["chosen"] = chosen

        for key in rejected_keys:
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example[key], tools=tools, tokenize=False
            )
            rejected = prompt_rejected[len(prompt):]
            output[key] = rejected

        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tools=tools, tokenize=False
            )
            completion = prompt_completion[len(prompt):]
            output["completion"] = completion
    else:
        # Implicit prompt
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(
                example["chosen"], tools=tools, tokenize=False)
            output["chosen"] = chosen
        for key in rejected_keys:
            rejected = tokenizer.apply_chat_template(
                example[key], tools=tools, tokenize=False)
            output[key] = rejected

    # Validation
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
            "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
        )
        if "chosen" in example and not prompt_chosen.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_chosen))
        for key in rejected_keys:
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example[key], tools=tools, tokenize=False)
            if not prompt_rejected.startswith(prompt):
                raise ValueError(error_message.format(prompt, prompt_rejected))
        if "completion" in example and not prompt_completion.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_completion))

    # Add other fields
    if "messages" in example:
        output["text"] = messages
    if "prompt" in example:
        output["prompt"] = prompt
    if "label" in example:
        output["label"] = example["label"]

    return output
