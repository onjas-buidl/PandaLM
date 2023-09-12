import os
import sys
import re
# import fire
import gradio as gr
import torch
import transformers
import traceback

from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from queue import Queue
from threading import Thread


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def build_prompt(instruction, input, resp1, resp2, result=None, explain=None, ref=None):
    rsp = f"### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}"

    if input:
        input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n{rsp}\n\n### Evaluation:\n"
    else:
        input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n{rsp}\n\n### Evaluation:\n"

    if result:
        output_sequence = f"{result}\n\n### Reason: {explain}\n\n### Reference: {ref}\n"
        return input_sequence, output_sequence
    else:
        return input_sequence


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def post_process_output(text):
    text = text.strip().split("### Evaluation:")[1].strip()
    pattern = re.compile(
        r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
    )
    pattern.sub("", text.strip()).strip()
    return text


# def main(
load_8bit: bool = False
base_model: str = "WeOpenML/PandaLM-7B-v1"
server_name: str = "0.0.0.0"  # Allows to listen on all interfaces by providing '0.
share_gradio: bool = False
server_port: int = 31228
# ):
base_model = base_model or os.environ.get("BASE_MODEL", "")
assert (
    base_model
), "Please specify a --base_model, e.g. --base_model='WeOpenML/PandaLM-7B-v1'"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_8bit,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def fib(a):
    a += 1 
    
    return a 

# str_code = """
def evaluate(
    instruction,
    input=None,
    response1=None,
    response2=None,
    temperature=0,
    top_p=1,
    top_k=1,
    num_beams=4,
    max_new_tokens=150,
    stream_output=False,
    repetition_penalty=1.2,
    early_stopping=True,
    **kwargs,
):
    # prompt = prompter.generate_prompt(instruction, input)
    prompt = build_prompt(instruction, input, response1, response2)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        early_stopping=early_stopping,
        repetition_penalty=repetition_penalty,
        **kwargs,
    )
    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }
    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.
        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(Stream(callback_func=callback))
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(generate_with_callback, kwargs, callback=None)

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)
                if output[-1] in [tokenizer.eos_token_id]:
                    break
                yield post_process_output(decoded_output)
        return  # early return for stream_output
    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield post_process_output(output)


# if __name__ != "__main__":
# Define the instruction and responses
# instruction = "Give me a summary of the public opinion on Trump."
# response1 = """Public opinion on Donald Trump, the 45th President of the United States, has been highly polarized and varied over the years, with significant differences among different demographic groups and political affiliations. As of my last knowledge update in September 2021, here's a general summary of the key trends in public opinion regarding Donald Trump:

# 1. **Polarization**: Trump's presidency was marked by a deeply divided public. His approval ratings often fell along party lines, with Republicans generally supporting him, and Democrats largely disapproving of his performance.

# 2. **Strong Base of Support**: Throughout his presidency, Trump maintained a dedicated and enthusiastic base of supporters. They appreciated his "America First" agenda, which included tax cuts, deregulation, and a focus on immigration control. They often praised his willingness to challenge political norms.

# 3. **Strong Disapproval**: Conversely, Trump faced significant and consistent disapproval from a large portion of the population, including many Democrats and some Independents. Critics cited concerns about his divisive rhetoric, handling of the COVID-19 pandemic, immigration policies, and ethical concerns.

# 4. **Media Influence**: Public opinion on Trump was influenced by media outlets that often had partisan leanings. Supporters claimed that he was unfairly treated by the mainstream media, while critics argued that he spread misinformation and attacked the press.

# 5. **Policy Achievements**: Trump's supporters often pointed to his achievements, such as tax reform, criminal justice reform, and conservative judicial appointments, as reasons for their support. Detractors, on the other hand, criticized some of these policies and their implications.

# 6. **Foreign Policy**: Opinions on Trump's foreign policy varied widely. Some praised his approach to China, Israel, and North Korea, while others were critical of his approach to international agreements like the Paris Climate Accord and the Iran Nuclear Deal.

# 7. **Impeachment and Scandals**: Trump's presidency included two impeachments by the House of Representatives, which further divided public opinion. Supporters generally saw these efforts as politically motivated, while critics believed they were necessary accountability measures.

# 8. **2020 Election and Post-Presidential Period**: Trump's refusal to concede the 2020 presidential election and his role in the January 6th Capitol insurrection led to a highly charged and divisive period in American politics. Some continued to support his claims of election fraud, while others viewed these claims as baseless.

# It's important to note that public opinion can change over time, and events and developments since September 2021 could have shifted perceptions of Donald Trump and his presidency. As of my last update, his influence on American politics remained significant, and discussions about his legacy continued to be a central topic in public discourse."""
# response2 = """As of my last update in September 2021, public opinion on Donald Trump, the 45th President of the United States, has been deeply polarized and varies widely depending on various factors, including political affiliation, demographics, and media consumption habits. Below are some broad strokes that outline the public sentiment on Trump:

# ### Supporters
# 1. **Strong Economy**: Many supporters credit Trump with fostering a strong economy before the COVID-19 pandemic, citing low unemployment rates and stock market performance.
# 2. **Straight Talk**: His direct, unfiltered communication style has been seen as refreshing to those who are frustrated with what they perceive as 'political correctness' or evasion in traditional politics.
# 3. **Conservative Policies**: From tax cuts to conservative judicial appointments, supporters appreciate his commitment to what they see as core conservative values.
# 4. **Populist Appeal**: Trump's critique of globalism and focus on America-first policies resonated with many who felt left behind by globalization and changing demographics.
# 5. **Anti-establishment**: His outsider status appealed to those who were frustrated with what they see as a corrupt or ineffective 'swamp' in Washington, D.C.

# ### Critics
# 1. **Divisiveness**: Critics argue that Trump's rhetoric and actions have fostered division in the country, often along racial, ethnic, or social lines.
# 2. **Handling of COVID-19**: His handling of the COVID-19 pandemic received widespread criticism for lack of a coordinated federal response and inconsistent messaging.
# 3. **Impeachments**: Trump was impeached twice by the House of Representatives — first for his interactions with Ukraine and later for incitement of insurrection related to the January 6, 2021 attack on the U.S. Capitol. Critics see these impeachments as evidence of abuse of power.
# 4. **Foreign Policy**: Skeptics question the effectiveness and wisdom of his foreign policy moves, including relationships with allies and adversaries.
# 5. **Truthfulness**: Critics frequently accuse Trump of lying or distorting facts, citing numerous fact-checking reports.

# ### Swing Voters and Independents
# - The perception among this group tends to be mixed, often contingent upon current events, policy outcomes, and the political climate.

# Public opinion on Donald Trump can also differ considerably depending on geographic location (urban vs. rural), age, and other demographic factors.

# It's also worth noting that public opinion continues to evolve, and what was true in 2021 may have shifted by your current date in 2023. Therefore, the perspectives presented here may have evolved over time."""
# # Call the evaluate function
# for output in evaluate(instruction, response1=response1, response2=response2):
#     print(output)

# print()
# print()
# print()
# print()

# for output in evaluate(instruction, response1=response2, response2=response1):
#     print(output)
def parse_output_score(output) -> int:
    output_score = output[24]
    try:
        output_score = int(output[24])
        return output_score
    except:
        if '1' in output[:30]:
            return 1
        else:
            return 2


import json

# Load the test cases from the JSON file
with open('../data/testset-v1.json', 'r') as f:
    test_cases = json.load(f)

from martian import openai as martian_openai
import openai

from secret_keys import MARTIAN_API_KEY, OPENAI_API_KEY
martian_openai.martian_api_key = MARTIAN_API_KEY
openai.api_key = OPENAI_API_KEY

def get_martian_response(prompt: str):
    chat_completion = martian_openai.ChatCompletion.create(
        # No need to set a model, the router handles that :]
        messages=[{"role": "user", "content": prompt}]
    )
    return chat_completion.choices[0].message.content

def get_openai_response(prompt: str):
    chat_completion = openai.ChatCompletion.create(
        # No need to set a model, the router handles that :]
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4",
    )
    return chat_completion.choices[0].message.content


# Iterate over the test cases
for test_case in test_cases[5:]:
    print(test_case)
    # print(test_case['idx'])
    instruction = test_case['instruction']
    input = test_case['input']
    response1 = get_martian_response(instruction + "\n" + input)
    print('Martian: ', response1)
    response2 = get_openai_response(instruction + "\n" + input)
    print('OpenAI: ', response2)
    annotator1 = test_case['annotator1']
    annotator2 = test_case['annotator2']
    annotator3 = test_case['annotator3']
    
    # Call the evaluate function
    for output in evaluate(instruction, response1=response1, response2=response2):
        try:
            output_score = parse_output_score(output)
        # halt the program if the output[24] is not a number
        except:
            print("The output is not a number.")
            input()
            continue
        print(f"======= PandaLM result: {output_score} =======")
        # print(f"Annotator1: {annotator1}, Annotator2: {annotator2}, Annotator3: {annotator3}")
        print()
    # Calculate the average of the annotators' scores
    # average_score = (annotator1 + annotator2 + annotator3) / 3
    # print(f"Average Annotator Score: {average_score}")
    
    print()
