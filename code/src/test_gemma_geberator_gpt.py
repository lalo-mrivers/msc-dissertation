from typing import Union
from sentence_transformers import InputExample
import random
import re
import os
import torch
import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_from_disk, load_dataset

# Importa o define aquí la función load_pre_train_medrag_pubmed
# ...

def load_pre_train_medrag_pubmed(max_samples: int,
                                 test_size:Union[float, int, None] = 0.2,
                                 val_size:Union[float, int, None] = 0.15,
                                 seed:int = 42,
                                 dataset_name: str = 'MedRAG/pubmed') -> list:
    """
    Loads the dataset, filters it, and prepares it as a list of InoutExample with positive pairs.
    It takes title of the medical text as the query and the 'content' as a positive passage.
    It only uses one positive pair per medical text.
    """
    import time
    print(f"Loading dataset [test_size: {test_size}; val_size: {val_size}]...")
    st = time.time()
    #full_dataset = load_dataset(dataset_name, split='train')
    if os.path.exists(dataset_name + '_cached'):
        full_dataset = load_from_disk(dataset_name + '_cached')
    else:
        full_dataset = load_dataset(dataset_name, split='train[:100000]')
        full_dataset.save_to_disk(dataset_name + '_cached')

    #full_dataset = load_dataset(dataset_name, split='train[:100000]')
    #full_dataset.save_to_disk(dataset_name+'_cached')
    #full_dataset = load_from_disk(dataset_name+'_cached')
    print(f'Dataset loaded. {time.time() - st:.2f}s')

    samples = []
    titles = set()
    for i in range(len(full_dataset)):
        data_point = full_dataset[i]
        if len(samples) % (max_samples // 10) == 0:
            print(f'{len(samples) / (max_samples // 100)}%')
        if len(samples) >= max_samples:
            break
        
        query = re.sub(r'^[@$!%&/=\?\[\]]+', '', data_point.get('title'), count=1)
        query = re.sub(r'](?=[^\]]*$)', '', query, count=1)

        positive_passage = data_point.get('content')

        if len(query) > 0 and not query[0].isalnum():
            continue
        
        if len(positive_passage) > 0 and not positive_passage[0].isalnum():
            continue
        
        if query and positive_passage and query not in titles:
            samples.append(InputExample(texts=[query, positive_passage]))
            titles.add(query)

    if test_size:
        if seed:
            random.seed(seed)
        
        random.shuffle(samples)

        n = len(samples)
        train_size_idx = int((1 - test_size) * n)
        train = samples[:train_size_idx]
        test = samples[train_size_idx:]

        val = []
        queries = {}
        corpus = {}
        relevant_docs = {}
        if val_size:
            val_size_idx = int((1 - val_size) * train_size_idx)
            print(f'train_size_idx: {train_size_idx}, val_size_idx: {val_size_idx}')
            for idx in range(train_size_idx):
                if idx >= val_size_idx:
                    val.append(train[idx])
                    queries[f"q_{idx}"] =  train[idx].texts[0]
                    relevant_docs[f"q_{idx}"] = {f"doc_{idx}"}
                corpus[f"doc_{idx}"] = train[idx].texts[1] 
        
    else:
        train = samples
        val = []
        test = []


    return train, val, test, (queries, relevant_docs, corpus)



# ------------------- CONFIG -------------------
HF_TOKEN = "HFTOKEN"  #
MODEL_ID = "google/medgemma-4b-it"#"google/medgemma-27b-text-it"#"google/medgemma-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
MAX_TOKENS = 768
TEMPERATURE = 0.7
TOP_P = 0.95
OUTPUT_CSV = "medgemma_triplets_full.csv"
MAX_SAMPLES = 12500
MAX_TO_GENERATE_PER_RUN = None  # Cambia esto a, por ejemplo, 100 si quieres probar en lotes

# ------------------- PROMPT -------------------
def get_prompt(source_text: str) -> str:
    a = f"""From the following SOURCE TEXT, generate a data triplet to train a medical information retrieval model. The response must strictly follow the XML format below:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

GENERATION:
<question>
Generate a complex and specific clinical question here that can only be answered with information from the SOURCE TEXT.
</question>
<correct_answer>
Generate a long, detailed, and factual textual answer to the previous question here. Use only information contained in the SOURCE TEXT. The answer must be precise and correct.
</correct_answer>
<incorrect_similar_answer>
Generate an incorrect answer to the question here. This answer should seem plausible, use similar medical terminology from the source text, but be conceptually wrong, contain false information, or describe a mistaken treatment or diagnosis. It should be a "hard negative".
</incorrect_similar_answer>
"""
    b = f"""You are a clinical reasoning assistant generating challenging medical question-answer pairs from academic clinical texts.

**Task:**
For each SOURCE TEXT (academic-style, technical medical description), generate exactly three items:
  1. A complex, specific <question> that can only be answered using the SOURCE TEXT.
  2. A long, accurate <correct_answer> based strictly on the SOURCE TEXT.
  3. A <incorrect_similar_answer> that is plausible but clearly wrong, based on common misunderstandings or incorrect reasoning.

**Style Requirements:**
- The question should be clinically relevant and realistic, like in board exams.
- The correct answer must cite or paraphrase key facts from the SOURCE TEXT.
- The incorrect answer must sound reasonable but contain a significant conceptual error.
- Use appropriate medical terminology.

**HARD RULES:**
- DO NOT invent info not in the source.
- DO NOT write vague questions or answers.
- The incorrect answer MUST contain a subtle clinical mistake.

**Example:**

<SOURCE_TEXT>
Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia. It is characterized by chaotic, rapid, and irregular atrial activation, resulting in an irregularly irregular ventricular response. The main risks associated with AF are ischemic stroke and heart failure. Management is based on three pillars: ventricular rate control (with beta-blockers or calcium channel blockers), rhythm control (pharmacological or electrical cardioversion), and, crucially, anticoagulation for thromboembolism prevention, stratifying risk with scales like CHA2DS2-VASc.
</SOURCE_TEXT>

<question>
What is the primary reason for initiating anticoagulation therapy in patients with atrial fibrillation?
</question>
<correct_answer>
The primary reason for anticoagulation in atrial fibrillation is to prevent thromboembolic events, particularly ischemic stroke, as patients with AF are at significantly increased risk due to irregular atrial activity. Risk is assessed using tools like the CHA2DS2-VASc score.
</correct_answer>
<incorrect_similar_answer>
The primary reason for anticoagulation in atrial fibrillation is to reduce ventricular rate and prevent heart failure exacerbations by improving hemodynamic control.
</incorrect_similar_answer>

Now, use the following SOURCE TEXT to generate a new triplet:

<SOURCE_TEXT>
{'source_text'}
</SOURCE_TEXT>
"""
    c = f"""From the following SOURCE TEXT, generate a data triplet to train a medical information retrieval model. The response must strictly follow the XML format below:

<SOURCE_TEXT>
Beta-glucosidase from almonds catalyzes the hydrolysis of amygdalin. The enzyme's activity is enhanced in the presence of sodium ions but inhibited by heavy metals like mercury.
</SOURCE_TEXT>

GENERATION:
<question>
How do sodium ions and mercury affect the catalytic activity of almond beta-glucosidase in hydrolyzing amygdalin?
</question>
<correct_answer>
Sodium ions enhance the catalytic activity of almond beta-glucosidase in the hydrolysis of amygdalin, while heavy metals such as mercury inhibit the enzyme's activity.
</correct_answer>
<incorrect_similar_answer>
Sodium ions inhibit the catalytic activity of almond beta-glucosidase in the hydrolysis of amygdalin, while heavy metals such as mercury enhance the enzyme's function.
</incorrect_similar_answer>

Now use the same format with this new SOURCE TEXT:
<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

GENERATION:
"""
    return c


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        use_auth_token=HF_TOKEN
    )
    return tokenizer, model

def parse_triplet(text):
    #print('+'*100)
    #print(text)
    #print('+'*100)
    try:
        q = re.search(r"<question>(.*?)</question>", text, re.DOTALL).group(1).strip()
        a = re.search(r"<correct_answer>(.*?)</correct_answer>", text, re.DOTALL).group(1).strip()
        n = re.search(r"<incorrect_similar_answer>(.*?)</incorrect_similar_answer>", text, re.DOTALL).group(1).strip()
        return q, a, n
    except AttributeError:
        return None, None, None

def generate_triplet(tokenizer, model, source_text):
    prompt = get_prompt(source_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    triplet_part = output_text.split("GENERATION:")[-1]
    return parse_triplet(triplet_part)

def count_existing_triplets(csv_file):
    if not os.path.exists(csv_file):
        return 0
    try:
        df = pd.read_csv(csv_file)
        return len(df)
    except:
        return 0

def generate_triplets_from_dataset(tokenizer, model, dataset_examples, output_file, start_idx=0, max_to_generate=None):
    fieldnames = ["question", "correct_answer", "incorrect_answer", "source_text", "title"]
    generated_count = 0

    for i in range(start_idx, len(dataset_examples)):
        if max_to_generate is not None and generated_count >= max_to_generate:
            break

        example = dataset_examples[i]
        title = example.texts[0]
        content = example.texts[1]
        print(f"Generating triplet {i+1}/{len(dataset_examples)} for title: {title[:50]}...")
        q, a, n = generate_triplet(tokenizer, model, content)
        if q and a and n:
            row = {
                "question": q,
                "correct_answer": a,
                "incorrect_answer": n,
                "source_text": content,
                "title": title
            }
            with open(output_file, mode='a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            generated_count += 1
        else:
            print(f"Warning: No valid triplet generated for example {i+1}")

# ------------------- MAIN -------------------
def main():
    print("Loading dataset...")
    train, val, test, _ = load_pre_train_medrag_pubmed(max_samples=MAX_SAMPLES)

    tokenizer, model = load_model_and_tokenizer()

    existing = count_existing_triplets(OUTPUT_CSV)

    if existing == 0:
        with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "correct_answer", "incorrect_answer", "source_text", "title"])
            writer.writeheader()
        print("CSV creado con encabezados.")

    print(f"Tripletas ya generadas: {existing}")
    if existing >= len(train):
        print("Todas las tripletas ya fueron generadas.")
        return

    print(f"Generando desde el ejemplo {existing + 1}...")
    generate_triplets_from_dataset(
        tokenizer, model, train,
        OUTPUT_CSV,
        start_idx=existing,
        max_to_generate=MAX_TO_GENERATE_PER_RUN
    )

    print("Generación completada.")

if __name__ == "__main__":
    main()
