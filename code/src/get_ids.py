import datasets as hfds
import pandas as pd

print('leyendo dataaset')
dataset = hfds.load_dataset('MedRAG/pubmed', trust_remote_code=True)

print('ordenando')
df = dataset['train'].sort('id')

print('seleccionando')
df = df.select(range(1000000))

print('select id')
ids = [example['id'] for example in df]

print('To Pandas')
df_ids = pd.DataFrame(ids, columns=["id"])

print('saving file')
df_ids.to_csv("train_hf_ids.csv", index=False)