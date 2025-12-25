import os
import json
import re
import time
import spacy
import argparse
import requests
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
import wikipediaapi
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Настройка аргументов
parser = argparse.ArgumentParser(description="Collect Data from Wikidata/Wikipedia")
parser.add_argument("--limit", type=int, default=1000, help="Number of entities to fetch")
args = parser.parse_args()

nlp = spacy.load("en_core_web_sm")
USER_AGENT = "GraphResearchBot/1.0 (student@university.edu)"

def get_wikidata_entities(limit):
    print(f"[1/4] Querying Wikidata for {limit} entities...")
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    
    # Запрос: Политики и Компании
    query = f"""
    SELECT DISTINCT ?entity ?entityLabel WHERE {{
      {{
        SELECT DISTINCT ?entity WHERE {{
          {{ ?entity wdt:P31 wd:Q5; wdt:P39 ?pos. }} UNION {{ ?entity wdt:P31/wdt:P279* wd:Q4830453. }}
          ?article schema:about ?entity; schema:isPartOf <https://en.wikipedia.org/>.
        }} LIMIT {limit}
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        entities = []
        for r in results["results"]["bindings"]:
            entities.append({
                "id": r["entity"]["value"].split("/")[-1],
                "name": r["entityLabel"]["value"]
            })
        return entities
    except Exception as e:
        print(f"Error fetching entities: {e}")
        return []

def get_facts_and_text(entities):
    print(f"[2/4] Fetching texts and facts for {len(entities)} entities...")
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    
    dataset = []
    kg_triples = [] # Для обучения TransE
    
    for ent in tqdm(entities):
        e_id = ent['id']
        name = ent['name']
        
        # 1. Текст
        try:
            page = wiki.page(name)
            if not page.exists(): continue
            summary = page.summary
            if len(summary) < 50: continue
        except: continue
            
        # 2. Факты (Live query)
        try:
            url = "https://query.wikidata.org/sparql"
            q = f"""SELECT ?pLabel ?oLabel WHERE {{ wd:{e_id} ?p ?o. ?prop wikibase:directClaim ?p; rdfs:label ?pLabel. ?o rdfs:label ?oLabel. FILTER(LANG(?pLabel)="en" && LANG(?oLabel)="en") }} LIMIT 15"""
            r = requests.get(url, params={'format': 'json', 'query': q}, headers={'User-Agent': USER_AGENT})
            data = r.json()
            
            ent_triples = []
            for item in data['results']['bindings']:
                pred = item['pLabel']['value']
                obj = item['oLabel']['value']
                
                # Фильтр мусора
                if "http" not in obj and len(obj) < 60:
                    triple = [name, pred, obj]
                    kg_triples.append(triple) # Глобальный список для TransE
                    ent_triples.append(triple) # Локальный для T5
            
            if not ent_triples: continue

            # 3. Выравнивание (Alignment)
            doc = nlp(summary)
            for sent in doc.sents:
                sent_text = sent.text.strip().replace("\n", " ")
                if len(sent_text) < 20: continue
                
                # Ищем триплеты, объекты которых есть в предложении
                matched = [t for t in ent_triples if t[2] in sent_text]
                
                if matched:
                    # Линеаризация
                    graph_str = " [SEP] ".join([f"<H> {s} <R> {p} <T> {o}" for s, p, o in matched])
                    
                    dataset.append({
                        "entity": name,
                        "input_graph": graph_str,
                        "target_text": sent_text
                    })
                    
        except Exception:
            continue
            
    return dataset, kg_triples

if __name__ == "__main__":
    ents = get_wikidata_entities(args.limit)
    data, triples = get_facts_and_text(ents)
    
    # Сохранение
    os.makedirs("data", exist_ok=True)
    
    with open("data/dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    with open("data/kg_triples.json", "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)
        
    print(f"[Done] Saved {len(data)} training samples and {len(triples)} graph triples.")