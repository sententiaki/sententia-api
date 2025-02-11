import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import re
from dotenv import load_dotenv
import tiktoken

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Recupera le chiavi API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not (OPENAI_API_KEY and GOOGLE_API_KEY and GOOGLE_CSE_ID):
    raise ValueError("Assicurati che le chiavi API siano correttamente impostate nel file .env.")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
CORS(app)

# Funzione per tradurre le parole chiave in tre lingue
def traduci_parole_chiave(parole_chiave):
    traduzioni = {
        "it": parole_chiave,
        "de": GoogleTranslator(source='it', target='de').translate(parole_chiave),
        "fr": GoogleTranslator(source='it', target='fr').translate(parole_chiave)
    }
    return traduzioni

# Funzione per cercare le sentenze tramite Google Custom Search
def cerca_sentenze_google(parole_chiave):
    risultati_finali = []
    traduzioni = traduci_parole_chiave(parole_chiave)
    
    for lang, query in traduzioni.items():
        url = f"https://www.googleapis.com/customsearch/v1?q={query}+site:bger.ch&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(url)
        response.raise_for_status()
        risultati = response.json().get('items', [])

        for item in risultati[:5]:  # Limitiamo a 5 risultati complessivi
            titolo = item.get('title', '')
            link = item.get('link', '')
            # Estrarre il codice sentenza dal titolo (es: 4A_61/2024)
            codice_match = re.search(r'(\d+[A-Z]_\d+/\d+|\d+\s+[IVXLCDM]+\s+\d+)', titolo)
            if codice_match:
                codice_sentenza = codice_match.group(1)
                risultati_finali.append({"codice": codice_sentenza, "link": link})
                
    return risultati_finali[:5]  # Restituiamo al massimo 5 risultati

# Funzione per costruire l'URL su bger.li per una sentenza
def costruisci_url_bgerli(codice_sentenza):
    codice_sentenza = codice_sentenza.strip().replace(" ", "-").replace("/", "-")
    return f"https://bger.li/{codice_sentenza}"

# Funzione per estrarre il testo della sentenza da bger.li
def estrai_testo_sentenze(url):
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", id="content")
        if content:
            return content.get_text(separator="\n").strip()
        else:
            return "Testo della sentenza non trovato."
    except Exception as e:
        return f"Errore nell'estrazione del testo della sentenza: {e}"

# Funzioni per il chunking e il riassunto iterativo

def split_text_into_chunks(text, max_tokens=15000, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    return chunks

def summarize_with_chunking(text, summary_function, max_tokens=15000, model="gpt-3.5-turbo"):
    chunks = split_text_into_chunks(text, max_tokens, model)
    if len(chunks) == 1:
        return summary_function(text)
    else:
        chunk_summaries = []
        for chunk in chunks:
            summary_chunk = summary_function(chunk)
            chunk_summaries.append(summary_chunk)
        combined_summary = "\n".join(chunk_summaries)
        final_summary = summary_function(combined_summary)
        return final_summary

# Funzione per sintetizzare ogni sentenza (10 righe per la ricerca)
def sintetizza_sentenza_10_righe(testo_sentenza):
    def call_api(text):
        prompt = f"""
Sei un assistente giuridico esperto. Sintetizza il seguente testo di sentenza nei seguenti punti:
- **Riassunto della sentenza**: Scrivi un riassunto completo del tema principale trattato in circa 10 righe.
- **Articoli principali rilevanti**: Elenca gli articoli di legge citati o discussi nella sentenza.

Ecco il testo della sentenza:
{text}
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sei un assistente giuridico esperto."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Errore durante la sintesi della sentenza: {e}"
    return summarize_with_chunking(testo_sentenza, call_api)

# Funzione per sintetizzare la sentenza in 4 punti (legal summarization)
def sintetizza_testo_sentenza_4_punti(testo_sentenza):
    def call_api(text):
        prompt = f"""
Sei un assistente giuridico esperto. Sintetizza il seguente testo di sentenza nei seguenti 4 punti:
1. Riassunto della fattispecie: Dettagli e contesto principale della sentenza.
2. Articoli principali rilevanti: Elenca gli articoli giuridici citati o discussi nella sentenza.
3. Considerazioni principali del tribunale: Presenta le motivazioni e interpretazioni principali.
4. Conclusioni: Riassumi l'esito finale della sentenza e le sue implicazioni.

Ecco il testo della sentenza:
{text}
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sei un assistente giuridico esperto."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Errore durante la sintesi della sentenza: {e}"
    return summarize_with_chunking(testo_sentenza, call_api)

# Route per la ricerca delle sentenze e la loro sintetizzazione (10 righe)
@app.route('/ricerca_sentenze', methods=['GET'])
def ricerca_sentenze():
    query = request.args.get('query')
    if not query:
        return jsonify({"errore": "Parole chiave mancanti per la ricerca"}), 400

    sentenze_trovate = cerca_sentenze_google(query)
    risultati_sintetizzati = []

    for sentenza in sentenze_trovate:
        codice = sentenza["codice"]
        url_bgerli = costruisci_url_bgerli(codice)
        testo_scaricato = estrai_testo_sentenze(url_bgerli)
        
        if "Errore" in testo_scaricato or "Testo della sentenza non trovato" in testo_scaricato:
            sintesi = "Impossibile scaricare il contenuto della sentenza."
        else:
            sintesi = sintetizza_sentenza_10_righe(testo_scaricato)

        risultati_sintetizzati.append({
            "titolo": codice,
            "riassunto": sintesi,
            "link": url_bgerli
        })

    return jsonify(risultati_sintetizzati)

# Route per la sintesi di una singola sentenza (4 punti)
@app.route('/sintesi', methods=['GET'])
def get_summary():
    codice_sentenza = request.args.get('codice')
    if not codice_sentenza:
        return jsonify({"errore": "Codice sentenza mancante"}), 400

    url_bgerli = costruisci_url_bgerli(codice_sentenza)
    testo_scaricato = estrai_testo_sentenze(url_bgerli)

    if "Errore" in testo_scaricato or "Testo della sentenza non trovato" in testo_scaricato:
        return jsonify({"errore": testo_scaricato}), 404

    sintesi = sintetizza_testo_sentenza_4_punti(testo_scaricato)
    return jsonify({"sintesi": sintesi})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
