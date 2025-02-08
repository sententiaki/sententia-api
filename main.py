import os
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai
from googletrans import Translator

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Chiavi API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not OPENAI_API_KEY or not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise ValueError("Assicurati che le chiavi API siano correttamente impostate nel file .env.")

# Imposta la chiave API di OpenAI
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
CORS(app)

translator = Translator()

# Funzione per costruire l'URL su bger.li usando il codice sentenza
def costruisci_url_bgerli(codice_sentenza):
    codice_sentenza = codice_sentenza.strip().replace("/", "-").replace(" ", "-")
    return f"https://bger.li/{codice_sentenza}"

# Funzione per estrarre il contenuto della sentenza da bger.li
def estrai_testo_sentenze_bgerli(url):
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", id="content")
        if content:
            return content.get_text(separator="\n").strip()
        else:
            return "Errore: Testo della sentenza non trovato."
    except Exception as e:
        return f"Errore nell'estrazione del testo: {e}"

# Funzione per sintetizzare una sentenza per il sistema di ricerca sentenze
def sintetizza_testo_sentenza(testo_sentenza, codice):
    try:
        prompt = f"""
        Sei un assistente giuridico esperto. Sintetizza la seguente sentenza:
        Titolo: {codice}
        1. **Riassunto della sentenza:** Fornisci un riassunto completo in circa 10 righe che copra il tema principale della sentenza.
        2. **Articoli rilevanti:** Elenca gli articoli giuridici rilevanti discussi nella sentenza.

        Testo della sentenza:
        {testo_sentenza}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un assistente giuridico esperto."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Errore durante la sintesi: {e}"

# Funzione per sintetizzare una sentenza in 4 punti per il sistema Legal Summarization
def sintetizza_testo_sentenza_4_punti(testo_sentenza):
    try:
        prompt = f"""
        Sei un assistente giuridico esperto. Sintetizza il seguente testo di sentenza suddividendolo nei seguenti punti:
        
        1. **Riassunto della fattispecie:** Dettagli e contesto principale della sentenza.
        2. **Articoli principali rilevanti:** Elenco degli articoli di legge utilizzati o menzionati nella sentenza.
        3. **Considerazioni principali del tribunale:** Motivazioni centrali e interpretazioni giuridiche.
        4. **Conclusioni:** Esito finale della sentenza e i suoi effetti.

        Testo della sentenza:
        {testo_sentenza}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un assistente giuridico esperto."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )

        # Ritorna il risultato diviso nei 4 punti
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Errore durante la sintesi della sentenza: {e}"

# Funzione per estrarre il codice sentenza dal link di Google
def estrai_codice_sentenza(link):
    match = re.search(r'(\d+[A-Za-z]_\d+/\d+|\d{1,3}\s+[IVXLCDM]+\s+\d+)', link)
    return match.group(1) if match else "Codice non trovato"

# Funzione per cercare le sentenze con Google Custom Search
def cerca_sentenze_google(parole_chiave):
    risultati_finali = []
    try:
        # Traduci la query in francese e tedesco
        query_it = parole_chiave
        query_fr = translator.translate(parole_chiave, src='it', dest='fr').text
        query_de = translator.translate(parole_chiave, src='it', dest='de').text

        # Unisci le query
        queries = [query_it, query_fr, query_de]

        # Cerca in tutte le lingue
        for query in queries:
            url = f"https://www.googleapis.com/customsearch/v1?q={query}+site:bger.ch&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
            response = requests.get(url)
            response.raise_for_status()
            risultati_google = response.json().get('items', [])

            for item in risultati_google[:5]:
                link = item.get('link')
                titolo = item.get('title', 'Titolo non disponibile')
                codice_sentenza = estrai_codice_sentenza(titolo)

                if codice_sentenza != "Codice non trovato":
                    url_bgerli = costruisci_url_bgerli(codice_sentenza)
                    testo_sentenze = estrai_testo_sentenze_bgerli(url_bgerli)

                    if "Errore" not in testo_sentenze:
                        sintesi = sintetizza_testo_sentenza(testo_sentenze, codice_sentenza)
                        risultati_finali.append({
                            "titolo": codice_sentenza,
                            "riassunto": sintesi,
                            "link": url_bgerli
                        })

    except Exception as e:
        return [{"errore": f"Errore durante la ricerca: {e}"}]

    return risultati_finali if risultati_finali else [{"errore": "Nessun risultato trovato"}]

# Route per la ricerca delle sentenze
@app.route('/ricerca_sentenze', methods=['GET'])
def ricerca_sentenze():
    query = request.args.get('query')
    if not query:
        return jsonify({"errore": "Parole chiave mancanti per la ricerca"}), 400

    risultati = cerca_sentenze_google(query)
    return jsonify(risultati)

# Route per la sintesi delle sentenze singole (Legal Summarization)
@app.route('/sintesi', methods=['GET'])
def get_summary():
    codice_sentenza = request.args.get('codice')
    if not codice_sentenza:
        return jsonify({"errore": "Codice sentenza mancante"}), 400

    url = costruisci_url_bgerli(codice_sentenza)
    testo_sentenza = estrai_testo_sentenze_bgerli(url)

    if "Errore" in testo_sentenza:
        return jsonify({"errore": testo_sentenza}), 404

    sintesi_4_punti = sintetizza_testo_sentenza_4_punti(testo_sentenza)
    return jsonify({"sintesi": sintesi_4_punti})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




