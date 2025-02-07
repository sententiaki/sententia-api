import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not OPENAI_API_KEY or not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise ValueError("Assicurati che le chiavi API siano correttamente impostate nel file .env.")


# Imposta la chiave API di OpenAI
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
CORS(app)

# Funzione per costruire l'URL per la ricerca della sentenza su bger.li
def costruisci_url_bgerli(codice_sentenza):
    # Rimuove spazi in eccesso
    codice_sentenza = codice_sentenza.strip()

    # Controlla se il formato è del tipo "105 II 16" e lo converte in "105-II-16"
    match = re.match(r'^(\d{1,3})\s+([IVXLCDM]+)\s+(\d+)$', codice_sentenza)
    if match:
        codice_sentenza = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    else:
        # Sostituisci eventuali "/" con "-" per formati tipo "4A_61/2024"
        codice_sentenza = codice_sentenza.replace("/", "-")

    return f"https://bger.li/{codice_sentenza}"

# Funzione per estrarre il testo della sentenza dal sito bger.li
def estrai_testo_sentenze(url):
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", id="content")
        if content:
            return content.get_text(separator="\n").strip()
        else:
            raise ValueError("Testo della sentenza non trovato.")
    except Exception as e:
        return f"Errore nell'estrazione del testo della sentenza: {e}"

# Funzione per sintetizzare il testo della sentenza e dividerlo in 4 punti
def sintetizza_testo_sentenza(testo_sentenza):
    try:
        prompt = f"""
        Sei un assistente giuridico esperto. Sintetizza il seguente testo di sentenza suddividendolo nei seguenti punti:
        
        1. **Riassunto della fattispecie**: Dettagli e contesto principale della sentenza.
        2. **Articoli principali rilevanti**: Elenco degli articoli di legge utilizzati o menzionati nella sentenza.
        3. **Considerazioni principali del tribunale**: Motivazioni centrali e interpretazioni giuridiche.
        4. **Conclusioni**: Esito finale della sentenza e i suoi effetti.

        Ecco il testo della sentenza:
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
        # Dividi la risposta nei 4 punti previsti
        sintesi_completa = response["choices"][0]["message"]["content"].strip()
        
        punti = sintesi_completa.split("\n\n")  # Divide ogni sezione della risposta

        # Assicurati che ci siano almeno 4 sezioni
        return {
            "riassunto": punti[0] if len(punti) > 0 else "Informazione non disponibile",
            "articoli": punti[1] if len(punti) > 1 else "Informazione non disponibile",
            "considerazioni": punti[2] if len(punti) > 2 else "Informazione non disponibile",
            "conclusioni": punti[3] if len(punti) > 3 else "Informazione non disponibile"
        }
    except Exception as e:
        return {
            "riassunto": f"Errore durante la sintesi: {e}",
            "articoli": "Errore durante la sintesi.",
            "considerazioni": "Errore durante la sintesi.",
            "conclusioni": "Errore durante la sintesi."
        }

# Funzione per cercare le sentenze tramite Google Custom Search
def cerca_sentenze_google(parole_chiave):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={parole_chiave}+site:bger.ch&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(url)
        response.raise_for_status()
        risultati = response.json()

        sentenze_trovate = []
        for item in risultati.get('items', [])[:5]:  # Prende solo i primi 5 risultati
            titolo = item.get('title', 'Titolo non disponibile')
            link = item.get('link', '#')
            descrizione = item.get('snippet', 'Descrizione non disponibile')

            # Sintetizza brevemente ogni sentenza trovata
            sintesi_breve = sintetizza_testo_sentenza(descrizione)

            sentenze_trovate.append({
                "titolo": titolo,
                "link": link,
                "riassunto": sintesi_breve["riassunto"]  # Sintesi breve del contenuto
            })

        return sentenze_trovate

    except Exception as e:
        return [{"errore": f"Errore durante la ricerca su Google: {e}"}]

# Route per la sintesi della sentenza
@app.route('/sintesi', methods=['GET'])
def get_summary():
    codice_sentenza = request.args.get('codice')
    if not codice_sentenza:
        return jsonify({"errore": "Codice sentenza mancante"}), 400

    url = costruisci_url_bgerli(codice_sentenza)
    testo_sentenza = estrai_testo_sentenze(url)

    if not testo_sentenza or "Errore" in testo_sentenza:
        return jsonify({"errore": testo_sentenza}), 404

    sintesi = sintetizza_testo_sentenza(testo_sentenza)
    return jsonify(sintesi)

# Route per la ricerca di sentenze tramite parole chiave
@app.route('/ricerca_sentenze', methods=['GET'])
def ricerca_sentenze():
    query = request.args.get('query')
    if not query:
        return jsonify({"errore": "Parole chiave mancanti per la ricerca"}), 400

    risultati = cerca_sentenze_google(query)
    return jsonify(risultati)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)