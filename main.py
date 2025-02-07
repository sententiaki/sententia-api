import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re  # Per gestire la conversione del codice sentenza

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Recupera la chiave API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Errore: La chiave API di OpenAI non è stata trovata. Assicurati di impostarla nel file .env.")

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

# Route per la sintesi delle sentenze
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

# 🔍 Nuovo endpoint per la ricerca intelligente delle sentenze
@app.route('/cerca-sentenze', methods=['GET'])
def cerca_sentenze():
    query = request.args.get('query')
    
    if not query:
        return jsonify({"errore": "Devi inserire delle parole chiave per la ricerca."}), 400
    
    # Simulazione di ricerca delle sentenze sul web (puoi personalizzare questa parte con il vero motore di ricerca)
    url = f"https://bger.li/search?query={query}"  # URL fittizio, da personalizzare
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Simulazione di risultati (modifica per elaborare i veri risultati)
        risultati = f"Risultati trovati per la query '{query}' (simulazione di risultati)"
        
        return jsonify({"risultati": risultati})
    except Exception as e:
        return jsonify({"errore": f"Errore durante la ricerca: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
