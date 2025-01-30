import os
from flask import Flask, request, jsonify
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Recupera la chiave API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Errore: La chiave API di OpenAI non è stata trovata. Assicurati di impostarla nel file .env.")

# Imposta la chiave API di OpenAI
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

# Funzione per costruire l'URL per la ricerca della sentenza
def costruisci_url_bgerli(codice_sentenza):
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

# Funzione per sintetizzare il testo della sentenza
def sintetizza_testo_sentenza(testo_sentenza):
    try:
        prompt = f"""
        Sei un assistente giuridico esperto. Sintetizza il seguente testo di sentenza:
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
        return f"Errore durante la sintesi della sentenza: {e}"

# Route principale dell'API
@app.route('/sintesi', methods=['GET', 'POST'])
def get_summary():
    codice_sentenza = request.args.get('codice')
    if not codice_sentenza:
        return jsonify({"errore": "Codice sentenza mancante"}), 400

    url = costruisci_url_bgerli(codice_sentenza)
    testo_sentenza = estrai_testo_sentenze(url)

    if not testo_sentenza or "Errore" in testo_sentenza:
        return jsonify({"errore": testo_sentenza}), 404

    sintesi = sintetizza_testo_sentenza(testo_sentenza)
    if sintesi:
        return jsonify({"sintesi": sintesi})
    else:
        return jsonify({"errore": "Errore durante la sintesi della sentenza."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)