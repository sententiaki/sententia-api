import os
import requests
from flask import Flask, request, jsonify

# Prendi la chiave API da una variabile d'ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Errore: La chiave API di OpenAI non è stata trovata. Assicurati di impostarla come variabile d'ambiente.")

app = Flask(__name__)

@app.route('/sintesi', methods=['GET', 'POST'])
def sintetizza_sentenza():
    data = request.json
    codice_sentenza = data.get("codice_sentenza")

    if not codice_sentenza:
        return jsonify({"errore": "Codice della sentenza mancante"}), 400

    # Costruisci l'URL della sentenza su bger.li
    url_sentenza = f"https://bger.li/{codice_sentenza}"

    try:
        response = requests.get(url_sentenza, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"errore": f"Errore nel recupero della sentenza: {str(e)}"}), 500

    # Estrarre il testo della sentenza dal sito (devi adattarlo se cambia il formato HTML)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    contenuto_sentenza = soup.find("div", {"id": "content"})

    if not contenuto_sentenza:
        return jsonify({"errore": "Testo della sentenza non trovato"}), 404

    testo_sentenza = contenuto_sentenza.get_text()

    # Chiamata a OpenAI per la sintesi
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "Sintetizza la sentenza nei seguenti 4 punti: \n"
                                          "1. Riassunto della fattispecie \n"
                                          "2. Articoli principali rilevanti (elenco numerico) \n"
                                          "3. Considerazioni principali del tribunale (frasi chiave) \n"
                                          "4. Conclusioni finali"},
            {"role": "user", "content": testo_sentenza}
        ],
        "temperature": 0.7
    }

    try:
        risposta = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        risposta.raise_for_status()
        sintesi = risposta.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return jsonify({"errore": f"Errore durante la sintesi: {str(e)}"}), 500

    return jsonify({"sintesi": sintesi})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

