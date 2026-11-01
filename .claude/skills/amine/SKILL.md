---
name: amine
description: Création des vidéos d'Amine (voix ElevenLabs, texte Groq, images/vidéos Replicate, montage ffmpeg).
---

# Vidéos Amine

Outils disponibles dans l'environnement :
- `ffmpeg` / `ffprobe` pour le montage
- `yt-dlp` pour télécharger des vidéos
- SDK Python `elevenlabs`, `groq`, `replicate` (les clés API sont injectées automatiquement par le proxy)

Les dépendances sont installées au démarrage par `setup.sh` à la racine du dépôt.

## Clients prêts à l'emploi

Toujours passer par `clients.py` (dans ce dossier), qui gère les clés factices et le proxy :

```python
import sys; sys.path.insert(0, ".claude/skills/amine")
from clients import elevenlabs, groq, replicate, GROQ_MODEL
```

## Pièges connus

- **Replicate** : le SDK ignore `HTTPS_PROXY` (il crée son propre transport httpx), d'où des erreurs 401.
  `clients.py` corrige ça. Si tu crées un client à la main :
  `replicate.Client(api_token="dummy", transport=httpx.HTTPTransport(proxy=os.environ["HTTPS_PROXY"]))`.
- **Replicate 402 "Insufficient credit"** : le compte doit être crédité sur https://replicate.com/account/billing.
- **Groq** : `llama-3.1-8b-instant` n'est pas disponible sur ce compte. Utiliser `openai/gpt-oss-120b`
  (ou `openai/gpt-oss-20b`). Liste : `groq.models.list()`.

## Exemple de montage (image fixe + voix, format vertical)

```bash
ffmpeg -y -loop 1 -i image.png -i voix.mp3 -vf "scale=720:1280,format=yuv420p" \
  -c:v libx264 -tune stillimage -c:a aac -shortest video.mp4
```
