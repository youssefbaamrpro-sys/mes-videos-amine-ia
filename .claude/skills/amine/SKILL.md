---
name: amine
description: Création des vidéos d'Amine (voix ElevenLabs, texte Groq, images/vidéos Replicate, montage ffmpeg).
---

# Vidéos Amine

Outils disponibles dans l'environnement :
- `ffmpeg` / `ffprobe` pour le montage
- `yt-dlp` pour télécharger des vidéos
- SDK Python `elevenlabs`, `groq`, `replicate` (les clés API sont injectées automatiquement)

Les dépendances sont installées au démarrage par `setup.sh` à la racine du dépôt.

## Notes

- Si un SDK exige une clé API, mets une valeur factice (ex. `GROQ_API_KEY=dummy`) : le proxy la remplace par la vraie.
- Groq : `llama-3.1-8b-instant` n'est pas disponible sur ce compte. Utiliser par exemple `openai/gpt-oss-120b` ou `openai/gpt-oss-20b` (liste complète : `Groq().models.list()`).
