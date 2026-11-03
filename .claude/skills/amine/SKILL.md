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
from clients import elevenlabs, groq, replicate, telecharger, televerser, lancer, GROQ_MODEL

sortie = replicate.run("black-forest-labs/flux-schnell",
                       input={"prompt": "...", "aspect_ratio": "9:16", "output_format": "png"})
telecharger(sortie[0], "image.png")
```

## Pièges connus

- **Replicate** : le SDK ignore `HTTPS_PROXY` (il crée son propre transport httpx), d'où des erreurs 401.
  `clients.py` corrige ça. Si tu crées un client à la main :
  `replicate.Client(api_token="dummy", transport=httpx.HTTPTransport(proxy=os.environ["HTTPS_PROXY"]))`.
- **Replicate `version: null`** : le SDK 1.0.7 plante (« none is not an allowed value ») sur les modèles
  officiels comme `flux-schnell`. `clients.py` applique un correctif.
- **Replicate téléchargement 400** : ne pas utiliser `sortie.read()` (le SDK envoie la clé au stockage,
  qui refuse). Utiliser `telecharger(sortie, chemin)`.
- **Replicate 402 "Insufficient credit"** : le compte doit être crédité sur https://replicate.com/account/billing.
- **Groq** : `llama-3.1-8b-instant` n'est pas disponible sur ce compte. Utiliser `openai/gpt-oss-120b`
  (ou `openai/gpt-oss-20b`). Liste : `groq.models.list()`.
- **Replicate 502 avec un fichier local** : `replicate.run(..., input={"audio": open(...)})` échoue sur
  les gros fichiers. Utiliser `televerser(chemin, type)` puis passer l'URL ; `lancer(modele, entrees)`
  crée la prédiction, gère le 429 (limite à 1 requête/10 s quand le crédit est < 5 $) et attend la fin.
- **Replicate `Prefer: wait`** : provoque « upstream request failed » via le proxy. Toujours sonder.
- **Modèles communautaires** (ex. `ryan5453/demucs`, `cjwbw/demucs`) : il faut le hash de version
  (`/v1/models/<owner>/<name>` → `latest_version.id`). `ryan5453/demucs` est resté bloqué une fois
  en « processing » 35 min (facturé) : préférer `cjwbw/demucs` (stems bass/drums/other/vocals).
- **ElevenLabs Starter** : `output_format` 192 kbps refusé (403) → utiliser `mp3_44100_128`.
- **Transcription Groq** : les horodatages ignorent le silence de début de fichier. Transcrire
  chaque clip TTS séparément puis ajouter son décalage, pas la piste montée entière.
- **Mixage de voix** : ne pas monter avec `adelay`+`amix` (piste tronquée/décalée constatée) :
  assembler en numpy (voir `pipeline/02_voix_fr.py`).

## Traduire une vidéo existante (pipeline)

Pour traduire une vidéo avec sous-titres incrustés **sans Vmake** (qui laisse du flou quand les
sous-titres ont un fond) : on pose le texte traduit par-dessus, avec un fond opaque identique.

1. `cjwbw/demucs` → `bg.wav` (bass+drums+other) et `vocals` ; cloner la voix (ElevenLabs IVC) à partir
   des **vocals isolées** (≥ 2 min), pas du mix.
2. Transcrire (Groq `whisper-large-v3`, mots horodatés), traduire en gardant chaque phrase dans le
   créneau de l'original → `script_fr.json` (`[début, fin, texte]`).
3. `pipeline/01_detect_soustitres.py` : repère les bandeaux de sous-titres par image (`navyboxes.json`).
4. `pipeline/02_voix_fr.py` : voix générée **par passages de ~15 s** (pas phrase par phrase, sinon
   coupures), `speed` ≈ 0,86–0,9 pour coller au débit de l'original, fondus 20 ms, sans coupe sèche.
5. `pipeline/03_rendu.py` : sous-titres/étiquettes FR par-dessus (liste `SPECS` à adapter à chaque
   vidéo), remplacement du logo produit, rendu 720×1280. Vérifier avec `test_rendu.py` (planche d'images).
6. `pipeline/04_lipsync.py` : `sync/lipsync-2-pro` **plan par plan** (jamais tous les plans collés,
   sinon visage déformé aux raccords). Ne recoller que le haut de l'image (visage) pour garder les
   sous-titres du rendu final.

Coûts (Replicate) : lip-sync ~0,05 $/s (standard) à ~0,08 $/s (pro) de vidéo. **Faire valider la voix
sur une version sans lip-sync avant de lancer le lip-sync**, pour ne le payer qu'une fois.

## Exemple de montage (image fixe + voix, format vertical)

```bash
ffmpeg -y -loop 1 -i image.png -i voix.mp3 -vf "scale=720:1280,format=yuv420p" \
  -c:v libx264 -tune stillimage -c:a aac -shortest video.mp4
```

## Voix la plus fidèle : doublage ElevenLabs

Pour garder la voix **et l'intonation** de l'orateur d'origine, préférer le doublage
(`POST /v1/dubbing`, `source_lang`, `target_lang`, `num_speakers=1`, `watermark=false`) au clone + TTS.
Coût constaté : ~840 crédits pour 3 min. Le résultat inclut la musique de fond.
L'API d'édition des segments (`/v1/dubbing/resource/...`) est en bêta fermée : on ne peut pas
imposer son texte. Pour changer un mot (ex. nom de produit), remplacer le mot **dans l'audio source**
avant le doublage (voix isolée Demucs + TTS du clone en langue source à l'emplacement du mot).
Les sous-titres se recalent ensuite sur la transcription Groq du doublage.

## Remplacer un texte sur un objet qui bouge (sans glissement)

Ne pas redessiner le texte à une position estimée image par image (ça glisse).
Méthode qui colle : corriger UNE image clé (inpaint + nouveau texte, en x2), puis suivre l'objet
(flux optique Lucas-Kanade + `estimateAffinePartial2D`, affiné par ECC quand c'est meilleur,
choix image par image sur la corrélation des zones de texte), et coller le patch corrigé
déformé par la transformation (`apply_device` dans `pipeline/03c_rendu_pub3.py`).
Caler chaque plan sur ses images de coupe exactes (`cutframes.json`), sinon 1 image d'allemand passe.
