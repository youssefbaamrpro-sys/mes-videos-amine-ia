"""Clients prêts à l'emploi pour les vidéos d'Amine.

Les vraies clés API sont ajoutées par le proxy de l'environnement : on passe des
valeurs factices. Le SDK Replicate crée son propre transport httpx, qui ignore
HTTPS_PROXY ; on lui en fournit un explicitement, sinon les appels échouent en 401.

Usage :
    import sys; sys.path.insert(0, ".claude/skills/amine")
    from clients import elevenlabs, groq, replicate
"""
import os

import httpx
import replicate as _replicate
from elevenlabs.client import ElevenLabs
from groq import Groq

_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

elevenlabs = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", "dummy"))
groq = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy"))
replicate = _replicate.Client(
    api_token=os.environ.get("REPLICATE_API_TOKEN", "dummy"),
    **({"transport": httpx.HTTPTransport(proxy=_proxy)} if _proxy else {}),
)

# Les modèles "officiels" (ex. black-forest-labs/flux-schnell) renvoient "version": null,
# que le SDK 1.0.7 refuse (erreur "none is not an allowed value"). On rend le champ optionnel.
from replicate.prediction import Prediction as _Prediction

_field = _Prediction.__fields__["version"]
_field.required = False
_field.allow_none = True

GROQ_MODEL = "openai/gpt-oss-120b"


def telecharger(sortie, chemin):
    """Enregistre une sortie Replicate (FileOutput ou URL) dans `chemin`.

    Ne pas utiliser `sortie.read()` : le SDK envoie l'en-tête Authorization au
    stockage des fichiers générés, qui répond 400. On télécharge l'URL sans clé.
    """
    url = getattr(sortie, "url", None) or str(sortie)
    r = httpx.get(url, follow_redirects=True, timeout=120)
    r.raise_for_status()
    with open(chemin, "wb") as f:
        f.write(r.content)
    return chemin


def televerser(chemin, type_mime):
    """Téléverse un fichier sur Replicate (API files) et renvoie son URL.

    Passer un fichier ouvert à `replicate.run` échoue souvent en 502 sur les
    gros fichiers (vidéo, audio long) : on téléverse d'abord, puis on passe l'URL.
    """
    r = httpx.post(
        "https://api.replicate.com/v1/files",
        headers={"Authorization": "Bearer dummy"},
        files={"content": (os.path.basename(chemin), open(chemin, "rb"), type_mime)},
        proxy=_proxy, timeout=300,
    )
    r.raise_for_status()
    return r.json()["urls"]["get"]


def lancer(modele, entrees, version=None, attente=10, max_essais=6):
    """Crée une prédiction Replicate (gère le 429) et attend la fin. Renvoie la sortie.

    `modele` = "owner/name" (modèle officiel) ; `version` = hash pour les modèles communautaires.
    Lève une erreur claire en cas de 402 (crédit insuffisant).
    """
    import time
    h = {"Authorization": "Bearer dummy"}
    c = httpx.Client(proxy=_proxy, timeout=120)
    url = ("https://api.replicate.com/v1/predictions" if version
           else f"https://api.replicate.com/v1/models/{modele}/predictions")
    body = {"input": entrees, **({"version": version} if version else {})}
    for _ in range(max_essais):
        r = c.post(url, headers=h, json=body)
        if r.status_code != 429:
            break
        time.sleep(12)
    if r.status_code == 402:
        raise RuntimeError("Replicate : crédit insuffisant (402). Ajouter du crédit sur replicate.com/account/billing.")
    r.raise_for_status()
    pid = r.json()["id"]
    while True:
        d = c.get(f"https://api.replicate.com/v1/predictions/{pid}", headers=h).json()
        if d["status"] == "succeeded":
            return d["output"]
        if d["status"] in ("failed", "canceled"):
            raise RuntimeError(f"Replicate {pid} : {d['status']} {d.get('error')}")
        time.sleep(attente)
