# VoxCPM2 — RunPod Serverless Worker

Worker RunPod serverless minimal pour [OpenBMB/VoxCPM2](https://github.com/OpenBMB/VoxCPM), construit directement à partir du dépôt officiel (pas de template communautaire tiers).

## Contenu
- `handler.py` — handler RunPod serverless (`runpod.serverless.start`) qui charge `openbmb/VoxCPM2` via la lib officielle `voxcpm` et expose la synthèse texte→audio (avec clonage de voix optionnel).
- `Dockerfile` — image CUDA 12.1 basée sur Ubuntu 22.04, avec les poids du modèle téléchargés au moment du build (pour éviter un cold start trop long).
- `requirements.txt`
- `test_input.json` — exemple d'entrée pour test local.

## Format d'entrée
```json
{
  "input": {
    "text": "Bonjour le monde",
    "prompt_wav_url": "https://.../voix_reference.wav",
    "prompt_text": "Transcription exacte du wav de reference",
    "cfg_value": 2.0,
    "inference_timesteps": 10,
    "denoise": false,
    "normalize": false,
    "seed": null
  }
}
```
Sortie : `{"audio_base64": "...", "sample_rate": 16000}`

## Déploiement sur RunPod

Deux options, au choix :

### Option A — Build via GitHub (recommandé, pas besoin de Docker Hub)
1. Poussez ce dossier dans un dépôt GitHub (public ou privé) vous appartenant.
2. Sur RunPod → Serverless → New Endpoint → **GitHub Repo**, sélectionnez le dépôt et la branche.
3. Renseignez le chemin du Dockerfile (`Dockerfile` à la racine si le dossier est à la racine du repo).
4. RunPod build l'image côté serveur — inutile de builder localement.

### Option B — Build local + push sur un registre (Docker Hub, GHCR, etc.)
```bash
docker build -t <votre_registre>/voxcpm2-runpod:latest .
docker push <votre_registre>/voxcpm2-runpod:latest
```
Puis sur RunPod → Serverless → New Endpoint → **Custom Image**, renseignez l'URL de l'image.

## Configuration du worker (16 Go VRAM, ~0.58 $/h)
Dans l'écran de création d'endpoint RunPod :
- **GPU** : choisissez un GPU avec 16 Go de VRAM correspondant au tarif ~0.58 $/h affiché dans la liste (le tarif exact dépend de la disponibilité/région au moment du déploiement — à vérifier dans l'UI).
- **Container Disk** : ≥ 20 Go (les poids VoxCPM2 + CUDA font plusieurs Go).
- **Workers actifs/max** : selon votre usage (0 actif = scale-to-zero, facturé seulement à l'exécution).
- **Variables d'environnement** (optionnel) :
  - `VOXCPM_MODEL_ID` (def. `openbmb/VoxCPM2`)
  - `VOXCPM_SAMPLE_RATE` (def. `16000`)

## Notes
- VRAM requise par VoxCPM2 : ~8 Go d'après la doc officielle — un GPU 16 Go a de la marge confortable.
- Le denoiser (ZipEnhancer) est chargé par défaut ; désactivable en modifiant `load_denoiser=False` dans `handler.py` si non nécessaire.
