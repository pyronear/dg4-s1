# georefcam

`georefcam` est un package Python permettant la création et l'utilisation de systèmes de géoréférencement pour la géolocalisation de points d'intérêt.

## Contenu

Les trois modules principaux sont :
- `georefcam.py` : contient la classe GeoRefCam utilisée pour la procédure complète de projection de point depuis le repère image vers le MNT dans le repère monde. Il permet également d'évaluer une correction des paramètres d'orientation de la caméra par comparaison d'une image témoin avec le MNT
- `dem.py` : contient les classes utilisées pour la représentation de MNTs ; contient aussi la classe abstraite codifiant leurs APIs
- `camera_model.py` : contient les classes utilisées pour la modélisation de caméras géolocalisées/orientées, et la projection de points depuis le repère image vers des rayons dans le repère monde ; contient aussi la classe abstraite codifiant leurs APIs 

## Utilisation

Le dossier **demo_notebooks** contient des notebooks de test/viz des capacités du package à projeter un point d'intérêt dans le champ de vue d'une caméra sur le MNT environnant (ici brison_4, cf. les [données et galeries d'images de caméras Pyronear](https://drive.google.com/file/d/1GsJIjNyjnZjV2tzMuB0xTZ2hwz-lpRjB/view?usp=sharing)). Les données IGN du département correspondant (Ardèche) au format ASCII grid résolution 25m sont téléchargeables [ici](https://wxs.ign.fr/aqd29otkz2hofiee5pb0fygn/telechargement/prepackage/BDALTI-25M_PACK_FXX_2023-02-01$BDALTIV2_2-0_25M_ASC_LAMB93-IGN69_D007_2022-12-16/file/BDALTIV2_2-0_25M_ASC_LAMB93-IGN69_D007_2022-12-16.7z) au format compressé 7zip.

Les points d'intérêt sont issus du fichier `df_annotations.csv` qui contient les coordonnées de GCPs relevés manuellement afin de pouvoir comparer les projections faites avec une vérité terrain (approximative).

Le dossier contient également
- un notebook qui benchmark la projection de multiple points en coordonnées pixels sur le MNT environnant
- un notebook qui démontre la correction automatique des paramètres d'orientation de la caméra

### Exécution des notebook test

Les étapes nécessaires à l'exécution d'un notebook sont donc les suivantes (depuis le dossier actuel):
- Installation des dépendances nécessaires : `pip install -r requirements.txt`
- Installation du package `georefcam` : `pip install .`
- Téléchargement des [données et galeries d'images de caméras](https://drive.google.com/file/d/1GsJIjNyjnZjV2tzMuB0xTZ2hwz-lpRjB/view?usp=sharing)
- Téléchargement et extraction des [données IGN utilisées](https://data.geopf.fr/telechargement/download/BDALTI/BDALTIV2_2-0_25M_ASC_LAMB93-IGN69_D007_2022-12-16/BDALTIV2_2-0_25M_ASC_LAMB93-IGN69_D007_2022-12-16.7z)
- Édition de la cellule du notebook qui indique les chemins de dossiers contenant les données
- Exécution séquentielle du notebook

Les notebooks a été testé sur Jupyter Lab, dans un environnement virtuel généré à l'aide du fichier `requirements.txt` avec Python 3.10.12 sur Ubuntu 22.04.
