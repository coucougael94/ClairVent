# ClairVent : études sur l'humidité.

La première opération **WeatherRetriever** a permis la constitution d’un fichier de données météorologiques. Ces données proviennent de l’[INSEE](https://www.insee.fr/fr/information/4190491), croisées avec **Open Meteo**, **CAMS Copernicus** et **ECMWF**.

---

## Requêtes et résultats

### 1. Moyenne de l’hygrométrie relative (2019)

**Requête :**
```sql
SELECT AVG(relative_humidity_2m) FROM weather_data;
```

- **Valeur 2019 :** 69,77 %
- **Biais :**
  - Heures de collecte : moyennes calculées entre 8h et 18h (heures actives).
  - Saisonnalité : les décès surviennent majoritairement en hiver, période plus humide.

**Interprétation :**
Les particules fines, via les interactions **Van der Waals**, alourdissent l’eau en suspension. La polarité de l’eau favorise la capture et la décantation des aérosols, modifiant la densité atmosphérique.

**Sources :**
- [ScienceDirect : Interaction particules fines/humidité](https://www.sciencedirect.com/science/article/pii/S0048969724037756)
- [PLOS ONE : Effets de la pollution atmosphérique](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0216550)
- [Atmo-France : Impacts sanitaires de la pollution](https://www.atmo-france.org/article/les-effets-nefastes-de-la-pollution)

---

### 2. Hygrométrie aux derniers instants

**Requête :**
```sql
SELECT AVG(relative_humidity_2m) as Hum FROM weather_data WHERE deltadaydeath=0;
```

- **Observation :** Les décès surviennent lors d’épisodes légèrement plus humides que la moyenne des 30 jours précédents.
- **Biais :** Identiques à ceux cités précédemment.

**Sources :**
- [ScienceDirect : Climat et santé en France](https://www.sciencedirect.com/science/article/pii/S014765132500082X)
- [Données mondiales : Climat en France](https://www.donneesmondiales.com/europe/france/climat.php)

---

### 3. Évolution de l’hygrométrie avant les décès

**Requête :**
```sql
SELECT deltadaydeath, AVG(relative_humidity_2m) as A2 FROM weather_data GROUP BY deltadaydeath;
```

- **Valeurs 2019 :** Augmentation de l’hygrométrie **4 jours avant les décès** (pics aux jours -6, -13, -17, -20, -24).
- **Biais supplémentaire :** Un épisode pluvieux peut influencer les moyennes sur plusieurs jours.

**Interprétation :**
Les décès pourraient être liés à un **cycle inflammatoire** :
- Première inflammation → réponse immunitaire.
- Seconde inflammation (2ᵉ–5ᵉ jour) → risque d’**orage de cytokines**.

**Sources :**
- [France Info : Orages de cytokines et COVID-19](https://www.franceinfo.fr/sante/maladie/coronavirus/coronavirus-quatre-questions-sur-les-orages-de-cytokine-soupconnes-d-etre-responsables-de-cas-graves-de-covid-19_3914925)
- [Nature : Mécanismes des orages de cytokines](https://www.nature.com/articles/s41392-025-02178-y)

---

## Synthèse

En 2019, en France, une majorité des décès pourrait être expliquée par la **pollution de l’air**. Le projet **ClairVent/CxW** étudie :
- Les **cycles d’hygrométrie** (liés aux épisodes pluvieux).
- Les **réponses immunitaires** (orages de cytokines).
- La **pollution atmosphérique** (particules fines et densité de l’air).

**Complément :**
- [Effets de la température et de la pression sur les infarctus](https://www.ahajournals.org/doi/10.1161/01.cir.100.1.e1)


