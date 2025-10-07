# ClairVent
Épidémiologie urbaine

## Outils

https://github.com/coucougael94/MinageINSEE
https://github.com/coucougael94/WeatherRetriever



## XGBoost (1st result)

Health Crématorium X death effects : 29 septembre 2025.

<img width="1920" height="1080" alt="2025-09-29-010323_1920x1080_scrot" src="https://github.com/user-attachments/assets/84694a24-0b48-4dd1-823c-66224b2b05bd" />
<img width="1920" height="1080" alt="2025-09-29-010333_1920x1080_scrot" src="https://github.com/user-attachments/assets/853d2b49-55c6-4f7a-8035-122585d55d83" />


Données chargées: 3292 lignes,
38 colonnes
Données d'entraînement: 2633 lignes
Données de test: 659 lignes
Entraînement du modèle XGBoost...
Mean Squared Error: 922.639192212711R²
Score: 0.7002957087197559


//Mes deux algos de coeff.
{
    // --- Conversion en vecteurs ---
    auto toVector = [](QLineSeries* s) {
        std::vector<double> v; v.reserve(s->count());
        for (int i = 0; i < s->count(); ++i) v.push_back(s->at(i).y());
        return v;
    };
    auto bleu  = toVector(series.first);
    auto vert  = toVector(series.second);
    auto rouge = toVector(seriesVilleCritique);

    // --- Alignement des séries ---
    // Si les séries représentent des angles (ex: 0° à 359°), ajoute un point à 360° pour fermer la boucle
    auto ajouterPoint360SiBesoin = [](std::vector<double>& v) {
        if (!v.empty() && v.size() == 360) {
            v.push_back(v.front()); // Ferme la boucle à 360°
        }
    };
    ajouterPoint360SiBesoin(bleu);
    ajouterPoint360SiBesoin(vert);
    ajouterPoint360SiBesoin(rouge);

    // Trouve la taille minimale commune
    size_t n = std::min({bleu.size(), vert.size(), rouge.size()});
    if (n == 0) {
        qWarning() << "Aucune donnée valide!";
        return; // ou gérer l'erreur comme tu le souhaites
    }

    // --- Paramètres ---
    const double eps = 1e-6;
    const double seuilBleu = 1e-6;

    // --- Log-ratios conditionnels ---
    std::vector<double> logAvec, logSans;
    for (size_t i = 0; i < n; ++i) {
        if (i >= bleu.size() || i >= vert.size() || i >= rouge.size()) continue;
        if (bleu[i] <= seuilBleu || vert[i] <= eps) continue;
        double logR = std::log((bleu[i] + eps) / (vert[i] + eps));
        if (rouge[i] > 0.0) logAvec.push_back(logR);
        else logSans.push_back(logR);
    }

    // --- Indices robustes ---
    auto medianLog = [](const std::vector<double>& v) { return median_stdvec(v); };
    double medAvec = !logAvec.empty() ? medianLog(logAvec) : NAN;
    double medSans = !logSans.empty() ? medianLog(logSans) : NAN;
    double indice = (!std::isnan(medAvec) && !std::isnan(medSans)) ? std::exp(medAvec - medSans) : NAN;

    // --- Proportion de bleu ---
    int countAvec = 0, countSans = 0, countBleuAvec = 0, countBleuSans = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i >= bleu.size() || i >= rouge.size()) continue;
        if (rouge[i] > 0.0) {
            countAvec++;
            if (bleu[i] > seuilBleu) countBleuAvec++;
        } else {
            countSans++;
            if (bleu[i] > seuilBleu) countBleuSans++;
        }
    }
    double propAvec = countAvec > 0 ? double(countBleuAvec) / countAvec : NAN;
    double propSans = countSans > 0 ? double(countBleuSans) / countSans : NAN;
    double indicePresence = (propAvec > 0 && propSans > 0) ? propAvec / propSans : NAN;

    // --- Affichage ---
    qDebug() << "Proportion de directions avec bleu (rouge présent) / countAvec=" << propAvec << countAvec;
    qDebug() << "Proportion de directions avec bleu (rouge absent) / countSans =" << propSans << countSans;
    qDebug() << "Indice de présence (rapport) =" << indicePresence;
    qDebug() << "log-ratio médiane (avec rouge) =" << medAvec;
    qDebug() << "log-ratio médiane (sans rouge) =" << medSans;
    qDebug() << "Indice d'influence (≈ médiane ratio) =" << indice;




    // Extraction des données des séries en QVector<double>
    auto toQVector = [](QLineSeries* s) {
        QVector<double> v;
        v.reserve(s->count());
        for (int i = 0; i < s->count(); ++i) {
            v.append(s->at(i).y());
        }
        for (int i = s->count(); i< 361;i++)
        {
            v.append(0);
        }
        return v;
    };


    QVector<double> bleu_ = toQVector(series.first);
    QVector<double> vert_ = toQVector(series.second);


    // Supposons que windCoeffs est un double* de 361 éléments
    QVector<double> rouge_;
    rouge.reserve(361);
    for (int i = 0; i < 361; ++i) {
        rouge_.append(windCoeffs[i]);
    }

    // Appel de votre fonction
    double globalInfluence = computeGlobalInfluenceCoefficient(rouge_, vert_, bleu_ );

    // Affichage ou utilisation des résultats
    qDebug() << "Résultat de computeGlobalInfluenceCoefficient:" << globalInfluence;



}
// distance circulaire en degrés entre a et b (0..359)
double MainWindow::circDistance(int a, int b) {
    int diff = qAbs(a - b) % 360;
    return (double) qMin(diff, 360 - diff);
}

// noyau gaussien circulaire
double MainWindow::kernel(double deltaDeg, double sigmaDeg) {
    double z = deltaDeg / sigmaDeg;
    return qExp(-0.5 * z * z);
}

void normalize(QVector<double>& signal) {
    double min_val = *std::min_element(signal.begin(), signal.end());
    double max_val = *std::max_element(signal.begin(), signal.end());
    double range = max_val - min_val;
    if (range <= 0.0) {
        std::fill(signal.begin(), signal.end(), 0.0);
        return;
    }
    for (double& x : signal) {
        x = (x - min_val) / range;
    }
}












double PowerCV::computeGlobalInfluenceCoefficient(
    const QVector<double>& red,
    const QVector<double>& green,
    const QVector<double>& blue,
    double sigmaDeg,
    int K)
{

    const int N = 361;
    QVector<int> redAngles;
    for (int i = 0; i < N; ++i) if (red[i] > 0.5) redAngles.append(i);
    int N_R = redAngles.size();
    if (N_R == 0 || N_R >= 0.8 * N) return 1.0; // Pas d'influence si trop/moins de rouges

    if (redAngles.isEmpty()) return 1.0;

    // 1. Discrétiser green en K classes
    QVector<double> greenSorted = green;
    std::sort(greenSorted.begin(), greenSorted.end());
    QVector<double> thresholds(K-1);
    for (int k = 1; k < K; ++k) {
        int idx = (k * N) / K;
        thresholds[k-1] = greenSorted[idx];
    }

    // 2. Pour chaque classe k, calculer I_k
    QVector<double> I(K, 1.0);
    QVector<int> N_k(K, 0);
    for (int k = 0; k < K; ++k) {
        double mu_R = 0.0, sum_w_R = 0.0;
        double mu_notR = 0.0, sum_w_notR = 0.0;

        for (int i = 0; i < N; ++i) {
            // Déterminer la classe de green[i]
            int k_i = 0;
            while (k_i < K-1 && green[i] > thresholds[k_i]) k_i++;
            N_k[k_i]++;

            // Calculer le poids w(i)
            double w_i = 0.0;
            for (int j : redAngles) {
                double d = circDistance(i, j);
                w_i += kernel(d, sigmaDeg);
            }

            // Mettre à jour mu_R ou mu_notR
            if (red[i] > 0.5) {
                mu_R += blue[i] * w_i;
                sum_w_R += w_i;
            } else {
                mu_notR += blue[i] * w_i;
                sum_w_notR += w_i;
            }
        }

        if (sum_w_R > 0) mu_R /= sum_w_R;
        if (sum_w_notR > 0) mu_notR /= sum_w_notR;
        I[k] = (mu_notR > 0) ? (mu_R / mu_notR) : 1.0;
    }

       // 3. Influence globale
    double C = 0.0;
    for (int k = 0; k < K; ++k) {
        C += (double)N_k[k] / N * I[k];
    }

    // Correction par rareté
    double rarity_factor = 1.0 - pow((double)N_R / N, 2.0);
    double C_corrected = C * rarity_factor;

    return C;
}




### Prochains sujet d'article :
Les emulsifiants (via INSERM).

### A venir :
 -  Approximation précise cré via runge Kutta (Euler récursion 4).
 -  altitude
 -  multi-emissaire (Outlier.s, rouge) 





## Crédits :
Merci à Open Meteo x Observatoire CAMS, UMAP x OSM, Python x Qt, XGBoost x Eigen etc.

XGBoost c'est génial !
