"""
Génération d'un jeu de données SYNTHÉTIQUE de transactions bancaires.

Important (honnêteté) : ces données sont générées artificiellement pour les besoins
du projet (pas de données bancaires réelles, pour des raisons évidentes de
confidentialité). Les features et la logique de fraude s'inspirent de schémas
réalistes utilisés dans les jeux de données publics de détection de fraude
(distance par rapport au domicile, écart par rapport au montant habituel,
utilisation de la puce/du code PIN, achat en ligne, etc.).

Usage : python3 generate_data.py
Produit : transactions.csv
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 20000
FRAUD_RATE = 0.035  # ~3.5% de transactions frauduleuses, réaliste pour ce type de données


def generate(n_samples=N_SAMPLES, fraud_rate=FRAUD_RATE, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    def make_block(n, fraud):
        # Distance par rapport au domicile (km) : les fraudes ont tendance à être
        # commises plus loin du domicile du client.
        distance_from_home = rng.gamma(shape=2.0, scale=15 if fraud else 5, size=n)

        # Distance par rapport à la dernière transaction (km)
        distance_from_last = rng.gamma(shape=1.5, scale=10 if fraud else 3, size=n)

        # Ratio entre le montant de la transaction et le montant médian habituel du client
        ratio_to_median_purchase = rng.gamma(shape=2.0, scale=3.0 if fraud else 0.9, size=n)

        # Montant de la transaction (euros)
        amount = np.round(rng.gamma(shape=2.0, scale=250 if fraud else 60, size=n), 2)

        # Variables binaires de contexte
        repeat_retailer = rng.binomial(1, 0.35 if fraud else 0.85, size=n)
        used_chip = rng.binomial(1, 0.15 if fraud else 0.75, size=n)
        used_pin = rng.binomial(1, 0.05 if fraud else 0.55, size=n)
        online_order = rng.binomial(1, 0.85 if fraud else 0.30, size=n)

        # Heure de la transaction (0-23) : les fraudes sont plus fréquentes la nuit
        if fraud:
            hour = rng.choice(range(24), size=n, p=_night_weighted_probs())
        else:
            hour = rng.choice(range(24), size=n, p=_day_weighted_probs())

        label = np.full(n, 1 if fraud else 0)

        return pd.DataFrame({
            "distance_from_home": distance_from_home,
            "distance_from_last_transaction": distance_from_last,
            "ratio_to_median_purchase_price": ratio_to_median_purchase,
            "amount": amount,
            "repeat_retailer": repeat_retailer,
            "used_chip": used_chip,
            "used_pin_number": used_pin,
            "online_order": online_order,
            "transaction_hour": hour,
            "is_fraud": label,
        })

    df_fraud = make_block(n_fraud, fraud=True)
    df_legit = make_block(n_legit, fraud=False)

    df = pd.concat([df_fraud, df_legit], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", range(1, len(df) + 1))
    return df


def _night_weighted_probs():
    # Plus de poids sur 0h-5h pour les fraudes
    weights = np.ones(24)
    weights[0:6] = 4.0
    weights[6:18] = 1.0
    weights[18:24] = 2.0
    return weights / weights.sum()


def _day_weighted_probs():
    # Plus de poids en journée pour les transactions normales
    weights = np.ones(24)
    weights[0:6] = 0.3
    weights[6:18] = 2.0
    weights[18:24] = 1.0
    return weights / weights.sum()


if __name__ == "__main__":
    df = generate()
    out_path = __file__.rsplit("/", 1)[0] + "/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Jeu de données généré : {out_path}")
    print(f"Lignes : {len(df)} | Fraudes : {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
